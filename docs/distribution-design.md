# Splitting the workload across four Mac minis — designs and the measurements that decide them

**Status: proposal. No code is written against this document yet.** Every figure is either
measured (and the record that measured it is named) or explicitly marked as a projection.
Nothing here asserts a tok/s that has not been observed.

## 1. What the engine is

This engine is an **SSD streamer**. It reads a model too large for RAM off disk, keeps what
it can in a cache, and computes with what it has. Residency is an optimisation, never an
assumption.

**Qwen3.6-35B-A3B is the test vehicle. The target is Qwen 3.8 at 120–180 B**, which will not
fit in the aggregate memory of this cluster at any quantisation we would serve. A design that
works because *the 35 B happens to fit* is therefore not a design — it would pass the test and
fail the job. Any plan here is judged by whether it scales past RAM.

## 2. The resources

| | |
| --- | --- |
| Compute | four Mac mini M2, 8 GB unified memory each, ~3.5–5 GB usable after macOS |
| GPU | 8-core M2 GPU, ~100 GB/s unified memory bandwidth |
| Local storage | one SSD per node. **Measured** cold sequential 1.65 GB/s, preload 1.82 GB/s |
| Wire | LAN via switch. **Measured** 118 MB/s, 0.565 ms RPC |
| Aggregate | ~14–20 GB of RAM usable as cache, ~6–7 GB/s of streaming bandwidth |

The wire figure is a **single-stream** measurement. It is consistent with 1 GbE, and also with
a faster link limited to one TCP stream. **It has not been measured with parallel streams, and
that measurement is step 3 below**, because it decides whether the latency-bound designs are
viable.

## 3. The law every design obeys

```
tok/s  =  min(  compute_per_stage ,
                local_SSD_rate / (active_bytes_per_token x (1 - hit_rate)) ,
                wire_rate / activation_bytes_per_token  )
```

**Measured inputs, for the 35 B:**

| | value | source |
| --- | --- | --- |
| Active expert bytes per token | 8 x 40 x 1,769,472 B = **566 MB** | expert size measured |
| Hidden state | 2048 x 2 B = **4 KB** | model shape |
| Single-node step | **130 ms** = 7.4 tok/s | measured, 48 tokens, 40 slots |
| Routed MoE share of the step | **19.4 ms of 137.0** | measured |
| Expert read share | **47.4 ms, hidden behind the GPU** | measured; reading a quarter of the experts changes the step by nothing |
| Peer request cost | **2.95 ms**, 37 per token = **109 ms** | measured |

**Three consequences.**

1. **Weights never cross the wire.** 566 MB per token at 118 MB/s is 0.21 tok/s. The wire may
   carry hidden states and nothing else.
2. **Streaming must be partitioned, not shared.** If every node streams the same misses, four
   nodes produce one node's throughput. Each node must stream only its own share.
3. **The cache is the only thing that beats the law.** 566 MB per token at 1.8 GB/s is
   314 ms/token with no cache. **21 tok/s needs roughly a 70% hit rate.** That number is an
   assumption today and is the first thing to measure.

**The target: 21 tok/s = 47.6 ms per step.**

## 4. Design A — layer pipeline, each stage streams its own experts

Node *i* owns ten consecutive layers, holds all 256 experts of those layers on its own SSD,
and keeps its own cache. Activations flow forward; nothing else crosses the wire.

| | 35 B | 120 B (projection, ~3.4x) |
| --- | --- | --- |
| Active bytes/token/stage | 10 x 8 x 1.77 MB = **142 MB** | ~480 MB |
| Stream at 1.8 GB/s, no cache | 79 ms | 267 ms |
| Stream at a 70% hit rate | **24 ms** | 80 ms |
| Compute per stage | **~21 ms** | ~71 ms |
| Wire per token | 3 x 4 KB = 12 KB = **0.1 ms** | 0.1 ms |
| **Projection** | **21–40 tok/s** | **7–12 tok/s** |

**Why it is the right shape.** Both the compute *and* the SSD bandwidth divide by the number of
nodes — the property no other design here has. The wire carries three hidden states per token,
so 118 MB/s is irrelevant. It degrades gracefully: at 180 B, add nodes and both axes scale.

**Where it hurts.** Streaming is on the critical path, so stage time is `max(compute, stream)`
— without overlap the design gains nothing. Each stage's cache is over its own ten layers,
which is a smaller and better-localised working set but also less room.

## 5. Design B — expert-parallel with activation routing

Every node holds the full model on SSD, owns a quarter of the experts, and routes an activation
to whichever node owns the expert it needs. **This is what exists today.**

| | measured / projected |
| --- | --- |
| Wire bytes/token | ~1.9 MB — inside 118 MB/s |
| Requests/token | **37 measured**, at **2.95 ms** each = **109 ms** |
| Dense work | **replicated** on all four nodes: 66.5 ms identical everywhere |
| SSD per node | divides correctly |
| **Measured end to end** | **0.85x a single node** |

**The failure is latency and replication, not bandwidth.** Two defects, both named:

- **Latency.** Thirty-seven round trips per token, each two small dispatches and a wait. A
  batching change that should have been worth a great deal measured **9%**; an eight-slot
  width change measured **11%**. The cost is not in the kernels.
- **Replication.** Attention, shared expert, router and head run identically on four nodes.
  **66.5 ms of a 47.6 ms budget, spent four times over.**

**Verdict.** It streams correctly and its bandwidth scales, but it cannot reach 21 without
solving the round-trip latency *and* dividing the dense work — at which point it has become
Design C or A.

## 6. Design C — tensor-parallel four ways, SSD-streamed

Every tensor split four ways by row; each node streams only its own rows' misses; all-reduce
after each block.

| | value |
| --- | --- |
| Compute/node | 130/4 = **32.5 ms** resident |
| Stream/node | 142 MB -> **79 ms** no cache, **24 ms** at a 70% hit rate |
| Cache locality | **poor** — a quarter of everything, so the hot set is spread across all nodes |
| Wire per token | 320 KB of all-reduce, but **40 layers x 2 collectives x 0.565 ms = 45 ms of latency** |
| **Projection** | **15–22 tok/s**, and only with the collectives fully overlapped |

**Verdict.** The most even division of compute, and the textbook answer — with two liabilities.
**45 ms of collective latency is 95% of the budget**, and row-splitting destroys cache locality:
a node's cache holds a quarter of every layer, so the routing skew that makes caching work is
diluted across the cluster. It trades a streaming problem for a latency problem.

## 7. Comparison

| | compute/node | stream/node (70% hit) | wire latency/token | 35 B projection | scales with nodes |
| --- | --- | --- | --- | --- | --- |
| **A. Layer pipeline** | 21 ms | 24 ms | **1.7 ms** | **21–40 tok/s** | both axes |
| B. Expert-parallel *(today)* | 21 ms + 66 ms replicated | 24 ms | **109 ms** | ~5.5 | bandwidth only |
| C. Tensor four-way | 32.5 ms | 24 ms | **45 ms** | 15–22 | compute only |

**The decisive difference is which term of the law each design attacks.** B divides the SSD but
not the dense compute, and adds latency. C divides both but adds more latency and ruins cache
locality. **A divides compute and SSD and its latency is a rounding error** — 12 KB on a wire
that carries 118 MB/s.

## 8. Routing affinity — placing experts by how they are actually used

The router already decides which eight experts a token uses, per layer, and the engine already
reads those indices at the point where it builds the phase-2 buffer. **The routing trace is
therefore almost free to record**: 40 layers x 8 ids x 2 B = 640 B per token, 82 KB per
128-token prompt. A thousand prompts is 82 MB.

**What it would buy, and it is aimed at a measured cost.** Today a token's eight experts for a
layer are scattered across four nodes, so a layer costs ~3.5 round trips and a token costs 37
requests. If experts that co-activate were co-located, that falls to ~1.2–1.5 nodes per layer
and ~12–15 requests per token — **the serving cost from 109 ms to roughly 40–50 ms, and the
serving node from 5.6 to ~9 tok/s.**

**What it would not buy.** It does not reduce the 566 MB of active bytes per token, and it does
not touch the replicated dense work. **It is a 1.5–2x on the serving path, not a route to 21 on
its own.** It applies to Designs B and C, where an expert's home node is a placement choice.
For Design A it buys something different: **frequency-ordered cache sizing per stage**, since a
stage's experts are already co-located by construction.

**The analyses the trace supports:**

1. **Per-layer expert frequency** — Gini and entropy over the 256 experts. **This is the
   go/no-go number.**
2. **Co-occurrence per layer** — P(*b* in top-8 | *a* in top-8), a 256 x 256 symmetric matrix.
3. **Cross-layer correlation** — does expert *a* at layer *L* predict expert *b* at *L+1*? If
   it does, the unit of placement is a layer band rather than a single layer.
4. **Prompt-class conditioning** — the same matrices split by code, prose, mathematics,
   multilingual and long context.
5. **Partition the affinity graph** into four balanced groups, minimising expected distinct
   nodes touched per layer per token.

**The placement policy that falls out is two tiers, not a flat shard:**

- **Replicate the hot core.** The top *k* experts per layer, by frequency, are copied onto every
  node. If the top 32 of 256 carry most of the mass, that is 32 x 40 x 1.77 MB = **2.3 GB
  replicated**, ~0.6 GB per node.
- **Shard the cold tail by co-activation**, so that what remains tends to arrive together.

**Risks, stated plainly.**

- **Routing may not be skewed enough to matter.** If the per-layer distribution is near-uniform
  the graph has no structure, the partition is arbitrary, and the idea buys nothing.
- **A static map against input-dependent routing.** A placement fitted on one corpus can be worse
  than random on another, so any result must be validated **held-out by prompt class**, never
  fitted and reported on the same data.
- **Context length shifts routing**, so the trace must be taken at the lengths the engine serves.

**And the prior evidence points both ways, which is why it must be measured.** The reference's
cache curve — 5.164, 6.019 and 7.075 tok/s at 1, 2 and 3 GB — is close to *linear*, which argues
for weak skew. But a slot-count sweep measured a **1.78x** lever, which argues there is real
structure to exploit. Those two readings cannot both be right.

## 9. The measurements that decide, in order, before any engine work

**1. Is routing skewed?** Twenty prompts x 128 tokens, a few minutes of a single node's normal
work, ~1.6 MB of trace. The per-layer frequency distribution answers it:

| Gini | reading | decision |
| --- | --- | --- |
| above 0.5 | heavy skew, co-activation likely | build the map and the partition |
| 0.2 to 0.5 | moderate | worth the replication tier, not the graph partition |
| below 0.2 | near-uniform | the idea is dead here — record it and stop |

**2. Does a per-stage cache hold its working set?** Measure the hit rate of a ~3 GB cache over
ten layers x 256 experts. **The whole plan rests on ~70%, and it is an assumption today.**

**3. What is the wire, really?** Four to eight parallel streams between two nodes. 118 MB/s
single-stream may be understating the switch, and this one number decides whether Design C is
viable at all.

**4. Is streaming hideable behind compute?** Stage time with ten layers' reads overlapped
against ten layers of arithmetic. If 142 MB/token of reads cannot hide behind ~21 ms of work,
Design A serialises and gains nothing.

**5. Only then**, a two-node prototype of whichever design survives.

## 10. What would falsify each design

| design | falsified by |
| --- | --- |
| A. Layer pipeline | reads not hideable behind compute; or per-stage cache hit rate far below 70% |
| B. Expert-parallel | per-request latency not reducible below ~0.5 ms once batched |
| C. Tensor four-way | wire latency materially above 0.565 ms per collective, or cache locality worthless |
| Routing affinity | per-layer routing entropy near-uniform (Gini below 0.2) |

## 11. The 35 B caveat that must travel with every result

Ten layers of the 35 B is 4.53 GB of experts, against ~4.5 GB of usable RAM per node — **so the
35 B very nearly fits, and a 35 B run will therefore look better than the target ever will.**
Any 35 B result presented as evidence for the 120–180 B case must be taken **with the cache
deliberately capped** at the fraction the larger model would allow. Otherwise it certifies the
accident, not the design.

## 12. Status against the objective

The objective asks for **21 tok/s across four nodes**. Measured today: **7.377–7.974 tok/s on
one node**, and **0.85x of that across three**. The gap is analysed above and is not closed.
The designs in this document are proposals; **none of them has been built or measured.**
