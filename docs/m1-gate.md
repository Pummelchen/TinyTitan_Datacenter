# The M1 gate

M1's gate is three things: **correct output**, a **recorded tok/s baseline**, and a **measured
cache hit rate**. One command runs all three on one checkpoint and writes a report:

```bash
.venv/bin/python tools/run_m1_gate.py \
    --snapshot .build/hf-cache/models--Qwen--Qwen3.6-35B-A3B/snapshots/<rev> \
    --model Qwen/Qwen3.6-35B-A3B --revision <sha>
```

| | |
| --- | --- |
| Correctness | every prompt in `tools/m1_prompts.json`, the engine's trace against the contract's, byte for byte, through `tools/trace_diff.py` — which also compares the **discrete** router decisions exactly, because I3 makes them a different kind of claim from "the numbers are close" |
| Throughput | greedy generation with `datacenter-generate`, its own timing parsed into the report |
| Cache | the engine's expert-traffic counters, written beside each trace in `metrics.json` |
| Memory | each engine run's **peak resident set size**, from the platform's `/usr/bin/time -l`, because `DC-032`'s gate is a budget and a budget needs a number. It counts clean file-backed pages, so it is an upper bound on the process rather than a claim about private dirty memory |

## THE GATE, on the real model: one prompt, all three parts

`.venv/bin/python tools/run_m1_gate.py --snapshot <35B> --only capital --max-new-tokens 4`

```
capital      IDENTICAL   41.87 s  peak    4.16 GB  hit rate 0
0.0191 tok/s mean over 4 steps
GATE PASSED
```

| Part | Result |
| --- | --- |
| Correct output | **IDENTICAL** — 83 tensors and 40 discrete decisions against the contract, digests equal |
| Peak memory | **4.16 GB** against ~4.5 GB of usable memory per node |
| Cache hit rate | **0.0000** over 2240 requests |
| Throughput | **0.0191 tok/s** — **52.2 s per token** |

The throughput is the number to sit with. It is not a tuning result; it is what a faithful fp32
CPU port with **no KV cache** costs: every generation step re-runs the whole sequence, so a
step's cost grows with the sequence and the mean over four steps after a five-token prompt is
52 s. At a hundred tokens of context that arithmetic is minutes per token. The brief's M0
deferred the cache deliberately — "a cache is a second numeric path through attention and M0's
job is to establish one correct path before there are two" — and that deferral has now been
paid for honestly. **For M1 the KV cache is not an optimisation, it is the next piece of
correctness-preserving work**, and it comes before Metal: a kernel is a constant factor, and
this is a growth term.

The expert traffic behind the token: 2240 requests, **3.52 G elements**, which is **7.05 GB read
from the SSD** in bf16 and 14.1 GB held in fp32.

## First result on the real model — one prompt, M1's correctness claim holds

`capital` (5 tokens) against the real 27 GB checkpoint, engine and contract:

| | engine | contract |
| --- | --- | --- |
| Wall time | **45.6 s** | 461.8 s |
| Peak resident memory | **4.31 GB** | not measured |
| Tensors, discrete | 83, 40 | 83, 40 |
| Digest | `b8c976c5e7ba8816…` | `b8c976c5e7ba8816…` |

`trace_diff` says **IDENTICAL — 83 tensors, 40 discrete decisions checked**, so the engine
reproduces the contract on the 35 B model **byte for byte, with the router's decisions asserted
separately**. That is M1's correctness claim, measured rather than argued, and it is the first
time the whole stack — sharded reader, importer, streaming provider, mixture, head — has run on
the real weights.

Three numbers worth reading carefully:

- **4.31 GB peak resident memory against the brief's ~4.5 GB of usable memory per node.** That is
  at the budget edge, and on macOS the figure includes clean file-backed pages, so most of it is
  the streamed weights the page cache is holding. It is a measured argument for the brief's own
  I/O rule — expert slabs read with `F_NOCACHE`/`O_DIRECT` — which is not yet implemented.
- **Cache hit rate 0.0000**: 2240 expert requests across 40 layers and no hits. A layer's slot
  bank is built as the layer loads and dropped with it, so nothing survives a token, and within
  one token two of five positions rarely agree on all eight experts.
- **Engine 10.1× faster than the contract** on the same prompt (45.6 s against 461.8 s), which is
  the first honest apples-to-apples figure for the fp32 path on this hardware.

Traffic per prompt: 40 layers × **28 distinct experts** × 2 projections = 2240 slices, which is
consistent with 5 positions × top-8 = 40 pairs and no repeats across layers. In fp32 that is
3.52 G elements ≈ 14.1 GB of decoded weights, or ≈ 7.05 GB as the bf16 the shards actually store.

## Status of the instrument

**The instrument is verified, and one prompt has now been measured end to end.** Run against the 236 KB
`tiny-qwen36` fixture, the gate reports `identical everywhere: True` on every prompt and exits
zero, and `tools/test_run_m1_gate.py` drives exactly that — so the gate is not untested code
waiting for the one input that matters. The numbers below are therefore the *shape* of the
report, not M1's result:

| | |
| --- | --- |
| Engine vs contract, tiny fixture | identical, digests equal |
| Cache hit rate, tiny fixture | **0.0000** |
| Throughput, tiny fixture | below the tool's tenth-of-a-second resolution |
| Peak resident memory, tiny fixture | 0.01 GB |

The hit rate of zero is worth reading rather than dismissing: the slot bank belongs to one layer
and is built as that layer loads, so a hit needs two positions **in the same layer of the same
forward pass** to choose the same expert. On a nine-token prompt that happens rarely, and nothing
is reused across tokens. Whether the real model's eight-of-256 routing repeats enough to make the
bank earn its memory is exactly the question M1's gate exists to answer — and if it does not, the
answer is a cache that survives a token, which is a design change and will be recorded as one.

## What the gate does *not* yet claim

- **No KV cache.** Every generation step re-runs the whole sequence, so the mean step time grows
  with it and the recorded figure is a baseline to improve on, not a steady-state rate. The
  report says so in its own `note`.
- **One node.** M2's bit-identity across nodes is a separate gate and the one that matters most.
- **The prompt set is short.** `long` is 67 tokens: beyond one delta-rule chunk, but nothing like
  the 128K the later milestones must hold at. Long-context conversion failures are real and this
  gate does not look for them.

## Status, 2026-09-16: correctness holds on one prompt of five, throughput is superseded

Written because the tables above are the only surviving record of the first real-model run — the
report JSON under `.build/` is gone — and **their throughput figures are no longer current**. Nobody
should quote `0.0191 tok/s` as M1's baseline without reading this section first.

### Proven

| | |
| --- | --- |
| Correct output | **IDENTICAL** on one prompt of five (`capital`, 5 tokens): 83 tensors and 40 discrete decisions, digests equal, through `trace_diff.py` |
| Discrete decisions | exact, asserted separately from the numbers, as `I3` requires |
| The cached path | now agrees with the uncached one on **tokens and margins** (`11751,11,264,3177`; `1.6400, 0.0972, 1.2532, 2.6414`) after the weight-layout fix in the attention cache |
| The 4-bit install | builds (20 GB, peak footprint 4.44 GB), and a full verification reads it uncached in ~20 s at a 33.7 MB peak with free disk steady |

### Superseded, and by how much

The measured throughput above was taken **before** three changes, each of which is bit-identical and
each of which was measured locally:

| change | measured effect |
| --- | --- |
| the `int4` row read decodes **one expert** instead of the whole stack | one expert of eight now costs 1184 of 9472 bytes where it cost all of them |
| the ordered matmul vectorised across outputs | **1.66×** (4.1 → 6.8 GFLOP/s) |
| the int4 unpack vectorised by group and widened to eight codes per load | **1.83×** (648 → 1185 M values/s) |

The first of those is the one that should matter most, and the arithmetic is why: the active experts
are ~2.0 B parameters per token, so unpacking them at the old rate is ~9.3 s, and decoding whole
stacks instead of one expert multiplied that by thirty-two — **~55 s**, which is the measured
52.2 s/step reached independently by a different route.

**So the prediction is that throughput improves by most of an order of magnitude. It is not a
result.** No real-model run has happened since those fixes, and this section exists so that the
stale number and the prediction cannot be confused for one another.

### Not measured, and what would close the gate

- the remaining **four prompts** of `tools/m1_prompts.json`;
- a **re-measured throughput baseline** on the fixed engine, and therefore the M1 gate's actual
  tok/s figure;
- the **cache hit rate** on the fixed engine — the `0.0000` above is real but was taken with the
  whole-stack decode, which changes the traffic it counts;
- the engine running **against the 4-bit install** on the real model, rather than against the
  checkpoint.

All four need the same thing: one real-model run on this node, which needs the operator's approval
because the process peaks at 4.16 GB against about 4.5 GB usable. The watchdog is armed, the disk
floor is enforced, and the run is a few minutes. **M1's gate is therefore open, with its
correctness claim holding on one prompt of five and its throughput claim retired pending that
measurement.**
