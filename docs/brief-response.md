# Answering the brief

The brief ends with a deliverable, not a task: *"do NOT start coding — respond with (a) your
understanding of the invariants, (b) any place where this brief is underspecified or where you
disagree, and (c) a concrete plan for Milestone 0 only."* This is that answer, written after the work
rather than before it, because the invariants cost something to understand and a restatement of the
brief's own words would be worth nothing.

Every figure below is measured and reproducible, or is explicitly marked as an estimate. Where I
disagree with the brief, there is a measurement attached.

## (a) The invariants, as the work taught them

**I1 — bit-reproducibility.** I took this to be about floating-point determinism. It is not, mostly.
The arithmetic has been right at every turn: the ordered matmul, the vectorised matmul, the eight-wide
unpack and five language features all compose to the contract's exact bytes on the real model, and two
runs of `capital` produced **byte-identical traces**. What actually threatened I1, four times, was
**machinery**: a slot bank rebuilt on every forward so nothing survived a token; a denormal flush the
GPU hardware performed and the CPU did not; a *test fixture* whose synthetic scales contained NaN
payloads the real install never has; and a memory budget that would have swapped the machine. I1 is not
a property of the kernels; it is a property of everything around them, and the only defence that has
worked is asserting it end to end rather than per component.

**I2 — sharding is semantically free.** Not yet testable, and the brief is right that this is the real
gate. What exists is `ShardedSafetensors` and the IR; what does not exist is a second node. I have no
opinion worth having about I2 until M2 runs, except this: because the reduction order is part of the
contract (I1's "fixed ring order, never arrival order"), **the ring membership must be pinned too**, or
two runs at the same N could disagree — the brief implies this and does not say it.

**I3 — discrete decisions match exactly.** Confirmed at fixture scale: the router's top-k index sets are
asserted as a **separate** assertion from the numeric comparison, so a marginal argmax flip cannot hide
behind a passing numeric check. The policy keeps routers at bf16 with the reason written next to it —
because "the output then diverges completely while every per-tensor check still looks green" is precisely
what would otherwise happen.

**I4 — policy is data, not code.** True and verifiable: the policy file is a per-role table with a `why`
block justifying each entry against an invariant, a role missing from it **stops the install** (tested
four times), and the same file serves two model families — `mlp.down`/`mlp.gate`/`mlp.up` appear in the
dense fixture and in no MoE tensor, which is the shared format working rather than rot. The **sharding**
half of I4 does not exist yet; that is M2's.

**I5 — transcode, do not requantize.** Not applicable yet, and saying so is more useful than pretending
otherwise: the M1 checkpoint ships bf16, so dequantize-then-requantize is the **only** option and no
error is being added on top of a vendor's. I5 becomes live at M4, where DeepSeek's FP4 experts exist to
be transcoded from, and the test of whether this project honours it will be whether the on-disk 4-bit
layout ends up close to the vendor's block scaling or merely convenient.

**I6 — provenance in the artifact.** Was **half-unimplemented** when I audited it, in the real install
and in both committed fixtures: `source.files` was `{}` and `revision` was the placeholder `"local"`.
Both are now fixed and pinned by tests that cannot be satisfied by the old behaviour — a digest checked
against an independently computed hash, and a property test that fails if the placeholder returns.

## (b) Where the brief is underspecified, or where I disagree

**1. The slot bank's size is missing, and the number in the code does not fit the machine.**
"Per-layer LRU slot banks" is the requirement; the code says `expertSlotsPerLayer = 16`. One expert is
**12.58 MB** of fp32 (8.39 gate+up, 4.19 down), so 16 slots is **201 MB per layer** and **8.05 GB across
40 layers** — against the brief's own **~4.5 GB usable**. A 1.5 GB cache allows **2.98 slots per layer**.
This is no longer an argument: `SlotBudgetTests` reports it as an **expected failure** with the numbers in
the message, and `SHARD_EXPERT_SLOTS` exists so the size is chosen from a measured hit-rate curve rather
than from taste. I disagree with the literal, not with the design.

**2. The 16 KB alignment and `O_DIRECT` are one requirement, not two.**
L3 asks for experts "contiguous as [gate|up|down], **16 KB aligned**"; the runtime rule names the read
path as `F_NOCACHE` / `O_DIRECT`. 16 KB is what **`O_DIRECT`** needs. This engine reads through
`F_NOCACHE` + `pread`, which needs none, and the install is **64-byte aligned** (0 of 693 offsets fail 64
bytes; **688 of 693** fail 16 KB). So the requirement is deferred *with the read path* — and `D13`
records the single measurement that would bring it back: if `F_NOCACHE` ever stops bypassing the cache,
the install is rewritten and every trace regenerated.

**3. "4–10× the single-node tokens/sec" is aggregate bandwidth, and the brief does not say what it is
4–10× *of*.**
Expert-parallel multiplies aggregate SSD read bandwidth and aggregate cache by N. It multiplies nothing
else: the dense backbone, the norms, the router, the shared expert and the all-reduce are replicated work
on every node. So the speedup approaches N only while **expert reads dominate**, and the honest form is a
read fraction rather than a range. This matters because it changes what to measure first: not tokens/sec,
but **bytes read per token**, which the sweep reports. On this model 256 experts are routed with top-8
and the routed stack is 92.9 % of the parameters, so the case looks favourable — and it is still a
prediction, not a result.

**4. "4 KB payload" is a latency claim, and the brief's own 10–20 ms is consistent with it — which is
the useful part.**
4 KB per layer over 40 layers is 160 KB/token, which on 1 GbE is ~1.3 ms of *wire* time; the brief's
10–20 ms is therefore a **round-trip latency** estimate, and it is the right shape of number: measured
LAN RTT here is **0.49–0.64 ms**, so 40 blocking round-trips is 20–26 ms. The conclusion I draw is
stronger than the brief's: on 1 GbE the all-reduce is not bandwidth-limited at all, it is
**serialisation**-limited, so overlapping it with shared-expert compute is not an optimisation but a
requirement, and a ring should be preferred to a tree for the same reason.

**5. What the brief does not mention and M1 needed.** A checkpoint's **reference build** must be pinned
as well as its revision — a `transformers` upgrade changes the arithmetic a golden trace was captured
against, which is why `tools/requirements-reference.txt` exists and why `compare_reference_modules.py`
does. The brief pins the source commit and would otherwise leave this hole.

## (c) Milestone 0, and its state

M0 asked for: one small dense model, bf16, no quantization, no streaming, matching the HuggingFace
reference **layer by layer**, plus the golden-trace capture tool and the diff harness. **It is done, and
the parts that mattered most were not the model.**

- **The harness came first, as the brief insisted**, and it is what made M1 possible: `datacenter-trace`
  emits a 64-byte-aligned container with a per-tensor sha256 and a canonical digest, `trace_diff.py`
  compares two traces, and `docs/trace-format.md` and `docs/ir-schema.md` are the contracts. When five
  prompts were later run on a 35 B model, comparing them took seconds.
- **The tiny-fixture discipline** is M0's most valuable inheritance and was not in the brief: a fixture
  in **megabytes** runs the same paths — importer, mixture, cache, quantisation, gate — as a 20 GB
  install. It exists because this node has 8 GB and **panicked twice** proving it. Every test added since
  has been fixture-scale for that reason.
- **Layer-by-layer matching** is asserted against a **Python contract** as well as against the reference,
  which is a stronger gate than the brief asked for: bit-exactness is asserted two ways — the trace bytes
  and the discrete decisions — and numeric tolerance is explicitly not a substitute.

The honest caveat: M0's gate is proven at **fixture scale**, not by running a ~1B dense model on this
node, because this node cannot hold one alongside its other work. The fixtures capture the arithmetic;
they do not capture scale. Scale is what M1 tested, and M1's gate is where the project actually stands.
