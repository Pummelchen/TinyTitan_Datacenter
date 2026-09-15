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
