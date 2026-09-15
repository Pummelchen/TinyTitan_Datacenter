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

## Status

**The instrument is verified; the measurement is not taken yet.** Run against the 236 KB
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
