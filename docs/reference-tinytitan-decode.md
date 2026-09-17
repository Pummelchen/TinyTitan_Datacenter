# The sister project's decode path — a study, and what to port

**Read, not copied.** The operator authorised reading `TinyTitan` for approach on 2026-09-17. It is
Apache-2.0 and this repository is MIT; the provenance position in `THIRD_PARTY_NOTICES.md` and `D36` is
**unchanged** — no code is copied, no `NOTICE` transfers — and `tools/check_provenance.py` still guards that
offline. This document describes *mechanisms* so that the changes below can be judged on their own arithmetic.
Nothing here is a substitute for the measurements in `docs/`.

## What it measures

The operator's own run of it, on a 35 B-A3B at 4-bit on node3 (8 GiB, M-series, macOS 27), 46-token prompt and
512 generated tokens, **identical output digest `bfe8fa42b239e55b` across all four configurations**:

| expert cache | decode tok/s | expert hit | I/O hidden | GPU busy | occupancy | host wait/token |
| --- | --- | --- | --- | --- | --- | --- |
| 1 GB | 5.164 | 60.5% | 8.1% | 34.5% | 32.9% | — |
| 2 GB | 6.019 | 72.3% | 11.1% | 40.1% | 37.9% | — |
| 3 GB | **7.075** | 79.8% | 21.4% | 45.7% | 42.8% | 55 ms |
| 4 GB | 2.756 | 84.6% | 46.6% | 17.9% | 17.4% | 219 ms |

The 4 GB row is the warning: the highest hit rate produced the slowest run, because a 4 GB wired cache plus the
dense weights, KV and prompt cache no longer fit in 8 GiB and the machine swaps against a cache its own
pressure paged out. TTFT is flat across all four (4.91-5.36 s), so **prefill is not the lever — expert I/O is.**

## The four mechanisms

Named by file, because each is a separable idea:

1. **`Runtime/Inference/ExpertPrefetchRing.swift`** — a fixed ring of raw-byte `MTLBuffer` slots that holds
   *predictions*, not cache entries. A completed prediction becomes resident only if the router actually selects
   that expert, and a wrong prediction is discarded without disturbing the cache. This is the overlap: bytes for
   the next token are in flight while the current token computes.
2. **`Infrastructure/Streaming/ExpertResidencyTable.swift`** — a GPU-visible table of `(slot, state,
   generation)` per `(layer, expert)`, with explicit `empty` / `loading` / `resident` states. The decode kernel
   reads it to find the slot for a selected expert, which is what lets the expert pool live in **GPU memory**
   rather than in a Swift array.
3. **`Infrastructure/Streaming/PreadExpertStreamer.swift`** — `pread`-based streaming with a **fixed per-layer
   slot cache**; misses are fanned across threads with `DispatchQueue.concurrentPerform`, and slot state is
   published only after every direct read finishes, so a concurrent planner never sees partial bytes as
   resident.
4. **`Runtime/Configuration/RuntimeConfiguration.swift`** — the sizing rule. `affordableExpertCacheBudget` is
   `min(wanted, physicalMemory / 2)` as a **hard clamp**, with the launcher's "at most 30% of physical RAM"
   warning as advisory guidance on top; slots are derived from the model's own `expertStride`, layer count and
   that budget.

## What this engine does instead, and the mapping

The operator's precondition is explicit (`D97`, `DC-117`): **7 tok/s decode on one Mac mini M2 before any
further network work.** This engine is at **0.230 tok/s**. Four changes, ordered by risk-adjusted value, each of
which is numerically neutral or independently verified:

| # | Change | Their mechanism | Why it is safe | Task |
| --- | --- | --- | --- | --- |
| 1 | Fan expert misses across threads instead of reading them one at a time | `PreadExpertStreamer`'s `concurrentPerform` | Reads are independent; the bytes and their order of arrival do not affect any value. The same shape as the dequantiser change in `D94`, which was verified by `Int4UnpackTests` | `DC-118` |
| 2 | Let the expert bank **outlive the layer** — a generation-scoped, `(layer, expert)`-keyed bank sized by `SHARD_EXPERT_BANK_MB`, plus the hit/bytes/seconds metrics this engine does not report at all | `ExpertResidencyTable` + the per-layer slot cache | Caching a decoded weight cannot change a value; the trace digest is the check. Today the bank is built inside `loadLayer` and dropped with the layer, which is why `D31` measured a hit rate of 0 — the lifetime, not the size | `DC-119` |
| 3 | Decode selected experts with an int4 GEMV straight from resident slots, never materialising fp32 | `Kernels/MoE/` + `MetalExpertReader`/`MetalExpertStagingPool` | `MetalMatmul` is already asserted bit-identical to `Ops.orderedMatmul` over 288 shapes (`D61`), so the accumulation order is preserved by construction; the trace digest gates it | `DC-120` |
| 4 | Prefetch the next token's likely experts while the current token computes | `ExpertPrefetchRing` + `ExpertIOEventCoordinator` | A prediction is a hint: it may only ever *add* resident bytes, never change a value. It is worth doing only after 1-3, because there is no point prefetching into a bank that dies with the layer | `DC-121` |

The first two are the ones the reference's own curve quantifies: its hit rate rises 60.5 → 79.8% from 1 → 3 GB
and is worth **+37%**, and it starts from a bank that *can* hit. Ours starts from 0% by construction, so the
first resident bytes are worth far more than the reference's marginal ones.

**The memory ceiling is not a tuning knob, it is the failure mode.** The reference's 4 GB run is 2.6x slower
than its 3 GB run *on the same machine*. This node has ~4.5 GB usable, a 4 GB memory watchdog (`D54`) and a
5 GB disk floor, so the bank is capped well below the reference's 3 GB until the dense payload, KV and prompt
cache are measured beside it — and the measurement to make is theirs: **hit rate and host wait per token at
each size**, not tok/s alone.
