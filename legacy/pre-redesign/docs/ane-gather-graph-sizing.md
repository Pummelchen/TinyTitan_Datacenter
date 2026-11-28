# Sizing the gather-graph variant: no-go, with the numbers

**Question.** Today the ANE prefill sidecar computes attention *densely* over the
whole context and folds the QSA selection in as an additive mask —
`[chunk, history + chunk]` scores per layer-chunk. The GPU path instead *gathers*
the ~2,051 keys the indexer kept, per query. Can an ANE graph do the same, and
would it win? If it did, it would cut the arithmetic by `total / budget` and, if
the arena is what the per-variant load cost tracks, make the sidecar load faster
too — which is where the 3.8 prefill regression lives
(`benchmark/ane-prefill/README.md`).

**Verdict: do not build it.** The ANE accepts a gather; the arithmetic is not the
problem. A gathered key is replicated across every query that selects it, so the
graph has to materialise `heads × chunk × budget × headDim` values where the
dense graph materialises `heads × chunk × total`. For this geometry that is
**64× more data, 103 GB at the real chunk**, and measured it is slower at the
sizes that build and stops running altogether one step up.

Everything below is measured on Mac15,3 (M3) 24 GB, macOS 27.0, coremltools 9.0,
with `benchmark/ane_gather_probe.py`. Machines and versions matter for ANE
behaviour; re-run the probe rather than quoting these as ceilings.

## The two graphs, at one geometry

Both are pure attention at the shipped 3.8 head geometry (24 heads × 256,
budget 2,051, total 8,192), differing only in how the keys are reached:

| | inputs | scores | per-head intermediate |
|---|---|---|---|
| `dense` (today) | `q`, `k`, `v`, `mask[1,1,T,total]` | `[heads, T, total]` | `T × total` |
| `gather` (proposed) | `q`, `k`, `v`, `idx[T,budget]` i32 | `[heads, T, budget]` | `T × budget × headDim` |

Per head and per query the dense graph holds `total` = 8,192 values; the gather
holds `budget × headDim` = 2,051 × 256 = 525,056. The score matrix *is* smaller
(2,051 vs 8,192), but it is not what has to be materialised to get there.

## What the probe measured

| chunk | graph | ANE ops | load | predict | gather/dense predict |
|---:|---|---:|---:|---:|---:|
| 32 | dense | 13 | 0.133 s | 0.042 s | — |
| 32 | gather | **9** | 0.186 s | 0.238 s | **5.70×** |
| 64 | dense | 13 | 0.144 s | 0.043 s | — |
| 64 | gather | **12** | 2.883 s | **fails** | — |

Records: `benchmark/ane-prefill/ane-gather-probe-v5.6-gather-{32,64}.json`.

Three things matter in that table:

1. **The ANE accepts the gather.** 9 and 12 operations are assigned to the Neural
   Engine, so this is not a "the compiler refuses it" limitation — the gather is a
   cost problem, which is a much harder thing to design around.
2. **It is slower where it runs.** At chunk 32 the gather predicts **5.70× slower**
   than the dense graph at the same geometry, despite its score matrix being 4×
   smaller, because the gathered keys are what move.
3. **It stops working as the chunk grows.** At chunk 64 the gathered keys reach
   1.6 GB and the ANE prediction fails outright — *"Unable to compute the
   prediction using ML Program. It can be an invalid input data or
   broken/unsupported model."* — while the dense graph runs a 1.6 GB score matrix
   at chunk 4,096 without trouble. Extrapolating the tensor the compiler must
   build: 24 × 4,096 × 2,051 × 256 × 2 B = **103 GB** at the real chunk.

## The other lever: the per-variant load

The gather would also have to fix the load, since that is what dominates (24
variant loads against a ~185 s GPU prefill). It does not: the probe's gather
package loads 1.4× slower at chunk 32 and **20× slower** at chunk 64.

The load driver was measured separately, by building the four 3.8 layer-3
variants and merging them in different combinations — loading the *same* function
from packages of different composition:

| package | load h0 | load h4096 | load h8192 | load h12288 |
|---|---:|---:|---:|---:|
| `h0` alone | 6.68 s | — | — | — |
| `h0 + h4096` | 6.94 s | 13.56 s | — | — |
| all four | 6.84 s | 13.66 s | 37.50 s | fails |

So the load is **per variant and proportional to the score arena**
(`heads × chunk × total`) — 6.7 s at 0.8 GB, 13.6 s at 1.6 GB, 37.5 s at 2.4 GB —
and package composition is irrelevant: h0 costs the same alone as beside a
variant that cannot load at all. A gather graph would not shrink that arena; on
the evidence above it enlarges the tensor the compiler has to place.

## What would have to change for the ANE to pay on 3.8

Not a gather. The arithmetic is not what the ANE is losing on:

- the GPU path's sparse attention is budget-bounded and already cheap;
- the ANE's per-prediction advantage on this graph is small (0.43 s on the ANE
  against 0.65 s on the CPU alone, where the 35B's blocks measured 26.7× the
  GPU);
- and the per-variant setup dominates: 24 loads ≈ 245 s against a 185 s GPU
  prefill.

The only lever the numbers leave is **fewer and smaller arenas per request** —
a smaller chunk pays more loads (the runtime's documented trade-off: 16 loads per
covered layer at chunk 1,024 against 4 at 4,096), and fewer covered layers pays
less attention than the model needs. Neither reaches break-even with the load
cost measured here.

**So the shipped answer stands: Qwen 3.8 stays on the GPU, no sidecar is
installed for it, and `tools/ane_sidecars.sh` skips the family.** Re-open this
only with a different ANE cost model (a new chip generation whose per-variant
specialisation is cheap) or an attention implementation that does not have to
materialise gathered keys — and re-run the probe first, because it is cheap.

## Reproducing

```bash
~/.venvs/coreml-py311/bin/python benchmark/ane_gather_probe.py \
    --chunk 32 --repeats 3 --record --label v5.6-gather-32
~/.venvs/coreml-py311/bin/python benchmark/ane_gather_probe.py \
    --chunk 64 --repeats 3 --record --label v5.6-gather-64
```

The probe builds both graphs, converts them for the ANE, reports the operations
assigned to the Neural Engine, and times load and prediction under
`CPU_AND_NE` and `CPU_ONLY`. At the real chunk it cannot build the gather graph
at all, which is the conclusion stated in one line.
