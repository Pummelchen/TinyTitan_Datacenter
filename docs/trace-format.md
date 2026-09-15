# The golden-trace format

The trace is what every M0 gate compares against, so the format is part of the
contract, not an implementation detail. It is implemented in
`tools/trace_format.py` (container), `tools/trace_diff.py` (comparison) and
`tools/make_synthetic_trace.py` (a model-free fixture); the tests that pin its
behaviour are in `tools/test_trace_harness.py`.

## Layout

```
<trace>/
  manifest.json   schema, provenance, and an ordered index of every tensor
  data.bin        the tensors, concatenated, each 64-byte aligned
```

A trace is a directory rather than an archive because both sides have to read it: the
Swift engine reads `data.bin` with one `mmap` and one offset per tensor, with no
archive decoder and no JSON parse per element.

## Why each rule exists

| Rule | The failure it prevents |
| --- | --- |
| Floats are raw little-endian words (`f32`, `f16`, `bf16`), never text | Decimal round-tripping is a lossy step in the middle of a bit-exactness gate, and it makes ULP distances meaningless |
| Every tensor carries its own sha256 in the index | A trace edited after capture would still compare; `Trace.verify()` refuses it instead |
| The manifest carries a whole-trace digest over the canonical index | I1 ("same input, same bytes") becomes one string comparison, and a doctored index cannot hide |
| Tensors are indexed explicitly — name, offset, nbytes, shape, dtype | No implicit ordering, so a truncated or reordered file is a validation failure rather than a silent mis-comparison |
| Discrete decisions live in their own section, with values and shape | Router top-k sets are compared for exact equality, never with a tolerance (I3); they can never "almost match" |
| The producer records the reference stack and the model revision | A trace is only comparable to another captured from the same reference build (I6) |

## The index

Each entry in `manifest.tensors`:

| Field | Meaning |
| --- | --- |
| `name` | dotted path: `embed.out`, `layer.03.hidden_in`, `layer.03.mlp_out`, `final_norm.out`, `logits` |
| `dtype` | `f32` · `f16` · `bf16` · `i32` · `i64` · `u8` |
| `shape` | the tensor's logical shape; `nbytes` must equal `elements × dtype_size` |
| `offset`, `nbytes` | byte range inside `data.bin` |
| `sha256` | hash of that byte range |

`manifest.discrete` entries carry `{name, shape, values}`, for example
`layer.03.router.topk`. A trace also records which algorithm produced it where the
reference has more than one — for the Gated DeltaNet, the chunked prefill path and the
recurrent decode path are different algorithms that must agree, so a trace that does
not say which one it came from cannot be used to gate the other.

## Comparison rules

`tools/trace_diff.py` reports, in this order:

1. tensors present in the reference and absent in the candidate, and vice versa;
2. a `dtype` or `shape` mismatch for a tensor that exists in both;
3. the **first** byte-level difference, located to the tensor, the element index, both
   values and (for floats) the fp32 ULP distance. Scanning stops there: a divergence at
   layer 3 makes everything after it noise;
4. every discrete decision, compared as an exact sequence — including order, because
   the routed order is part of what a kernel must reproduce.

Exit status is `0` identical, `1` different, `2` unreadable. A discrete mismatch is
reported even when every float tensor matches, because that is precisely the failure
mode (I3) where all the numbers look right and the model is wrong.

## What M0a has proven so far

The harness is implemented and its seeded-failure suite passes: a one-ULP change is
located to the element, a second change after the first is *not* reported, a transposed
tensor is a shape finding rather than a byte diff, a flipped top-k with identical floats
is caught by the discrete comparator, missing and extra tensors are both caught, a
reordered decision set is caught, and a trace edited after capture is refused.

`tools/trace_capture.py` fills a trace from the reference: one forward pass with hooks at
every layer boundary, fp32 compute, `attn_implementation="eager"`, deterministic
algorithms, a pinned thread count and seed — all recorded in the trace. It has a `--tiny`
mode that builds a small random `qwen3` from a config, so the plumbing and the
reproducibility question were settled without a 4.5 GB download.

Measured on that tiny model (4 layers, 64 wide):

| Comparison | Result |
| --- | --- |
| fp32, same configuration, run against run | **bit-identical** — the reference obeys I1 |
| fp32, 1 thread versus 4 threads | **bit-identical** — measured, not assumed; the real checkpoint re-measures it |
| fp32 versus bf16 | every element differs, and the absolute divergence grows with depth (1.2e-4 at the embedding → 2.8e-2 at the final norm, on values of scale 2.7) |

The last row is why the gate is bit-exactness rather than a tolerance: relative error on
these activations reaches 13285% because the values themselves sit near zero, so a
relative bound is either meaningless or impossible, and no tolerance can see a top-k
index flip at all.

Two traces whose recorded reference stacks differ are **refused** rather than compared.
This is not theoretical: the machine this was developed on carries an unpinned
`transformers 5.16.1` in its system interpreter and a pinned `5.17.0` in the project venv,
and a gate that mixed them would report a difference that belongs to the reference, not
to the engine.

What is **not** yet done: running the capture against the real checkpoint (it needs the
memory-mapped per-layer path of `DC-021`), and recording which delta-rule path produced a
Gated DeltaNet trace (`DC-024`).
