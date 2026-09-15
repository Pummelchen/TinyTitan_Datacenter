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

What is **not** yet done: the torch-side capture tool that fills a trace from the real
model, and the end-to-end bit-match against a conventional model (`DC-024`, `DC-028`).
