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

Before any of that, the two traces must be **comparable**: if both were captured by the
reference tool and their recorded stacks differ (`transformers`, `torch`, compute and
weight dtype, attention implementation), the comparison is refused rather than performed.
Two builds are not two measurements of the same thing, and this is not theoretical — the
development machine carries an unpinned `transformers 5.16.1` in its system interpreter
beside the pinned `5.17.0` in the project venv. The check runs *before* the digest
shortcut, because identical bytes from different builds can still be incomparable.

Exit status is `0` identical, `1` different, `2` unreadable. A discrete mismatch is
reported even when every float tensor matches, because that is precisely the failure
mode (I3) where all the numbers look right and the model is wrong.

## What M0a has proven so far

The harness is implemented and its seeded-failure suite passes: a one-ULP change is
located to the element, a second change after the first is *not* reported, a transposed
tensor is a shape finding rather than a byte diff, a flipped top-k with identical floats
is caught by the discrete comparator, missing and extra tensors are both caught, a
reordered decision set is caught, and a trace edited after capture is refused.

The reference is captured two ways, and they agree:

| Path | What it does |
| --- | --- |
| `capture` (resident) | builds the model and runs it in memory — the straightforward reference |
| `capture_from_disk` | builds the model on the **meta** device and loads one decoder layer at a time, releasing it after use — the only way an fp32 2 B model fits an 8 GB node (D6) |

Between them they are the **semantic oracle**, not the numeric contract: torch's matmul
accumulates in an order that belongs to its BLAS kernels, and an ordered fp32 sum does not
reproduce it (measured: 3544 of 4096 outputs differ on a real layer's shapes, with both
results equally close to fp64). Bit-exactness is therefore defined against
`tools/ordered_reference.py`, which states the order of every sum, and the torch trace is
used for exact discrete decisions and per-tensor closeness. See D3 in
[`m0-decisions.md`](m0-decisions.md).

## The engine's trace, and the comparison that matters

The engine writes the same container: `datacenter-trace <snapshot> <out> <tokens…>`
produces a trace from `sources/DatacenterEngine`, and the two are compared with the differ
above. `tools/check_engine_contract.py` runs both and compares them, which is M0's central
claim as a single command:

```
[1/4] building the engine (release)
[2/4] running the contract in Python ... 87 tensors, digest 101195ec6be8839c…
[3/4] running the forward in the engine ... 87 tensors, digest 101195ec6be8839c…
[4/4] comparing
      IDENTICAL — 87 tensor(s) ... (matching digests)
      data.bin identical: 7680000 bytes
OK — the engine reproduces the contract exactly
```

Measured on `Qwen/Qwen3-0.6B` at revision `c1899de2…`, 8 tokens, fp32: **87 of 87 tensors
byte-identical**, whole-trace digest
`101195ec6be8839c7f3340d0be444aa1c83aef706283f0b89379d5546d60de4b`, 7,680,000 bytes of
`data.bin` identical, 5.7 s at `-O` (8.3 s unoptimised), peak resident set 2.43 GB.

Two details worth keeping in mind when reading a trace:

- The **digest is computed independently by each implementation** — the Swift writer
  recomputes it over a key-sorted JSON encoding rather than copying the Python value — so
  a matching digest is already two implementations agreeing on every tensor's bytes. The
  raw `data.bin` comparison is run as well, because "the digests agree" and "the bytes
  agree" should not be assumed to be the same statement.
- The comparison runs the differ **and** the byte comparison. A digest shortcut that
  skipped the comparison without one of them would be a gate that reports success without
  looking.

**Results on the real checkpoints**, not on a fixture:

| Model | Result |
| --- | --- |
| `Qwen3-0.6B` (conventional, `qwen3`) | resident and layer-by-layer captures are **bit-identical** — same digest, 87 tensors |
| `Qwen3.5-2B` (the M0 model, 18 Gated DeltaNet layers) | **two independent runs are bit-identical**, 75 tensors, peak RSS **2.78–3.17 GiB** — for weights that are ~9.2 GB in fp32 and cannot be resident at all |

The layer-by-layer capture of the M0 model additionally records what produced it:
`delta_rule_path: chunked (prefill: one full-sequence forward)`, `linear_attention_layers:
18`, and `optional_kernels: {causal_conv1d: false, fla: false}` — meaning the reference
used its own PyTorch fallback rather than a fused kernel. A trace that did not say so
could not be used to gate the fused path, because the two are not guaranteed to agree.

Two checkpoint facts the tool now refuses to gloss over:

- `Qwen3-0.6B`'s config says `tie_word_embeddings: true` while the file **ships a separate
  `lm_head.weight`**. The file wins, and when the config claims tying the two tensors must
  be equal — if they ever disagree, the capture stops rather than guessing which is meant.
- A checkpoint tensor that no module claims is a hard error, not a warning: an unread
  tensor is a silent omission.

The capture also demonstrated the reason that guard exists, at its own expense: a forward
**pre-hook that returns non-None replaces the module's arguments**, and the loader hook was
returning `load_into`'s tensor count. The first real run failed with an `int` where the
hidden state should have been — which is exactly the class of bug the harness exists to
catch, found by running it rather than by reading it.
