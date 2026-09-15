# The model IR

The IR is the description of a model that the whole pipeline dispatches on. It is
**data on disk**, not Swift code (L1): a new family is a new spec plus whatever kernels
its ops need, and a vendor point release with unchanged ops is a new spec with a new
revision and the same ops.

Implemented in `sources/DatacenterIR/`:

| File | Holds |
| --- | --- |
| `TensorRole.swift` | the role vocabulary and each role's shape contract |
| `IRSpec.swift` | the spec types, the policy tables, JSON coding |
| `Validation.swift` | every diagnostic a spec can raise |
| `Qwen3Importer.swift` | the `qwen3` name-to-role map (L2) |
| `tests/DatacenterIRTests/IRTests.swift` | 13 tests, run against a real checkpoint's inventory |

## Shape of a spec

```json
{
  "irVersion": 1,
  "family": "qwen3",
  "source": {
    "repo": "Qwen/Qwen3-0.6B",
    "revision": "c1899de289a04d12100db370d81485cdf75e47ca",
    "files": { "model.safetensors": "<sha256>" },
    "passes": [],
    "policies": []
  },
  "config": { "hiddenSize": 1024, "numLayers": 28, "headDim": 128, "…": "…" },
  "blocks": [
    { "id": "embed", "kind": "embedding" },
    { "id": "layer.00", "kind": "decoder", "index": 0 },
    { "id": "final", "kind": "final-norm" },
    { "id": "head", "kind": "output-head" }
  ],
  "tensors": [
    {
      "name": "model.layers.0.self_attn.q_norm.weight",
      "role": "attn.q_norm",
      "shape": [128],
      "block": "layer.00"
    }
  ],
  "policy": {
    "quant": { "attn.q_norm": "bf16" },
    "shard": { "attn.q_norm": "replicate" }
  }
}
```

`config` carries only what the pipeline needs. Nothing downstream reads a vendor's
config file: the importer is the single place that knows a vendor's key names (L2), and
it is also where a vendor's *legacy* key layout is handled — Qwen3 checkpoints still ship
`rope_theta` while `transformers` v5 reads `rope_parameters`, and that mapping belongs in
the importer rather than in every consumer.

## Roles are a closed vocabulary

Dispatch is on **role**, never on tensor name. The vocabulary
(`TensorRole.allCases`) is versioned with the IR, and a spec naming a role this version
does not know is **refused by name**, listing what is known — a new head from a vendor
becomes a conversation, not a silent omission. The same rule applies to importers: a
checkpoint tensor that no role claims is a hard error, because a weight the engine
silently does not use is the worst available failure.

Each role also carries a **shape contract**: `attn.q_norm` is `[headDim]`, not
`[hiddenSize]`; `attn.q` is `[numAttentionHeads × headDim, hiddenSize]`. The importer's
output is checked against the contract before anything downstream trusts it, which is how
a plausible-looking transcription error becomes a diagnostic instead of a wrong answer.
Where a contract genuinely cannot be expressed from the configuration — the n-gram table,
an MTP head, a compressed-KV projection — the contract is `nil` and says so rather than
inventing a shape.

The vocabulary includes roles for the families later milestones run (routed experts,
the router, the Gated DeltaNet, the n-gram table) because I4 requires **one policy
format** to be expressible for every role. Declaring the vocabulary is not building the
pipeline for it: M0 implements the dense roles only.

## Policy is data (I4)

`policy.quant` and `policy.shard` are keyed by role, so changing what is quantized or
what is sharded is a data change. A role in use with no policy entry is a diagnostic —
never a default, because a default is how a policy quietly stops describing the model.
Values: `fp32 · bf16 · fp16 · fp8-block · fp4-block · int4-affine` and
`replicate · shard-by-expert · shard-by-head · shard-by-row`.

## Provenance is in the artifact (I6)

Every spec carries the source repo, the revision hash, a sha256 per source weight file,
and — once conversion has run — the ordered list of transform passes and the policy files
used. A spec built before conversion says `passes: []`, which is a fact worth recording
rather than an absence.

## What is refused, and why it is refused loudly

| Diagnostic | The failure it prevents |
| --- | --- |
| `unsupported-ir-version` | a reader guessing at a format it does not know |
| `duplicate-tensor` | two entries for one weight, where the last one silently wins |
| `unknown-block` | a tensor in a block the execution sequence does not contain |
| `shape-mismatch` | a mapping that is plausible and wrong |
| `missing-policy` | a role in use that no policy describes |
| `layer-count-mismatch` | a spec whose block list disagrees with its own configuration |
| `missing-output-head` | an untied model with no head, which would generate from nothing |

`diagnostics()` returns **all** of them, because an importer that reports one problem per
run turns a ten-second fix into ten runs. `validate()` is the throwing form.

## The importer (L2)

`Qwen3Importer` is a pure name-to-role map: a `switch` on the tensor's own name and
nothing else. No math, no quantization, no reshaping decisions beyond what the role's
shape contract already states. It is tested against the **real** inventory of
`Qwen/Qwen3-0.6B` — 311 tensors with their shapes, dumped from the safetensors header —
because a mapping tested only against names we invented is a mapping of our own
assumptions.

Source of truth for the mapping: the checkpoint's tensor names and the reference
implementation's module structure (`transformers` v5.17.0,
`models/qwen3/modeling_qwen3.py`), recorded in
[`reference-qwen3-dense.md`](reference-qwen3-dense.md).

## Not yet here

- The `qwen3_5` importer for the M0 model (Gated DeltaNet roles are declared, the mapping
  is not written) — `DC-022`.
- A CLI that turns a checkpoint into a spec file on disk — the pipeline half of `DC-022`.
- Op registration in the engine (L4) and the transform passes (L3), which belong to the
  milestone after the first forward pass matches.
