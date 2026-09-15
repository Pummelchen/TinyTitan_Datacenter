# M1 reference contract — Qwen3.6-35B-A3B (`qwen3_5_moe`)

The validation model for M1. Everything here is read from the checkpoint's own configuration
and `safetensors` headers at `Qwen/Qwen3.6-35B-A3B` (Apache-2.0), not from the model card:
the card states no parameter counts, and the brief's numbers are checked against the shapes
rather than assumed.

## Provenance

| | |
| --- | --- |
| Repository | `Qwen/Qwen3.6-35B-A3B` (Apache-2.0) |
| `model_type` | `qwen3_5_moe` — the text tower is `qwen3_5_moe_text` |
| Tensors | 1045 in 26 shards, 67 GiB of bf16 weights |
| Inventory | `tests/DatacenterIRTests/Fixtures/qwen36-35b-a3b-tensors.json`, read from the shard headers by `tools/make_qwen36_fixture.py` — no weights downloaded |

## Parameters, counted from the shapes

| | |
| --- | --- |
| Text tower | **34.66 B** |
| Routed experts (40 layers) | **32.21 B** — **92.9 %** of the model |
| Dense, always active | **2.45 B** (of which 1.02 B is the embedding and the untied head) |
| Active experts per token | 1.01 B (top-8 of 256 is 1/32 of the stack) |
| **Active per token** | **3.45 B** |

So the brief's "35B total / A3B active" is **right**, and the reason it is 3.45 rather than
3.0 is that 1.02 B of the always-active part is the embedding and the head. An earlier
arithmetic slip of mine subtracted the experts of only the thirty Gated DeltaNet layers and
produced 11.26 B; the experts are in **all forty** layers. The correction is recorded here
because the number is quoted in planning.

## Architecture, from the checkpoint

| | |
| --- | --- |
| Layers | 40: **30 Gated DeltaNet + 10 full attention** (`full_attention_interval: 4`, so indices 3, 7, … 39) |
| Hidden / vocab | 2048 / 248320 |
| Attention | 16 heads, 2 KV heads, `head_dim` 256, query carries its output gate |
| Linear attention | 16 key heads, **32 value heads**, head dims 128, conv kernel 4 |
| Experts | **256 routed, top-8**, intermediate 512, plus a shared expert of 512 with a scalar gate |
| Head | `lm_head.weight` present: this family is **not** tied |
| Norms | `A_log` and the Gated DeltaNet's `norm` are stored **bf16** here, where `qwen3_5` stores them fp32 |

Every layer is a mixture of experts — there is no dense `intermediate_size` in the
configuration at all.

## The layout that differs from `qwen3_5`, and why it matters

The routed experts are **stacked**, one tensor per projection:

| Tensor | Shape | Meaning |
| --- | --- | --- |
| `mlp.gate.weight` | `[256, 2048]` | the router |
| `mlp.experts.gate_up_proj` | `[256, 1024, 2048]` | all 256 experts, **gate and up fused** |
| `mlp.experts.down_proj` | `[256, 2048, 512]` | all 256 experts |
| `mlp.shared_expert.{gate,up,down}_proj.weight` | `[512, 2048]`, `[512, 2048]`, `[2048, 512]` | the shared expert |
| `mlp.shared_expert_gate.weight` | `[1, 2048]` | the scalar gate on the shared expert's output |

The IR therefore has **two sets of expert roles**: `expert.stack_gate_up` and
`expert.stack_down` for what a checkpoint ships, and `expert.gate`, `expert.up`,
`expert.down` for the per-expert layout L3's repack pass *produces*. An importer maps names
to roles and does not reshape, so the two layouts are two roles rather than one role plus a
convention — and a checkpoint whose stack held only the gate would be refused by the shape
contract, which is exactly what a test asserts.

## Excluded by a declared prefix

| Prefix | What it is | Count |
| --- | --- | --- |
| `model.visual.` | the vision tower — a different architecture from `qwen3_5`'s, with biased LayerNorms | 333 |
| `mtp.` | the multi-token-prediction head (`fc`, two pre-norms, one attention layer, `norm`) | 19 |

333 + 19 + 693 = 1045, the whole checkpoint. Both groups are in the file and neither is in M1. As with `qwen3_5`, the exclusion is a named
prefix with a reason and the tests assert that every tensor is either mapped or excluded.

## Still to extract before the kernels are written — do not guess these

- the router's exact arithmetic: whether the logits are computed in fp32 or the model dtype,
  whether `norm_topk_prob` is set for this checkpoint, and how the shared expert's scalar
  gate combines with the routed sum. I3 makes the top-8 index set a discrete decision that
  must match exactly, and it is the first thing quantisation breaks;
- whether the stack's `gate_up` halves are `[gate | up]` or interleaved, which the shape does
  not say;
- `intermediate_size` is absent from the configuration, so anything that assumed a dense MLP
  width would silently use the MoE width;
- the Gated DeltaNet and attention are the `qwen3_5` ones by name and shape — but "by name and
  shape" is not "by arithmetic", and this family's own reference file must be read before the
  kernels are trusted.
