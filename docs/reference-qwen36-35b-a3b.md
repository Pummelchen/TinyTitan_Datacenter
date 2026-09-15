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

## The mixture of experts, transcribed (`Qwen3_5MoeSparseMoeBlock:903`)

Read from `transformers` v5.17.0 `models/qwen3_5_moe/modeling_qwen3_5_moe.py`.

```python
shared = shared_expert(hidden)                       # an ordinary gate/up/down MLP
_, routing_weights, selected = gate(hidden)          # below
routed = experts(hidden, selected, routing_weights)  # below
shared = sigmoid(shared_expert_gate(hidden)) * shared
return routed + shared                               # the shared expert is added, not ranked
```

### The router (`Qwen3_5MoeTopKRouter:884`)

```python
router_logits = F.linear(hidden, weight)                              # (tokens, experts)
router_probs  = softmax(router_logits, dtype=torch.float, dim=-1)     # ← FP32, explicitly
router_top_value, router_indices = torch.topk(router_probs, top_k, dim=-1)
router_top_value /= router_top_value.sum(dim=-1, keepdim=True)        # ← renormalised
router_top_value = router_top_value.to(router_logits.dtype)           # ← back to the model dtype
```

Four things worth stating plainly:

- **The softmax is fp32** while everything around it is bf16. That is a fp32 island in the
  same sense as the attention softmax, and the router is exactly where I3 says not to
  economise.
- **The top-k is taken on the probabilities, not the logits.** Equivalent in ordering,
  different in the numbers that end up being compared.
- **The top-k weights are renormalised** to sum to one over the chosen experts. Note that
  this is *unconditional* here: the implementation does not consult `norm_topk_prob`, so a
  checkpoint whose configuration disabled it would still be renormalised.
- **The weights are cast back to the model dtype** before use, so in a bf16 run the router's
  arithmetic ends at bf16 even though its softmax did not.

**Ties are not specified by the reference.** `torch.topk` does not promise which of two equal
probabilities comes first, so the *order* of a tied top-k is implementation-defined there.
The contract states its own rule — **lowest expert index first**, the same rule `argmax` uses
— because I3 requires the index *set* to be comparable and a set with an undefined order is
not.

### The experts (`Qwen3_5MoeExperts:845`)

```python
for expert_idx in expert_hit:                        # ascending index order
    top_k_pos, token_idx = where(expert_mask[expert_idx])
    gate, up = linear(x, gate_up_proj[expert_idx]).chunk(2, dim=-1)   # ← [gate | up]
    out = act(gate) * up
    out = linear(out, down_proj[expert_idx])
    out = out * top_k_weights[token_idx, top_k_pos, None]
    final.index_add_(0, token_idx, out)
```

- The fused stack is **`[gate | up]`** — the first half is the gate — which the shape could
  not say and the code does.
- The accumulation runs in **ascending expert index**, not in top-k rank order. That is
  exactly what D4 chose for the distributed reduction ("ascending global expert id"), so the
  single-node contract and the future ring reduction agree by construction rather than by
  coincidence.

### The norms are the same as `qwen3_5`'s

`Qwen3_5MoeRMSNorm:925` initialises its weight to **zeros** and multiplies by `(1 + weight)`,
the offset convention `qwen3_5` uses and `qwen3` does not. A kernel written from the `qwen3`
convention would be wrong here, and the decoder layer is otherwise identical: residual,
mixer, residual, `post_attention_layernorm`, mixture.

## Still to extract — do not guess these

- the `attention_mask` path: what the reference does at padded positions in a batch, which
  M1's single-sequence runs do not exercise but M2's might;
- whether the Gated DeltaNet and attention blocks in *this* module differ in any arithmetic
  from `qwen3_5`'s beyond the names — the shapes and the class names match, and "matches by
  name and shape" is not "matches by arithmetic". The transcription above covers the mixture
  only, and the two attention families must be compared line by line before M1's kernels
  are trusted;
- the MTP head, which is M5's feature.
