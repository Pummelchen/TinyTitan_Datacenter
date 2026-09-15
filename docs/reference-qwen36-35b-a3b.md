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

## This family's attention and Gated DeltaNet are `qwen3_5`'s arithmetic — verified

"Matches by name and shape" is not "matches by arithmetic", and reading two 2100-line modules
side by side is not a method. `tools/compare_reference_modules.py` parses both modules, renames
`Qwen3_5Moe` onto `Qwen3_5`, and compares every top-level entity's unparsed body line by line:

```
$ .venv/bin/python tools/compare_reference_modules.py \
    .../qwen3_5/modeling_qwen3_5.py .../qwen3_5_moe/modeling_qwen3_5_moe.py \
    --rename Qwen3_5Moe=Qwen3_5 --allow-differ <reviewed>
OK — 25 entities share their arithmetic, 7 reviewed difference(s)
```

**Identical** — the same arithmetic, byte for byte after renaming:

| | |
| --- | --- |
| `Qwen3_5GatedDeltaNet` | the whole layer, conv and gated norm included |
| `Qwen3_5Attention` | including the per-head `[query \| gate]` split |
| `Qwen3_5RMSNormGated`, `Qwen3_5TextRotaryEmbedding` | |
| `torch_chunk_gated_delta_rule`, `torch_recurrent_gated_delta_rule`, `l2norm` | the rule itself |
| `causal_conv1d_fn`, `causal_conv1d_update`, `apply_mask_to_padding_states` | |
| `apply_rotary_pos_emb`, `rotate_half`, `eager_attention_forward`, `repeat_kv` | |

**The seven differences, all reviewed:**

| Entity | Difference |
| --- | --- |
| `Qwen3_5DecoderLayer` | `Qwen3_5MLP(config, intermediate_size)` becomes `Qwen3_5SparseMoeBlock(config)`, plus two lines unpacking the mixture's tuple. The residual structure is unchanged. |
| `Qwen3_5RMSNorm` | **a decorator**: `@use_kernel_forward_from_hub('RMSNormZeroCentered')` is present in `qwen3_5` and absent in `qwen3_5_moe`. The body is identical. |
| `Qwen3_5ForCausalLM`, `Qwen3_5ForConditionalGeneration`, `Qwen3_5PreTrainedModel` | the wrapper classes: the mixture's parameter names, and the multimodal entry point |
| `Qwen3_5ModelOutputWithPast`, `Qwen3_5CausalLMOutputWithPast` | output dataclasses |

The norm's decorator is worth a second look: **`RMSNormZeroCentered`** is the reference's own
name for the convention M0b found the hard way (a 5 % error at layer 0 from assuming
`weight` rather than `(1 + weight)`). Two independent routes to the same fact, which is what
makes it trustworthy.

The consequence for M1: **the M0b kernels are reusable without modification.** The Gated
DeltaNet, the attention including the output gate, the RoPE, the conv, `l2norm` and the
chunked rule are the same code in both families, so M1's kernel work is the mixture — the
router, the expert stack and the shared expert — and the streaming around it, not a second
implementation of the attention.

## The Gated DeltaNet is not the same shape, and that was two silent bugs

The AST diff above says the two families' `Qwen3_5GatedDeltaNet` is *identical code*. It does
not say the code is *exercised identically*: the arithmetic depends on the configuration, and
this family's configuration is lopsided.

| | `qwen3_5` (2 B) | `qwen3_5_moe` (35 B) |
| --- | --- | --- |
| key heads | 16 | **16** |
| value heads | 16 | **32** |
| key head dim | 128 | 128 |

Two bugs came out of that single asymmetry, both of which the 2 B model could not expose:

1. **The key head count was derived from the value head count.** The IR carried
   `linearValueHeads` and the total `linearKeyDim`, and both the Python contract and
   `Qwen3_5Forward.swift:147` computed `keyHeads = valueHeads`. For this family that makes the
   key head width `2048 / 32 = 64` instead of `128` — every Gated DeltaNet layer wrong, with no
   shape error anywhere to catch it. The IR now carries `linearKeyHeads` as its own field,
   because the two counts are independent facts about the model.

2. **The grouped-query head expansion was missing.** `Qwen3_5MoeGatedDeltaNet.forward:645`:

   ```python
   if self.num_v_heads // self.num_k_heads > 1:
       query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
       key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
   ```

   Each key head serves `32 / 16 = 2` value heads, and without the repeat the delta rule pairs
   the wrong heads. The contract now does it; **the Swift `GatedDeltaNet` still does not** and
   is filed as `DC-038` rather than patched silently — its recurrence is flat over all heads,
   which is valid because the heads are block-diagonal, but the flat vectors have to be
   expanded first.

Both were found by a tiny configuration with **sixteen key heads to thirty-two value heads**,
copied from the real model, rather than the symmetric one the 2 B tests use. That is the
argument for building fixtures from the model's own numbers instead of convenient ones.

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

### The kernel, and what it is checked against

`sources/DatacenterEngine/MixtureOfExperts.swift` implements all of the above and
`MixtureOfExpertsTests` asserts it against golden **bit patterns** emitted by
`tools/ordered_moe.py`: the block's output, the renormalised weights, and — as its own
assertion, because I3 says so — **the chosen experts, in order**. It passes under `-Onone`
and `-O`, so the compiler is not quietly contracting a multiply-add into the fused operation
the contract forbids.

The tie-break is tested directly rather than through a vector that happens to contain no
ties, because Swift's `sorted(by:)` is not a stable sort: the comparator carries the index,
and a tie above the cut has its own case.

### The norms are the same as `qwen3_5`'s

`Qwen3_5MoeRMSNorm:925` initialises its weight to **zeros** and multiplies by `(1 + weight)`,
the offset convention `qwen3_5` uses and `qwen3` does not. A kernel written from the `qwen3`
convention would be wrong here, and the decoder layer is otherwise identical: residual,
mixer, residual, `post_attention_layernorm`, mixture.

## Still to extract — do not guess these

- the `attention_mask` path: what the reference does at padded positions in a batch, which
  M1's single-sequence runs do not exercise but M2's might;
- `full_attention_interval` is consumed by the reference's configuration *constructor* and is
  not an attribute afterwards — `layer_types` is the authoritative list — so the checkpoint's
  `config.json` is the only place the interval can be read from, which is where the importer
  reads it;
- the MTP head, which is M5's feature.
