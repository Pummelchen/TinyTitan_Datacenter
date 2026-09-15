# M0 reference contract — Qwen3.5-2B (`qwen3_5`)

**Status:** verified against the reference implementation, 2026-09-15. Nothing is
implemented yet. Every statement carries the file and line it came from; nothing here is
inferred.

This is the contract for the **M0 model** chosen in `D1`. The conventional-transformer
contract used for the harness-validation stage (M0a) is
[`reference-qwen3-dense.md`](reference-qwen3-dense.md).

## Provenance

| | |
| --- | --- |
| Model | `Qwen/Qwen3.5-2B` |
| Model revision | `15852e8c16360a2fea060d615a32b45270f8a8fc` |
| Checkpoint | one shard, 4,548,221,488 bytes, `Qwen3_5ForConditionalGeneration` |
| Reference implementation | `transformers`, tag `v5.17.0` |
| Reference file | `src/transformers/models/qwen3_5/modeling_qwen3_5.py` |
| File sha256 | `762feb6c7426a7f15b5bf830df54c07438bf9e7c27b8cdb23179045920412c3b` |
| Config sha256 | `19966c3200cee92cc4bccaef5f94550c56d731dbf03858a9bd2ed6aebcc3f7da` |

## Architecture, from the checkpoint's own config

24 layers, alternating **three `linear_attention` then one `full_attention`** — 18 Gated
DeltaNet layers and 6 full-attention layers. Dense MLP (`intermediate_size: 6144`), tied
embeddings, one MTP layer, hidden 2048, `rms_norm_eps: 1e-6`, context 262,144, vocab
248,320. It is **not** a conventional transformer, and the first layer is already a
Gated DeltaNet.

**It is a vision-language checkpoint.** `modeling_qwen3_5.py` carries a full vision tower
(`Qwen3_5VisionModel:1124`, patch embed `:961`, merger `:981`, vision blocks `:1094`) and
the repo ships `preprocessor_config.json` and `video_preprocessor_config.json`. The
importer maps **the text tower only** (`Qwen3_5TextModel:1222`); vision tensors must be
filtered by name rather than loaded and ignored.

## The decoder layer

`Qwen3_5DecoderLayer:861` — pre-norm, token mixer selected by `config.layer_types[i]`,
then MLP, two residual adds against the unnormalized input:

```
residual = x
x = input_layernorm(x)
x = linear_attn(x) | self_attn(x)     # by block type
x = residual + x
residual = x
x = post_attention_layernorm(x)
x = mlp(x)
x = residual + x
```

## The Gated DeltaNet layer (`Qwen3_5GatedDeltaNet:504`)

Parameters, exactly as the reference constructs them:

| Parameter | Shape / value |
| --- | --- |
| `in_proj_qkv` | `Linear(hidden → key_dim*2 + value_dim)`, **one fused projection** (key_dim = 16×128 = 2048, value_dim = 16×128 = 2048) |
| `in_proj_z` | `Linear(hidden → value_dim)` — the gate |
| `in_proj_b` | `Linear(hidden → num_v_heads)` — beta |
| `in_proj_a` | `Linear(hidden → num_v_heads)` — decay pre-activation |
| `conv1d` | `Conv1d(conv_dim, conv_dim, kernel_size=4, groups=conv_dim, bias=False, padding=3)` — **depthwise**, causal |
| `dt_bias` | `Parameter(ones(num_v_heads))` |
| `A_log` | `Parameter(log(uniform(0.01, 16, num_v_heads)))` — bounded away from 0 |
| `norm` | `Qwen3_5RMSNormGated(head_v_dim=128, eps=1e-6)` |
| `out_proj` | `Linear(value_dim → hidden, bias=False)` |

Forward, in order, with the dtype boundaries the reference actually uses:

| # | Step | Source | dtype |
| --- | --- | --- | --- |
| 1 | `mixed_qkv = concat(q, k, v)` through the depthwise causal conv | `:504` | bf16 |
| 2 | split into query / key / value | `:504` | bf16 |
| 3 | `beta = b.sigmoid()` | `:504` | bf16 |
| 4 | `g = -A_log.float().exp() * softplus(a.float() + dt_bias)` — **computed in fp32** | `:504` | fp32 |
| 5 | `torch_chunk_gated_delta_rule(..., chunk_size=64, use_qk_l2norm_in_kernel=True)` (prefill) or `torch_recurrent_gated_delta_rule(...)` (decode) | `:301`, `:438` | fp32 internals, bf16 I/O |
| 6 | `core_attn_out = norm(core_attn_out, z)` — **gated** RMSNorm | `:218` | see below |
| 7 | `out_proj` | `:504` | bf16 |

**`Qwen3_5RMSNormGated:218` — the exact order**, which is not the same as the plain norm:

1. variance in **fp32** (`x.to(float32).pow(2).mean(-1)`), normalize with `rsqrt(var+eps)`;
2. cast back to the input dtype, then multiply by the bf16 weight;
3. multiply by `silu(gate.to(torch.float32))` — the gate is activated **in fp32**, so this
   multiply promotes to fp32;
4. cast the result back to the input dtype.

**Two delta-rule implementations exist** (`:301` chunked for prefill, `:438` recurrent for
decode) and they are different algorithms achieving the same recurrence. The engine must
match **both**, and prefill and decode must agree with each other bit-for-bit, or I1 fails
at the prefill/decode boundary rather than between nodes. Which path a trace came from
therefore belongs in the trace header.

## What the checkpoint itself settles (verified 2026-09-15)

The tensor names and shapes below are read from the safetensors header of
`Qwen/Qwen3.5-2B` at revision `15852e8c…`, not from a description of it. They are what the
importer maps, and `tests/DatacenterIRTests/Qwen3_5ImporterTests.swift` checks all 632 of
them.

| Fact | Value |
| --- | --- |
| Text tower prefix | `model.language_model.` (the model is a conditional-generation wrapper) |
| Text tensors | 320 = 1 embedding + 18 × 14 GDN + 6 × 11 attention + 1 final norm |
| Full-attention layers | exactly those with `index % 4 == 3` — 3, 7, 11, 15, 19, 23 |
| The head | **absent**: `tie_word_embeddings: true` and no `lm_head.weight` in the file |
| Vision tower | `model.visual.` — 297 tensors, excluded by a declared prefix |
| MTP head | `mtp.` — 15 tensors (`fc`, `pre_fc_norm_embedding`, `pre_fc_norm_hidden`, `norm`, one attention layer), excluded: present in the file, not part of M0 |

GDN layer shapes (`model.language_model.layers.0.*`): `in_proj_qkv` `[6144, 2048]` —
Q and K are 16 heads of 128, V is 16 heads of 128, concatenated; `in_proj_z` `[2048, 2048]`;
`in_proj_a` and `in_proj_b` `[16, 2048]`; `conv1d` `[6144, 1, 4]`; `norm` `[128]` **stored
fp32**; `A_log` `[16]` **fp32**; `dt_bias` `[16]` bf16.

Full-attention layer shapes (`layers.3.*`): `q_proj` `[4096, 2048]` — twice the query width,
because `attn_output_gate: true` and the two halves are `[query | gate]`; `k_proj`,
`v_proj` `[512, 2048]`; `o_proj` `[2048, 2048]`; `q_norm`, `k_norm` `[256]`.

That last row is the reason the IR grew an `attnOutputGate` flag: a shape contract that
assumed a bare query would have rejected the checkpoint, and a guess about which half is
the gate would have produced a plausible, wrong forward pass. The **layout** (`[query |
gate]`) is established by the shape; the **application order** is still on the list below.

## The attention output gate, exactly (`Qwen3_5Attention.forward:776`)

```python
query_states, gate = torch.chunk(self.q_proj(hidden).view(*shape, -1, self.head_dim * 2), 2, dim=-1)
...
attn_output = attn_output.reshape(*shape, -1)
attn_output = attn_output * torch.sigmoid(gate)      # :819
attn_output = self.o_proj(attn_output)               # :821
```

Two facts, both easy to get wrong:

- The split is **per head, along the last axis**. The projection's output is viewed as
  `[tokens, heads, 2·head_dim]` and halved into `(query, gate)` of `[tokens, heads,
  head_dim]` each. It is **not** a global `[all queries | all gates]` split: that reading
  produces identical shapes and a wrong model.
- The gate multiplies the attention output **after** attention and **before** `o_proj`,
  through a `sigmoid`, on the already-reshaped `[tokens, heads·head_dim]` tensor.

QK-norm is unchanged from `qwen3`: applied to the per-head `[tokens, heads, head_dim]`
view, on the head dim only, and to query and key but not value (`:791`–`:792`).

## The Gated DeltaNet layer, transcribed (`Qwen3_5GatedDeltaNet.forward:550`)

| Step | What happens | Line |
| --- | --- | --- |
| 1 | `mixed_qkv = in_proj_qkv(x)` then `.transpose(1, 2)` → `[B, 6144, S]` | `:562` |
| 2 | `z = in_proj_z(x).reshape(B, S, 16, 128)` — the output gate, per value head | `:566` |
| 3 | `b = in_proj_b(x)`, `a = in_proj_a(x)` — both `[B, S, 16]` | `:570` |
| 4 | depthwise causal conv: weight `[6144, 1, 4]`, **bias is None**, `padding = 3`, truncate to `S`, then **silu** | `:588`, `:270` |
| 5 | split the 6144 channels into `q`, `k` (2048 each) and `v` (2048) → `[B, S, 16, 128]` | `:597` |
| 6 | `beta = b.sigmoid()` — in the **model dtype** (bf16), not fp32 | `:610` |
| 7 | `g = -exp(A_log.float()) * softplus(a.float() + dt_bias)` — **fp32**, and the negation is outside the exponential | `:612` |
| 8 | the chunked delta rule, `chunk_size=64`, `use_qk_l2norm_in_kernel=True` | `:633` |
| 9 | gated RMSNorm over the value head dim, gate `z` | `:648` |
| 10 | `out_proj` | `:651` |

`num_v_heads // num_k_heads` is 1 here, so the `repeat_interleave` at `:613`–`:615` does
nothing in this model — it is written down because a family that needs it would otherwise
be silently mis-ported.

## The chunked delta rule, in order (`torch_chunk_gated_delta_rule:301`)

1. transpose to `[B, H, S, D]` and cast **fp32** (`:330`);
2. `l2norm` query and key over the last dim, then `query *= D^-0.5` (`:337`, `:342`);
3. pad `S` up to a multiple of `chunk_size` with zeros (`:345`);
4. `v_beta = v * beta`, `k_beta = k * beta` (`:354`);
5. reshape into chunks of 64; `cum_decay = decay.cumsum(dim=3)` (`:364`);
6. `pairwise_decay = exp(cum_decay_i − cum_decay_j)`, with the strictly-upper triangle set
   to `-inf` **before** the `exp` (`:367`–`:369`);
7. `ut_system = (k_beta @ kᵀ) * pairwise_decay`, `intra_chunk_attn = (q @ kᵀ) *
   pairwise_decay`, `decayed_k_beta = k_beta * exp(cum_decay)` (`:372`–`:374`);
8. `new_values = solve_triangular(ut_system, v_beta, upper=False, unitriangular=True)`,
   `k_cumdecay = solve_triangular(ut_system, decayed_k_beta, …)` (`:381`–`:382`) — a forward
   substitution, since the system is unit lower triangular. The exported-graph path at
   `:384`–`:390` builds the same inverse by substitution and adds the identity; the two
   agree, and the *order* of the substitution is the thing to fix in the contract;
9. `query *= exp(cum_decay)`, `key *= exp(cum_decay[-1] − cum_decay)`, `chunk_decay =
   exp(cum_decay[-1])` (`:396`–`:398`);
10. per chunk, in sequence (`:402`–`:410`):
    ```
    v_new    = new_values[i] − k_cumdecay[i] @ S
    out[i]   = query[i] @ S + intra_chunk_attn[i] @ v_new
    S        = S * chunk_decay[i] + key[i]ᵀ @ v_new
    ```

The initial state is zeros unless one is supplied (`:392`), and the final state is returned
only when asked for (`:412`).

`l2norm:294` is `x * rsqrt((x·x).sum(-1, keepdim=True) + 1e-6)` — epsilon `1e-6`, and the
reference's own comment notes FLA computes `x / sqrt(…)` instead, so the two differ in the
last bits by construction.

## The gated RMSNorm, in order (`Qwen3_5RMSNormGated:225`)

```python
hidden  = hidden.to(torch.float32)
variance = hidden.pow(2).mean(-1, keepdim=True)
hidden  = hidden * torch.rsqrt(variance + eps)     # eps = 1e-6
hidden  = weight * hidden.to(input_dtype)          # ← cast back to bf16 BEFORE the weight
hidden  = hidden * silu(gate.to(torch.float32))    # ← gate activated in fp32
return hidden.to(input_dtype)
```

The two orderings are the whole point of the class: the normalised value is rounded to
bf16 **before** the weight multiply, and the gate is activated in fp32 **after** it.

## Still to extract — do not guess these

- `Qwen3_5TextRotaryEmbedding:143` and how the full-attention layers apply RoPE here: the
  `qwen3` contract's rotate-half is **not** assumed to carry over, and there is a
  `recomposition_frequencies` at `:204` that `qwen3` has no equivalent for;
- the prefill/decode split between `causal_conv1d_fn:270` and `causal_conv1d_update:250` —
  M0 has no cache, so the prefill path is the one to port first, but the conv state's
  `state_len` handling at `:265` needs transcribing before any cache exists;
- `torch_recurrent_gated_delta_rule:438`, the single-token path: it is *not* the chunked
  rule with `chunk_size=1`, and M1 needs it;
- the MTP head wiring — present in the checkpoint, deliberately out of M0's scope;
- `apply_mask_to_padding_states:237` only matters with a padded batch, which M0 does not
  have.
