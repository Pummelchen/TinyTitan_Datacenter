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

## Still to extract before the GDN kernel is written — do not guess these

- the intra-chunk algorithm in full: `chunk_size=64` is known, but the decay masking,
  the `ut_system` construction (`:399`–`:403`), the `torch.eye` regularisation and the
  chunk-level state update order are not yet transcribed;
- the exact `l2norm` (`:294`, "aligns with the FLA implementation") epsilon and reduction;
- how `causal_conv1d_fn` (`:270`) treats the `padding = kernel_size - 1` at sequence
  boundaries, and the prefill/decode split between `causal_conv1d_fn` and
  `causal_conv1d_update` (`:250`);
- `Qwen3_5Attention:749` in full (QK-norm placement and the `attn_output_gate`
  application order — the `qwen3` contract's equivalent is already known and is *not*
  assumed to carry over);
- `Qwen3_5TextRotaryEmbedding:143` and how the full-attention layers apply RoPE here;
- the MTP head wiring (`mtp_num_hidden_layers: 1`) and whether M0 includes it (M0 should
  not — it is an M5 concern for the other family);
- the vision-tensor name list, to filter the checkpoint without loading it.
