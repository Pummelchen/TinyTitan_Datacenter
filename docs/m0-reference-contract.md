# M0 reference contract — Qwen3 dense (`qwen3`)

**Status:** verified against the reference implementation, 2026-09-15. Nothing here is
implemented yet, and nothing here is inferred — every statement carries the file and line
it came from.

The brief forbids guessing at norm placement, epsilon values or RoPE application order,
because a plausible guess produces output that looks fine and is wrong. This file exists
so that no kernel has to guess.

## Provenance

| | |
| --- | --- |
| Model (candidate, `DC-020` open) | `Qwen/Qwen3-1.7B` |
| Model revision | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| Reference implementation | `transformers`, tag `v5.17.0` |
| Reference file | `src/transformers/models/qwen3/modeling_qwen3.py` |
| File sha256 | `cbb7f2dc274c2f5592746c0dc6985ca50353efa07376f92cc922b77680a74f69` |
| Config file sha256 | `49d9ccd0f29ccd977b93dba2004f84b867891b3d0509b71fd8ba3ed01054e64d` |
| Retrieved | 2026-09-15 |

Line numbers below refer to that file at that tag. **Kernel comments must cite the line and
the model revision**, per I6.

## The layer, in order, with its dtype boundaries

`B`=batch, `T`=sequence, `H`=16 query heads, `KV`=8 key/value heads, `D`=128 (`head_dim`).
The model checkpoint is bf16, so "bf16" below means the activation dtype.

| # | Step | Source | dtype in → out |
| --- | --- | --- | --- |
| 1 | `embed_tokens` lookup | `:346` | ids → bf16 |
| 2 | `input_layernorm` = RMSNorm | `:50` | bf16 → **fp32 internally** → cast back to bf16 → **× bf16 weight** |
| 3 | `q_proj` / `k_proj` / `v_proj`, no bias | `:211` | bf16 → bf16 |
| 4 | **`q_norm` / `k_norm`**, RMSNorm over `D`=128 **per head** | `:211` | bf16 → fp32 internals → bf16 |
| 5 | reshape to `[B, T, heads, D]`, transpose to `[B, heads, T, D]` | `:211` | bf16 |
| 6 | RoPE: `(q*cos) + (rotate_half(q)*sin)` | `:148`, `:140` | cos/sin computed in fp32, **cast to bf16 before use** |
| 7 | scores `= matmul(q, kᵀ) * scaling`, `scaling = D**-0.5` | `:185` | bf16 (the multiply stays bf16) |
| 8 | `+ causal mask` | `:185` | bf16 |
| 9 | **softmax in fp32**, then cast to `query.dtype` | `:185` | fp32 → bf16 |
| 10 | `matmul(attn, v)`, `repeat_kv` first (`H/KV` = 2) | `:185`, `:173` | bf16 → bf16 |
| 11 | transpose + `contiguous` + `o_proj` | `:185`, `:211` | bf16 |
| 12 | residual: `× = residual + attention_out` | `:283` | bf16 (residual is the **unnormalized** input) |
| 13 | `post_attention_layernorm` | `:50`, `:283` | as step 2 |
| 14 | MLP: `down_proj(silu(gate_proj(x)) * up_proj(x))` | `:70` | bf16, `silu` |
| 15 | residual: `× = residual + mlp_out` | `:283` | bf16 |
| 16 | final `norm` | `:346` | as step 2 |
| 17 | `lm_head`, **tied to `embed_tokens.weight`** | `:431` | bf16 → logits |

## The details that are easy to get wrong

1. **QK-norm is real and it is per head.** `q_norm`/`k_norm` are `Qwen3RMSNorm(head_dim)`
   applied to the reshaped `[.., heads, D]` view, **before RoPE**, before the transpose,
   and **not** on values. The source even comments on it: *"unlike olmo, only on the head
   dim!"*. Missing this changes the model, not just the numerics.
2. **The norm's cast happens before the weight multiply.** `forward` computes
   `variance = x32.pow(2).mean(-1)`, `x32 * rsqrt(variance + eps)`, then
   `self.weight * hidden_states.to(input_dtype)`. The weight multiply is bf16 × bf16 — not
   an fp32 multiply that is rounded at the end.
3. **Softmax is the only fp32 island in attention.** Scores, mask, and both matmuls are
   bf16; only the softmax is promoted to fp32 and cast straight back.
4. **RoPE is rotate-half, not interleaved.** `rotate_half(x) = cat((-x2, x1))` and
   `emb = cat((freqs, freqs))` agree with each other. The `cos`/`sin` handed to the
   rotation are **bf16**, computed in fp32 but cast down first.
5. **`scaling` is a Python float multiplied after the matmul**, so it does not promote the
   accumulator to fp32.
6. **Two residual adds per layer**, both against the *unnormalized* input (standard
   pre-norm), and the second residual is the output of the attention sub-layer.
7. **No biases anywhere:** `attention_bias: false` in the checkpoint, `bias=False` on every
   projection.
8. **Tied head.** The checkpoint sets `tie_word_embeddings: true`, while the class default
   at `configuration_qwen3.py:74` is `false`. The checkpoint wins: there is no separate
   `lm_head` tensor in the safetensors, so an importer that expects one will fail.
9. **A config-layout trap.** The checkpoint carries the legacy keys
   `rope_theta: 1000000` and `rope_scaling: null`; the v5 reference expects
   `rope_parameters` (null in the checkpoint) and maps the legacy keys internally. The IR
   importer must implement that same mapping — reading `rope_parameters` directly off the
   checkpoint yields `None` and silently falls back to a default base.
10. **`eps = 1e-6`** (`rms_norm_eps`), **`head_dim = 128`** (explicit in the checkpoint),
    **`max_position_embeddings = 40960`**, `hidden_size = 2048`,
    `intermediate_size = 6144`, 28 layers, `vocab_size = 151936`, `hidden_act = silu`.
11. **No YaRN at M0.** `rope_scaling` is null for this model, so M0 is plain RoPE with
    `theta = 1e6`. Long-context scaling is an M4/M5 concern.

## Reference harness configuration (proposed; `D3` decides the gate)

- `attn_implementation="eager"` — the sdpa/flash paths are not documented as
  bit-reproducible, and the eager path is short enough to mirror exactly.
- `torch.use_deterministic_algorithms(True)`, thread count pinned, dtype pinned, model
  revision pinned, prompt set frozen.
- Traces captured from forward hooks at every layer boundary (input, post-attention
  residual, post-MLP residual) plus final logits.
- The reproducibility of this configuration is **being measured, not assumed**; if PyTorch
  CPU is not bit-stable across thread counts, the thread count becomes part of the frozen
  configuration and run-to-run stability is asserted in the capture tool.

## Not yet established — do not guess these either

- the chat template and exact prompt formatting for the frozen prompt set
  (`tokenizer_config.json`, not yet read),
- whether Metal bf16 matmul reproduces PyTorch CPU bf16 exactly — this is the `D3`
  question and the largest single risk to M0's gate,
- the reduction order inside `mean`/`rsqrt` as executed on the GPU versus torch's CPU
  kernel (the reason `D3` may have to define the gate on fp32 accumulation),
- padding/tokenization edge cases for the base (non-instruct) checkpoint.
