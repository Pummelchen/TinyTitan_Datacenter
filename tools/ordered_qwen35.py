#!/usr/bin/env python3
"""The Qwen3.5 Gated DeltaNet *layer*, in the contract's order (M0b).

`ordered_gdn.py` holds the chunked delta rule itself; this file holds everything around
it, transcribed from `Qwen3_5GatedDeltaNet.forward:550`:

    in_proj_qkv -> depthwise causal conv (kernel 4, no bias, left padding 3, silu)
                -> split q | k | v
    in_proj_z    -> the output gate, per value head
    in_proj_b    -> beta = sigmoid(b)
    in_proj_a    -> g = -exp(A_log) * softplus(a + dt_bias)     (fp32)
                -> chunked delta rule (l2norm, chunk 64)
                -> gated RMSNorm over the value head dim
                -> out_proj

Checked against the reference's own `Qwen3_5GatedDeltaNet` module by
`test_ordered_qwen35.py`, on a tiny configuration so the comparison does not need a 2 B
checkpoint.
"""

from __future__ import annotations

import numpy as np

from ordered_gdn import chunk_gated_delta_rule, gated_rms_norm
from ordered_reference import exp32, f32, ordered_matmul, ordered_sum


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Stable sigmoid, the same branch structure as the dense contract's."""
    x = f32(x)
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = f32(np.float32(1.0) / (np.float32(1.0) + exp32(f32(-x[positive]))))
    exponential = exp32(x[~positive])
    out[~positive] = f32(exponential / (np.float32(1.0) + exponential))
    return out


def silu(x: np.ndarray) -> np.ndarray:
    x = f32(x)
    return f32(x * sigmoid(x))


def softplus(x: np.ndarray, beta: float = 1.0, threshold: float = 20.0) -> np.ndarray:
    """`F.softplus` with torch's default threshold: above it, the identity.

    The threshold is part of the reference's behaviour rather than an optimisation detail:
    `softplus(x) = x` exactly for `x * beta > 20`, which is a different number from
    `log1p(exp(x))` in the last bits.
    """
    x = f32(x)
    scaled = f32(x * np.float32(beta))
    out = np.empty_like(x)
    large = scaled > np.float32(threshold)
    out[large] = x[large]
    # The contract's transcendental rule: evaluated in double, rounded to Float. Stating
    # it that way rather than calling float32 `log1p`/`exp` is what lets the Swift
    # implementation reproduce these bits.
    out[~large] = f32(np.log1p(np.exp(np.float64(scaled[~large]))) / np.float64(beta))
    return out


def depthwise_causal_conv(x: np.ndarray, weight: np.ndarray, activation: str = "silu") -> np.ndarray:
    """`causal_conv1d_fn:270`: depthwise, left-padded, truncated, then activated.

    `x` is `[B, C, S]` and `weight` is `[C, 1, K]`. The reference pads by `K - 1` on the
    left and truncates back to `S`, which is what makes it causal: output `s` sees inputs
    `s - K + 1 … s` and nothing later.
    """
    x = f32(x)
    weight = f32(weight)
    batch, channels, length = x.shape
    kernel = weight.shape[-1]
    out = np.zeros((batch, channels, length), dtype=np.float32)
    for index in range(batch):
        for channel in range(channels):
            taps = [f32(weight[channel, 0, k]) for k in range(kernel)]
            for position in range(length):
                accumulator = np.float32(0.0)
                for k in range(kernel):
                    source = position + k - (kernel - 1)
                    if source >= 0:
                        accumulator = f32(accumulator + f32(taps[k] * x[index, channel, source]))
                out[index, channel, position] = accumulator
    return silu(out) if activation == "silu" else out


def gated_delta_net_layer(hidden: np.ndarray, weights: dict, config) -> np.ndarray:
    """One Gated DeltaNet layer. `hidden` is `[B, S, hidden_size]`; returns the same shape."""
    hidden = f32(hidden)
    batch, length, _ = hidden.shape
    heads = config.linear_num_value_heads
    head_k = config.linear_key_head_dim
    head_v = config.linear_value_head_dim
    key_dim = head_k * config.linear_num_key_heads
    value_dim = head_v * heads

    # 1-4: the projection, the conv, and back to sequence-major.
    mixed = ordered_matmul(hidden, weights["in_proj_qkv"])           # [B, S, 3*D]
    mixed = np.transpose(mixed, (0, 2, 1))                            # [B, 3*D, S]
    mixed = depthwise_causal_conv(mixed, weights["conv1d"], config.hidden_act)
    mixed = np.transpose(mixed, (0, 2, 1))                            # [B, S, 3*D]

    query = mixed[..., :key_dim]
    key = mixed[..., key_dim : 2 * key_dim]
    value = mixed[..., 2 * key_dim :]

    z = ordered_matmul(hidden, weights["in_proj_z"]).reshape(batch, length, heads, head_v)
    b = ordered_matmul(hidden, weights["in_proj_b"])
    a = ordered_matmul(hidden, weights["in_proj_a"])

    beta = sigmoid(b)
    # fp32, and the negation is outside the exponential.
    gate = f32(-exp32(f32(weights["A_log"])) * softplus(f32(a + weights["dt_bias"])))

    query = query.reshape(batch, length, config.linear_num_key_heads, head_k)
    key = key.reshape(batch, length, config.linear_num_key_heads, head_k)
    value = value.reshape(batch, length, heads, head_v)

    # `Qwen3_5MoeGatedDeltaNet.forward:645` -- grouped-query style, and the step the 2 B model
    # never exercised because it has sixteen key heads *and* sixteen value heads:
    #
    #     if self.num_v_heads // self.num_k_heads > 1:
    #         query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
    #         key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
    #
    # The MoE family has sixteen key heads to thirty-two value heads, so each key head serves
    # two value heads and the delta rule sees them repeated consecutively.
    if heads // config.linear_num_key_heads > 1:
        factor = heads // config.linear_num_key_heads
        query = np.repeat(query, factor, axis=2)
        key = np.repeat(key, factor, axis=2)

    core, _ = chunk_gated_delta_rule(
        query, key, value, gate, beta, chunk_size=64, use_qk_l2norm=True
    )

    normalized = gated_rms_norm(
        core.reshape(-1, head_v), z.reshape(-1, head_v), weights["norm"], eps=config.rms_norm_eps
    )
    normalized = normalized.reshape(batch, length, value_dim)
    return ordered_matmul(normalized, weights["out_proj"])


# --------------------------------------------------------------------------------------
# The full-attention layer, the decoder layer and the text model.
# --------------------------------------------------------------------------------------


def rope_tables(config, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The text RoPE tables, width `int(head_dim * partial_rotary_factor)`.

    `Qwen3_5TextRotaryEmbedding:164` builds `inv_freq` over `dim = int(head_dim ×
    partial_rotary_factor)` — 64 of 256 for this checkpoint — and
    `recomposition_frequencies:204` doubles it. The multimodal grid recomposition is a
    no-op for text-only input: `Qwen3_5TextModel.forward:1260` expands one grid into four
    identical ones, so the interleaved copy between grids copies identical values.
    """
    head_dim = config.head_dim
    factor = 1.0
    parameters = getattr(config, "rope_parameters", None) or {}
    factor = parameters.get("partial_rotary_factor", 1.0)
    base = parameters.get("rope_theta", 10_000.0)
    dim = int(head_dim * factor)

    exponents = f32(np.arange(0, dim, 2, dtype=np.float32) / np.float32(dim))
    inverse = f32(np.float32(1.0) / f32(np.power(np.float32(base), exponents)))
    angles = np.outer(np.float64(positions), np.float64(inverse))
    doubled = np.concatenate([angles, angles], axis=-1)
    return f32(np.cos(doubled)), f32(np.sin(doubled))


def apply_rope_partial(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """`apply_rotary_pos_emb:674`, which is **partial** by construction.

    The first `rotary_dim` channels rotate with rotate-half *within themselves* and the
    rest pass through untouched. `x` is `[T, heads, head_dim]`.
    """
    x = f32(x)
    rotary = cos.shape[-1]
    half = rotary // 2
    rotating, passing = x[..., :rotary], x[..., rotary:]
    rotated = np.concatenate([-rotating[..., half:], rotating[..., :half]], axis=-1)
    embedded = f32(f32(rotating * cos[:, None, :]) + f32(rotated * sin[:, None, :]))
    return np.concatenate([embedded, passing], axis=-1)


def full_attention_layer(hidden: np.ndarray, weights: dict, config, cos: np.ndarray, sin: np.ndarray,
                         mask: np.ndarray) -> np.ndarray:
    """`Qwen3_5Attention.forward:776`, including the output gate that is unique to it."""
    hidden = f32(hidden)
    tokens = hidden.shape[0]
    heads = config.num_attention_heads
    kv_heads = config.num_key_value_heads
    head_dim = config.head_dim
    eps = config.rms_norm_eps
    scaling = np.float32(head_dim**-0.5)

    # The projection is viewed as [tokens, heads, 2*head_dim] and halved along the last
    # axis: the gate sits *inside* each head, not in a second bank after all queries.
    projected = ordered_matmul(hidden, weights["q_proj"]).reshape(tokens, heads, head_dim * 2)
    query, gate = projected[..., :head_dim], projected[..., head_dim:]

    key = ordered_matmul(hidden, weights["k_proj"]).reshape(tokens, kv_heads, head_dim)
    value = ordered_matmul(hidden, weights["v_proj"]).reshape(tokens, kv_heads, head_dim)

    query = rms_norm(query, weights["q_norm"], eps)
    key = rms_norm(key, weights["k_norm"], eps)
    query = apply_rope_partial(query, cos, sin)
    key = apply_rope_partial(key, cos, sin)

    groups = heads // kv_heads
    key = np.repeat(key, groups, axis=1)
    value = np.repeat(value, groups, axis=1)

    mixer = np.zeros((tokens, heads, head_dim), dtype=np.float32)
    for head in range(heads):
        scores = f32(ordered_matmul(query[:, head, :], key[:, head, :]) * scaling)
        scores = f32(scores + mask)
        attention = softmax(scores)
        mixer[:, head, :] = ordered_matmul(attention, f32(value[:, head, :].T))

    mixer = mixer.reshape(tokens, heads * head_dim)
    mixer = f32(mixer * sigmoid(gate.reshape(tokens, heads * head_dim)))
    return ordered_matmul(mixer, weights["o_proj"])


def softmax(x: np.ndarray) -> np.ndarray:
    """Max-subtracted, ordered sum, fp32 — the reference softmaxes in fp32 here too."""
    x = f32(x)
    shifted = f32(x - x.max(axis=-1, keepdims=True))
    exponentials = exp32(shifted)
    total = ordered_sum(exponentials, axis=-1)
    return f32(exponentials / total[..., None])


def decoder_layer(hidden: np.ndarray, weights: dict, config, layer_index: int,
                  cos: np.ndarray, sin: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """`Qwen3_5DecoderLayer.forward:874`: residual, mixer, residual, feed-forward."""
    hidden = f32(hidden)
    residual = hidden
    normed = rms_norm(hidden, weights["input_layernorm"], config.rms_norm_eps)
    if config.layer_types[layer_index] == "full_attention":
        mixed = full_attention_layer(normed, weights["self_attn"], config, cos, sin, mask)
    else:
        # The Gated DeltaNet layer keeps the reference module's `[batch, sequence, hidden]`
        # shape -- it was checked against that module directly -- so the unbatched model
        # forward adds and drops the batch axis here rather than duplicating the layer.
        mixed = gated_delta_net_layer(normed[None], weights["linear_attn"], config)[0]
    hidden = f32(residual + mixed)

    residual = hidden
    normed = rms_norm(hidden, weights["post_attention_layernorm"], config.rms_norm_eps)
    gated = f32(silu(ordered_matmul(normed, weights["mlp"]["gate_proj"]))
                * ordered_matmul(normed, weights["mlp"]["up_proj"]))
    hidden = f32(residual + ordered_matmul(gated, weights["mlp"]["down_proj"]))
    return hidden


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    """The backbone RMSNorm — and this family's is **weight-offset**.

    `Qwen3_5RMSNorm:841` initialises its parameter to **zeros** and multiplies by
    `(1.0 + weight)`, not by `weight`:

        output = x * rsqrt(mean(x^2) + eps)
        output = output * (1.0 + weight)

    so a zero weight is the identity and a checkpoint's stored values are offsets from
    one. The `qwen3` family's norm is the ordinary `weight * normalised`, and the GDN's
    own `Qwen3_5RMSNormGated` in the same file is *also* the ordinary kind — three norms,
    two conventions, which is exactly why this was found by comparing against the module
    rather than by assuming the family was consistent with itself.

    The reference also casts last: `output * (1 + weight)` in fp32, and only then to the
    model dtype.
    """
    x = f32(x)
    variance = f32(ordered_sum(f32(x * x), axis=-1) / np.float32(x.shape[-1]))
    inverse = f32(np.float32(1.0) / np.sqrt(f32(variance + np.float32(eps))))
    return f32(f32(np.float32(1.0) + weight) * f32(x * inverse[..., None]))


def text_model_forward(weights: dict, config, tokens, capture: dict | None = None) -> np.ndarray:
    """The text tower end to end, returning the logits.

    `weights` is `{"embed_tokens", "norm", "layers": [...]}` with each layer holding the
    role-keyed arrays the importer maps to.
    """
    tokens = list(tokens)
    hidden = f32(weights["embed_tokens"][tokens])
    if capture is not None:
        capture["embed.out"] = hidden

    cos, sin = rope_tables(config, np.arange(len(tokens), dtype=np.float64))
    # Causal mask as an additive -inf matrix, applied to the scaled scores.
    mask = np.triu(np.full((len(tokens), len(tokens)), -np.inf, dtype=np.float32), k=1)

    for index, layer in enumerate(weights["layers"]):
        if capture is not None:
            capture[f"layer.{index:02d}.hidden_in"] = hidden
        hidden = decoder_layer(hidden, layer, config, index, cos, sin, mask)
        if capture is not None:
            capture[f"layer.{index:02d}.hidden_out"] = hidden

    hidden = rms_norm(hidden, weights["norm"], config.rms_norm_eps)
    if capture is not None:
        capture["final_norm.out"] = hidden
    # The family is tied: the embedding matrix is the head.
    logits = ordered_matmul(hidden, weights["embed_tokens"])
    if capture is not None:
        capture["logits"] = logits
    return logits


# --------------------------------------------------------------------------------------
# Streaming: the same forward, with each layer's weights fetched and released in turn.
# --------------------------------------------------------------------------------------


class SpecConfig:
    """The IR spec's configuration, in the shape these functions expect.

    The spec carries everything the contract needs, which is the point of L1: the Python
    side reads the *engine's* spec and stays ignorant of every tensor name, so the importer
    remains the only place where names live (L2).
    """

    def __init__(self, config: dict):
        self.hidden_size = config["hiddenSize"]
        self.num_hidden_layers = config["numLayers"]
        self.num_attention_heads = config["numAttentionHeads"]
        self.num_key_value_heads = config["numKeyValueHeads"]
        self.head_dim = config["headDim"]
        self.intermediate_size = config["intermediateSize"]
        self.vocab_size = config["vocabSize"]
        self.rms_norm_eps = config["rmsNormEps"]
        self.hidden_act = "silu"
        self.tie_word_embeddings = config["tieWordEmbeddings"]
        self.attn_output_gate = config["attnOutputGate"]
        interval = config.get("fullAttentionInterval")
        self.layer_types = [
            "full_attention" if interval and index % interval == interval - 1 else "linear_attention"
            for index in range(self.num_hidden_layers)
        ]
        value_heads = config.get("linearValueHeads")
        self.linear_num_value_heads = value_heads
        # The IR stores the totals; the per-head width follows from them, as in the engine.
        # The key and value counts are **separate** fields: `qwen3_5` has sixteen of each and
        # `qwen3_5_moe` has sixteen keys to thirty-two values, so deriving one from the other
        # silently halved the key head width for the second family.
        key_heads = config.get("linearKeyHeads") or value_heads
        self.linear_num_key_heads = key_heads
        key_dim = config.get("linearKeyDim")
        self.linear_key_head_dim = key_dim // key_heads if key_dim and key_heads else None
        self.linear_value_head_dim = config.get("linearValueHeadDim")
        self.linear_conv_kernel_dim = config.get("linearConvKernelDim")
        self.rope_parameters = {
            "rope_theta": config.get("ropeTheta", 10_000.0),
            "partial_rotary_factor": config.get("partialRotaryFactor", 1.0) or 1.0,
            "rope_type": "default",
        }


def roles_by_block(spec: dict) -> dict:
    """`block -> role -> tensor name`, straight out of the spec."""
    blocks: dict = {}
    for tensor in spec["tensors"]:
        blocks.setdefault(tensor["block"], {})[tensor["role"]] = tensor["name"]
    return blocks


_LAYER_ROLES = {
    "input_layernorm": "norm.attn",
    "post_attention_layernorm": "norm.mlp",
}


def layer_weights(names: dict, source, materialise) -> dict:
    """Assemble one layer's role-keyed weights through the source.

    `materialise` maps a role to the key the contract's functions use, so the layer
    dictionaries are built once here instead of in every caller.
    """
    weights = {
        "input_layernorm": materialise(names["norm.attn"]),
        "post_attention_layernorm": materialise(names["norm.mlp"]),
    }
    # A family with a dense feed-forward has these roles and a mixture does not; the caller
    # fills in `mlp` itself when it is a mixture, so the mixer is written once for both.
    if "mlp.gate" in names:
        weights["mlp"] = {
            "gate_proj": materialise(names["mlp.gate"]),
            "up_proj": materialise(names["mlp.up"]),
            "down_proj": materialise(names["mlp.down"]),
        }
    if "attn.q" in names:
        weights["self_attn"] = {
            "q_proj": materialise(names["attn.q"]),
            "k_proj": materialise(names["attn.k"]),
            "v_proj": materialise(names["attn.v"]),
            "o_proj": materialise(names["attn.o"]),
            "q_norm": materialise(names["attn.q_norm"]),
            "k_norm": materialise(names["attn.k_norm"]),
        }
    else:
        weights["linear_attn"] = {
            "in_proj_qkv": materialise(names["linear.in_qkv"]),
            "in_proj_z": materialise(names["linear.in_z"]),
            "in_proj_b": materialise(names["linear.in_b"]),
            "in_proj_a": materialise(names["linear.in_a"]),
            "conv1d": materialise(names["linear.conv"]),
            "A_log": materialise(names["linear.a_log"]),
            "dt_bias": materialise(names["linear.dt_bias"]),
            "norm": materialise(names["linear.norm"]),
            "out_proj": materialise(names["linear.out"]),
        }
    return weights


def streamed_text_forward(spec: dict, source, tokens, capture: dict | None = None, head_block: int = 8192) -> np.ndarray:
    """The text tower, one layer resident at a time.

    `source.tensor(name)` returns a whole tensor in fp32 and `source.rows(name, start, end)`
    a row range — the same two operations the Swift engine's reader offers, so the two
    implementations stream identically rather than one of them being special.
    """
    config = SpecConfig(spec["config"])
    names = roles_by_block(spec)
    tokens = list(tokens)

    def materialise(name):
        return source.tensor(name)

    embedding = names["embed"].get("token.embedding")
    if embedding is None:
        raise ValueError("the spec has no token.embedding tensor")
    hidden = f32(source.rows(embedding, tokens[0], tokens[0] + 1))
    if len(tokens) > 1:
        hidden = np.concatenate([source.rows(embedding, token, token + 1) for token in tokens], axis=0)
        hidden = f32(hidden)
    if capture is not None:
        capture["embed.out"] = hidden

    positions = np.arange(len(tokens), dtype=np.float64)
    cos, sin = rope_tables(config, positions)
    mask = np.triu(np.full((len(tokens), len(tokens)), -np.inf, dtype=np.float32), k=1)

    for index in range(config.num_hidden_layers):
        block = f"layer.{index:02d}"
        if capture is not None:
            capture[f"{block}.hidden_in"] = hidden
        weights = layer_weights(names[block], source, materialise)
        hidden = decoder_layer(hidden, weights, config, index, cos, sin, mask)
        if capture is not None:
            capture[f"{block}.hidden_out"] = hidden
        del weights  # released before the next layer is read, as the engine does

    final_name = names["final"]["norm.final"]
    hidden = rms_norm(hidden, source.tensor(final_name), config.rms_norm_eps)
    if capture is not None:
        capture["final_norm.out"] = hidden

    head_name = names.get("head", {}).get("head.lm", embedding)
    logits = np.zeros((len(tokens), config.vocab_size), dtype=np.float32)
    start = 0
    while start < config.vocab_size:
        end = min(start + head_block, config.vocab_size)
        block = f32(source.rows(head_name, start, end))
        logits[:, start:end] = ordered_matmul(hidden, block)
        start = end
    if capture is not None:
        capture["logits"] = logits
    return logits
