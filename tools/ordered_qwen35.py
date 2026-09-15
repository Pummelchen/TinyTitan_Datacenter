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
    out[~large] = f32(np.log1p(np.exp(scaled[~large])) / np.float32(beta))
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

    core, _ = chunk_gated_delta_rule(
        query, key, value, gate, beta, chunk_size=64, use_qk_l2norm=True
    )

    normalized = gated_rms_norm(
        core.reshape(-1, head_v), z.reshape(-1, head_v), weights["norm"], eps=config.rms_norm_eps
    )
    normalized = normalized.reshape(batch, length, value_dim)
    return ordered_matmul(normalized, weights["out_proj"])
