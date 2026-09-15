#!/usr/bin/env python3
"""Emit golden vectors for the engine's ops, as fp32 bit patterns.

The contract in `ordered_reference.py` is only useful if another language can reproduce
it, and "reproduce" here means bit patterns, not closeness. This writes the inputs and
the contract's exact outputs to a JSON fixture the Swift tests assert against, so a
disagreement shows up as a failing test on the op that caused it rather than as a
mysterious drift in a 28-layer trace.

Regenerate with:
    .venv/bin/python tools/make_contract_vectors.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import ordered_reference as ref  # noqa: E402

FIXTURE = (
    Path(__file__).parent.parent
    / "tests"
    / "DatacenterEngineTests"
    / "Fixtures"
    / "contract-vectors.json"
)


def vector(array) -> dict:
    """A tensor as a shape plus the bit patterns of its fp32 values."""
    array = ref.f32(array)
    return {
        "shape": list(array.shape),
        "bits": [int(v) for v in array.reshape(-1).view(np.uint32)],
    }


def main() -> None:
    rng = np.random.default_rng(20260915)

    matmul_x = ref.f32(rng.standard_normal((3, 7)) * 0.7)
    matmul_w = ref.f32(rng.standard_normal((5, 7)) * 0.7)

    norm_x = ref.f32(rng.standard_normal((2, 4)) * 2)
    norm_weight = ref.f32(rng.standard_normal(4))

    # A silu input with both extremes: the stable-sigmoid branch is part of the contract.
    silu_x = ref.f32(np.concatenate([rng.standard_normal(6) * 2, [-200.0, -1.0, 0.0, 1.0, 200.0]]))

    softmax_x = ref.f32(rng.standard_normal((2, 5)) * 4)

    head_dim, tokens, heads = 8, 6, 2
    positions = np.arange(tokens, dtype=np.float64)
    cos, sin = ref.rope_tables(head_dim, positions, 1_000_000.0)
    rope_x = ref.f32(rng.standard_normal((tokens, heads, head_dim)))
    rope_out = ref.apply_rope(rope_x, cos, sin)

    payload = {
        "note": (
            "Golden vectors for the numeric contract in tools/ordered_reference.py. "
            "Bit patterns, not values: the engine must reproduce the contract exactly. "
            "Generate with tools/make_contract_vectors.py."
        ),
        "matmul": {
            "x": vector(matmul_x),
            "w": vector(matmul_w),
            "out": vector(ref.ordered_matmul(matmul_x, matmul_w)),
        },
        "sum": {
            "x": vector(ref.f32([1e8, 1.0, -1e8, 1.0, 0.5, -0.25])),
            "out": vector(np.asarray(ref.ordered_sum(ref.f32([1e8, 1.0, -1e8, 1.0, 0.5, -0.25])))),
        },
        "rms_norm": {
            "x": vector(norm_x),
            "weight": vector(norm_weight),
            "eps": 1e-6,
            "out": vector(ref.rms_norm(norm_x, norm_weight, 1e-6)),
        },
        "silu": {"x": vector(silu_x), "out": vector(ref.silu(silu_x))},
        "sigmoid": {"x": vector(silu_x), "out": vector(ref.sigmoid(silu_x))},
        "softmax": {"x": vector(softmax_x), "out": vector(ref.softmax(softmax_x))},
        "rope": {
            "positions": [int(p) for p in positions],
            "head_dim": head_dim,
            "theta": 1_000_000.0,
            "cos": vector(cos),
            "sin": vector(sin),
        },
        "apply_rope": {
            "x": vector(rope_x),
            "cos": vector(cos),
            "sin": vector(sin),
            "out": vector(rope_out),
        },
    }

    # --- The Gated DeltaNet, family two -------------------------------------------------
    # A tiny configuration of the real geometry: two key heads and two value heads of
    # width 4, a convolution kernel of 4, and five positions, so the chunked rule runs
    # with a single chunk but every reshape, gate and boundary is exercised.
    import ordered_qwen35 as q35

    class TinyConfig:
        hidden_size = 16
        linear_num_key_heads = 2
        linear_num_value_heads = 2
        linear_key_head_dim = 4
        linear_value_head_dim = 4
        linear_conv_kernel_dim = 4
        rms_norm_eps = 1e-6
        hidden_act = "silu"
        head_dim = 8
        rope_parameters = {"rope_theta": 1e7, "partial_rotary_factor": 0.5}

    tiny = TinyConfig()
    key_dim = tiny.linear_num_key_heads * tiny.linear_key_head_dim
    value_dim = tiny.linear_num_value_heads * tiny.linear_value_head_dim
    conv_dim = key_dim * 2 + value_dim
    positions, hidden_size = 5, tiny.hidden_size

    gdn_weights = {
        "in_proj_qkv": ref.f32(rng.standard_normal((conv_dim, hidden_size)) * 0.4),
        "in_proj_z": ref.f32(rng.standard_normal((value_dim, hidden_size)) * 0.4),
        "in_proj_b": ref.f32(rng.standard_normal((tiny.linear_num_value_heads, hidden_size)) * 0.4),
        "in_proj_a": ref.f32(rng.standard_normal((tiny.linear_num_value_heads, hidden_size)) * 0.4),
        "conv1d": ref.f32(rng.standard_normal((conv_dim, 1, tiny.linear_conv_kernel_dim)) * 0.4),
        "A_log": ref.f32(rng.standard_normal(tiny.linear_num_value_heads)),
        "dt_bias": ref.f32(rng.standard_normal(tiny.linear_num_value_heads) * 0.2),
        "norm": ref.f32(rng.standard_normal(tiny.linear_value_head_dim) * 0.2 + 1.0),
        "out_proj": ref.f32(rng.standard_normal((hidden_size, value_dim)) * 0.4),
    }
    gdn_hidden = ref.f32(rng.standard_normal((1, positions, hidden_size)) * 0.7)
    gdn_out = q35.gated_delta_net_layer(gdn_hidden, gdn_weights, tiny)

    # A second case long enough to need two chunks: the sequential scan over chunks is
    # where a padding or state-threading error would hide, and a single-chunk case never
    # runs it.
    long_positions = 70
    gdn_hidden_long = ref.f32(rng.standard_normal((1, long_positions, hidden_size)) * 0.7)
    gdn_out_long = q35.gated_delta_net_layer(gdn_hidden_long, gdn_weights, tiny)

    payload["gdn_multichunk"] = {
        "config": {
            "hidden_size": hidden_size,
            "num_key_heads": tiny.linear_num_key_heads,
            "num_value_heads": tiny.linear_num_value_heads,
            "key_head_dim": tiny.linear_key_head_dim,
            "value_head_dim": tiny.linear_value_head_dim,
            "conv_kernel": tiny.linear_conv_kernel_dim,
            "eps": tiny.rms_norm_eps,
            "positions": long_positions,
        },
        "hidden": vector(gdn_hidden_long),
        "out": vector(gdn_out_long),
        "weights": {name: vector(value) for name, value in gdn_weights.items()},
    }

    payload["gdn"] = {
        "config": {
            "hidden_size": hidden_size,
            "num_key_heads": tiny.linear_num_key_heads,
            "num_value_heads": tiny.linear_num_value_heads,
            "key_head_dim": tiny.linear_key_head_dim,
            "value_head_dim": tiny.linear_value_head_dim,
            "conv_kernel": tiny.linear_conv_kernel_dim,
            "eps": tiny.rms_norm_eps,
            "positions": positions,
        },
        "hidden": vector(gdn_hidden),
        "out": vector(gdn_out),
        "weights": {name: vector(value) for name, value in gdn_weights.items()},
    }

    # --- The mixture of experts, M1's model ---------------------------------------------
    # A tiny mixture with the real shape of the problem: 8 experts, top-2, and a shared
    # expert of the same width. The discrete decision (the index set) is part of the vector,
    # because I3 asserts it separately from the numbers.
    import ordered_moe as moe

    tokens, hidden_size, experts, top_k, inter = 4, 24, 8, 2, 12
    moe_weights = {
        "router_weight": ref.f32(rng.standard_normal((experts, hidden_size)) * 0.6),
        "gate_up": ref.f32(rng.standard_normal((experts, 2 * inter, hidden_size)) * 0.4),
        "down": ref.f32(rng.standard_normal((experts, hidden_size, inter)) * 0.4),
        "shared_gate": ref.f32(rng.standard_normal((inter, hidden_size)) * 0.4),
        "shared_up": ref.f32(rng.standard_normal((inter, hidden_size)) * 0.4),
        "shared_down": ref.f32(rng.standard_normal((hidden_size, inter)) * 0.4),
        "shared_scalar_gate": ref.f32(rng.standard_normal((1, hidden_size)) * 0.3),
    }
    moe_hidden = ref.f32(rng.standard_normal((tokens, hidden_size)) * 0.7)
    moe_out, moe_indices, moe_weights_out = moe.sparse_moe_block(
        moe_hidden, top_k=top_k, **moe_weights
    )
    payload["moe"] = {
        "config": {
            "tokens": tokens,
            "hidden_size": hidden_size,
            "experts": experts,
            "top_k": top_k,
            "intermediate": inter,
        },
        "hidden": vector(moe_hidden),
        "out": vector(moe_out),
        "indices": {"shape": list(moe_indices.shape), "values": [int(v) for v in moe_indices.reshape(-1)]},
        "weights": {name: vector(value) for name, value in moe_weights.items()},
        "top_k_weights": vector(moe_weights_out),
    }

    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(payload, indent=2) + "\n")
    tensors = sum(1 for value in payload.values() if isinstance(value, dict))
    print(f"wrote {FIXTURE} ({FIXTURE.stat().st_size} bytes, {tensors} vector groups)")


if __name__ == "__main__":
    main()
