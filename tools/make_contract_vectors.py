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

    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(payload, indent=2) + "\n")
    tensors = sum(1 for value in payload.values() if isinstance(value, dict))
    print(f"wrote {FIXTURE} ({FIXTURE.stat().st_size} bytes, {tensors} vector groups)")


if __name__ == "__main__":
    main()
