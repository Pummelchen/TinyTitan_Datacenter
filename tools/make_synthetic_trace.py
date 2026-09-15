#!/usr/bin/env python3
"""Synthetic fixture trace — the model-free half of M0a.

The harness must be proven able to *fail* before it is pointed at a real model, so
this generates a trace from a tiny deterministic computation that needs no torch, no
checkpoint and no download. The numbers are arbitrary; the properties are not:

- **Deterministic by construction.** A hand-rolled LCG, not ``random``, so the bytes
  cannot move when a Python version changes its generator.
- **Order-sensitive.** The per-layer sums run in a fixed index order, so a kernel
  that reorders its accumulation produces a different trace.
- **It carries a discrete decision** (a router-style top-3 over the layer's own
  values, ties broken by ascending index), so the I3 comparator has something real
  to compare instead of only float tensors.

Usage::

    python3 tools/make_synthetic_trace.py <out-dir> [--seed N] [--layers N]
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

from trace_format import write_trace

HIDDEN = 8
TOKENS = 2
LAYERS = 3


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _pack_f32(values) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


class _LCG:
    """Numerical Recipes' linear congruential generator: 32-bit, stable, tiny."""

    def __init__(self, seed: int):
        self.state = seed & 0xFFFFFFFF

    def next_u32(self) -> int:
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return self.state

    def next_f32(self) -> float:
        return _f32(self.next_u32() / 2147483648.0 - 1.0)


def _matmul_vec(matrix, vector, width: int):
    """Fixed accumulation order, fp32-rounded at every step, like a real kernel."""
    out = []
    for row in range(width):
        acc = _f32(0.0)
        for col in range(width):
            acc = _f32(acc + _f32(matrix[row][col] * vector[col]))
        out.append(acc)
    return out


def _swish_gate(x, gate):
    """A SwiGLU-shaped elementwise op, enough to be order-sensitive."""
    out = []
    for i, value in enumerate(x):
        sigmoid = _f32(1.0 / (1.0 + pow(2.718281828459045, -gate[i])))
        out.append(_f32(value * sigmoid))
    return out


def _topk(values, k):
    """Top-k with an explicit tie-break: value descending, then index ascending.
    Written as a comparator rather than relying on a library's stability."""
    order = sorted(range(len(values)), key=lambda i: (-values[i], i))
    return order[:k]


def synthetic_tensors(seed: int = 1, layers: int = LAYERS, hidden: int = HIDDEN, tokens: int = TOKENS):
    """Return ``(tensors, discrete)`` for :func:`trace_format.write_trace`."""
    rng = _LCG(seed)
    attn = [[rng.next_f32() for _ in range(hidden)] for _ in range(hidden)]
    gate = [[rng.next_f32() for _ in range(hidden)] for _ in range(hidden)]
    down = [[rng.next_f32() for _ in range(hidden)] for _ in range(hidden)]

    tensors = []
    discrete = []

    hidden_states = [[rng.next_f32() for _ in range(hidden)] for _ in range(tokens)]
    flat = [value for token in hidden_states for value in token]
    tensors.append(("embed.out", "f32", [tokens, hidden], _pack_f32(flat)))

    for layer in range(layers):
        name = f"layer.{layer:02d}"
        tensors.append(
            (
                f"{name}.hidden_in",
                "f32",
                [tokens, hidden],
                _pack_f32([value for token in hidden_states for value in token]),
            )
        )

        # Router decision on the layer input: a real discrete decision, derived from
        # the same numbers, with the deterministic tie-break the engine must match.
        logits = _matmul_vec(attn, hidden_states[0], hidden)
        chosen = _topk(logits, min(3, hidden))
        discrete.append((f"{name}.router.topk", [len(chosen)], chosen))

        new_states = []
        for token in hidden_states:
            mixed = _matmul_vec(attn, token, hidden)
            gated = _swish_gate(mixed, _matmul_vec(gate, token, hidden))
            out = _matmul_vec(down, gated, hidden)
            residual = [_f32(token[i] + out[i]) for i in range(hidden)]
            new_states.append(residual)
        hidden_states = new_states
        tensors.append(
            (
                f"{name}.mlp_out",
                "f32",
                [tokens, hidden],
                _pack_f32([value for token in hidden_states for value in token]),
            )
        )

    tensors.append(
        (
            "final_norm.out",
            "f32",
            [tokens, hidden],
            _pack_f32([value for token in hidden_states for value in token]),
        )
    )
    tensors.append(
        (
            "logits",
            "f32",
            [tokens, hidden],
            _pack_f32([value for token in hidden_states for value in token]),
        )
    )
    return tensors, discrete


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--layers", type=int, default=LAYERS)
    args = parser.parse_args(argv)

    tensors, discrete = synthetic_tensors(seed=args.seed, layers=args.layers)
    manifest = write_trace(
        args.out,
        tensors=tensors,
        discrete=discrete,
        model={"repo": "synthetic", "revision": f"seed-{args.seed}"},
        reference={"compute_dtype": "f32", "note": "no model; deterministic fixture"},
        prompt={"token_ids": list(range(TOKENS))},
        producer="make_synthetic_trace",
    )
    print(f"wrote {len(tensors)} tensor(s) and {len(discrete)} discrete decision(s)")
    print(f"digest {manifest['digest']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
