#!/usr/bin/env python3
"""The controlled-order reference forward — what the engine's gate actually compares against.

Why this exists, measured rather than argued: a matmul accumulated in ascending ``k``
does **not** reproduce PyTorch's matmul bit-for-bit. On a 8x1024 by 512x1024 fp32 case,
3544 of 4096 outputs differed, mean 17 ULP and max 22587 ULP — while both results sat
exactly the same distance (3.148e-07) from an fp64 computation of the same product. In
other words the difference is not error, it is *summation order*, and PyTorch's order is
a property of its BLAS kernels rather than of the model. No engine can be required to
reproduce it, and no Metal kernel ever will.

So M0 has two references, with two different jobs:

- **torch (``trace_capture.py``)** — the semantic oracle. It answers "is this the right
  model": per-tensor closeness, and exact agreement on discrete decisions (I3).
- **this module** — the numeric contract. Every sum has an order this file states, so
  the engine can be bit-identical to something real, and I1/I2 stay testable.

The order is chosen to be the obvious one, because the engine has to reproduce it in
Metal: ascending index, one product added per step, no reassociation, no fused
multiply-add, fp32 throughout.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def f32(x) -> np.ndarray:
    """fp32, shape preserved.

    ``np.ascontiguousarray`` promotes a scalar to a 1-element array, which quietly turned
    a 1-D ``ordered_sum`` into an array of shape (1,) instead of a scalar. A contract
    whose helpers change the shape of a value is not a contract.
    """
    array = np.asarray(x, dtype=np.float32)
    return array if array.ndim == 0 else np.ascontiguousarray(array, dtype=np.float32)


def ordered_matmul(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """``x @ w.T`` with every output accumulated in ascending ``k``, fp32, no FMA.

    ``w`` is ``[out, in]``, the layout every checkpoint uses. Each iteration performs one
    multiply and one add per output, so the rounding sequence is fully determined by the
    shapes and by this loop — which is the whole point.
    """
    x = f32(x)
    w = f32(w)
    out = np.zeros((*x.shape[:-1], w.shape[0]), dtype=np.float32)
    for k in range(x.shape[-1]):
        out = f32(out + f32(x[..., k, None] * w[None, :, k]))
    return out


def ordered_sum(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """A sum with the same contract: ascending index, one addition per step."""
    x = f32(x)
    moved = np.moveaxis(x, axis, -1)
    total = np.zeros(moved.shape[:-1], dtype=np.float32)
    for k in range(moved.shape[-1]):
        total = f32(total + moved[..., k])
    return total


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    """The reference's order: fp32, mean of squares, ``1/sqrt``, then the weight.

    ``1.0/np.sqrt`` rather than a hardware rsqrt: the two differ in the last bits, and
    a contract has to name one of them.
    """
    x32 = f32(x)
    variance = f32(ordered_sum(f32(x32 * x32), axis=-1) / np.float32(x32.shape[-1]))
    inverse = f32(np.float32(1.0) / np.sqrt(f32(variance + np.float32(eps))))
    return f32(f32(weight) * f32(x32 * inverse[..., None]))


def exp32(x: np.ndarray) -> np.ndarray:
    """``exp`` evaluated in double precision and rounded to fp32.

    Every transcendental here is defined that way, because that is the only formulation
    measured to agree bit-for-bit between this file and Swift. Measured on 6000 inputs
    across the ranges the model uses: numpy's own float32 ``sin``/``cos`` differ from
    Swift's in 12–18% of cases, Swift's differ from libm's ``sinf``/``cosf`` in ~1%, and
    computing in double and rounding agrees in **0 of 6000**. ``exp`` happens to agree
    either way; it is stated the same way as the others so the contract is one sentence
    rather than a table of exceptions.
    """
    return f32(np.exp(np.float64(x)))


def sin32(x: np.ndarray) -> np.ndarray:
    return f32(np.sin(np.float64(x)))


def cos32(x: np.ndarray) -> np.ndarray:
    return f32(np.cos(np.float64(x)))


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Stable sigmoid, with ``exp`` in the contract's precision.

    The branch keeps the exponential from overflowing on either side, and which branch
    is used is part of the contract: ``1/(1+exp(-x))`` and ``exp(x)/(1+exp(x))`` agree
    mathematically and can differ in the last bit.
    """
    x32 = f32(x)
    out = np.empty_like(x32)
    positive = x32 >= 0
    out[positive] = f32(np.float32(1.0) / (np.float32(1.0) + exp32(f32(-x32[positive]))))
    exponential = exp32(x32[~positive])
    out[~positive] = f32(exponential / (np.float32(1.0) + exponential))
    return out


def silu(x: np.ndarray) -> np.ndarray:
    """``x * sigmoid(x)``, in fp32, with the stable sigmoid above."""
    x32 = f32(x)
    return f32(x32 * sigmoid(x32))


def softmax(x: np.ndarray) -> np.ndarray:
    """Max-subtracted, ascending-index sum, fp32 throughout, ``exp`` per the contract."""
    x32 = f32(x)
    shifted = f32(x32 - x32.max(axis=-1, keepdims=True))
    exponentials = exp32(shifted)
    total = ordered_sum(exponentials, axis=-1)
    return f32(exponentials / total[..., None])


def rope_tables(head_dim: int, positions: np.ndarray, theta: float):
    """Rotate-half inverse frequencies in fp32, matching the reference's layout.

    The angles are duplicated to the full head width — the reference computes
    ``emb = cat((freqs, freqs))`` — because rotate-half pairs element ``i`` with
    element ``i + head_dim/2``, and both halves carry the same angle.
    """
    # angles, cos and sin are all computed in double and rounded, per the contract's
    # one-sentence rule for transcendentals. This matters most here: RoPE's angles are
    # the only place the model's arithmetic depends on sin/cos at all.
    exponents = f32(np.arange(0, head_dim, 2, dtype=np.float32) / np.float32(head_dim))
    frequencies = f32(np.float32(1.0) / f32(np.power(np.float32(theta), exponents)))
    angles = np.outer(np.float64(positions), np.float64(frequencies))
    doubled = np.concatenate([angles, angles], axis=-1)
    return f32(np.cos(doubled)), f32(np.sin(doubled))


def apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """``(x*cos) + (rotate_half(x)*sin)``, rotate-half being ``cat(-x2, x1)``."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated = np.concatenate([-x2, x1], axis=-1)
    return f32(f32(f32(x * cos[:, None, :]) + f32(rotated * sin[:, None, :])))


class OrderedQwen3:
    """A `qwen3` dense forward in the order this module defines.

    Deliberately written as one readable pass rather than a module graph: this file *is*
    the numeric specification the Swift and Metal implementations are checked against,
    so it should be readable as one.
    """

    def __init__(self, snapshot_dir: Path):
        from safetensors import safe_open

        snapshot = Path(snapshot_dir)
        config = json.loads((snapshot / "config.json").read_text())
        self.layers = config["num_hidden_layers"]
        self.heads = config["num_attention_heads"]
        self.kv_heads = config["num_key_value_heads"]
        self.head_dim = config.get("head_dim", config["hidden_size"] // self.heads)
        self.eps = config["rms_norm_eps"]
        self.theta = config.get("rope_theta", 1_000_000.0)
        self.tied = config.get("tie_word_embeddings", False)
        self.scale = np.float32(self.head_dim**-0.5)

        self.weights = {}
        # torch is used here for one reason only: the checkpoint stores bfloat16, which
        # numpy has no dtype for. The cast to fp32 is exact, and no arithmetic in this
        # module goes through torch — the whole point of the file is that the order of
        # the arithmetic is visible.
        import torch
        from safetensors.torch import load_file

        for name, tensor in load_file(snapshot / "model.safetensors").items():
            self.weights[name] = f32(tensor.to(torch.float32).numpy())

    def _get(self, name: str) -> np.ndarray:
        return self.weights[name]

    def forward(self, token_ids) -> dict[str, np.ndarray]:
        ids = np.asarray(token_ids, dtype=np.int64)
        tokens = len(ids)
        positions = np.arange(tokens, dtype=np.float32)

        captured: dict[str, np.ndarray] = {}
        hidden = f32(self._get("model.embed_tokens.weight")[ids])
        captured["embed.out"] = hidden

        cos, sin = rope_tables(self.head_dim, positions, self.theta)
        groups = self.heads // self.kv_heads
        # Causal mask: -inf above the diagonal, applied after the scale.
        mask = np.triu(np.full((tokens, tokens), -np.inf, dtype=np.float32), k=1)

        for layer in range(self.layers):
            tag = f"layer.{layer:02d}"
            prefix = f"model.layers.{layer}."
            captured[f"{tag}.hidden_in"] = hidden

            residual = hidden
            normed = rms_norm(hidden, self._get(prefix + "input_layernorm.weight"), self.eps)

            q = ordered_matmul(normed, self._get(prefix + "self_attn.q_proj.weight"))
            k = ordered_matmul(normed, self._get(prefix + "self_attn.k_proj.weight"))
            v = ordered_matmul(normed, self._get(prefix + "self_attn.v_proj.weight"))

            q = q.reshape(tokens, self.heads, self.head_dim)
            k = k.reshape(tokens, self.kv_heads, self.head_dim)
            v = v.reshape(tokens, self.kv_heads, self.head_dim)
            # QK-norm is per head, over head_dim, before RoPE and not on values.
            q = rms_norm(q, self._get(prefix + "self_attn.q_norm.weight"), self.eps)
            k = rms_norm(k, self._get(prefix + "self_attn.k_norm.weight"), self.eps)

            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

            k = np.repeat(k, groups, axis=1)
            v = np.repeat(v, groups, axis=1)

            mixer = np.zeros((tokens, self.heads, self.head_dim), dtype=np.float32)
            for head in range(self.heads):
                # ordered_matmul(x, w) contracts x's last axis with w's last axis, so
                # scores are q @ kᵀ by passing k un-transposed, and the output is
                # weights @ v by passing v transposed.
                scores = f32(ordered_matmul(q[:, head, :], k[:, head, :]) * self.scale)
                scores = f32(scores + mask)
                weights = softmax(scores)
                mixer[:, head, :] = ordered_matmul(weights, f32(v[:, head, :].T))
            mixer = f32(mixer.reshape(tokens, self.heads * self.head_dim))
            attention = ordered_matmul(mixer, self._get(prefix + "self_attn.o_proj.weight"))
            captured[f"{tag}.mixer_out"] = attention
            hidden = f32(residual + attention)

            residual = hidden
            normed = rms_norm(hidden, self._get(prefix + "post_attention_layernorm.weight"), self.eps)
            gated = f32(silu(ordered_matmul(normed, self._get(prefix + "mlp.gate_proj.weight")))
                        * ordered_matmul(normed, self._get(prefix + "mlp.up_proj.weight")))
            mlp = ordered_matmul(gated, self._get(prefix + "mlp.down_proj.weight"))
            captured[f"{tag}.mlp_out"] = mlp
            hidden = f32(residual + mlp)

        hidden = rms_norm(hidden, self._get("model.norm.weight"), self.eps)
        captured["final_norm.out"] = hidden
        head = "lm_head.weight" if "lm_head.weight" in self.weights else "model.embed_tokens.weight"
        captured["logits"] = ordered_matmul(hidden, self._get(head))
        return captured


def _parse_tokens(text: str) -> list[int]:
    return [int(part) for part in text.replace(" ", "").split(",") if part]


def main(argv: list[str] | None = None) -> int:
    """Write the contract's trace for a prompt, so another implementation can be diffed.

    The trace this writes is the artifact half of the reference: it is what the engine's
    own trace is compared against, tensor for tensor, byte for byte.
    """
    import argparse
    from pathlib import Path

    import trace_format

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path, help="a checkpoint directory with config.json and model.safetensors")
    parser.add_argument("out", type=Path, help="trace directory to write")
    parser.add_argument("--tokens", required=True, help="comma-separated token ids")
    parser.add_argument("--model", default="", help="model id recorded in the manifest")
    parser.add_argument("--revision", default="", help="source revision recorded in the manifest")
    args = parser.parse_args(argv)

    tokens = _parse_tokens(args.tokens)
    model = OrderedQwen3(args.snapshot)
    captured = model.forward(tokens)

    tensors = [(name, "f32", values.shape, values.tobytes()) for name, values in captured.items()]
    manifest = trace_format.write_trace(
        args.out,
        tensors=tensors,
        model={"id": args.model, "revision": args.revision, "compute": "fp32", "contract": "ordered_reference.py"},
        prompt={"tokens": [int(t) for t in tokens]},
        producer="ordered-reference-python",
    )
    print(f"wrote {args.out}: {len(tensors)} tensors, digest {manifest['digest'][:16]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
