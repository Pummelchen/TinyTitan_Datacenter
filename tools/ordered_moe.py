#!/usr/bin/env python3
"""The `qwen3_5_moe` mixture of experts, in the contract's order (M1).

Transcribed from `transformers` v5.17.0 `models/qwen3_5_moe/modeling_qwen3_5_moe.py`
(`Qwen3_5MoeTopKRouter:884`, `Qwen3_5MoeExperts:845`, `Qwen3_5MoeSparseMoeBlock:903`) and
recorded step by step in `docs/reference-qwen36-35b-a3b.md`.

Three things here are decisions rather than transcriptions, and each is stated because the
reference leaves it open or leaves it implicit:

- **the top-k tie-break is ours**: lowest expert index first, the same rule `argmax` uses.
  `torch.topk` does not promise an order for equal probabilities, and I3 needs a comparable
  index set;
- **the accumulation is in ascending expert index**, which the reference does do (it iterates
  `expert_hit`, which `nonzero` returns sorted) and which is therefore also what the future
  ring reduction will do — the single-node contract and the distributed one agree by
  construction;
- **the fp32 islands are named**: the softmax that produces the probabilities is fp32 while
  everything around it is the model's dtype.
"""

from __future__ import annotations

import numpy as np

from ordered_reference import exp32, f32, ordered_matmul, ordered_sum, silu


def softmax(x: np.ndarray) -> np.ndarray:
    """Max-subtracted, ordered sum, fp32 — the router's island."""
    x = f32(x)
    shifted = f32(x - x.max(axis=-1, keepdims=True))
    exponentials = exp32(shifted)
    total = ordered_sum(exponentials, axis=-1)
    return f32(exponentials / total[..., None])


def router(hidden: np.ndarray, weight: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The router: logits, the chosen experts, and their renormalised weights.

    Returns `(logits, indices, weights)` with `indices`/`weights` shaped `[tokens, top_k]`.
    The indices are what I3 asserts exactly — a tolerance cannot express "the same experts".
    """
    hidden = f32(hidden)
    logits = ordered_matmul(hidden, weight)
    probabilities = softmax(logits)

    tokens = probabilities.shape[0]
    indices = np.zeros((tokens, top_k), dtype=np.int64)
    weights = np.zeros((tokens, top_k), dtype=np.float32)
    for token in range(tokens):
        row = probabilities[token]
        # Descending by probability, and by ascending index on a tie: `argsort` is not stable
        # for this purpose, so the tie-break is applied explicitly rather than hoped for.
        order = sorted(range(row.shape[0]), key=lambda expert: (-float(row[expert]), expert))
        chosen = order[:top_k]
        indices[token] = chosen
        selected = f32(np.array([row[expert] for expert in chosen], dtype=np.float32))
        # Renormalised over the chosen experts, unconditionally: the reference does not
        # consult `norm_topk_prob`.
        weights[token] = f32(selected / ordered_sum(selected))
    return logits, indices, weights


def experts(
    hidden: np.ndarray, gate_up: np.ndarray, down: np.ndarray, indices: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """The routed experts, accumulated in ascending expert index.

    `gate_up` is `[experts, 2·intermediate, hidden]` with the gate in the **first** half,
    `down` is `[experts, hidden, intermediate]`.

    Either may also be a **provider**: a callable taking an expert index and returning that expert's rows.
    The layout makes this cheap — the checkpoint's leading axis *is* the expert, so one expert is one row of
    the stacked tensor — and it is what lets the reference run where a layer's stack does not fit: the real
    model's `gate_up` is 3.2 GB in fp32 against about 4.5 GB usable per node, and no account of the
    arithmetic changes when the same values arrive one expert at a time. `DC-032` did the same thing on the
    engine side; this is the contract catching up with it.
    """
    hidden = f32(hidden)
    tokens = hidden.shape[0]
    output = np.zeros_like(hidden)

    # Which (token, rank) pairs each expert serves, then the experts in ascending order.
    pairs: dict[int, list[tuple[int, int]]] = {}
    for token in range(tokens):
        for rank in range(indices.shape[1]):
            pairs.setdefault(int(indices[token, rank]), []).append((token, rank))

    def expert_rows(tensor, expert: int) -> np.ndarray:
        return tensor(expert) if callable(tensor) else tensor[expert]

    for expert in sorted(pairs):
        rows = [token for token, _ in pairs[expert]]
        current = hidden[rows]
        fused = ordered_matmul(current, expert_rows(gate_up, expert))
        half = fused.shape[-1] // 2
        gate, up = fused[:, :half], fused[:, half:]
        activated = f32(silu(gate) * up)
        projected = ordered_matmul(activated, expert_rows(down, expert))
        for position, (token, rank) in enumerate(pairs[expert]):
            output[token] = f32(output[token] + f32(projected[position] * weights[token, rank]))
    return output


def expert_mlp(hidden: np.ndarray, gate: np.ndarray, up: np.ndarray, down: np.ndarray) -> np.ndarray:
    """One ordinary gate/up/down MLP — the shared expert, and the M0b layers' form."""
    hidden = f32(hidden)
    return ordered_matmul(f32(silu(ordered_matmul(hidden, gate)) * ordered_matmul(hidden, up)), down)


def sparse_moe_block(
    hidden: np.ndarray,
    *,
    router_weight: np.ndarray,
    gate_up: np.ndarray,
    down: np.ndarray,
    shared_gate: np.ndarray,
    shared_up: np.ndarray,
    shared_down: np.ndarray,
    shared_scalar_gate: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The whole block: routed sum plus the gated shared expert.

    Returns `(output, indices, weights)`, the last two so a caller can record the discrete
    decision rather than infer it from the numbers.
    """
    hidden = f32(hidden)
    shared = expert_mlp(hidden, shared_gate, shared_up, shared_down)
    _, indices, weights = router(hidden, router_weight, top_k)
    routed = experts(hidden, gate_up, down, indices, weights)
    # The shared expert's own scalar gate, applied to its output and then *added* to the
    # routed sum rather than ranked with it.
    gated_shared = f32(sigmoid(ordered_matmul(hidden, shared_scalar_gate)) * shared)
    return f32(routed + gated_shared), indices, weights


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Stable sigmoid, the same branch structure as everywhere else in the contract."""
    x = f32(x)
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = f32(np.float32(1.0) / (np.float32(1.0) + exp32(f32(-x[positive]))))
    exponential = exp32(x[~positive])
    out[~positive] = f32(exponential / (np.float32(1.0) + exponential))
    return out
