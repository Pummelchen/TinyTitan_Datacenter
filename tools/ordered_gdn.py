#!/usr/bin/env python3
"""The Gated DeltaNet's chunked delta rule, in the contract's order (M0b).

Transcribed from `transformers` v5.17.0 `models/qwen3_5/modeling_qwen3_5.py`
(`torch_chunk_gated_delta_rule:301`, `l2norm:294`, `Qwen3_5RMSNormGated:225`) and recorded
step by step in `docs/reference-qwen35-2b.md`. Nothing here is inferred from "how a delta
rule usually works": the reference is the authority, and `test_ordered_gdn.py` checks this
implementation against the reference's *own function* on random inputs.

The order is stated rather than inherited. Every contraction is an ascending-index fp32
accumulation (D3), the triangular solve is forward substitution with rows taken in order,
and the cumulative decay is a left-to-right prefix sum. Torch's blocked LAPACK solve and
its BLAS matmuls use different orders, so agreement with torch is a *semantic* check —
closeness, not bit-equality — exactly as for the dense path.
"""

from __future__ import annotations

import numpy as np

from ordered_reference import exp32, f32, ordered_matmul, ordered_sum, silu


def cumulative_sum(x: np.ndarray, axis: int) -> np.ndarray:
    """A prefix sum with the contract's order: left to right, one addition per step."""
    x = f32(x)
    moved = np.moveaxis(x, axis, -1)
    total = np.zeros_like(moved)
    running = np.zeros(moved.shape[:-1], dtype=np.float32)
    for index in range(moved.shape[-1]):
        running = f32(running + moved[..., index])
        total[..., index] = running
    return np.moveaxis(total, -1, axis)


def l2norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """`x * rsqrt((x·x).sum(-1) + eps)`, with the sum in the contract's order.

    The reference uses `torch.rsqrt`, which is not required to be correctly rounded; the
    contract states `1/sqrt` instead, as it does everywhere else.
    """
    x = f32(x)
    squares = f32(x * x)
    total = ordered_sum(squares, axis=-1)
    inverse = f32(np.float32(1.0) / np.sqrt(f32(total + np.float32(eps))))
    return f32(x * inverse[..., None])


def solve_unit_lower(lower: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Forward substitution for a unit lower triangular system, rows in order.

    `torch.linalg.solve_triangular(..., upper=False, unitriangular=True)` solves the same
    system; the *order* the rows are eliminated in is what the contract fixes here.
    """
    lower = f32(lower)
    rhs = f32(rhs)
    rows = lower.shape[-1]
    solution = np.zeros_like(rhs)
    for row in range(rows):
        accumulator = rhs[..., row, :].copy()
        for column in range(row):
            accumulator = f32(accumulator - f32(lower[..., row, column, None] * solution[..., column, :]))
        solution[..., row, :] = accumulator
    return solution


def gated_rms_norm(hidden: np.ndarray, gate: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """`Qwen3_5RMSNormGated:225` — two orderings that are the point of the class.

    The reference rounds the normalised value to bf16 *before* the weight multiply, then
    activates the gate in fp32 *after* it. In an all-fp32 contract the first rounding does
    not exist, which is a deliberate difference from the bf16 oracle and is recorded here
    rather than glossed.
    """
    hidden = f32(hidden)
    variance = f32(ordered_sum(f32(hidden * hidden), axis=-1) / np.float32(hidden.shape[-1]))
    inverse = f32(np.float32(1.0) / np.sqrt(f32(variance + np.float32(eps))))
    normalised = f32(weight * f32(hidden * inverse[..., None]))
    # The *one* silu in the contract, imported rather than spelled again: this function
    # originally carried its own single-branch form, which differs from the stable one in
    # the last bit for negative inputs. The cross-language test found it as a 1-ULP
    # disagreement, which is exactly what a second implementation of a stated function
    # costs.
    return f32(normalised * silu(gate))


def recurrent_gated_delta_rule(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    decay: np.ndarray,
    beta: np.ndarray,
    initial_state: np.ndarray | None = None,
    use_l2norm: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """`torch_recurrent_gated_delta_rule:440` — the **decode** path, one step at a time.

    This is the other half of the reference's Gated DeltaNet, and the reason a cache is a second
    numeric path rather than a free speedup: `Qwen3_5MoeGatedDeltaNet.forward:625` calls *this*
    when `use_precomputed_states and seq_len == 1`, and `chunk_gated_delta_rule` otherwise. The
    two are algebraically equivalent and not bit-identical, because the chunked rule groups its
    sums over a chunk of 64 positions while this accumulates one step at a time.

    Transcribed per line, because the order *is* the arithmetic:

        decay_t = decay[..., i].exp()
        state   = state * decay_t
        kv_mem  = (state * k_t).sum(dim=-2)
        delta   = (v_t - kv_mem) * beta_t
        state   = state + k_t ⊗ delta
        out     = (state * q_t).sum(dim=-2)

    Shapes follow the chunked rule's convention, which is the reference's: `query`/`key` are
    `[batch, length, heads, k_head_dim]`, `value` is `[batch, length, heads, v_head_dim]`,
    `decay`/`beta` are `[batch, length, heads]`, and the state is `[batch, heads, k_head_dim,
    v_head_dim]`. (The reference transposes to `[batch, heads, length, …]` internally and hands
    the state back in the `[batch, heads, k, v]` layout, which is what a cache stores.)

    The reference casts to fp32 before the loop and normalises query and key there too, so the
    whole of this runs in fp32. Getting the axis order wrong is not a shape error in numpy — a
    transposed rule runs happily and returns different numbers — so the test pins it against the
    reference's own function rather than against a shape.
    """
    query = f32(query)
    key = f32(key)
    value = f32(value)
    beta = f32(beta)
    decay = f32(decay)
    batch, length, heads, key_dim = query.shape
    value_dim = value.shape[-1]

    if use_l2norm:
        query = l2norm(query)
        key = l2norm(key)

    # "And always normalize queries by the head dimension" — unconditional, after the l2norm:
    #
    #     query = query / (query.shape[-1] ** 0.5)
    #
    # Missing this line left the output out by a constant factor, which is exactly the shape of
    # bug that looks like a numerical difference rather than a mistake.
    query = f32(query / np.float32(query.shape[-1] ** 0.5))

    if initial_state is None:
        state = np.zeros((batch, heads, key_dim, value_dim), dtype=np.float32)
    else:
        state = f32(initial_state)
    output = np.zeros((batch, length, heads, value_dim), dtype=np.float32)

    for position in range(length):
        q_t = query[:, position]
        k_t = key[:, position]
        v_t = value[:, position]
        # The decay multiplies the whole state before anything else touches it.
        decay_t = exp32(decay[:, position])[..., None, None]
        state = f32(state * decay_t)
        beta_t = beta[:, position][..., None]
        # `(state * k_t).sum(dim=-2)`: the reduction is over the key dimension, in ascending
        # order, which is what `ordered_sum` does and what numpy's own `sum` does not promise.
        kv_memory = ordered_sum(f32(state * k_t[..., None]), axis=-2)
        delta = f32((v_t - kv_memory) * beta_t)
        state = f32(state + f32(k_t[..., None] * delta[..., None, :]))
        output[:, position] = ordered_sum(f32(state * q_t[..., None]), axis=-2)

    return output, state


def chunk_gated_delta_rule(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    decay: np.ndarray,
    beta: np.ndarray,
    chunk_size: int = 64,
    initial_state: np.ndarray | None = None,
    use_qk_l2norm: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """The chunked rule, exactly as `torch_chunk_gated_delta_rule:301` orders it.

    `query`, `key`, `value` are `[B, S, H, D]`; `decay` (the log-space `g`) and `beta` are
    `[B, S, H]`. Returns the output `[B, S, H, D_v]` and the final state `[B, H, D, D_v]`.
    """
    query = np.moveaxis(f32(query), 1, 2)          # [B, H, S, D]
    key = np.moveaxis(f32(key), 1, 2)
    value = np.moveaxis(f32(value), 1, 2)
    beta = np.moveaxis(f32(beta), 1, 2)            # [B, H, S]
    decay = np.moveaxis(f32(decay), 1, 2)

    if use_qk_l2norm:
        query = l2norm(query, eps=1e-6)
        key = l2norm(key, eps=1e-6)
    query = f32(query * np.float32(query.shape[-1] ** -0.5))

    # Pad the sequence up to a whole number of chunks, exactly as the reference does.
    batch, heads, length, key_dim = key.shape
    value_dim = value.shape[-1]
    padding = (chunk_size - length % chunk_size) % chunk_size
    if padding:
        query = np.concatenate([query, np.zeros((batch, heads, padding, key_dim), np.float32)], axis=2)
        key = np.concatenate([key, np.zeros((batch, heads, padding, key_dim), np.float32)], axis=2)
        value = np.concatenate([value, np.zeros((batch, heads, padding, value_dim), np.float32)], axis=2)
        beta = np.concatenate([beta, np.zeros((batch, heads, padding), np.float32)], axis=2)
        decay = np.concatenate([decay, np.zeros((batch, heads, padding), np.float32)], axis=2)

    total = length + padding
    chunks = total // chunk_size
    shape = (batch, heads, chunks, chunk_size, key_dim)
    query = query.reshape(*shape)
    key = key.reshape(*shape)
    value_chunks = value.reshape(batch, heads, chunks, chunk_size, value_dim)
    beta = beta.reshape(batch, heads, chunks, chunk_size)
    decay = decay.reshape(batch, heads, chunks, chunk_size)

    value_beta = f32(value_chunks * beta[..., None])
    key_beta = f32(key * beta[..., None])

    cum_decay = cumulative_sum(decay, axis=3)
    # exp(cum_i - cum_j) with the strictly upper triangle masked to -inf *before* the exp,
    # so no overflow can occur and no future position can contribute.
    pairwise = f32(cum_decay[..., :, None] - cum_decay[..., None, :])
    upper = np.triu(np.ones((chunk_size, chunk_size), dtype=bool), k=1)
    pairwise = np.where(upper, np.float32("-inf"), pairwise)
    pairwise = exp32(pairwise)

    ut_system = f32(ordered_matmul(key_beta, key) * pairwise)
    intra_chunk_attn = f32(ordered_matmul(query, key) * pairwise)
    decayed_key_beta = f32(key_beta * exp32(cum_decay)[..., None])

    new_values = solve_unit_lower(ut_system, value_beta)
    key_cumdecay = solve_unit_lower(ut_system, decayed_key_beta)

    query = f32(query * exp32(cum_decay)[..., None])
    key = f32(key * exp32(f32(cum_decay[..., -1:] - cum_decay))[..., None])
    chunk_decay = exp32(cum_decay[..., -1])

    if initial_state is None:
        state = np.zeros((batch, heads, key_dim, value_dim), dtype=np.float32)
    else:
        state = f32(initial_state)

    output = np.zeros((batch, heads, chunks, chunk_size, value_dim), dtype=np.float32)
    for index in range(chunks):
        v_new = f32(
            new_values[:, :, index] - ordered_matmul(key_cumdecay[:, :, index], f32(state.transpose(0, 1, 3, 2)))
        )
        inter = ordered_matmul(query[:, :, index], f32(state.transpose(0, 1, 3, 2)))
        output[:, :, index] = f32(
            inter + ordered_matmul(intra_chunk_attn[:, :, index], f32(v_new.transpose(0, 1, 3, 2)))
        )
        state = f32(
            state * chunk_decay[:, :, index, None, None]
            + ordered_matmul(f32(key[:, :, index].transpose(0, 1, 3, 2)), f32(v_new.transpose(0, 1, 3, 2)))
        )

    output = output.reshape(batch, heads, total, value_dim)[:, :, :length]
    return np.moveaxis(output, 1, 2).copy(), state
