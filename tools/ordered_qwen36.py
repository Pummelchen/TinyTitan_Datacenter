#!/usr/bin/env python3
"""The `qwen3_5_moe` text tower in the contract's order (M1), streamed and discrete-recording.

This is `ordered_qwen35`'s model forward with the mixture substituted for the dense
feed-forward, and it reuses that module rather than restating it — which is not laziness but
the *result* of `tools/compare_reference_modules.py`: the two families' Gated DeltaNet,
attention, RoPE, conv, `l2norm` and chunked rule are identical entity for entity, so a second
transcription would be a second thing to keep in step for no gain.

What is new here is the mixture (from `ordered_moe`) and the **discrete decision**: I3 requires
the router's top-k index sets to match the reference *exactly* and to be checked separately
from any tolerance, so the forward records them as data. They are not inferred from the
numbers afterwards, because the numbers are exactly what a quantisation error would perturb —
by the time you could infer the indices from the logits, the interesting failure has already
happened.
"""

from __future__ import annotations

import numpy as np

import ordered_moe as moe
import ordered_qwen35 as q35
from ordered_reference import f32, ordered_matmul

__all__ = ["SpecConfig", "moe_decoder_layer", "mixer_weights", "text_model_forward", "streamed_text_forward"]


class SpecConfig(q35.SpecConfig):
    """The spec's configuration, plus the mixture's fields."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.num_experts = config.get("numExperts")
        # The key is the IR's own spelling. Reading `numExpertsPerTok` here left `top_k` None
        # and only failed when the contract was driven by a spec the *engine* had emitted —
        # which is exactly what reading the spec instead of a private copy is supposed to
        # catch, and did.
        self.num_experts_per_tok = config["numExpertsPerToken"]
        self.moe_intermediate_size = config.get("moeIntermediateSize")
        # The shared expert's width is its own field; it happens to equal the routed experts'
        # width in this family, and a contract that assumed so would be right here and wrong
        # somewhere else.
        self.shared_expert_intermediate_size = (
            config.get("sharedExpertIntermediateSize") or self.moe_intermediate_size
        )


def mixer_weights(names: dict, materialise, provider=None) -> dict:
    """One layer's roles, with the mixture in place of a dense feed-forward.

    `provider` names a callable factory: given a tensor name it returns a callable that fetches one expert's
    rows. With it the routed experts are never materialised as a stack; without it the whole layer is, which
    is the form that does not fit on an 8 GB node for the real model. The shared expert is small either way.
    """
    weights = q35.layer_weights(names, materialise)
    stack = provider or materialise
    weights["mlp"] = {
        "router_weight": materialise(names["router.logits"]),
        "gate_up": stack(names["expert.stack_gate_up"]),
        "down": stack(names["expert.stack_down"]),
        "shared_gate": materialise(names["expert.shared.gate"]),
        "shared_up": materialise(names["expert.shared.up"]),
        "shared_down": materialise(names["expert.shared.down"]),
        "shared_scalar_gate": materialise(names["expert.shared.scalar"]),
    }
    return weights


def moe_decoder_layer(
    hidden: np.ndarray, weights: dict, config: SpecConfig, layer_index: int,
    cos: np.ndarray, sin: np.ndarray, mask: np.ndarray, discrete: dict | None = None,
) -> np.ndarray:
    """`Qwen3_5MoeDecoderLayer.forward:945`: residual, mixer, residual, mixture."""
    hidden = f32(hidden)
    residual = hidden
    normed = q35.rms_norm(hidden, weights["input_layernorm"], config.rms_norm_eps)
    if config.layer_types[layer_index] == "full_attention":
        mixed = q35.full_attention_layer(normed, weights["self_attn"], config, cos, sin, mask)
    else:
        mixed = q35.gated_delta_net_layer(normed[None], weights["linear_attn"], config)[0]
    hidden = f32(residual + mixed)

    residual = hidden
    normed = q35.rms_norm(hidden, weights["post_attention_layernorm"], config.rms_norm_eps)
    output, indices, _ = moe.sparse_moe_block(
        normed, top_k=config.num_experts_per_tok, **weights["mlp"]
    )
    if discrete is not None:
        discrete[f"layer.{layer_index:02d}.router.topk"] = indices
    return f32(residual + output)


def text_model_forward(
    weights: dict, config: SpecConfig, tokens, capture: dict | None = None, discrete: dict | None = None
) -> np.ndarray:
    """The text tower end to end from in-memory weights, returning the logits.

    The in-memory twin of `streamed_text_forward`, and the one the reference module is
    compared against: `weights` is `{"embed_tokens", "norm", "layers": [...]}`, each layer
    holding the role-keyed arrays an importer maps to.
    """
    tokens = list(tokens)
    hidden = f32(weights["embed_tokens"][tokens])
    if capture is not None:
        capture["embed.out"] = hidden

    cos, sin = q35.rope_tables(config, np.arange(len(tokens), dtype=np.float64))
    mask = np.triu(np.full((len(tokens), len(tokens)), -np.inf, dtype=np.float32), k=1)

    for index, layer in enumerate(weights["layers"]):
        if capture is not None:
            capture[f"layer.{index:02d}.hidden_in"] = hidden
        hidden = moe_decoder_layer(hidden, layer, config, index, cos, sin, mask, discrete)
        if capture is not None:
            capture[f"layer.{index:02d}.hidden_out"] = hidden

    hidden = q35.rms_norm(hidden, weights["norm"], config.rms_norm_eps)
    if capture is not None:
        capture["final_norm.out"] = hidden
    # This family is NOT tied: the head is its own tensor.
    logits = ordered_matmul(hidden, weights["lm_head"])
    if capture is not None:
        capture["logits"] = logits
    return logits


def streamed_text_forward(
    spec: dict, source, tokens, capture: dict | None = None, discrete: dict | None = None,
    head_block: int = 8192, stream_experts: bool = False
) -> np.ndarray:
    """The text tower, one layer resident at a time, recording the router's decisions.

    `source.tensor(name)` returns a whole tensor in fp32 and `source.rows(name, start, end)` a
    row range — the same two operations the Swift engine's reader offers.

    With `stream_experts` it fetches the routed experts **by index** instead of materialising a layer's
    stack, which for the real model is 3.2 GB in fp32 against about 4.5 GB usable per node. The values are
    the same ones — one expert is one row of the stacked tensor — so the arithmetic is unchanged, and
    `tools/test_ordered_qwen36.py` asserts the two paths produce byte-identical captures and decisions
    rather than trusting that argument.
    """
    config = SpecConfig(spec["config"])
    names = q35.roles_by_block(spec)
    tokens = list(tokens)

    def provider(name):
        """Fetch one expert's rows on demand, so a layer's stack is never materialised.

        Raises if the source cannot do row ranges: silently falling back to the whole stack would be the
        kind of quiet substitution this project keeps writing tests against.
        """
        if not hasattr(source, "rows"):
            raise SystemExit(f"the source cannot read rows, so experts cannot be streamed for {name}")

        def fetch(expert: int) -> np.ndarray:
            return source.rows(name, expert, expert + 1)[0]

        return fetch

    def materialise(name):
        return source.tensor(name)

    embedding = names["embed"].get("token.embedding")
    if embedding is None:
        raise ValueError("the spec has no token.embedding tensor")
    hidden = f32(np.concatenate([source.rows(embedding, token, token + 1) for token in tokens], axis=0))
    if capture is not None:
        capture["embed.out"] = hidden

    cos, sin = q35.rope_tables(config, np.arange(len(tokens), dtype=np.float64))
    mask = np.triu(np.full((len(tokens), len(tokens)), -np.inf, dtype=np.float32), k=1)

    for index in range(config.num_hidden_layers):
        block = f"layer.{index:02d}"
        if capture is not None:
            capture[f"{block}.hidden_in"] = hidden
        weights = mixer_weights(names[block], materialise, provider if stream_experts else None)
        hidden = moe_decoder_layer(hidden, weights, config, index, cos, sin, mask, discrete)
        if capture is not None:
            capture[f"{block}.hidden_out"] = hidden
        del weights  # released before the next layer is read, as the engine does

    hidden = q35.rms_norm(hidden, source.tensor(names["final"]["norm.final"]), config.rms_norm_eps)
    if capture is not None:
        capture["final_norm.out"] = hidden

    head_name = names.get("head", {}).get("head.lm", embedding)
    logits = np.zeros((len(tokens), config.vocab_size), dtype=np.float32)
    start = 0
    while start < config.vocab_size:
        end = min(start + head_block, config.vocab_size)
        logits[:, start:end] = ordered_matmul(hidden, f32(source.rows(head_name, start, end)))
        start = end
    if capture is not None:
        capture["logits"] = logits
    return logits
