#!/usr/bin/env python3
"""Write a tiny `qwen3_5_moe` checkpoint and the contract's golden output for it.

The M1 counterpart of `make_tiny_qwen35_checkpoint.py`, and the same idea: a checkpoint small
enough to commit, carrying the family's **real** naming, its **real** configuration shape and
its **real** asymmetries — sixteen key heads to thirty-two value heads becomes two to four, so
the grouped-query path is exercised rather than bypassed — plus the golden output that the
engine's forward has to reproduce byte for byte.

    .venv/bin/python tools/make_tiny_qwen36_checkpoint.py

Writes into `tests/DatacenterEngineTests/Fixtures/tiny-qwen36/`:

| | |
| --- | --- |
| `model.safetensors` | the text tower under `model.language_model.` plus an untied `lm_head` |
| `config.json` | the real nesting: `model_type: qwen3_5_moe` with a `text_config` |
| `spec.json` | emitted by the engine's own importer, so the Python side carries no second mapping |
| `golden.json` | the contract's tensors, its **discrete** router decisions, and the argmax |
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

FIXTURE = ROOT / "tests" / "DatacenterEngineTests" / "Fixtures" / "tiny-qwen36"
TOKENS = [3, 1, 4, 1, 5, 9, 2, 6, 5]

# Small, but shaped like the 35 B model rather than like a convenient toy: two layers (one
# Gated DeltaNet and one full attention), **two key heads to four value heads**, eight experts
# with a top-2, a shared expert, and an untied head.
TINY = dict(
    vocab_size=128,
    hidden_size=32,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=8,
    moe_intermediate_size=16,
    shared_expert_intermediate_size=16,
    num_experts=8,
    num_experts_per_tok=2,
    rms_norm_eps=1e-6,
    hidden_act="silu",
    full_attention_interval=2,
    attn_output_gate=True,
    tie_word_embeddings=False,
    linear_num_key_heads=2,
    linear_num_value_heads=4,
    linear_key_head_dim=8,
    linear_value_head_dim=8,
    linear_conv_kernel_dim=4,
    rope_parameters={"rope_theta": 1e7, "rope_type": "default", "partial_rotary_factor": 0.5},
)


def vector(array) -> dict:
    import numpy as np

    flat = np.asarray(array, dtype=np.float32).reshape(-1)
    return {"shape": list(np.asarray(array).shape), "bits": [int(v) for v in flat.view(np.uint32)]}


def main() -> int:
    import numpy as np
    import torch
    from safetensors.torch import save_file
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextModel

    import ordered_qwen36 as q36

    torch.manual_seed(20260915)
    config = Qwen3_5MoeTextConfig(**TINY)
    model = Qwen3_5MoeTextModel(config).eval()
    # `Qwen3_5MoeExperts` allocates with `torch.empty` -- its parameters are meant to be
    # *loaded* -- so a model built here holds uninitialised memory until it is filled. Skipping
    # this produced a fixture of NaNs the first time the same trap was hit in the M1 contract.
    with torch.no_grad():
        for layer in model.layers:
            layer.mlp.experts.gate_up_proj.normal_(0.0, 0.3)
            layer.mlp.experts.down_proj.normal_(0.0, 0.3)
            layer.mlp.gate.weight.normal_(0.0, 0.4)
    head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    with torch.no_grad():
        head.weight.normal_(0.0, 0.3)

    state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    # The checkpoint's own naming: the real file keeps the text tower under
    # `model.language_model.` and ships a separate `lm_head.weight`, because this family is
    # not tied. The importer maps those names; the engine reads them.
    weights = {f"model.language_model.{name}": tensor for name, tensor in state.items()}
    weights["lm_head.weight"] = head.weight.detach().clone()

    FIXTURE.mkdir(parents=True, exist_ok=True)
    save_file({name: tensor.contiguous() for name, tensor in weights.items()}, str(FIXTURE / "model.safetensors"))

    # The config's real nesting, and the keys the real checkpoint carries -- including
    # `full_attention_interval`, which the reference's own configuration object consumes in its
    # constructor and does not keep, so the file is the only place to read it from.
    text_config = {
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "head_dim": config.head_dim,
        "moe_intermediate_size": config.moe_intermediate_size,
        "shared_expert_intermediate_size": config.shared_expert_intermediate_size,
        "num_experts": config.num_experts,
        "num_experts_per_tok": config.num_experts_per_tok,
        "vocab_size": config.vocab_size,
        "rms_norm_eps": config.rms_norm_eps,
        "hidden_act": config.hidden_act,
        "full_attention_interval": TINY["full_attention_interval"],
        "attn_output_gate": True,
        "tie_word_embeddings": False,
        "linear_num_key_heads": config.linear_num_key_heads,
        "linear_num_value_heads": config.linear_num_value_heads,
        "linear_key_head_dim": config.linear_key_head_dim,
        "linear_value_head_dim": config.linear_value_head_dim,
        "linear_conv_kernel_dim": config.linear_conv_kernel_dim,
        "rope_parameters": TINY["rope_parameters"],
    }
    (FIXTURE / "config.json").write_text(
        json.dumps(
            {"model_type": "qwen3_5_moe", "text_config": text_config, "tie_word_embeddings": False},
            indent=1,
        )
        + "\n"
    )

    # The contract's forward on the same weights, in the role-keyed form the engine assembles.
    contract_weights = {
        "embed_tokens": state["embed_tokens.weight"].numpy().astype(np.float32),
        "norm": state["norm.weight"].numpy().astype(np.float32),
        "lm_head": head.weight.detach().numpy().astype(np.float32),
        "layers": [],
    }
    for index in range(config.num_hidden_layers):
        prefix = f"layers.{index}."
        layer = {
            "input_layernorm": state[prefix + "input_layernorm.weight"].numpy().astype(np.float32),
            "post_attention_layernorm": state[prefix + "post_attention_layernorm.weight"].numpy().astype(np.float32),
            "mlp": {
                "router_weight": state[prefix + "mlp.gate.weight"].numpy().astype(np.float32),
                "gate_up": state[prefix + "mlp.experts.gate_up_proj"].numpy().astype(np.float32),
                "down": state[prefix + "mlp.experts.down_proj"].numpy().astype(np.float32),
                "shared_gate": state[prefix + "mlp.shared_expert.gate_proj.weight"].numpy().astype(np.float32),
                "shared_up": state[prefix + "mlp.shared_expert.up_proj.weight"].numpy().astype(np.float32),
                "shared_down": state[prefix + "mlp.shared_expert.down_proj.weight"].numpy().astype(np.float32),
                "shared_scalar_gate": state[prefix + "mlp.shared_expert_gate.weight"].numpy().astype(np.float32),
            },
        }
        if config.layer_types[index] == "full_attention":
            layer["self_attn"] = {
                key: state[f"{prefix}self_attn.{key}.weight"].numpy().astype(np.float32)
                for key in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm")
            }
        else:
            layer["linear_attn"] = {
                "in_proj_qkv": state[prefix + "linear_attn.in_proj_qkv.weight"].numpy().astype(np.float32),
                "in_proj_z": state[prefix + "linear_attn.in_proj_z.weight"].numpy().astype(np.float32),
                "in_proj_b": state[prefix + "linear_attn.in_proj_b.weight"].numpy().astype(np.float32),
                "in_proj_a": state[prefix + "linear_attn.in_proj_a.weight"].numpy().astype(np.float32),
                "conv1d": state[prefix + "linear_attn.conv1d.weight"].numpy().astype(np.float32),
                "A_log": state[prefix + "linear_attn.A_log"].numpy().astype(np.float32),
                "dt_bias": state[prefix + "linear_attn.dt_bias"].numpy().astype(np.float32),
                "norm": state[prefix + "linear_attn.norm.weight"].numpy().astype(np.float32),
                "out_proj": state[prefix + "linear_attn.out_proj.weight"].numpy().astype(np.float32),
            }
        contract_weights["layers"].append(layer)

    spec_config = q36.SpecConfig(
        {
            "hiddenSize": config.hidden_size,
            "numLayers": config.num_hidden_layers,
            "numAttentionHeads": config.num_attention_heads,
            "numKeyValueHeads": config.num_key_value_heads,
            "headDim": config.head_dim,
            "intermediateSize": config.moe_intermediate_size,
            "vocabSize": config.vocab_size,
            "rmsNormEps": config.rms_norm_eps,
            "tieWordEmbeddings": False,
            "attnOutputGate": True,
            "fullAttentionInterval": TINY["full_attention_interval"],
            "numExperts": config.num_experts,
            "numExpertsPerTok": config.num_experts_per_tok,
            "moeIntermediateSize": config.moe_intermediate_size,
            "sharedExpertIntermediateSize": config.shared_expert_intermediate_size,
            "linearKeyHeads": config.linear_num_key_heads,
            "linearValueHeads": config.linear_num_value_heads,
            "linearKeyDim": config.linear_num_key_heads * config.linear_key_head_dim,
            "linearValueHeadDim": config.linear_value_head_dim,
            "linearConvKernelDim": config.linear_conv_kernel_dim,
            "ropeTheta": 1e7,
            "partialRotaryFactor": 0.5,
        }
    )

    captured: dict = {}
    discrete: dict = {}
    logits = q36.text_model_forward(
        contract_weights, spec_config, TOKENS, capture=captured, discrete=discrete
    )

    # Check the contract against the reference module on the very weights just written, so the
    # fixture cannot be golden for the wrong model.
    with torch.no_grad():
        reference = model(input_ids=torch.tensor([TOKENS]), use_cache=False).last_hidden_state.numpy()[0]
    error = float(np.abs(captured["final_norm.out"] - reference).max())
    scale = float(np.abs(reference).max())
    print(f"contract vs reference: max |Δ| {error:.3e} against scale {scale:.3e}")
    if error > max(1e-5, scale * 1e-4):
        print("FAIL: the contract does not match the reference on this fixture")
        return 1

    expected = {
        "note": (
            "Golden output of the numeric contract (tools/ordered_qwen36.py) on the tiny "
            "qwen3_5_moe checkpoint beside this file, for a fixed token sequence. Bit patterns, "
            "because the engine must reproduce the contract exactly -- and the router's "
            "decisions as their own entry, because I3 asserts them apart from any tolerance. "
            "Regenerate with tools/make_tiny_qwen36_checkpoint.py."
        ),
        "tokens": TOKENS,
        "layer_types": list(config.layer_types),
        "tensors": {name: vector(values) for name, values in captured.items()},
        "discrete": {
            name: {"shape": list(values.shape), "values": [int(v) for v in values.reshape(-1)]}
            for name, values in discrete.items()
        },
        "argmax": [int(v) for v in logits.argmax(-1)],
        "source": {
            "repo": "tiny-qwen36-fixture",
            "revision": "generated",
            "transformers": __import__("transformers").__version__,
        },
    }
    (FIXTURE / "golden.json").write_text(json.dumps(expected, indent=1) + "\n")

    binary = ROOT / ".build" / "release" / "datacenter-trace"
    if binary.exists():
        import subprocess

        result = subprocess.run(
            [str(binary), "--emit-spec", str(FIXTURE / "spec.json"), str(FIXTURE)],
            capture_output=True, text=True,
        )
        print("spec:", (result.stdout or result.stderr).strip())
    else:
        print("spec: not emitted -- build the release binary first")
    print(f"wrote {FIXTURE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
