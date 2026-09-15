#!/usr/bin/env python3
"""Write a tiny `qwen3_5` checkpoint and the contract's golden output for it.

The real checkpoint is 2 B parameters and the engine reads it a layer at a time, so the
whole path — the safetensors reader, the importer, the per-layer loading, both layer kinds,
the partial RoPE, the tied head — can be exercised end to end only on a model small enough
to commit. This builds one from the real geometry, in the real checkpoint's own naming, and
records what the contract computes for a fixed token sequence so the Swift tests can assert
it bit for bit.

    .venv/bin/python tools/make_tiny_qwen35_checkpoint.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

FIXTURE = ROOT / "tests" / "DatacenterEngineTests" / "Fixtures" / "tiny-qwen35"
TOKENS = [3, 17, 5, 42, 8]
HIDDEN = 32

# Small, but not so small that a mistake can hide: four layers so three are Gated DeltaNet
# and one is full attention, four value heads with distinct widths, and a partial RoPE.
TINY = dict(
    vocab_size=64,
    hidden_size=HIDDEN,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    intermediate_size=48,
    full_attention_interval=4,
    linear_num_key_heads=2,
    linear_num_value_heads=2,
    linear_key_head_dim=8,
    linear_value_head_dim=8,
    linear_conv_kernel_dim=4,
    rms_norm_eps=1e-6,
    hidden_act="silu",
    tie_word_embeddings=True,
    rope_parameters={
        "rope_theta": 1e7,
        "rope_type": "default",
        "partial_rotary_factor": 0.5,
        "mrope_section": [11, 11, 10],
        "mrope_interleaved": True,
    },
)


def vector(array) -> dict:
    array = np.asarray(array, dtype=np.float32)
    return {"shape": list(array.shape), "bits": [int(v) for v in array.reshape(-1).view(np.uint32)]}


def main() -> int:
    from safetensors.torch import save_file
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

    import ordered_qwen35 as q35

    torch.manual_seed(20260915)
    config = Qwen3_5TextConfig(**TINY)
    model = Qwen3_5TextModel(config).eval()
    state = {name: tensor.detach().float() for name, tensor in model.state_dict().items()}

    # The checkpoint's own naming: the text tower lives under `model.language_model.`, which
    # is what the importer maps and what the engine therefore has to read.
    weights = {f"model.language_model.{name}": tensor for name, tensor in state.items()}
    FIXTURE.mkdir(parents=True, exist_ok=True)
    save_file(weights, str(FIXTURE / "model.safetensors"))

    # The real config's nesting, so the importer is exercised by the same shape of file.
    text_config = {key: value for key, value in config.to_dict().items() if not key.startswith("_")}
    # `Qwen3_5TextConfig` has no `attn_output_gate` attribute at all, yet
    # `Qwen3_5Attention.__init__` hardcodes `2 * num_attention_heads * head_dim` and its
    # forward always chunks the pair — so the module doubles the query *unconditionally* and
    # never reads a flag. The real checkpoint's config nevertheless carries
    # `attn_output_gate: true`, and the IR's shape contract reads it, so the key is written
    # here to match the checkpoint the contract is written against. Recorded in
    # docs/reference-qwen35-2b.md: the flag describes the checkpoint, it does not switch
    # anything.
    text_config["attn_output_gate"] = True
    # Likewise `full_attention_interval`: the class derives `layer_types` but does not
    # necessarily serialise the interval, and the real checkpoint carries it.
    text_config["full_attention_interval"] = TINY["full_attention_interval"]
    (FIXTURE / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5", "text_config": text_config, "tie_word_embeddings": True}, indent=1)
        + "\n"
    )

    # The contract's own forward, on the same weights, in the same role-keyed form the
    # engine assembles.
    contract_weights = {
        "embed_tokens": state["embed_tokens.weight"].numpy().astype(np.float32),
        "norm": state["norm.weight"].numpy().astype(np.float32),
        "layers": [],
    }
    for index in range(config.num_hidden_layers):
        prefix = f"layers.{index}."
        layer = {
            "input_layernorm": state[prefix + "input_layernorm.weight"].numpy().astype(np.float32),
            "post_attention_layernorm": state[prefix + "post_attention_layernorm.weight"].numpy().astype(np.float32),
            "mlp": {
                key: state[f"{prefix}mlp.{key}.weight"].numpy().astype(np.float32)
                for key in ("gate_proj", "up_proj", "down_proj")
            },
        }
        if config.layer_types[index] == "full_attention":
            layer["self_attn"] = {
                key: state[f"{prefix}self_attn.{key}.weight"].numpy().astype(np.float32)
                for key in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm")
            }
        else:
            layer["linear_attn"] = {
                key: state[f"{prefix}linear_attn.{key}"].numpy().astype(np.float32)
                for key in ("in_proj_qkv.weight", "in_proj_z.weight", "in_proj_b.weight", "in_proj_a.weight",
                            "conv1d.weight", "A_log", "dt_bias", "norm.weight", "out_proj.weight")
            }
            layer["linear_attn"] = {
                "in_proj_qkv": layer["linear_attn"]["in_proj_qkv.weight"],
                "in_proj_z": layer["linear_attn"]["in_proj_z.weight"],
                "in_proj_b": layer["linear_attn"]["in_proj_b.weight"],
                "in_proj_a": layer["linear_attn"]["in_proj_a.weight"],
                "conv1d": layer["linear_attn"]["conv1d.weight"],
                "A_log": layer["linear_attn"]["A_log"],
                "dt_bias": layer["linear_attn"]["dt_bias"],
                "norm": layer["linear_attn"]["norm.weight"],
                "out_proj": layer["linear_attn"]["out_proj.weight"],
            }
        contract_weights["layers"].append(layer)

    captured: dict[str, np.ndarray] = {}
    q35.text_model_forward(contract_weights, config, TOKENS, capture=captured)

    expected = {
        "note": (
            "Golden output of the numeric contract (tools/ordered_qwen35.py) on the tiny "
            "checkpoint beside this file, for a fixed token sequence. Bit patterns, because "
            "the engine must reproduce the contract exactly. Regenerate with "
            "tools/make_tiny_qwen35_checkpoint.py."
        ),
        "tokens": TOKENS,
        "layer_types": list(config.layer_types),
        "tensors": {name: vector(values) for name, values in captured.items()},
        "argmax": [int(v) for v in captured["logits"].argmax(-1)],
    }
    (FIXTURE / "golden.json").write_text(json.dumps(expected, indent=1) + "\n")

    # The spec, emitted by the engine's own importer: the Python side reads *this* rather
    # than carrying a second copy of the mapping (L2). Requires the release binary to have
    # been built, which the end-to-end gate builds anyway.
    binary = ROOT / ".build" / "release" / "datacenter-trace"
    if binary.exists():
        import subprocess

        result = subprocess.run(
            [str(binary), "--emit-spec", str(FIXTURE / "spec.json"), str(FIXTURE)], capture_output=True, text=True
        )
        print("spec:", (result.stdout or result.stderr).strip())
    else:
        print("spec: skipped, build first with `swift build -c release`")

    size = (FIXTURE / "model.safetensors").stat().st_size
    print(f"wrote {FIXTURE}: checkpoint {size} bytes, {len(captured)} captured tensors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
