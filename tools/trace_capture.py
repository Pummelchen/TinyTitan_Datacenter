#!/usr/bin/env python3
"""Reference-side trace capture: run the pinned model and write a golden trace.

The reference defines what "correct" means for M0, so its configuration is part of
the evidence and is recorded in every trace it writes (I6):

- **fp32 compute** (D3). Weights may be stored bf16, but every op runs in IEEE
  fp32 here, because bit-equality between two fp32 implementations with a fixed
  order is provable and bit-equality against an implementation-defined bf16
  matmul is not.
- **`attn_implementation="eager"`** — the sdpa/flash paths are not documented as
  bit-reproducible and their reduction order is not ours to control.
- **Deterministic algorithms, pinned thread count, pinned seed.** Reduction order
  in a threaded CPU matmul is a property of the thread count, so the thread count
  is part of the measurement, not a local preference.
- **One layer resident at a time is the ambition** (D6): an fp32 2 B model is
  ~9.2 GB and does not fit in an 8 GB node. This tool loads what transformers can
  load and says so in the trace; the memory-mapped per-layer path is DC-021.

Runs against a real checkpoint with ``--model``, or against a tiny random model
built from a config with ``--tiny`` — the second needs no download, which is how
the plumbing and the reproducibility question are settled before a 4.5 GB fetch.

Run it with the project venv, not the system interpreter::

    .venv/bin/python tools/trace_capture.py .build/tiny-trace --tiny
"""

from __future__ import annotations

import argparse
import hashlib
import platform
import sys
import time
from pathlib import Path

from trace_format import write_trace

TINY_CONFIG = {
    "vocab_size": 256,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "max_position_embeddings": 128,
    "tie_word_embeddings": True,
}


def _torch():
    import torch  # imported here so the module can be inspected without torch

    return torch


def torch_available() -> bool:
    try:
        _torch()
        return True
    except Exception:
        return False


def configure_determinism(torch, threads: int, seed: int) -> dict:
    """Pin everything that moves a reduction, and report what actually happened."""
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    record = {"threads": torch.get_num_threads(), "seed": seed, "deterministic_algorithms": False}
    try:
        torch.use_deterministic_algorithms(True)
        record["deterministic_algorithms"] = True
    except Exception as exc:  # the flag can be refused by the build
        record["deterministic_algorithms_error"] = f"{type(exc).__name__}: {exc}"
    record["matmul_precision"] = torch.get_float32_matmul_precision()
    return record


def build_tiny_model(seed: int = 1234):
    """A small `qwen3` model with random weights: same code path, no download."""
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch = _torch()
    config = Qwen3Config(**TINY_CONFIG)
    config._attn_implementation = "eager"
    torch.manual_seed(seed)
    model = Qwen3ForCausalLM(config)
    model.eval()
    return model, config


def load_model(model_id: str, revision: str | None, dtype):
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_id, revision=revision)
    config._attn_implementation = "eager"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, config=config, dtype=dtype, low_cpu_mem_usage=True
    )
    model.eval()
    return model, config


def _as_f32_bytes(tensor) -> bytes:
    torch = _torch()
    return tensor.detach().to(torch.float32).contiguous().numpy().tobytes()


def capture_forward(model, input_ids, dtype) -> tuple[list, list]:
    """Run one forward pass, capturing a tensor at every layer boundary.

    Hooks rather than a reimplementation: the numbers come from the reference's own
    modules, so a difference is a difference in the model, not in a transcription
    of it.
    """
    torch = _torch()
    tensors: list[tuple[str, str, list, bytes]] = []
    discrete: list[tuple[str, list, list]] = []

    layers = getattr(model, "model", model).layers
    handles = []

    def record(value, name):
        if value is None:
            return
        if isinstance(value, tuple):
            value = value[0]
        tensors.append((name, "f32", list(value.shape), _as_f32_bytes(value)))

    handles.append(
        getattr(model, "model", model).embed_tokens.register_forward_hook(
            lambda _m, _i, out: record(out, "embed.out")
        )
    )
    for index, layer in enumerate(layers):
        tag = f"layer.{index:02d}"
        handles.append(layer.register_forward_pre_hook(lambda _m, args, tag=tag: record(args[0], f"{tag}.hidden_in")))
        token_mixer = getattr(layer, "self_attn", None) or getattr(layer, "linear_attn", None)
        if token_mixer is not None:
            handles.append(token_mixer.register_forward_hook(lambda _m, _i, out, tag=tag: record(out, f"{tag}.mixer_out")))
        handles.append(layer.mlp.register_forward_hook(lambda _m, _i, out, tag=tag: record(out, f"{tag}.mlp_out")))

    final_norm = getattr(model, "model", model).norm
    handles.append(final_norm.register_forward_hook(lambda _m, _i, out: record(out, "final_norm.out")))

    try:
        with torch.no_grad():
            output = model(input_ids=input_ids)
    finally:
        for handle in handles:
            handle.remove()

    logits = output.logits.detach().to(torch.float32).contiguous()
    tensors.append(("logits", "f32", list(logits.shape), logits.numpy().tobytes()))

    # A router decision is a discrete decision (I3). Dense models have none, and
    # saying so explicitly is better than an empty section nobody notices.
    if getattr(model.config, "num_experts", None) or getattr(model.config, "n_routed_experts", None):
        raise NotImplementedError(
            "router capture belongs with the first MoE model (M1); a dense M0 model has "
            "no discrete decisions, and inventing a placeholder here would be worse than "
            "an explicit gap"
        )
    _ = dtype
    return tensors, discrete


def _file_hashes(model_id: str, revision: str | None) -> dict:
    """sha256 of the source files, for I6. Empty when the model is not local."""
    from huggingface_hub import HfApi

    try:
        info = HfApi().model_info(model_id, revision=revision, files_metadata=True)
    except Exception:
        return {}
    return {s.rfilename: (s.blob_id or "") for s in info.siblings if s.rfilename.endswith((".safetensors", ".json"))}


def capture(
    out: Path,
    *,
    tiny: bool = False,
    model_id: str | None = None,
    revision: str | None = None,
    dtype_name: str = "f32",
    threads: int = 1,
    seed: int = 1234,
    token_ids=None,
    prompt_text: str | None = None,
    producer_notes: str = "",
) -> dict:
    torch = _torch()
    dtype = torch.float32 if dtype_name == "f32" else torch.bfloat16
    record = configure_determinism(torch, threads, seed)

    if tiny:
        model, config = build_tiny_model(seed)
        model = model.to(dtype)
        source = {"repo": "synthetic-config", "revision": "built-in", "params": sum(p.numel() for p in model.parameters())}
        files: dict = {}
    else:
        if not model_id:
            raise SystemExit("--model is required unless --tiny is given")
        model, config = load_model(model_id, revision, dtype)
        source = {"repo": model_id, "revision": revision or "main", "params": sum(p.numel() for p in model.parameters())}
        files = _file_hashes(model_id, revision)

    ids = token_ids if token_ids is not None else [1, 2, 3, 4, 5, 6, 7, 8]
    input_ids = torch.tensor([ids], dtype=torch.long)

    started = time.time()
    tensors, discrete = capture_forward(model, input_ids, dtype)
    elapsed = time.time() - started

    import transformers

    manifest = write_trace(
        out,
        tensors=tensors,
        discrete=discrete,
        model=source,
        reference={
            "tool": "trace_capture",
            "compute_dtype": dtype_name,
            "weight_dtype": dtype_name,
            "attn_implementation": "eager",
            "determinism": record,
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "python": platform.python_version(),
            "machine": platform.platform(),
            "elapsed_seconds": round(elapsed, 3),
            "notes": producer_notes or "one forward pass, hooks at every layer boundary",
        },
        prompt={
            "token_ids": list(ids),
            "text_sha256": hashlib.sha256((prompt_text or "").encode()).hexdigest() if prompt_text else None,
        },
        producer=f"trace_capture (torch {torch.__version__}, transformers {transformers.__version__})",
        extra={"file_blob_ids": files} if files else None,
    )
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("--tiny", action="store_true", help="build a small random model, no download")
    parser.add_argument("--model", help="HuggingFace model id")
    parser.add_argument("--revision", help="model revision (commit hash); pin it for a gate")
    parser.add_argument("--dtype", default="f32", choices=["f32", "bf16"])
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args(argv)

    if not torch_available():
        print("error: torch is not installed in this interpreter", file=sys.stderr)
        return 2

    manifest = capture(
        args.out,
        tiny=args.tiny,
        model_id=args.model,
        revision=args.revision,
        dtype_name=args.dtype,
        threads=args.threads,
        seed=args.seed,
    )
    print(f"wrote {len(manifest['tensors'])} tensor(s)")
    print(f"digest {manifest['digest']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
