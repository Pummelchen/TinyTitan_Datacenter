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


class DiskWeights:
    """Tensors read from a safetensors snapshot on demand, never all at once.

    D6's problem: an fp32 2 B model is ~9.2 GB and an 8 GB node cannot hold it. The
    answer is not a smaller model but a smaller *working set* — one layer resident,
    cast to fp32 as it is used, released afterwards. This class is the reader; it
    also refuses to guess, so a tensor in the checkpoint that no module claims is an
    error rather than a silent omission.
    """

    def __init__(self, snapshot_dir: Path, dtype):
        from safetensors import safe_open

        self._torch = _torch()
        self._safe_open = safe_open
        self.dir = Path(snapshot_dir)
        self.dtype = dtype
        self.claimed: set[str] = set()

        index = self.dir / "model.safetensors.index.json"
        self.map: dict[str, str] = {}
        if index.is_file():
            import json as _json

            self.map = _json.loads(index.read_text())["weight_map"]
        else:
            shards = sorted(
                p for p in self.dir.iterdir() if p.suffix == ".safetensors" or ".safetensors-" in p.name
            )
            for shard in shards:
                with safe_open(shard, framework="pt") as handle:
                    for name in handle.keys():
                        self.map[name] = shard.name
        if not self.map:
            raise SystemExit(f"no safetensors found under {self.dir}")

    def get(self, name: str):
        shard = self.dir / self.map[name]
        with self._safe_open(shard, framework="pt") as handle:
            return handle.get_tensor(name).to(self.dtype)

    def load_into(self, module, prefix: str, *, required: bool = True) -> int:
        """Assign every tensor under ``prefix`` into ``module``, replacing meta
        placeholders. Returns how many were loaded."""
        state = {}
        for name in self.map:
            if name.startswith(prefix):
                state[name[len(prefix) :]] = self.get(name)
                self.claimed.add(name)
        if not state:
            if required:
                raise SystemExit(f"nothing in the checkpoint under {prefix!r}")
            return 0
        missing, unexpected = module.load_state_dict(state, strict=False, assign=True)
        # Non-persistent buffers (a rotary table, for instance) are computed rather
        # than stored, so their absence is expected and the caller rebuilds them.
        missing = [m for m in missing if m not in getattr(module, "_non_persistent_buffers_set", set())]
        if missing or unexpected:
            raise SystemExit(f"{prefix}: missing {missing}, unexpected {unexpected}")
        return len(state)

    def report_unclaimed(self, prefix: str = "") -> list[str]:
        return sorted(n for n in self.map if n not in self.claimed and n.startswith(prefix))

    def release(self, module) -> None:
        """Hand a layer's memory back: replace its parameters and buffers with empty
        meta placeholders. Assigning a meta tensor to ``param.data`` is refused by
        PyTorch's variable hooks, so the entries are replaced rather than mutated."""
        torch = self._torch
        for name, param in list(module._parameters.items()):
            if param is not None:
                module._parameters[name] = torch.nn.Parameter(torch.empty(0, device="meta"), requires_grad=False)
        for name, buffer in list(module._buffers.items()):
            if buffer is not None:
                module._buffers[name] = torch.empty(0, device="meta")


def optional_kernel_availability() -> dict:
    """Which fused kernels the reference actually used.

    The Gated DeltaNet has a reference PyTorch implementation and an optional fused
    one; they are correct but not necessarily bit-identical to each other. A trace
    that does not say which was in play cannot be used to gate the other, so the
    answer goes in the manifest (DC-024).
    """
    import importlib.util

    return {name: importlib.util.find_spec(name) is not None for name in ("causal_conv1d", "fla")}


def peak_rss_bytes() -> int:
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def capture_from_disk(
    out: Path,
    *,
    snapshot_dir: Path,
    dtype_name: str = "f32",
    threads: int = 1,
    seed: int = 1234,
    token_ids=None,
    model_id: str | None = None,
    revision: str | None = None,
    producer_notes: str = "",
) -> dict:
    """Capture with one decoder layer resident at a time (D6, DC-021).

    The forward is driven layer by layer rather than through ``model(...)``, because
    the whole point is that the model does not exist in memory at any moment. Every
    step still calls the reference's *own* modules — a transcription of the forward
    pass would be a second implementation to be wrong in.
    """
    torch = _torch()
    from transformers import AutoConfig, AutoModelForCausalLM

    dtype = torch.float32 if dtype_name == "f32" else torch.bfloat16
    determinism = configure_determinism(torch, threads, seed)

    snapshot_dir = Path(snapshot_dir)
    config = AutoConfig.from_pretrained(snapshot_dir)
    config._attn_implementation = "eager"
    text_config = getattr(config, "text_config", config)

    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
    weights = DiskWeights(snapshot_dir, dtype)

    # The checkpoint prefix differs by wrapper: a plain causal LM stores `model.*`,
    # a conditional-generation checkpoint nests the text tower.
    prefix = "model."
    if not any(name.startswith(prefix + "embed_tokens.") for name in weights.map):
        prefix = "model.language_model."
    if not any(name.startswith(prefix + "embed_tokens.") for name in weights.map):
        raise SystemExit(
            "could not find the text tower's embedding in the checkpoint; "
            f"first keys: {sorted(weights.map)[:4]}"
        )

    ids = token_ids if token_ids is not None else [1, 2, 3, 4, 5, 6, 7, 8]
    input_ids = torch.tensor([ids], dtype=torch.long)
    tensors: list[tuple[str, str, list, bytes]] = []

    def record(value, name):
        if value is None:
            return
        if isinstance(value, tuple):
            value = value[0]
        tensors.append((name, "f32", list(value.shape), _as_f32_bytes(value)))

    # The forward is the reference's own `model(...)`: it builds the causal mask and
    # dispatches to the attention interface exactly as the resident path does. The
    # only thing this tool changes is *when* a layer's weights exist — loaded by a
    # pre-hook, released by a post-hook — which is why the two paths must agree byte
    # for byte, and a test asserts that they do.
    handles = []
    for index, layer in enumerate(model.model.layers):
        tag = f"layer.{index:02d}"

        def load_layer(module, _args, index=index):
            # Returning a value from a forward pre-hook REPLACES the module's
            # arguments. `load_into` returns a count, so returning it here would
            # hand the next hook an int where the hidden state should be — which is
            # exactly what happened the first time this ran. Return None, always.
            weights.load_into(module, f"{prefix}layers.{index}.")
            return None

        handles.append(layer.register_forward_pre_hook(load_layer))
        handles.append(
            layer.register_forward_pre_hook(lambda _m, args, tag=tag: record(args[0], f"{tag}.hidden_in"))
        )
        mixer = getattr(layer, "self_attn", None) or getattr(layer, "linear_attn", None)
        if mixer is not None:
            handles.append(
                mixer.register_forward_hook(lambda _m, _i, out, tag=tag: record(out, f"{tag}.mixer_out"))
            )
        handles.append(layer.mlp.register_forward_hook(lambda _m, _i, out, tag=tag: record(out, f"{tag}.mlp_out")))
        handles.append(
            layer.register_forward_hook(lambda module, _i, _o: weights.release(module))
        )

    with torch.no_grad():
        weights.load_into(model.model.embed_tokens, prefix + "embed_tokens.")
        handles.append(model.model.embed_tokens.register_forward_hook(lambda _m, _i, out: record(out, "embed.out")))
        # A rotary table is computed, not stored, so it is rebuilt rather than
        # inherited as a meta placeholder.
        model.model.rotary_emb = type(model.model.rotary_emb)(text_config)
        weights.load_into(model.model.norm, prefix + "norm.")
        handles.append(model.model.norm.register_forward_hook(lambda _m, _i, out: record(out, "final_norm.out")))

        # The head: a checkpoint may ship a separate `lm_head.weight` even when the
        # config says the weights are tied, which is what Qwen3-0.6B does. The file
        # wins, and if the config claims tying then the two tensors must agree —
        # a checkpoint that disagrees with its own config is a finding, not a detail.
        head_names = [n for n in weights.map if n.endswith("lm_head.weight")]
        tied = bool(getattr(text_config, "tie_word_embeddings", False))
        if head_names:
            weights.load_into(model.lm_head, head_names[0][: -len("weight")])
            head_note = "separate tensor in the checkpoint"
            if tied:
                if not torch.equal(model.lm_head.weight, model.model.embed_tokens.weight):
                    raise SystemExit(
                        "config says tie_word_embeddings but the checkpoint's lm_head.weight "
                        "differs from embed_tokens.weight; refusing to guess which is meant"
                    )
                head_note += ", verified identical to embed_tokens (config says tied)"
        elif tied:
            model.lm_head.weight = model.model.embed_tokens.weight
            head_note = "tied to embed_tokens, absent from the checkpoint"
        else:
            raise SystemExit("no lm_head in the checkpoint and tie_word_embeddings is false")
        try:
            output = model(input_ids=input_ids)
        finally:
            for handle in handles:
                handle.remove()
    record(output.logits, "logits")

    unclaimed = [n for n in weights.report_unclaimed() if "visual" not in n and "mtp" not in n]
    if unclaimed:
        raise SystemExit(
            f"the checkpoint carries {len(unclaimed)} tensor(s) no module claimed, "
            f"e.g. {unclaimed[:3]} — an unread tensor is a silent omission, not a success"
        )

    import transformers

    manifest = write_trace(
        out,
        tensors=tensors,
        discrete=[],
        model={
            "repo": model_id or str(snapshot_dir),
            "revision": revision or "unpinned",
            "snapshot": str(snapshot_dir),
            "checkpoint_tensors": len(weights.map),
        },
        reference={
            "tool": "trace_capture",
            "mode": "one-layer-resident",
            "compute_dtype": dtype_name,
            "weight_dtype": dtype_name,
            "attn_implementation": "eager",
            "determinism": determinism,
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "peak_rss_bytes": peak_rss_bytes(),
            "linear_attention_layers": sum(
                1 for t in (getattr(text_config, "layer_types", None) or []) if t == "linear_attention"
            ),
            "delta_rule_path": (
                "chunked (prefill: one full-sequence forward)"
                if any(t == "linear_attention" for t in (getattr(text_config, "layer_types", None) or []))
                else None
            ),
            "optional_kernels": optional_kernel_availability(),
            "notes": producer_notes or "layer-by-layer forward with per-layer weight loading",
        },
        prompt={"token_ids": list(ids)},
        producer=f"trace_capture disk (torch {torch.__version__}, transformers {transformers.__version__})",
    )
    return manifest


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
    determinism = configure_determinism(torch, threads, seed)

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
            "determinism": determinism,
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "python": platform.python_version(),
            "machine": platform.platform(),
            "elapsed_seconds": round(elapsed, 3),
            "optional_kernels": optional_kernel_availability(),
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
    parser.add_argument("--from-disk", action="store_true", help="one decoder layer resident at a time (DC-021)")
    parser.add_argument("--snapshot", type=Path, help="a local snapshot directory; implies --from-disk")
    parser.add_argument("--cache", type=Path, default=Path(".build/hf-cache"), help="where to keep downloaded snapshots")
    parser.add_argument("--dtype", default="f32", choices=["f32", "bf16"])
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args(argv)

    if not torch_available():
        print("error: torch is not installed in this interpreter", file=sys.stderr)
        return 2

    if args.snapshot or args.from_disk:
        snapshot = args.snapshot
        if snapshot is None:
            if not args.model:
                print("error: --from-disk needs --model or --snapshot", file=sys.stderr)
                return 2
            from huggingface_hub import snapshot_download

            snapshot = Path(
                snapshot_download(
                    args.model,
                    revision=args.revision,
                    cache_dir=args.cache,
                    allow_patterns=["*.json", "*.safetensors*", "*.txt", "*.jinja", "*.model"],
                )
            )
        manifest = capture_from_disk(
            args.out,
            snapshot_dir=snapshot,
            dtype_name=args.dtype,
            threads=args.threads,
            seed=args.seed,
            model_id=args.model,
            revision=args.revision,
        )
        print(f"wrote {len(manifest['tensors'])} tensor(s) from {snapshot}")
        print(f"peak RSS {manifest['reference']['peak_rss_bytes'] / 2**30:.2f} GiB")
        print(f"digest {manifest['digest']}")
        return 0

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
