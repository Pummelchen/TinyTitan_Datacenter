#!/usr/bin/env python3
"""Measure what 4-bit costs, using the M0 gate's own instruments (DC-031, M0c).

The question M0c asks is not "does it run" but "what did it cost", and the gate already
knows how to answer that: the contract runs against the install with the dequantizer in
place of the checkpoint reader, and the result is compared to the fp32 reference on the
frozen prompt set — numbers *and* discrete decisions, separately (I3).

    .venv/bin/python tools/measure_quantization.py --snapshot <snapshot> --spec spec.json \\
        --reference .build/m0-gate --work .build/m0c
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import ordered_qwen35 as q35  # noqa: E402
import ordered_qwen36 as q36  # noqa: E402
import quantize  # noqa: E402
import trace_format  # noqa: E402


def logits_of(trace, tokens: int) -> np.ndarray:
    entry = trace.entry("logits")
    values = np.frombuffer(trace.tensor("logits").payload, dtype=np.float32)
    return values.reshape(entry["shape"]) if len(entry["shape"]) == 2 else values.reshape(tokens, -1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=ROOT / "tools" / "quant_policy.json")
    parser.add_argument("--prompts", type=Path, default=ROOT / "tools" / "m0_prompts.json")
    parser.add_argument("--reference", type=Path, default=ROOT / ".build" / "m0-gate",
                        help="directory holding the fp32 engine traces named <id>-engine")
    parser.add_argument("--work", type=Path, default=ROOT / ".build" / "m0c")
    parser.add_argument("--only", help="run a single prompt id")
    parser.add_argument("--rebuild", action="store_true", help="rebuild the install even if it exists")
    args = parser.parse_args(argv)

    spec = json.loads(args.spec.read_text())
    policy = quantize.load_policy(args.policy)
    prompts = json.loads(args.prompts.read_text())["prompts"]
    if args.only:
        prompts = [p for p in prompts if p["id"] == args.only]
    args.work.mkdir(parents=True, exist_ok=True)
    install = args.work / "install"

    if args.rebuild or not (install / "install.json").exists():
        print("building the install")
        started = time.time()
        manifest = quantize.build_install(args.snapshot, install, spec, policy, str(args.policy))
        print(f"  {len(manifest['tensors'])} tensors quantized, {len(manifest['skipped'])} kept higher, {time.time() - started:.1f} s")
    else:
        manifest = quantize.verify_install(install)
        print(f"reusing the install: {len(manifest['tensors'])} tensors")

    weights = sum(e["shape"][0] * e["padded_columns"] for e in manifest["tensors"])
    payload = sum(e["nbytes"] for e in manifest["tensors"])
    checkpoint_bytes = sum(f.stat().st_size for f in args.snapshot.glob("model.safetensors*"))
    entry_bytes = payload + (install / "install.json").stat().st_size
    print(f"\n  quantized weights: {weights:,}")
    print(f"  quantized payload: {payload:,} bytes ({payload * 8 / max(weights, 1):.2f} bits/weight incl. scales and zeros)")
    print(f"  install total:     {entry_bytes:,} bytes")
    print(f"  checkpoint:        {checkpoint_bytes:,} bytes (bf16, all tensors)")
    print(f"  the quantized tensors alone are {payload * 8 / max(weights, 1):.2f} bits/weight; the rest of the model stays bf16 or fp32")

    source = quantize.InstallSource(install)
    report = {"model": spec["source"]["repo"], "revision": spec["source"]["revision"], "prompts": []}

    for prompt in prompts:
        tokens = prompt["tokens"]
        reference = args.reference / f"{prompt['id']}-engine"
        if not reference.exists():
            print(f"  {prompt['id']}: no fp32 reference at {reference}; run the gate first", file=sys.stderr)
            return 2

        captured: dict[str, np.ndarray] = {}
        decisions: dict[str, np.ndarray] = {}
        started = time.time()
        # The contract follows the spec's own family, the same way the engine does: a family
        # with a mixture records its router's decisions, and I3 makes those a separate
        # measurement from any tolerance.
        if spec.get("family", "qwen3_5") == "qwen3_5_moe":
            q36.streamed_text_forward(spec, source, tokens, capture=captured, discrete=decisions)
        else:
            q35.streamed_text_forward(spec, source, tokens, capture=captured)
        seconds = time.time() - started
        quantized_trace = args.work / f"{prompt['id']}-quantized"
        trace_format.write_trace(
            quantized_trace,
            tensors=[(n, "f32", v.shape, v.tobytes()) for n, v in captured.items()],
            model={"id": spec["source"]["repo"], "revision": spec["source"]["revision"], "quant": "int4-affine"},
            prompt={"tokens": [int(t) for t in tokens]},
            producer="ordered-reference-python",
        )

        fp32 = trace_format.read_trace(reference)
        shared = [n for n in fp32.tensor_names if n in captured]
        worst_relative, worst_name = 0.0, ""
        for name in shared:
            a = np.frombuffer(fp32.tensor(name).payload, dtype=np.float32)
            b = captured[name].reshape(-1)
            scale = float(np.abs(a).max())
            relative = float(np.abs(a - b).max()) / max(scale, 1e-30)
            if relative > worst_relative:
                worst_relative, worst_name = relative, name

        fp32_logits = logits_of(fp32, len(tokens))
        quantized_logits = logits_of(trace_format.read_trace(quantized_trace), len(tokens))
        fp32_tokens = fp32_logits.argmax(-1)
        quantized_tokens = quantized_logits.argmax(-1)
        agreement = int((fp32_tokens == quantized_tokens).sum())
        margins = np.sort(quantized_logits, axis=-1)[:, -2:]
        smallest = float((margins[:, 1] - margins[:, 0]).min())

        entry = {
            "id": prompt["id"],
            "tokens": len(tokens),
            # How many top-k rows this prompt's mixture produced, so a reader can see the
            # decisions were recorded rather than skipped (the comparison itself lives in
            # `test_ordered_qwen36_quant`, which has both sides).
            "router_decision_rows": sum(len(values) for values in decisions.values()),
            "seconds": round(seconds, 1),
            "positions": len(tokens),
            "discrete_agree": agreement,
            "discrete_total": len(tokens),
            "worst_relative": worst_relative,
            "worst_tensor": worst_name,
            "smallest_margin": smallest,
            "argmax_fp32": [int(t) for t in fp32_tokens],
            "argmax_4bit": [int(t) for t in quantized_tokens],
        }
        report["prompts"].append(entry)
        print(
            f"\n  {prompt['id']}: {agreement}/{len(tokens)} positions agree on the next token, "
            f"worst relative {worst_relative:.2e} at {worst_name}, smallest margin {smallest:.4f}"
        )
        if agreement != len(tokens):
            for index, (a, b) in enumerate(zip(fp32_tokens, quantized_tokens)):
                if a != b:
                    print(f"    position {index}: fp32 {int(a)} vs 4-bit {int(b)}", file=sys.stderr)

    (args.work / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"\nreport at {args.work / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
