#!/usr/bin/env python3
"""Run the `qwen3_5` contract on a checkpoint, streaming one layer at a time.

The engine cannot hold 2 B parameters in fp32 and neither can this: both read a layer's
tensors, use them and release them, and both read the embedding one row at a time and the
tied head in blocks of vocabulary rows. That symmetry is deliberate — the point of this
tool is to be the *bit-exactness target* for the engine on the real model, so it has to run
where the engine runs.

The tensor names come from the engine's own IR spec (`datacenter-trace --emit-spec`), not
from a second copy of the mapping: L2 says the importer is the only place that knows names,
and this keeps that true across languages.

    .venv/bin/python tools/ordered_qwen35_trace.py <snapshot> <out> --spec spec.json \\
        --tokens 1,2,3,4,5,6,7,8 --model Qwen/Qwen3.5-2B --revision <sha>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import ordered_qwen35 as q35  # noqa: E402
import trace_format  # noqa: E402


# The reader is shared with the install builder and the mixture's CLI, so all three agree
# about what a sharded checkpoint is.
from safetensors_source import SafetensorsSource  # noqa: E402,F401


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--spec", type=Path, required=True, help="IR spec emitted by datacenter-trace --emit-spec")
    parser.add_argument("--tokens", required=True, help="comma-separated token ids")
    parser.add_argument("--model", default="")
    parser.add_argument("--revision", default="")
    args = parser.parse_args(argv)

    spec = json.loads(args.spec.read_text())
    tokens = [int(part) for part in args.tokens.replace(" ", "").split(",") if part]
    source = SafetensorsSource(args.snapshot)

    captured: dict[str, np.ndarray] = {}
    q35.streamed_text_forward(spec, source, tokens, capture=captured)

    tensors = [(name, "f32", values.shape, values.tobytes()) for name, values in captured.items()]
    manifest = trace_format.write_trace(
        args.out,
        tensors=tensors,
        model={
            "id": args.model,
            "revision": args.revision,
            "compute": "fp32",
            "contract": "ordered_qwen35.py",
            "spec_family": spec["family"],
        },
        prompt={"tokens": [int(t) for t in tokens]},
        producer="ordered-reference-python",
    )
    print(f"wrote {args.out}: {len(tensors)} tensors, digest {manifest['digest'][:16]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
