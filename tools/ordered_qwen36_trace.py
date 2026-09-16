#!/usr/bin/env python3
"""Run the `qwen3_5_moe` contract on a checkpoint, streaming one layer at a time.

The engine cannot hold 2 B parameters in fp32 and neither can this: both read a layer's
tensors, use them and release them, and both read the embedding one row at a time and the
tied head in blocks of vocabulary rows. That symmetry is deliberate — the point of this
tool is to be the *bit-exactness target* for the engine on the real model, so it has to run
where the engine runs.

The tensor names come from the engine's own IR spec (`datacenter-trace --emit-spec`), not
from a second copy of the mapping: L2 says the importer is the only place that knows names,
and this keeps that true across languages.

    .venv/bin/python tools/ordered_qwen36_trace.py <snapshot> <out> --spec spec.json \\
        --tokens 1,2,3,4,5,6,7,8 --model Qwen/Qwen3.6-35B-A3B --revision <sha>

It differs from its `qwen3_5` sibling in one way that matters: the mixture's **router decisions
are written into the trace's discrete section**, not as tensors. I3 asserts them apart from any
tolerance, and a trace that folded them into a float tensor would make the one comparison that
cannot be approximate into an approximate one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import ordered_qwen36 as q36  # noqa: E402
import trace_format  # noqa: E402
from check_disk_headroom import require_headroom  # noqa: E402


from contract_source import add_uncached_argument, open_source  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--spec", type=Path, required=True, help="IR spec emitted by datacenter-trace --emit-spec")
    parser.add_argument("--tokens", required=True, help="comma-separated token ids")
    parser.add_argument("--model", default="")
    parser.add_argument("--revision", default="")
    parser.add_argument(
        "--capture-internals",
        action="store_true",
        help=(
            "also record what is inside each layer (attn_out, ff_out), not just at its boundaries. Off by "
            "default because the trace's digest covers its tensor list: a trace with extra tensors is a "
            "different artifact."
        ),
    )
    parser.add_argument(
        "--stream-experts",
        action="store_true",
        help=(
            "fetch the routed experts by index instead of materialising a layer's stack. For the real model "
            "that stack is 3.2 GB in fp32 against about 4.5 GB usable per node, which is why the contract "
            "could not be re-run here; the values are identical (one expert is one row of the stacked "
            "tensor) and tools/test_ordered_moe.py asserts the two paths byte for byte."
        ),
    )
    add_uncached_argument(parser)
    args = parser.parse_args(argv)
    require_headroom(purpose="the contract run")

    spec = json.loads(args.spec.read_text())
    tokens = [int(part) for part in args.tokens.replace(" ", "").split(",") if part]
    # An install directory is read through the install's own dequantiser, which is what the Swift reader
    # mirrors. That is how the gate's real question gets asked: same weights on both sides, so a difference
    # is a difference in arithmetic rather than in what was quantised (`D55`).
    source = open_source(args.snapshot, uncached=args.uncached)

    captured: dict[str, np.ndarray] = {}
    decisions: dict[str, np.ndarray] = {}
    q36.streamed_text_forward(
        spec, source, tokens, capture=captured, discrete=decisions,
        stream_experts=args.stream_experts, internals=args.capture_internals,
    )

    tensors = [(name, "f32", values.shape, values.tobytes()) for name, values in captured.items()]
    discrete = [
        (name, list(values.shape), [int(v) for v in values.reshape(-1)])
        for name, values in sorted(decisions.items())
    ]
    manifest = trace_format.write_trace(
        args.out,
        tensors=tensors,
        model={
            "id": args.model,
            "revision": args.revision,
            "compute": "fp32",
            "contract": "ordered_qwen36.py",
            "spec_family": spec["family"],
        },
        prompt={"tokens": [int(t) for t in tokens]},
        producer="ordered-reference-python",
        discrete=discrete,
    )
    print(
        f"wrote {args.out}: {len(tensors)} tensors, {len(discrete)} discrete, "
        f"digest {manifest['digest'][:16]}…"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
