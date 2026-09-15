#!/usr/bin/env python3
"""Check the engine against the numeric contract, end to end, on a real checkpoint.

This is M0's central claim as a command: run the contract in Python, run the same forward
in the engine, and compare the two traces byte for byte. The op-level tests in
`tests/DatacenterEngineTests` pin each primitive; this pins the wiring, which is what they
cannot see — a transposed weight or a norm applied on the wrong side would leave every
primitive perfect and the model wrong.

    .venv/bin/python tools/check_engine_contract.py \\
        --snapshot .build/hf-cache/models--Qwen--Qwen3-0.6B/snapshots/<rev> \\
        --tokens 1,2,3,4,5,6,7,8 --model Qwen/Qwen3-0.6B --revision <sha>

Needs the checkpoint and the venv (numpy), so it is not part of the stdlib-only gate; CI
runs the fixtures instead. Exits non-zero on any difference.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(command, cwd=ROOT, capture_output=True, text=True, **kwargs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True, help="checkpoint directory")
    parser.add_argument("--tokens", default="1,2,3,4,5,6,7,8")
    parser.add_argument("--model", default="")
    parser.add_argument("--revision", default="")
    parser.add_argument("--work", type=Path, default=ROOT / ".build" / "engine-contract")
    parser.add_argument("--configuration", default="release", choices=["debug", "release"])
    args = parser.parse_args(argv)

    contract_trace = args.work / "contract"
    engine_trace = args.work / "engine"

    print(f"[1/4] building the engine ({args.configuration})")
    built = run(["swift", "build", "-c", args.configuration, "--product", "datacenter-trace"])
    if built.returncode != 0:
        print(built.stdout + built.stderr, file=sys.stderr)
        return 2

    print("[2/4] running the contract in Python")
    contract = run(
        [
            sys.executable if Path(sys.executable).name.startswith("python") else "python3",
            str(ROOT / "tools" / "ordered_reference.py"),
            str(args.snapshot),
            str(contract_trace),
            "--tokens",
            args.tokens,
            "--model",
            args.model,
            "--revision",
            args.revision,
        ]
    )
    if contract.returncode != 0:
        print(contract.stdout + contract.stderr, file=sys.stderr)
        return 2
    print("      " + contract.stdout.strip())

    print("[3/4] running the forward in the engine")
    binary = ROOT / ".build" / args.configuration / "datacenter-trace"
    engine = run(
        [
            str(binary),
            str(args.snapshot),
            str(engine_trace),
            args.tokens,
            "--model",
            args.model,
            "--revision",
            args.revision,
        ]
    )
    if engine.returncode != 0:
        print(engine.stdout + engine.stderr, file=sys.stderr)
        return 2
    print("      " + engine.stdout.strip())

    print("[4/4] comparing")
    # The differ takes a digest shortcut when the digests match, and the digest is
    # computed by each implementation independently — so a matching digest is already two
    # implementations agreeing on every tensor's bytes. The raw comparison is run as well,
    # because "the digests agree" and "the bytes agree" should never be assumed to be the
    # same statement.
    differ = run(
        [
            "python3",
            str(ROOT / "tools" / "trace_diff.py"),
            str(contract_trace),
            str(engine_trace),
        ]
    )
    print("      " + (differ.stdout.strip() or differ.stderr.strip()))
    if differ.returncode != 0:
        return 1

    left = (contract_trace / "data.bin").read_bytes()
    right = (engine_trace / "data.bin").read_bytes()
    if left != right:
        first = next((i for i, (a, b) in enumerate(zip(left, right)) if a != b), min(len(left), len(right)))
        print(f"FAIL: data.bin differs at byte {first}", file=sys.stderr)
        return 1
    print(f"      data.bin identical: {len(left)} bytes")
    print("OK — the engine reproduces the contract exactly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
