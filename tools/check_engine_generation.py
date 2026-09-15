#!/usr/bin/env python3
"""Check the engine's generation against the contract, and show what it wrote.

This is `DC-023`'s done-when as a command: the engine generates text on one Mac, the
token ids match the contract **exactly** (I3 — a discrete decision, not a tolerance), and
the text is printed so a human can see that it is coherent rather than merely equal.

    .venv/bin/python tools/check_engine_generation.py \\
        --snapshot .build/hf-cache/models--Qwen--Qwen3-0.6B/snapshots/<rev> \\
        --prompt "The capital of France is" --max-new-tokens 8

Needs the checkpoint, the venv and the engine's release binary, so it is not part of the
stdlib-only gate. Exits non-zero if the token ids differ.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def generated_ids(text: str) -> list[int]:
    for line in text.splitlines():
        if line.startswith("generated: "):
            return [int(part) for part in line.split(": ", 1)[1].split(",") if part]
    return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--model", default="")
    parser.add_argument("--revision", default="")
    parser.add_argument("--work", type=Path, default=ROOT / ".build" / "engine-generation")
    parser.add_argument("--configuration", default="release")
    args = parser.parse_args(argv)

    print("[1/5] tokenizing the prompt")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.snapshot))
    prompt_ids = tokenizer.encode(args.prompt)
    print(f"      {args.prompt!r} -> {prompt_ids}")

    contract_text = args.work / "contract"
    engine_text = args.work / "engine"
    tokens_arg = ",".join(str(t) for t in prompt_ids)

    print("[2/5] building the engine")
    built = subprocess.run(
        ["swift", "build", "-c", args.configuration, "--product", "datacenter-generate"],
        cwd=ROOT, capture_output=True, text=True,
    )
    if built.returncode != 0:
        print(built.stdout + built.stderr, file=sys.stderr)
        return 2

    print("[3/5] generating with the contract (greedy, full sequence each step)")
    contract = subprocess.run(
        [
            "python3", str(ROOT / "tools" / "ordered_reference.py"), str(args.snapshot), str(contract_text),
            "--tokens", tokens_arg, "--max-new-tokens", str(args.max_new_tokens),
            "--model", args.model, "--revision", args.revision,
        ],
        cwd=ROOT, capture_output=True, text=True,
    )
    if contract.returncode != 0:
        print(contract.stdout + contract.stderr, file=sys.stderr)
        return 2
    reference_ids = generated_ids(contract.stdout)
    print(f"      {len(reference_ids)} tokens: {reference_ids}")

    print("[4/5] generating with the engine")
    engine = subprocess.run(
        [
            str(ROOT / ".build" / args.configuration / "datacenter-generate"), str(args.snapshot), str(engine_text),
            tokens_arg, str(args.max_new_tokens), "--model", args.model, "--revision", args.revision,
        ],
        cwd=ROOT, capture_output=True, text=True,
    )
    if engine.returncode != 0:
        print(engine.stdout + engine.stderr, file=sys.stderr)
        return 2
    engine_ids = generated_ids(engine.stdout)
    print("      " + engine.stdout.strip().splitlines()[-1])

    print("[5/5] comparing")
    if engine_ids != reference_ids:
        first = next((i for i, (a, b) in enumerate(zip(engine_ids, reference_ids)) if a != b), min(len(engine_ids), len(reference_ids)))
        print(
            f"FAIL: token {first} differs — engine {engine_ids[first:first + 3]}, contract {reference_ids[first:first + 3]}",
            file=sys.stderr,
        )
        return 1

    differ = subprocess.run(
        ["python3", str(ROOT / "tools" / "trace_diff.py"), str(contract_text), str(engine_text)],
        cwd=ROOT, capture_output=True, text=True,
    )
    print("      " + (differ.stdout.strip() or differ.stderr.strip()))
    if differ.returncode != 0:
        return 1

    print(f"\n  prompt:    {args.prompt!r}")
    print(f"  generated: {tokenizer.decode(engine_ids)!r}")
    print(f"  full text: {tokenizer.decode(prompt_ids + engine_ids)!r}")
    print(f"\nOK — {len(engine_ids)} generated token(s), identical to the contract, trace identical")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
