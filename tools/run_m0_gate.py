#!/usr/bin/env python3
"""The M0 gate: run the frozen prompt set and report both halves of the claim (DC-026).

Two things are asserted, and they are different things (I3):

- **the engine against the contract**, byte for byte. This is the reproducibility claim:
  two implementations of the same stated arithmetic, one in Swift and one in Python, must
  produce identical bytes. Any difference is a bug in one of them.
- **the engine against the reference implementation**, whose numbers *cannot* match
  (bit-matching torch is impossible: D3, R13) but whose **discrete decisions** must. The
  argmax of the logits is compared as an index set, separately from any tolerance, because
  a 1-ULP logit difference can flip a marginal token and take the whole continuation with
  it while every per-tensor check still looks green.

The prompt set is frozen in `tools/m0_prompts.json` with its token ids resolved once, so the
gate does not depend on a tokenizer version to be reproducible.

    .venv/bin/python tools/run_m0_gate.py --snapshot <snapshot> --model Qwen/Qwen3.5-2B --revision <sha>

Needs the checkpoint, the venv and the release binary, so it is not part of the stdlib-only
CI gate. It writes a record to `.build/m0-gate/report.json` and exits non-zero on any
failure.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import trace_format  # noqa: E402


def run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(command, cwd=ROOT, capture_output=True, text=True, **kwargs)


def logits(trace, tokens: int, name: str = "logits") -> np.ndarray:
    """`[tokens, vocabulary]`, whether the trace stored it flat or shaped.

    The two writers do not have to agree here — one stores what its capture produced —
    so the reader derives the shape from the token count rather than assuming it.
    """
    entry = trace.entry(name)
    values = np.frombuffer(trace.tensor(name).payload, dtype=np.float32)
    if len(entry["shape"]) == 2:
        return values.reshape(entry["shape"])
    return values.reshape(tokens, -1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--model", default="")
    parser.add_argument("--revision", default="")
    parser.add_argument("--prompts", type=Path, default=ROOT / "tools" / "m0_prompts.json")
    parser.add_argument("--work", type=Path, default=ROOT / ".build" / "m0-gate")
    parser.add_argument("--configuration", default="release")
    parser.add_argument("--only", help="run a single prompt id")
    args = parser.parse_args(argv)

    prompt_set = json.loads(args.prompts.read_text())
    prompts = prompt_set["prompts"]
    if args.only:
        prompts = [p for p in prompts if p["id"] == args.only]
    args.work.mkdir(parents=True, exist_ok=True)

    binary = ROOT / ".build" / args.configuration / "datacenter-trace"
    python = sys.executable if Path(sys.executable).name.startswith("python") else "python3"

    print(f"M0 gate: {len(prompts)} prompt(s) from {args.prompts.name}")
    built = run(["swift", "build", "-c", args.configuration, "--product", "datacenter-trace"])
    if built.returncode != 0:
        print(built.stdout + built.stderr, file=sys.stderr)
        return 2

    spec_path = args.work / "spec.json"
    emitted = run([str(binary), "--emit-spec", str(spec_path), str(args.snapshot)])
    if emitted.returncode != 0:
        print(emitted.stdout + emitted.stderr, file=sys.stderr)
        return 2

    # The flags this gate passes to the contract. `--uncached` is the flag `D74` gave the 2B contract and made
    # this gate pass: without it the run goes through `safe_open`, which *maps* the checkpoint. It is
    # recorded in the report because the invocation is evidence, and a recorded flag is one a test can assert.
    contract_flags = ["--uncached"]

    report: dict = {
        "model": args.model,
        "revision": args.revision,
        "contract_flags": contract_flags,
        "prompts": [],
    }
    failed = False
    for prompt in prompts:
        tokens = ",".join(str(t) for t in prompt["tokens"])
        print(f"\n=== {prompt['id']}: {prompt['text']!r} ({len(prompt['tokens'])} tokens) ===")
        engine_trace = args.work / f"{prompt['id']}-engine"
        contract_trace = args.work / f"{prompt['id']}-contract"
        oracle_trace = args.work / f"{prompt['id']}-oracle"
        entry: dict = {"id": prompt["id"], "text": prompt["text"], "tokens": prompt["tokens"]}

        started = time.time()
        engine = run(
            [str(binary), str(args.snapshot), str(engine_trace), tokens,
             "--model", args.model, "--revision", args.revision]
        )
        if engine.returncode != 0:
            print(engine.stdout + engine.stderr, file=sys.stderr)
            failed = True
            continue
        entry["engine_seconds"] = round(time.time() - started, 1)
        print("  engine:   " + engine.stdout.strip().splitlines()[-1])

        started = time.time()
        contract = run(
            [python, str(ROOT / "tools" / "ordered_qwen35_trace.py"), str(args.snapshot), str(contract_trace),
             "--spec", str(spec_path), "--tokens", tokens, "--model", args.model, "--revision", args.revision,
             # This gate could not pass this flag, because until `D74` the 2B contract had no such flag: its
             # reader was hard-coded to `safe_open`, which *maps* the checkpoint. The pairing that makes a
             # real-model contract run survivable is in `docs/m1-gate.md` -- read through pread and fetch
             # experts by index -- and the dense family has no experts, so for M0 the first half is the one
             # that applies.
             *contract_flags]
        )
        if contract.returncode != 0:
            print(contract.stdout + contract.stderr, file=sys.stderr)
            failed = True
            continue
        entry["contract_seconds"] = round(time.time() - started, 1)
        print("  contract: " + contract.stdout.strip().splitlines()[-1])

        # Half one: byte equality against the contract.
        differ = run([python, str(ROOT / "tools" / "trace_diff.py"), str(contract_trace), str(engine_trace)])
        left = (contract_trace / "data.bin").read_bytes()
        right = (engine_trace / "data.bin").read_bytes()
        entry["contract_identical"] = differ.returncode == 0 and left == right
        entry["trace_bytes"] = len(left)
        print(f"  contract comparison: {'IDENTICAL' if entry['contract_identical'] else 'DIFFERS'}"
              f" ({len(left)} bytes)")
        if not entry["contract_identical"]:
            print("    " + differ.stdout.strip(), file=sys.stderr)
            failed = True

        # Half two: the discrete decisions against the reference implementation.
        started = time.time()
        oracle = run(
            [python, str(ROOT / "tools" / "trace_capture.py"), str(oracle_trace),
             "--snapshot", str(args.snapshot), "--dtype", "f32", "--tokens", tokens]
        )
        if oracle.returncode != 0:
            print(oracle.stdout + oracle.stderr, file=sys.stderr)
            failed = True
            continue
        entry["oracle_seconds"] = round(time.time() - started, 1)

        engine_logits = logits(trace_format.read_trace(engine_trace), len(prompt["tokens"]))
        oracle_logits = logits(trace_format.read_trace(oracle_trace), len(prompt["tokens"]))
        engine_tokens = engine_logits.argmax(-1)
        oracle_tokens = oracle_logits.argmax(-1)
        difference = np.abs(engine_logits - oracle_logits)
        scale = float(np.abs(oracle_logits).max())
        entry["oracle_discrete_match"] = bool(np.array_equal(engine_tokens, oracle_tokens))
        entry["oracle_max_relative"] = float(difference.max()) / max(scale, 1e-30)
        entry["oracle_argmax"] = [int(t) for t in oracle_tokens]
        entry["engine_argmax"] = [int(t) for t in engine_tokens]
        order = np.sort(oracle_logits, axis=-1)[:, -2:]
        entry["oracle_smallest_margin"] = float((order[:, 1] - order[:, 0]).min())
        print(f"  oracle comparison:    discrete "
              f"{'MATCH' if entry['oracle_discrete_match'] else 'DIFFER'}, "
              f"worst relative {entry['oracle_max_relative']:.2e}, "
              f"smallest margin {entry['oracle_smallest_margin']:.4f}")
        if not entry["oracle_discrete_match"]:
            failed = True

        report["prompts"].append(entry)

    report["passed"] = not failed
    (args.work / "report.json").write_text(json.dumps(report, indent=1) + "\n")

    print("\n=== M0 gate ===")
    for entry in report["prompts"]:
        identical = entry.get("contract_identical")
        discrete = entry.get("oracle_discrete_match")
        print(f"  {entry['id']:10s} contract {'IDENTICAL' if identical else 'DIFFERS':9s} "
              f"discrete {'MATCH' if discrete else 'DIFFER':7s} "
              f"worst rel {entry.get('oracle_max_relative', float('nan')):.1e}")
    print(f"\n{'PASS' if report['passed'] else 'FAIL'} — report at {args.work / 'report.json'}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
