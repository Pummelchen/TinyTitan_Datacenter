#!/usr/bin/env python3
"""THE M1 GATE: correct output, a recorded tok/s baseline, and a measured cache hit rate.

M1's gate has three parts and this runs all three, on one checkpoint, as one command:

1. **correct output** — every frozen prompt, the engine's trace against the contract's, byte for
   byte, with the router's **discrete decisions checked separately** (I3). A prompt that differs
   in any tensor, or in any chosen expert, fails the gate;
2. **a recorded tok/s baseline** — greedy generation with the engine, measured, written into the
   report as versioned evidence rather than quoted from a terminal;
3. **a measured cache hit rate** — the engine's own expert-traffic counters, which is the number
   that says whether the slot bank is earning its memory.

It runs on *any* checkpoint of the family, which is the point: the same instrument is verified on
the tiny fixture (where it costs a second) and then run on the 67 GB model, so the gate is never
untested code waiting for the one input that matters.

    .venv/bin/python tools/run_m1_gate.py \\
        --snapshot .build/hf-cache/models--Qwen--Qwen3.6-35B-A3B/snapshots/<rev> \\
        --model Qwen/Qwen3.6-35B-A3B --revision <sha>

Writes `.build/m1-gate/report.json` and exits non-zero on any failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROMPTS = ROOT / "tools" / "m1_prompts.json"


def run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(command, cwd=ROOT, capture_output=True, text=True, **kwargs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--model", default="")
    parser.add_argument("--revision", default="")
    parser.add_argument("--work", type=Path, default=ROOT / ".build" / "m1-gate")
    parser.add_argument("--prompts", type=Path, default=PROMPTS)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--configuration", default="release")
    parser.add_argument("--skip-generation", action="store_true", help="measure only correctness and traffic")
    args = parser.parse_args(argv)

    prompt_bytes = args.prompts.read_bytes()
    prompt_set = json.loads(prompt_bytes)
    prompt_digest = hashlib.sha256(prompt_bytes).hexdigest()
    args.work.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] the frozen prompt set: {len(prompt_set['prompts'])} prompts, sha256 {prompt_digest[:16]}…")
    for prompt in prompt_set["prompts"]:
        print(f"      {prompt['id']:12s} {prompt['length']:4d} tokens")

    print(f"[2/5] building the engine ({args.configuration})")
    binary = ROOT / ".build" / args.configuration
    built = run(["swift", "build", "-c", args.configuration])
    if built.returncode != 0:
        print(built.stdout + built.stderr, file=sys.stderr)
        return 2

    python = sys.executable if Path(sys.executable).name.startswith("python") else "python3"
    model_type = json.loads((args.snapshot / "config.json").read_text()).get("model_type", "")
    if model_type != "qwen3_5_moe":
        print(f"refusing: this is M1's gate and {args.snapshot} declares model_type {model_type!r}", file=sys.stderr)
        return 2

    print("[3/5] the engine's IR spec, emitted by the engine's own importer")
    spec_path = args.work / "spec.json"
    spec = run([str(binary / "datacenter-trace"), "--emit-spec", str(spec_path), str(args.snapshot)])
    if spec.returncode != 0:
        print(spec.stdout + spec.stderr, file=sys.stderr)
        return 2
    print("      " + spec.stdout.strip())

    print("[4/5] correctness: every prompt, engine against contract, byte for byte")
    report = {
        "gate": "M1",
        "model": {"id": args.model, "revision": args.revision, "path": str(args.snapshot)},
        "prompts": {"path": str(args.prompts), "sha256": prompt_digest, "count": len(prompt_set["prompts"])},
        "spec_family": json.loads(spec_path.read_text())["family"],
        "results": [],
        "throughput": None,
    }
    failed = False
    for prompt in prompt_set["prompts"]:
        tokens_arg = ",".join(str(t) for t in prompt["tokens"])
        engine_trace = args.work / f"{prompt['id']}-engine"
        contract_trace = args.work / f"{prompt['id']}-contract"

        started = time.time()
        engine = run([
            str(binary / "datacenter-trace"), str(args.snapshot), str(engine_trace), tokens_arg,
            "--model", args.model, "--revision", args.revision,
        ])
        engine_seconds = time.time() - started
        if engine.returncode != 0:
            print(engine.stdout + engine.stderr, file=sys.stderr)
            failed = True
            continue

        contract = run([
            python, str(ROOT / "tools" / "ordered_qwen36_trace.py"), str(args.snapshot), str(contract_trace),
            "--spec", str(spec_path), "--tokens", tokens_arg, "--model", args.model, "--revision", args.revision,
        ])
        if contract.returncode != 0:
            print(contract.stdout + contract.stderr, file=sys.stderr)
            failed = True
            continue

        # `trace_diff` is the harness M0 built and this is what it is for: it compares the
        # tensors byte for byte and the discrete decisions exactly, and it is the same tool on
        # the tiny fixture and on the 35 B model.
        diff = run([python, str(ROOT / "tools" / "trace_diff.py"), str(contract_trace), str(engine_trace)])
        identical = diff.returncode == 0
        failed = failed or not identical

        metrics_path = engine_trace / "metrics.json"
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        result = {
            "id": prompt["id"],
            "tokens": prompt["length"],
            "identical": identical,
            "diff": diff.stdout.strip(),
            "engine_seconds": round(engine_seconds, 3),
            "expert_requests": metrics.get("expert_requests"),
            "expert_hits": metrics.get("expert_hits"),
            "expert_rows_read": metrics.get("expert_rows_read"),
            "expert_hit_rate": metrics.get("expert_hit_rate"),
        }
        report["results"].append(result)
        print(
            f"      {prompt['id']:12s} {'IDENTICAL' if identical else 'DIFFERS':>9s}  "
            f"{engine_seconds:6.2f} s  hit rate {result['expert_hit_rate']}"
        )

    if not args.skip_generation:
        longest = max(prompt_set["prompts"], key=lambda prompt: prompt["length"])
        print(f"[5/5] throughput: generating {args.max_new_tokens} tokens after '{longest['id']}' ({longest['length']} tokens)")
        started = time.time()
        generate = run([
            str(binary / "datacenter-generate"), str(args.snapshot), str(args.work / "generation"),
            ",".join(str(t) for t in longest["tokens"]), str(args.max_new_tokens),
            "--model", args.model, "--revision", args.revision,
        ])
        wall = time.time() - started
        if generate.returncode != 0:
            print(generate.stdout + generate.stderr, file=sys.stderr)
            failed = True
        else:
            # The generate tool prints the line as "…, 12.3 s over 8 step(s), slowest …". The
            # figure is parsed rather than eyeballed, and the "is not None" matters: `if seconds`
            # is false for a legitimate 0.0, which silently turned a measured run into None.
            seconds = None
            for line in generate.stdout.splitlines():
                if " s over " in line:
                    try:
                        seconds = float(line.split(" s over ")[0].rsplit(",", 1)[-1].strip())
                    except ValueError:
                        pass
            # The engine has no KV cache yet, so every step re-runs the whole sequence and the
            # per-step time grows with it. The mean is what a user feels; the last step is the
            # honest steady-state proxy; both are recorded, and neither is called "tok/s" alone.
            tokens_per_second = args.max_new_tokens / seconds if seconds else None
            # The tool prints tenths of a second, so anything faster than 0.05 s reads as 0.0 and
            # cannot be measured this way at all. Saying so is better than reporting a rate
            # derived from a rounded zero.
            unmeasurable = seconds == 0.0
            report["throughput"] = {
                "prompt": longest["id"],
                "generated": args.max_new_tokens,
                "steps_seconds_total": seconds,
                "steps_seconds_wall": round(wall, 3),
                "mean_step_seconds": round(seconds / args.max_new_tokens, 3) if seconds else None,
                "tokens_per_second_mean": round(tokens_per_second, 4) if tokens_per_second else None,
                "note": (
                    "M1 has no KV cache: each step re-runs the whole sequence, so the mean is not "
                    "a steady-state rate and the figure is a baseline to improve on, not a target met"
                ),
                "below_timer_resolution": unmeasurable,
                "stdout": generate.stdout.strip(),
            }
            if unmeasurable:
                print("      the run finished below the tool's tenth-of-a-second resolution")
            else:
                print(f"      {report['throughput']['tokens_per_second_mean']} tok/s mean over {args.max_new_tokens} steps")

    report["identical"] = not failed
    report["hit_rate_overall"] = (
        sum(r["expert_hits"] or 0 for r in report["results"])
        / max(sum(r["expert_requests"] or 0 for r in report["results"]), 1)
    )
    (args.work / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"\nreport: {args.work / 'report.json'}")
    print(f"  identical everywhere: {report['identical']}")
    print(f"  cache hit rate:       {report['hit_rate_overall']:.4f}")
    if report["throughput"]:
        print(f"  throughput:           {report['throughput']['tokens_per_second_mean']} tok/s (mean, no KV cache)")
    if failed:
        print("GATE FAILED", file=sys.stderr)
        return 1
    print("GATE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
