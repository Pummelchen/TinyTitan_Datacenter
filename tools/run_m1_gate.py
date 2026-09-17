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
sys.path.insert(0, str(ROOT / "tools"))
from heavy_job import require_heavy_headroom  # noqa: E402
PROMPTS = ROOT / "tools" / "m1_prompts.json"


def generated_tokens(stdout: str) -> list[int]:
    """The engine's generated token ids, as the tool prints them.

    `M1`'s claim includes "identical generated tokens" between the cached and full-sequence paths, and that
    was a manual observation until this gate checked it. The line is parsed rather than eyeballed for the same
    reason the seconds are: a number a reader has to copy is a number that goes stale.
    """
    for line in stdout.splitlines():
        if line.startswith("generated: "):
            return [int(part) for part in line.split(": ", 1)[1].split(",") if part.strip()]
    return []


def run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(command, cwd=ROOT, capture_output=True, text=True, **kwargs)


TIME_TOOL = Path("/usr/bin/time")


def peak_rss_bytes(stderr: str) -> int | None:
    """The engine's peak resident memory, from `/usr/bin/time -l`.

    `DC-032`'s gate says resident memory stays inside a budget, and the brief gives about 4.5 GB
    of usable memory per node — so the number has to be measured rather than argued. Note what
    this counts: on macOS the figure includes clean file-backed pages, which is why the M0 runs
    reported 3.4 GB for a 2 B model whose weights were mostly mapped and shared. It is an upper
    bound on the process, not a claim about private dirty memory.
    """
    for line in stderr.splitlines():
        if "maximum resident set size" in line:
            digits = "".join(character for character in line if character.isdigit())
            if digits:
                return int(digits)
    return None


def measured(command: list[str]) -> tuple[subprocess.CompletedProcess, int | None]:
    """Run a command, measuring its peak memory when the platform's `time` can."""
    if TIME_TOOL.exists():
        result = run([str(TIME_TOOL), "-l", *command])
        return result, peak_rss_bytes(result.stderr)
    return run(command), None


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
    parser.add_argument(
        "--only", default="", metavar="ID,ID",
        help="run a subset of the frozen prompts; the contract side costs about ten times the engine's wall time",
    )
    args = parser.parse_args(argv)

    prompt_bytes = args.prompts.read_bytes()
    prompt_set = json.loads(prompt_bytes)
    prompt_digest = hashlib.sha256(prompt_bytes).hexdigest()
    args.work.mkdir(parents=True, exist_ok=True)

    if args.only:
        wanted = {part.strip() for part in args.only.split(",") if part.strip()}
        known = {prompt["id"] for prompt in prompt_set["prompts"]}
        unknown = wanted - known
        if unknown:
            print(f"refusing: no such prompt(s) {sorted(unknown)}; the set has {sorted(known)}", file=sys.stderr)
            return 2
        prompt_set["prompts"] = [p for p in prompt_set["prompts"] if p["id"] in wanted]
        print(f"      running a subset: {sorted(wanted)}")

    print(f"[1/5] the frozen prompt set: {len(prompt_set['prompts'])} prompts, sha256 {prompt_digest[:16]}…")
    for prompt in prompt_set["prompts"]:
        print(f"      {prompt['id']:12s} {prompt['length']:4d} tokens")

    print(f"[2/5] building the engine ({args.configuration})")
    binary = ROOT / ".build" / args.configuration
    built = run(["swift", "build", "-c", args.configuration])
    if built.returncode != 0:
        print(built.stdout + built.stderr, file=sys.stderr)
        return 2

    # The gate is handed either a checkpoint or an install, and it has to know which, because the two make
    # different claims. A checkpoint on both sides is the model as published; an install on both sides is
    # M1's restated claim — the engine against a contract reading the SAME install, which is what `D56`
    # settled — and it is also the form that needs no 70 GB mapping, so it runs on the 8 GB node. The kind
    # is discovered from the directory rather than named by a flag: the caller already said what they meant
    # by choosing the path.
    manifest_path = args.snapshot / "install.json"
    if manifest_path.exists():
        family = json.loads(manifest_path.read_text()).get("family", "")
        kind = "install"
        # Measured, not guessed: the whole gate on the real install -- engine and contract, all five frozen
        # prompts -- peaks at **1.41 GB**, on the longest prompt, with the contract alone at 1.21 GB. The
        # declaration is the measurement plus margin rather than the measurement, and the margin was widened
        # after that run: 1.41 against 1.5 is 6%, which is not a guard, it is a coincidence. `D73` recorded the
        # same lesson from the other side -- an under-declared guard admits a job the machine cannot take.
        needs_gb = 2.0
    else:
        family = json.loads((args.snapshot / "config.json").read_text()).get("model_type", "")
        kind = "checkpoint"
        # 4.2 GB is the checkpoint path's measured peak, and the measurement was redone this round rather
        # than inherited. The *contract* half is now cheap -- 0.397 GB for a five-token prompt, streaming its
        # experts and reading through `pread`, against the 4.16 GB `docs/m1-gate.md` recorded before either
        # flag existed -- but the gate's peak is **3.71 GB**, because the **engine's** trace over the bf16
        # checkpoint is what a checkpoint run actually costs. So streaming lowered the contract and the
        # declaration stays where the engine puts it: an under-declared guard admits a job the machine cannot
        # take, which is worse than a conservative one.
        needs_gb = 4.2

    require_heavy_headroom(needs_gb, purpose=f"the M1 gate on the {kind}")

    python = sys.executable if Path(sys.executable).name.startswith("python") else "python3"
    if family != "qwen3_5_moe":
        print(f"refusing: this is M1's gate and {args.snapshot} declares {family!r}", file=sys.stderr)
        return 2
    print(f"      source: a {kind} declaring {family!r}; BOTH sides of every comparison read this one source")

    print("[3/5] the engine's IR spec, emitted by the engine's own importer")
    spec_path = args.work / "spec.json"
    spec = run([str(binary / "datacenter-trace"), "--emit-spec", str(spec_path), str(args.snapshot)])
    if spec.returncode != 0:
        print(spec.stdout + spec.stderr, file=sys.stderr)
        return 2
    print("      " + spec.stdout.strip())

    print("[4/5] correctness: every prompt, engine against contract, byte for byte")
    # The flags this gate passes to the contract, in one place so that the command and the report cannot
    # disagree. The report records them because the invocation is part of the evidence: two rounds were spent
    # discovering that a gate was not passing what its own document required (`D69`, `D73`), and a recorded
    # flag is one a test can assert.
    contract_flags = ["--stream-experts", *(["--uncached"] if kind == "checkpoint" else [])]

    report = {
        "gate": "M1",
        "contract_flags": contract_flags,
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
        engine, peak_rss = measured([
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
            # `contract_flags`, defined above: `--stream-experts` is the only way the contract can read an
            # install's expert stacks at all, and `--uncached` is the other half of the pair `docs/m1-gate.md`
            # records -- without it the checkpoint path goes through `safe_open`, which *maps* the file. The
            # gate passed neither once, and the flag list is recorded in the report so a test can assert it.
            *contract_flags,
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
            "peak_rss_bytes": peak_rss,
            "expert_requests": metrics.get("expert_requests"),
            "expert_hits": metrics.get("expert_hits"),
            "expert_elements_read": metrics.get("expert_elements_read"),
            "expert_bytes_from_ssd": metrics.get("expert_bytes_from_ssd"),
            "expert_hit_rate": metrics.get("expert_hit_rate"),
        }
        report["results"].append(result)
        memory = f"{peak_rss / 1e9:.2f} GB" if peak_rss else "not measured"
        print(
            f"      {prompt['id']:12s} {'IDENTICAL' if identical else 'DIFFERS':>9s}  "
            f"{engine_seconds:6.2f} s  peak {memory:>10s}  hit rate {result['expert_hit_rate']}"
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

            # The other half of M1's generation claim: the cached path must generate **the same tokens** as
            # the full-sequence path, which is a discrete decision and not a tolerance (I3). It was checked
            # by hand once; a claim that is checked by hand is a claim that drifts.
            uncached_tokens = generated_tokens(generate.stdout)
            cached = run([
                str(binary / "datacenter-generate"), str(args.snapshot), str(args.work / "generation-cached"),
                ",".join(str(t) for t in longest["tokens"]), str(args.max_new_tokens),
                "--model", args.model, "--revision", args.revision, "--cached",
            ])
            cached_tokens = generated_tokens(cached.stdout)
            report["throughput"]["tokens_uncached"] = uncached_tokens
            report["throughput"]["tokens_cached"] = cached_tokens
            if cached.returncode != 0:
                print("      the cached generation failed", file=sys.stderr)
                failed = True
            elif not uncached_tokens or not cached_tokens:
                # An empty parse must not read as agreement: two absences are equal and say nothing.
                print("      could not read the generated tokens from one of the runs", file=sys.stderr)
                failed = True
            elif uncached_tokens != cached_tokens:
                print(f"      TOKENS DIFFER: cached {cached_tokens} against uncached {uncached_tokens}",
                      file=sys.stderr)
                failed = True
            else:
                print(f"      cached and uncached agree on {len(cached_tokens)} token(s)")

    report["identical"] = not failed
    report["hit_rate_overall"] = (
        sum(r["expert_hits"] or 0 for r in report["results"])
        / max(sum(r["expert_requests"] or 0 for r in report["results"]), 1)
    )
    (args.work / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"\nreport: {args.work / 'report.json'}")
    print(f"  identical everywhere: {report['identical']}")
    print(f"  cache hit rate:       {report['hit_rate_overall']:.4f}")
    peaks = [r["peak_rss_bytes"] for r in report["results"] if r.get("peak_rss_bytes")]
    if peaks:
        report["peak_rss_bytes"] = max(peaks)
        print(f"  peak resident memory: {max(peaks) / 1e9:.2f} GB (includes clean file-backed pages)")
    if report["throughput"]:
        print(f"  throughput:           {report['throughput']['tokens_per_second_mean']} tok/s (mean, no KV cache)")
    if failed:
        print("GATE FAILED", file=sys.stderr)
        return 1
    print("GATE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
