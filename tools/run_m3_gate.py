#!/usr/bin/env python3
"""M3's gate: the sharded cluster's throughput against one node, with correctness first (`DC-053`).

    python3 tools/run_m3_gate.py --install .build/m1-install \\
        --mesh node4@<addr>,node1@<addr>,node2@<addr>,node3@<addr> --steps 4

**Why it refuses to run on a busy farm.** This gate measures *seconds*, so the only honest way to take it
is when the machines are not doing anything else. It asks every node for its one-minute load average first
and stops if any is above the threshold, naming the numbers it saw — a throughput figure from a busy farm is
not a weak result, it is a different measurement. `--allow-busy-farm` runs it anyway and reports the numbers
as an **observation**: the ratio is printed and the exit status depends only on bit-identity, never on speed.

**Correctness is asserted before speed**, and never traded for it: every node's generated tokens and trace
digest must equal the single-node baseline's, or the run fails no matter how fast it was.

The measured speedup is the baseline's seconds per step divided by the **slowest node's** — a cluster step
finishes when its slowest member does, so an average over nodes would report a speed no user of the cluster
can obtain. The per-node figures come from each node's own `metrics.json` (`D37`), so the join and the file
staging are not counted as compute.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_m2_gate as gate  # noqa: E402

ROOT = gate.ROOT
OUT = ROOT / ".build" / "m3-gate"
LOAD = re.compile(r"load averages?:\s*([0-9.]+)")


def load_average(target: str, local: bool = False, timeout: int = 15) -> float | None:
    """The one-minute load average on a node, or None when it cannot be read.

    `None` is not zero: a node nobody could ask is NOT CHECKED, and the caller has to decide what to do
    about that rather than being handed a quiet-looking number. The first version passed the bare hostname
    to `ssh`, which logged in as *this* machine's user on every peer and failed — `BatchMode` turned that
    into a fast refusal rather than a prompt, which is how it was noticed.
    """
    if local:
        try:
            return os.getloadavg()[0]
        except OSError:
            return None
    host = target
    try:
        run = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={timeout}", host, "uptime"],
            capture_output=True, text=True, timeout=timeout + 10,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    match = LOAD.search(run.stdout)
    return float(match.group(1)) if match else None


def signal_explanation(returncode: int) -> str:
    """Why a node reported nothing, in words, because "no output" is a fact worth stating.

    A process killed by a signal has no stderr of its own: the shell reports it as a negative return code,
    or 128 + the signal number. Saying so is the difference between "the peer was killed" and "the tool
    lost the message".
    """
    if returncode < 0:
        return f"no output: the process was killed by signal {-returncode}"
    if returncode == 255:
        # 255 is `ssh`'s own code for "the remote command failed or the connection died", not a signal:
        # the first version of this said "128 + signal 127", which is arithmetic on a number that never
        # meant that. When a peer is killed with `pkill`, this is the exit the harness sees.
        return "no output: exit 255, which is what `ssh` reports when the remote command was killed"
    if returncode > 128:
        return f"no output: exit {returncode}, which a shell reports as 128 + signal {returncode - 128}"
    return "no output, and no error message: the process died before it could report"


def _install_bytes(install: Path) -> int:
    """The size of an install, for the one message that says how much is about to be copied."""
    total = 0
    for path in install.rglob("*"):
        if path.is_file():
            total += path.stat().st_size
    return total


def heavy_preflight() -> None:
    """Disk, memory and the one-heavy-job lock, before anything loads.

    The install path's measured peak is 348.6 MB (`docs/m1-gate.md`), so 0.35 GB is what this declares —
    the smallest declared need in the project, and still worth taking the lock: a cluster run's node here
    plus an install build is the pairing that panicked this machine.
    """
    from heavy_job import require_heavy_headroom

    require_heavy_headroom(0.35, purpose="the M3 gate (one node runs here)")


def farm_state(loads: dict[str, float | None], threshold: float) -> tuple[list[str], list[str]]:
    """Which nodes are too busy, and which could not be asked at all."""
    busy = [name for name, value in loads.items() if value is not None and value > threshold]
    unknown = [name for name, value in loads.items() if value is None]
    return busy, unknown


def speedup(baseline_seconds_per_step: float, node_seconds_per_step: list[float]) -> float:
    """Baseline over the **slowest** node: a cluster step is done when its slowest member is."""
    if not node_seconds_per_step:
        raise ValueError("no node reported its step time, so there is no cluster measurement")
    slowest = max(node_seconds_per_step)
    if slowest <= 0:
        raise ValueError(f"a node reported {slowest} s per step, which cannot be divided by")
    return baseline_seconds_per_step / slowest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", type=Path, required=True)
    parser.add_argument("--mesh", required=True, help="user@host for every node, THIS host first")
    parser.add_argument("--remote-install", default=None)
    parser.add_argument("--remote-dir", default="Downloads/m3-gate")
    parser.add_argument("--prompt", default="760,6511,314,9338,369")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--min-speedup", type=float, default=3.0)
    parser.add_argument("--quiet-load", type=float, default=1.0)
    parser.add_argument("--allow-busy-farm", action="store_true")
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)

    heavy_preflight()
    entries = [entry.strip() for entry in args.mesh.split(",") if entry.strip()]
    if len(entries) < 2:
        raise SystemExit("--mesh needs at least two entries, this host first")
    addresses = [entry.split("@")[-1] for entry in entries]
    binaries = {
        "generate": ROOT / ".build" / "release" / "datacenter-generate",
    }
    for name, binary in binaries.items():
        if not binary.exists():
            raise SystemExit(f"{binary} is missing: run `swift build -c release` first ({name})")

    # 1. Is the farm quiet enough for a throughput claim to mean anything?
    loads = {
        entry: load_average(entry, local=(index == 0)) for index, entry in enumerate(entries)
    }
    busy, unknown = farm_state(loads, args.quiet_load)
    unknown = [host for host, value in loads.items() if value is None]
    busy = [host for host, value in loads.items() if value is not None and value > args.quiet_load]
    print("[1/4] quiet-farm check (one-minute load average, threshold " f"{args.quiet_load})")
    for host, value in loads.items():
        print(f"      {host}: {'NOT CHECKED' if value is None else f'{value:.2f}'}")
    if unknown:
        print(f"      {len(unknown)} node(s) could not be asked; they are NOT CHECKED", file=sys.stderr)
    if busy and not args.allow_busy_farm:
        print(
            "M3 GATE REFUSED: the farm is not quiet — "
            + ", ".join(f"{host} at {loads[host]:.2f}" for host in busy)
            + f". A throughput figure from a busy farm is a different measurement, not a weak one. "
            f"Re-run when the nodes are idle, or pass --allow-busy-farm to record an OBSERVATION "
            f"(which cannot pass this gate).",
            file=sys.stderr,
        )
        return 2

    OUT.mkdir(parents=True, exist_ok=True)
    plan_path = OUT / "plan.json"
    plan = gate.write_plan(args.install, len(entries), plan_path)
    print(f"[2/4] plan: {plan['experts']} experts over {len(entries)} nodes; baseline is one node, same install")

    # 2. The baseline: one node, the same prompt, the same decode mode.
    baseline_dir = OUT / "baseline"
    baseline = gate.run(
        [
            str(binaries["generate"]), str(args.install), str(baseline_dir), args.prompt,
            str(args.steps), "--cached",
        ]
    )
    if baseline.returncode != 0:
        print(baseline.stdout + baseline.stderr, file=sys.stderr)
        raise SystemExit("the single-node baseline failed")
    baseline_metrics = json.loads((baseline_dir / "metrics.json").read_text())
    baseline_per_step = baseline_metrics["step_seconds_total"] / max(1, baseline_metrics["steps"])
    baseline_tokens = _tokens(baseline_dir)
    print(f"      baseline: {baseline_per_step:.3f} s/step, tokens {baseline_tokens}")

    # 3. The cluster: every node generates the same tokens, in a full mesh.
    base_port = random.randint(41_000, 59_000)
    config = {
        "endpoints": [{"host": host, "port": base_port + index} for index, host in enumerate(addresses)],
    }
    config_path = OUT / "cluster.json"
    config_path.write_text(json.dumps(config, sort_keys=True, separators=(",", ":")))
    for index, entry in enumerate(entries[1:], start=1):
        gate.stage_remote(
            entry, args.remote_dir, args.install, binaries["generate"], plan_path,
            args.remote_install, extra=[config_path],
        )
    print(f"      staged on {len(entries) - 1} peer(s)")

    processes: list[tuple[int, subprocess.Popen]] = []
    # Say what is about to cross the LAN **before** it does. Without `--remote-install` this copies the
    # whole install to every node — minutes per node, silently, which reads as a hang: that is exactly how
    # it was debugged, by finding an `scp` of 20 GB mid-flight. The installs are usually already there.
    for entry in addresses[1:]:
        already = args.remote_install is not None
        print(
            f"[3/4] staging to {entry}: the binary and the plan"
            + (" (the install is already on the peer)" if already else
               f", plus the {_install_bytes(args.install) / 1e9:.1f} GB install"
               " because --remote-install was not given")
        )
    for index, host in enumerate(addresses):
        install = str(args.install) if index == 0 else (args.remote_install or "./install")
        if index == 0:
            command = [
                str(binaries["generate"]), install, str(OUT / f"node-{index}"), args.prompt,
                str(args.steps), "--cached", "--plan", str(plan_path), "--config", str(config_path),
                "--node", str(index),
            ]
        else:
            remote = " ".join([
                "./datacenter-generate", install, f"./node-{index}", args.prompt, str(args.steps),
                "--cached", "--plan", "./plan.json", "--config", "./cluster.json", "--node", str(index),
            ])
            command = ["ssh", entries[index], f"cd {args.remote_dir} && {remote}"]
        processes.append(
            (index, subprocess.Popen(
                command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            ))
        )

    failed = False
    for index, process in processes:
        try:
            stdout, stderr = process.communicate(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            print(f"node {index} did not finish inside {args.timeout}s", file=sys.stderr)
            failed = True
            continue
        if process.returncode != 0:
            # An empty reason is not a reason. A node that was killed reports nothing, and printing a blank
            # line after "node 1 failed:" reads like the tool lost the message rather than like the process
            # was shot — which is what happened when a peer was killed mid-exchange to test the failure path.
            detail = stderr.strip() or signal_explanation(process.returncode)
            print(f"node {index} failed (exit {process.returncode}):\n{detail}", file=sys.stderr)
            failed = True
    if failed:
        print("M3 GATE FAILED: the cluster could not complete the run", file=sys.stderr)
        return 1

    for index, entry in enumerate(entries[1:], start=1):
        local = OUT / f"node-{index}"
        if local.exists():
            shutil.rmtree(local)
        local.mkdir(parents=True)
        subprocess.run(
            ["scp", "-q", "-r", f"{entry}:{args.remote_dir}/node-{index}/.", str(local)],
            check=True, cwd=ROOT,
        )

    per_node = []
    for index in range(len(entries)):
        path = OUT / f"node-{index}" / "metrics.json"
        if not path.exists():
            print(f"      node {index}: NOT REPORTED (no metrics.json)", file=sys.stderr)
            return 1
        metrics = json.loads(path.read_text())
        seconds = metrics["step_seconds_total"] / max(1, metrics["steps"])
        per_node.append({"node": index, "seconds_per_step": seconds, "reduces": metrics.get("exchange_reduces")})

    # 4. Correctness first, then speed.
    print("[3/4] bit-identity: every node's tokens and digest against the baseline")
    identical = True
    for index in range(len(entries)):
        tokens = _tokens(OUT / f"node-{index}")
        if tokens != baseline_tokens:
            identical = False
            print(f"      node {index}: tokens {tokens} != baseline {baseline_tokens}", file=sys.stderr)
            continue
        diff = gate.run(
            [sys.executable, str(ROOT / "tools" / "trace_diff.py"),
             str(baseline_dir), str(OUT / f"node-{index}")]
        )
        print(f"      node {index}: {'IDENTICAL' if diff.returncode == 0 else 'DIFFERS'} — tokens {tokens}")
        if diff.returncode != 0:
            identical = False
    if not identical:
        print("M3 GATE FAILED: the cluster did not reproduce the single-node result", file=sys.stderr)
        return 1

    ratio = speedup(baseline_per_step, [entry["seconds_per_step"] for entry in per_node])
    print("[4/4] throughput")
    for entry in per_node:
        print(f"      node {entry['node']}: {entry['seconds_per_step']:.3f} s/step")
    print(f"      baseline {baseline_per_step:.3f} s/step vs slowest node {max(e['seconds_per_step'] for e in per_node):.3f} s/step = {ratio:.2f}x")

    report = {
        "install": str(args.install),
        "nodes": len(entries),
        "steps": args.steps,
        "load_average": loads,
        "quiet_load_threshold": args.quiet_load,
        "busy_farm": bool(busy),
        "baseline_seconds_per_step": baseline_per_step,
        "nodes_seconds_per_step": per_node,
        "speedup": ratio,
        "min_speedup": args.min_speedup,
        "tokens": baseline_tokens,
        "bit_identical": True,
        "observation_only": bool(busy or args.allow_busy_farm),
    }
    if args.report:
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"      report: {args.report}")

    if report["observation_only"]:
        print(
            "M3 OBSERVATION ONLY: bit-identity holds and the ratio is "
            f"{ratio:.2f}x, but the farm was busy or --allow-busy-farm was given, so this is not a gate "
            "result and the threshold was not asserted."
        )
        return 0
    if ratio < args.min_speedup:
        print(f"M3 GATE FAILED: {ratio:.2f}x is below the {args.min_speedup}x the roadmap asks for", file=sys.stderr)
        return 1
    print(f"M3 GATE PASSED: {ratio:.2f}x on {len(entries)} nodes with bit-identity intact")
    return 0


def _tokens(trace: Path) -> list[int]:
    manifest = json.loads((trace / "manifest.json").read_text())
    for discrete in manifest.get("discrete", []):
        if discrete.get("name") == "generated.tokens":
            return list(discrete["values"])
    raise SystemExit(f"{trace}: no generated.tokens in the manifest")


if __name__ == "__main__":
    raise SystemExit(main())
