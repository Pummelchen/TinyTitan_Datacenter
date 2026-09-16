#!/usr/bin/env python3
"""M2's gate at fixture scale: two **processes**, one shard plan, one trace to match.

M2's claim is that N nodes produce what one node produces. Everything before this ran the two nodes on
two threads in one address space, which shares a heap and a failure domain; this runs them as separate
processes over TCP, each with its own install reader, and compares their traces with the project's own
`trace_diff.py` — byte for byte, discrete decisions included.

    python3 tools/run_m2_gate.py
    python3 tools/run_m2_gate.py --install <path> --tokens 1,2,3 --keep

Writes under `.build/m2-gate/` and exits non-zero on any difference. The fixture is about a megabyte, so
this runs anywhere; the real model at two nodes does not fit on the 8 GB development host, which is what
`DC-045` is waiting for.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "DatacenterEngineTests" / "Fixtures" / "tiny-qwen36" / "install"
OUT = ROOT / ".build" / "m2-gate"


def expert_count(install: Path) -> int:
    """The number of experts, read from the install's own manifest rather than assumed."""
    manifest = json.loads((install / "install.json").read_text())
    for tensor in manifest["tensors"]:
        if tensor.get("role") == "expert.stack_gate_up":
            return int(tensor["shape"][0])
    raise SystemExit(f"{install}: no expert.stack_gate_up, so there is nothing to shard")


def install_family(install: Path) -> str:
    """The family, read from the install's own manifest.

    A hardcoded guess here does not fail quietly: the node validates the plan against the model it
    opened and refuses, which is `D20` working — but the first version of this harness guessed
    `tiny-qwen36` while the fixture's family is `qwen3_5_moe`, and the refusal arrived as a Swift trap.
    """
    manifest = json.loads((install / "install.json").read_text())
    return str(manifest["family"])


def write_plan(install: Path, nodes: int, path: Path) -> dict:
    """The plan every node will read. Contiguous blocks, which is the engine's default too."""
    experts = expert_count(install)
    if experts < nodes:
        raise SystemExit(f"{experts} experts over {nodes} nodes leaves nodes idle; not a useful gate")
    base, extra = divmod(experts, nodes)
    owners: list[int] = []
    for node in range(nodes):
        owners.extend([node] * (base + (1 if node < extra else 0)))
    plan = {
        "schema": 1,
        "family": install_family(install),
        "experts": experts,
        "nodes": nodes,
        "distribution": "contiguous",
        "owners": owners,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, sort_keys=True, separators=(",", ":")))
    return plan


def wait_for_port(process: subprocess.Popen, seconds: float) -> int:
    """Read the `PORT n` line a listening node prints once it is bound."""
    deadline = time.time() + seconds
    while time.time() < deadline:
        line = process.stdout.readline() if process.stdout else ""
        if not line:
            if process.poll() is not None:
                raise SystemExit(f"the listening node exited before binding (status {process.returncode})")
            continue
        line = line.strip()
        print(f"      node: {line}")
        if line.startswith("PORT "):
            return int(line.split()[1])
    raise SystemExit("the listening node never reported a port")


def run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(command, cwd=ROOT, capture_output=True, text=True, **kwargs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", type=Path, default=FIXTURE)
    parser.add_argument("--tokens", default="1,2,3")
    parser.add_argument("--nodes", type=int, default=2, help="2 for M2's gate; the CLI is two-node for now")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--keep", action="store_true", help="leave .build/m2-gate in place")
    args = parser.parse_args(argv)

    if args.nodes != 2:
        # The engine's ShardExecution takes any number of peers, and the plan covers any N, but the
        # `datacenter-node` CLI takes exactly one --listen or --connect. Saying so is better than a
        # harness that appears to support four nodes and hangs at the second.
        raise SystemExit(f"this CLI runs two nodes; asked for {args.nodes} (multi-peer is M3's work)")
    if not (args.install / "install.json").exists():
        raise SystemExit(f"{args.install} is not an install (no install.json)")
    binaries = {
        "node": ROOT / ".build" / "release" / "datacenter-node",
        "trace": ROOT / ".build" / "release" / "datacenter-trace",
    }
    for name, binary in binaries.items():
        if not binary.exists():
            raise SystemExit(f"{binary} is missing: run `swift build -c release` first ({name})")

    OUT.mkdir(parents=True, exist_ok=True)
    plan_path = OUT / "plan.json"
    plan = write_plan(args.install, args.nodes, plan_path)
    print(
        f"[1/4] plan: {plan['experts']} experts over {plan['nodes']} nodes, "
        f"{plan['distribution']}, written to {plan_path.relative_to(ROOT)}"
    )

    # The single-node reference. Same install, same tokens, no shard context.
    reference = OUT / "reference"
    single = run([str(binaries["trace"]), str(args.install), str(reference), args.tokens])
    if single.returncode != 0:
        print(single.stdout + single.stderr, file=sys.stderr)
        raise SystemExit("the single-node reference trace failed")
    print(f"[2/4] reference: {single.stdout.strip().splitlines()[-1]}")

    # The cluster: node 0 listens and reports its port, node 1 connects to it.
    processes: list[subprocess.Popen] = []
    try:
        print(f"[3/4] {args.nodes} processes over TCP")
        listener = subprocess.Popen(
            [
                str(binaries["node"]), str(args.install), str(OUT / "node-0"), args.tokens,
                str(plan_path), "--node", "0", "--listen", "127.0.0.1:0",
            ],
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        processes.append(listener)
        port = wait_for_port(listener, args.timeout)
        print(f"      node 0 bound port {port}")

        for node in range(1, args.nodes):
            connector = subprocess.Popen(
                [
                    str(binaries["node"]), str(args.install), str(OUT / f"node-{node}"), args.tokens,
                    str(plan_path), "--node", str(node), "--connect", f"127.0.0.1:{port}",
                ],
                cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            processes.append(connector)

        failed = False
        for node, process in enumerate(processes):
            try:
                stdout, stderr = process.communicate(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                print(f"node {node} did not finish inside {args.timeout}s", file=sys.stderr)
                failed = True
                continue
            for line in stdout.strip().splitlines():
                print(f"      node {node}: {line}")
            if process.returncode != 0:
                print(f"node {node} failed:\n{stderr}", file=sys.stderr)
                failed = True
        if failed:
            return 1

        # The judge is the harness M0 built: byte-for-byte tensors and exact discrete decisions.
        print("[4/4] trace_diff, every node against the reference and against each other")
        for node in range(args.nodes):
            diff = run(
                [
                    sys.executable, str(ROOT / "tools" / "trace_diff.py"),
                    str(reference), str(OUT / f"node-{node}"),
                ]
            )
            verdict = diff.stdout.strip() or diff.stderr.strip()
            print(f"      node {node}: {verdict}")
            if diff.returncode != 0:
                failed = True
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
        if not args.keep:
            pass  # outputs are evidence; they stay until the next run replaces them

    if failed:
        print("M2 GATE (fixture scale) FAILED", file=sys.stderr)
        return 1
    print(f"M2 GATE (fixture scale) PASSED: {args.nodes} processes, one plan, one trace")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
