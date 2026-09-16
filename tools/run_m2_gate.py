#!/usr/bin/env python3
"""M2's gate at fixture scale: two **nodes**, one shard plan, one trace to match.

With no `--remote`, the two nodes are two processes on this host. With `--remote node1@node1`, node 1
runs on **another machine**, which is what M2's gate actually asks for: a different address space, a
different install reader, and a real link between them.

    python3 tools/run_m2_gate.py --remote node1@node1

The test is deliberately tiny — a ~2 MB fixture and the node binary, no model install, no benchmarks —
so it does not compete with anything else the farm is doing.

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
import random
import shutil
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


def stage_remote(
    remote: str, directory: str, install: Path, binary: Path, plan: Path, remote_install: str | None,
    extra: list[Path] | None = None,
) -> None:
    """Copy what a node needs onto another machine: the binary and the plan, plus the fixture when the
    remote has no install of its own.

    A 20 GB install is not copied here — `--remote-install` names one that is already on the peer, which
    is the difference between a functional test and a data migration.
    """
    subprocess.run(["ssh", remote, f"mkdir -p {directory}"], check=True, cwd=ROOT)
    sources = [str(binary), str(plan)] if remote_install else [str(binary), str(install), str(plan)]
    sources += [str(path) for path in (extra or [])]
    subprocess.run(["scp", "-q", "-r", *sources, f"{remote}:{directory}/"], check=True, cwd=ROOT)


def remote_address(host: str) -> str:
    """Resolve the peer's address, and say which one it is.

    The farm resolves node names over a mesh VPN, which the Testbed notes is ~2.5x the round-trip of the
    direct Ethernet segment. For a functional run that is irrelevant; for the timing work it is the whole
    difference, so the address is printed rather than assumed.
    """
    import socket

    return socket.gethostbyname(host)



def run_mesh(args, binaries, plan_path: Path, reference: Path, plan: dict) -> int:
    """Run every node as its own process on its own machine, in a full mesh.

    Every node must be starting at once: the mesh rule has node *i* connect to lower ids and accept from
    higher ones, so no node can finish joining before the others begin. They are started in parallel and
    their traces are fetched back for the differ.
    """
    entries = [entry.strip() for entry in args.mesh.split(",") if entry.strip()]
    if len(entries) < 2:
        raise SystemExit("--mesh needs at least two entries, this host first")
    if plan["nodes"] != len(entries):
        raise SystemExit(f"the plan covers {plan['nodes']} nodes and --mesh names {len(entries)}")

    base = random.randint(41_000, 59_000)
    endpoints = [{"host": e.split("@")[-1], "port": base + i} for i, e in enumerate(entries)]
    config_path = OUT / "cluster.json"
    config_path.write_text(json.dumps({"endpoints": endpoints}, sort_keys=True, separators=(",", ":")))
    print(f"[3/4] mesh of {len(entries)} nodes: " + ", ".join(f"{e}({endpoints[i]['host']})" for i, e in enumerate(entries)))

    for index, remote in enumerate(entries[1:], start=1):
        stage_remote(
            remote, args.remote_dir, args.install, binaries["node"], plan_path, args.remote_install,
            extra=[config_path],
        )
        print(f"      staged node {index} on {remote}")

    processes: list[tuple[int, subprocess.Popen]] = []
    processes.append(
        (
            0,
            subprocess.Popen(
                [
                    str(binaries["node"]), str(args.install), str(OUT / "node-0"), args.tokens,
                    str(plan_path), "--node", "0", "--config", str(config_path),
                ],
                cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            ),
        )
    )
    for index, remote in enumerate(entries[1:], start=1):
        install = args.remote_install or "./install"
        processes.append(
            (
                index,
                subprocess.Popen(
                    [
                        "ssh", remote,
                        f"cd {args.remote_dir} && ./datacenter-node {install} ./node-{index} "
                        f"{args.tokens} ./plan.json --node {index} --config ./cluster.json",
                    ],
                    cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                ),
            )
        )

    failed = False
    for node, process in processes:
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

    for index, remote in enumerate(entries[1:], start=1):
        local = OUT / f"node-{index}"
        if local.exists():
            shutil.rmtree(local)
        local.mkdir(parents=True)
        subprocess.run(
            ["scp", "-q", "-r", f"{remote}:{args.remote_dir}/node-{index}/.", str(local)],
            check=True, cwd=ROOT,
        )
    print(f"      fetched {len(entries) - 1} trace(s)")

    print("[4/4] trace_diff, every node against the reference")
    for node in range(len(entries)):
        diff = run(
            [
                sys.executable, str(ROOT / "tools" / "trace_diff.py"),
                str(reference), str(OUT / f"node-{node}"),
            ]
        )
        print(f"      node {node}: {diff.stdout.strip() or diff.stderr.strip()}")
        if diff.returncode != 0:
            failed = True
    if failed:
        print("M2 GATE FAILED", file=sys.stderr)
        return 1
    what = args.remote_install or "the fixture"
    print(f"M2 GATE PASSED: a mesh of {len(entries)} machines, install {what}, one plan, one trace")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", type=Path, default=FIXTURE)
    parser.add_argument("--tokens", default="1,2,3")
    parser.add_argument("--nodes", type=int, default=2, help="2 for M2's gate; the CLI is two-node for now")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--keep", action="store_true", help="leave .build/m2-gate in place")
    parser.add_argument(
        "--remote", default=None,
        help="run node 1 on another machine, as user@host (e.g. node1@node1); "
             "the fixture and the node binary are copied, and the trace is fetched back",
    )
    parser.add_argument("--remote-dir", default="m2-gate", help="where to stage files on the remote")
    parser.add_argument(
        "--mesh", default=None,
        help="user@host for every node with THIS host first, comma separated "
             "(e.g. node4@10.0.0.4,node1@10.0.0.1). Runs a full mesh: the all-reduce is pairwise, so a "
             "star would leave a leaf holding only the coordinator's terms",
    )
    parser.add_argument(
        "--remote-install", default=None,
        help="an install already present on the remote (e.g. Downloads/m1-install); the fixture is not "
             "copied in that case, which is what a 20 GB install needs",
    )
    args = parser.parse_args(argv)

    if args.mesh:
        pass
    elif args.nodes != 2:
        # The engine's ShardExecution takes any number of peers, and the plan covers any N, but the
        # `datacenter-node` CLI takes exactly one --listen or --connect. Saying so is better than a
        # harness that appears to support four nodes and hangs at the second.
        raise SystemExit(f"without --mesh this runs two nodes; asked for {args.nodes}")
    if not (args.install / "install.json").exists():
        raise SystemExit(f"{args.install} is not an install (no install.json)")
    binaries = {
        "node": ROOT / ".build" / "release" / "datacenter-node",
        "trace": ROOT / ".build" / "release" / "datacenter-trace",
    }
    for name, binary in binaries.items():
        if not binary.exists():
            raise SystemExit(f"{binary} is missing: run `swift build -c release` first ({name})")

    if args.mesh:
        # The mesh decides how many nodes there are; --nodes is for the single-host path.
        args.nodes = len([entry for entry in args.mesh.split(",") if entry.strip()])

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

    if args.mesh:
        return run_mesh(args, binaries, plan_path, reference, plan)

    # The cluster: node 0 listens and reports its port, node 1 connects to it.
    processes: list[subprocess.Popen] = []
    try:
        # Who listens depends on where the other node is. On one host the local node listens and the
        # peers connect to it; with a remote peer the **remote** node listens, because this host is the
        # one that can reach the other machine's address — and only one of them may hold the role.
        if args.remote:
            stage_remote(
                args.remote, args.remote_dir, args.install, binaries["node"], plan_path, args.remote_install
            )
            print(f"[3/4] two machines: node 1 listening on {args.remote}, node 0 connecting from here")
            remote_install = args.remote_install or "./install"
            remote = subprocess.Popen(
                [
                    "ssh", args.remote,
                    f"cd {args.remote_dir} && ./datacenter-node {remote_install} ./node-1 {args.tokens} "
                    f"./plan.json --node 1 --listen 0.0.0.0:0",
                ],
                cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            processes.append(remote)
            port = wait_for_port(remote, args.timeout)
            host = args.remote.split("@")[-1]
            address = remote_address(host)
            print(f"      node 1 bound port {port} on {address} (the farm's names take the VPN)")
            processes.append(
                subprocess.Popen(
                    [
                        str(binaries["node"]), str(args.install), str(OUT / "node-0"), args.tokens,
                        str(plan_path), "--node", "0", "--connect", f"{address}:{port}",
                    ],
                    cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                )
            )
        else:
            print(f"[3/4] {args.nodes} processes over TCP on one host")
            processes.append(
                subprocess.Popen(
                    [
                        str(binaries["node"]), str(args.install), str(OUT / "node-0"), args.tokens,
                        str(plan_path), "--node", "0", "--listen", "127.0.0.1:0",
                    ],
                    cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                )
            )
            port = wait_for_port(processes[0], args.timeout)
            print(f"      node 0 bound port {port}")
            for node in range(1, args.nodes):
                processes.append(
                    subprocess.Popen(
                        [
                            str(binaries["node"]), str(args.install), str(OUT / f"node-{node}"),
                            args.tokens, str(plan_path), "--node", str(node),
                            "--connect", f"127.0.0.1:{port}",
                        ],
                        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                    )
                )

        failed = False
        # The label is the **node id**, not the process index: with a remote peer the listener is node 1
        # and it is started first, and printing "node 0" beside its output is how a passing run gets read
        # as a failing one.
        labels = [1, 0] if args.remote else list(range(len(processes)))
        for slot, process in enumerate(processes):
            label = labels[slot]
            try:
                stdout, stderr = process.communicate(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                print(f"node {label} did not finish inside {args.timeout}s", file=sys.stderr)
                failed = True
                continue
            for line in stdout.strip().splitlines():
                print(f"      node {label}: {line}")
            if process.returncode != 0:
                print(f"node {label} failed:\n{stderr}", file=sys.stderr)
                failed = True
        if failed:
            return 1

        if args.remote:
            # Fetch the remote trace so the differ runs here, where the reference is. A trace is a
            # directory, so this is a recursive copy into a destination that must not already exist —
            # the first version copied without -r and scp said so.
            local = OUT / "node-1"
            if local.exists():
                shutil.rmtree(local)
            local.mkdir(parents=True)
            subprocess.run(
                ["scp", "-q", "-r", f"{args.remote}:{args.remote_dir}/node-1/.", str(local)],
                check=True, cwd=ROOT,
            )
            print(f"      fetched node 1's trace from the other machine ({len(list(local.iterdir()))} files)")

        # The judge is the harness M0 built: byte-for-byte tensors and exact discrete decisions.
        print("[4/4] trace_diff, every node against the reference")
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
        print("M2 GATE FAILED", file=sys.stderr)
        return 1
    where = f"node 0 here and node 1 on {args.remote}" if args.remote else f"{args.nodes} processes on one host"
    what = args.remote_install or "the fixture"
    print(f"M2 GATE PASSED: {where}, install {what}, one plan, one trace")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
