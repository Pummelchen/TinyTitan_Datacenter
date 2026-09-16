#!/usr/bin/env python3
"""Generation across a shard, on real machines (`DC-109`).

M2's gate compares a *forward*; this compares a **generation** — the cached decode path, one token at a
time, with an all-reduce inside every mixture layer of every step. It is the instrument M3's throughput
gate needs, and the reason it exists is a defect it would have hidden: the cached path used to call the
mixture directly, so a sharded `--cached` run computed only its own experts, all-reduced nothing, and
produced a token sequence that looked entirely reasonable.

    python3 tools/run_sharded_generation.py --install .build/m1-install \\
        --mesh node4@<addr>,node3@<addr> --remote-install /Users/node3/Downloads/m1-install \\
        --prompt 760,6511,314,9338,369 --steps 2

**No timings are claimed.** The run reports how long it took because a log without it is hard to read,
and the standing rule for this phase is functional tests: the farm is shared and nothing is isolated.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_m2_gate as gate  # noqa: E402

ROOT = gate.ROOT
OUT = ROOT / ".build" / "sharded-generation"


def generated_tokens(trace: Path) -> list[int]:
    """The token ids a run produced, from its own manifest — the trace is the evidence, not stdout."""
    manifest = json.loads((trace / "manifest.json").read_text())
    for discrete in manifest.get("discrete", []):
        if discrete.get("name") == "generated.tokens":
            return list(discrete["values"])
    raise SystemExit(f"{trace}: no generated.tokens in the manifest")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", type=Path, required=True)
    parser.add_argument("--mesh", required=True, help="user@host for both nodes, THIS host first")
    parser.add_argument("--remote-install", default=None, help="install path on the remote node")
    parser.add_argument("--remote-dir", default="Downloads/sharded-generation")
    parser.add_argument("--prompt", default="760,6511,314,9338,369")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args(argv)

    entries = [entry.strip() for entry in args.mesh.split(",") if entry.strip()]
    if len(entries) != 2:
        raise SystemExit("this driver runs two nodes; the mesh join supports any N, this comparison does not")
    binaries = {
        "generate": ROOT / ".build" / "release" / "datacenter-generate",
        "trace": ROOT / ".build" / "release" / "datacenter-trace",
    }
    for name, binary in binaries.items():
        if not binary.exists():
            raise SystemExit(f"{binary} is missing: run `swift build -c release` first ({name})")

    OUT.mkdir(parents=True, exist_ok=True)
    plan_path = OUT / "plan.json"
    plan = gate.write_plan(args.install, 2, plan_path)
    print(f"[1/4] plan: {plan['experts']} experts over 2 nodes, written to {plan_path.relative_to(ROOT)}")

    # The reference: one node, the same decode mode the cluster will use.
    reference = OUT / "reference"
    command = [
        str(binaries["generate"]), str(args.install), str(reference), args.prompt, str(args.steps),
        "--cached",
    ]
    single = gate.run(command)
    if single.returncode != 0:
        print(single.stdout + single.stderr, file=sys.stderr)
        raise SystemExit("the single-node reference generation failed")
    reference_tokens = generated_tokens(reference)
    print(f"[2/4] reference (one node): {reference_tokens}")

    base = random.randint(41_000, 59_000)
    config = {
        "endpoints": [
            {"host": entry.split("@")[-1], "port": base + index} for index, entry in enumerate(entries)
        ]
    }
    config_path = OUT / "cluster.json"
    config_path.write_text(json.dumps(config, sort_keys=True, separators=(",", ":")))
    remote = entries[1]
    gate.stage_remote(
        remote, args.remote_dir, args.install, binaries["generate"], plan_path, args.remote_install,
        extra=[config_path],
    )
    print(f"[3/4] node 1 staged on {remote}; both nodes generating {args.steps} token(s)")

    processes: list[tuple[int, subprocess.Popen]] = []
    processes.append(
        (
            0,
            subprocess.Popen(
                [
                    str(binaries["generate"]), str(args.install), str(OUT / "node-0"), args.prompt,
                    str(args.steps), "--cached", "--plan", str(plan_path), "--config", str(config_path),
                    "--node", "0",
                ],
                cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            ),
        )
    )
    install = args.remote_install or "./install"
    processes.append(
        (
            1,
            subprocess.Popen(
                [
                    "ssh", remote,
                    f"cd {args.remote_dir} && ./datacenter-generate {install} ./node-1 {args.prompt} "
                    f"{args.steps} --cached --plan ./plan.json --config ./cluster.json --node 1",
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

    local = OUT / "node-1"
    if local.exists():
        shutil.rmtree(local)
    local.mkdir(parents=True)
    subprocess.run(
        ["scp", "-q", "-r", f"{remote}:{args.remote_dir}/node-1/.", str(local)], check=True, cwd=ROOT
    )

    print("[4/4] the token list each node produced, against the reference")
    tokens = {0: generated_tokens(OUT / "node-0"), 1: generated_tokens(local)}
    for node in (0, 1):
        print(f"      node {node}: {tokens[node]}")
    if tokens[0] != reference_tokens or tokens[1] != reference_tokens:
        print(
            f"SHARDED GENERATION FAILED: reference {reference_tokens}, node 0 {tokens[0]}, "
            f"node 1 {tokens[1]}",
            file=sys.stderr,
        )
        return 1

    # And the traces themselves, which is the stronger claim: identical bytes, not just tokens.
    for node, path in ((0, OUT / "node-0"), (1, local)):
        diff = gate.run(
            [sys.executable, str(ROOT / "tools" / "trace_diff.py"), str(reference), str(path)]
        )
        print(f"      node {node}: {diff.stdout.strip() or diff.stderr.strip()}")
        if diff.returncode != 0:
            failed = True

    if failed:
        print("SHARDED GENERATION FAILED", file=sys.stderr)
        return 1
    print(f"SHARDED GENERATION PASSED: two machines, {len(reference_tokens)} token(s), {tokens[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
