#!/usr/bin/env python3
"""Run M1's sweep: the same prompt at several expert-bank sizes, and the curve that decides `D12`.

Why a script rather than a sequence in a message. M1's gate wants "a measured cache hit rate", `D12`
wants a size chosen from a measurement, and the run that produces both is a heavy one on a node that
has **panicked twice** while doing heavy things. A sequence typed from memory is how that goes wrong;
this is the sequence, with the preconditions checked before it starts and a refusal if they are not met.

    tools/run_m1_sweep.py --snapshot .build/m1-install --tokens 1,2,3,4,5 \
        --binary .build/release/datacenter-trace --slots 1,2,8,16 --out .build/m1-sweep

`--dry-run` prints the plan and the preconditions and runs nothing, which is also how the tests exercise
it without a model.

**Preconditions, checked and not promised** (the same rule the install builder follows):

- `tools/check_disk_headroom.py`'s 5 GB floor, because exhausting the disk here means exhausting swap;
- `sysctl vm.swapusage`, because a machine already well into swap is the state that preceded both
  panics. Reported always; a refusal above `--swap-used-limit-gb` (default 1.0).

The metrics file's schema is **read, not assumed**: `metrics.json` is reported whole, and the expert
figures are extracted when the keys are present under any of several plausible names. A script that
invented the shape of a file it did not read would fail exactly where this session keeps failing.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_disk_headroom import require_headroom  # noqa: E402


def swap_usage() -> dict[str, float]:
    """`sysctl vm.swapusage`, as megabytes. Empty when the key is absent (not macOS, or no swap)."""
    out = subprocess.run(["sysctl", "vm.swapusage"], capture_output=True, text=True).stdout
    values: dict[str, float] = {}
    for field in out.replace("=", " ").split():
        if field.endswith("M") and values:
            try:
                values[list(values)[-1] + "_next"] = float(field[:-1])
            except ValueError:
                pass
    # The output is `total = 2048.00M used = 12.50M free = 2035.50M`; parse it by name.
    tokens = out.replace("=", " ").split()
    for index, token in enumerate(tokens):
        if token in ("total", "used", "free") and index + 1 < len(tokens):
            raw = tokens[index + 1]
            if raw.endswith("M"):
                try:
                    values[token] = float(raw[:-1])
                except ValueError:
                    pass
    return {k: v for k, v in values.items() if not k.endswith("_next")}


def extract_expert_metrics(metrics: dict) -> dict:
    """Whatever the file records about the expert bank, under any of the names it might use."""
    found: dict = {}
    # **In document order**, so "the last layer's figure wins" is a property of the file rather than of
    # which end of a stack happened to be popped. The first version used a LIFO stack and its own test
    # caught the reversal.
    queue = [metrics]
    head = 0
    while head < len(queue):
        node = queue[head]
        head += 1
        if isinstance(node, dict):
            for key, value in node.items():
                lowered = key.lower()
                if any(word in lowered for word in ("hit", "miss", "request", "element", "resident", "bytes")):
                    found[key] = value
                elif isinstance(value, (dict, list)):
                    queue.append(value)
        elif isinstance(node, list):
            queue.extend(item for item in node if isinstance(item, (dict, list)))
    return found


def plan(args, swap: dict[str, float]) -> list[list[str]]:
    """One command per bank size. Printed by `--dry-run` and executed otherwise."""
    commands = []
    for slots in args.slots:
        out = args.out / f"slots-{slots}"
        commands.append([str(args.binary), str(args.snapshot), str(out), args.tokens])
    return commands


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True, help="the install directory")
    parser.add_argument("--tokens", required=True, help="comma-separated token ids, as the CLI takes them")
    parser.add_argument("--binary", type=Path, default=Path(".build/release/datacenter-trace"))
    parser.add_argument("--slots", default="1,2,8,16", help="bank sizes to sweep, comma separated")
    parser.add_argument("--out", type=Path, default=Path(".build/m1-sweep"))
    parser.add_argument("--swap-used-limit-gb", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true", help="check the preconditions, run nothing")
    args = parser.parse_args(argv)
    args.slots = [int(part) for part in args.slots.split(",") if part.strip()]

    print(f"sweep: {args.snapshot} at bank sizes {args.slots}")
    print(f"tokens: {args.tokens}   binary: {args.binary}   out: {args.out}")

    swap = swap_usage()
    if swap:
        print(f"swap: total {swap.get('total', float('nan')):.0f} MB, used {swap.get('used', float('nan')):.0f} MB, "
              f"free {swap.get('free', float('nan')):.0f} MB")
    else:
        print("swap: not reported by this system")

    # The floor, before anything heavy. `require_headroom` refuses below it and refuses if the watchdog
    # left a stop marker, which is deliberate: a run that tripped the limit must not resume by itself.
    require_headroom(purpose="the M1 sweep")

    used_gb = swap.get("used", 0.0) / 1024.0
    if used_gb > args.swap_used_limit_gb:
        print(
            f"REFUSING: {used_gb:.2f} GB of swap already in use, above the {args.swap_used_limit_gb:.2f} GB "
            f"limit. Both panics this node has had were preceded by swap growth, so a machine already "
            f"swapping is not the machine to start a 35 B run on.",
            file=sys.stderr,
        )
        return 2

    for slots, command in zip(args.slots, plan(args, swap)):
        environment = dict(os.environ, SHARD_EXPERT_SLOTS=str(slots))
        if args.dry_run:
            print(f"  would run: SHARD_EXPERT_SLOTS={slots} " + " ".join(command))
            continue
        print(f"  running bank of {slots} ...", flush=True)
        (args.out / f"slots-{slots}").mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(command, env=environment)
        if completed.returncode != 0:
            print(f"  bank {slots}: the trace exited {completed.returncode}", file=sys.stderr)
            return completed.returncode

    if args.dry_run:
        print("dry run: nothing was executed")
        return 0

    report: dict = {"snapshot": str(args.snapshot), "tokens": args.tokens, "slots": {}}
    for slots in args.slots:
        metrics_path = args.out / f"slots-{slots}" / "metrics.json"
        if not metrics_path.exists():
            report["slots"][slots] = {"error": "no metrics.json was written"}
            continue
        metrics = json.loads(metrics_path.read_text())
        report["slots"][slots] = {"metrics": metrics, "expert": extract_expert_metrics(metrics)}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "sweep.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")

    print("\nbank size | expert metrics")
    for slots in args.slots:
        entry = report["slots"][slots]
        print(f"{slots:>9} | {entry.get('expert', entry.get('error'))}")
    print(f"\nreport: {args.out / 'sweep.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
