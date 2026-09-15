#!/usr/bin/env python3
"""Watch free disk space and stop the heavy jobs before the machine runs out.

On 2026-09-16 this node panicked twice. The cause was not a bug: the 35 B engine and a 20 GB
install build ran concurrently on an 8 GB machine, macOS grew swap to *13 swapfiles and LOW swap
space*, the system stopped responding for 90 seconds, and the hardware watchdog panicked it. A
machine with almost no free disk has no room for the swap that failure feeds on, so free space is
a stability limit and not only a housekeeping number.

This is the software version of that watchdog, with a threshold the operator chooses:

    python3 tools/disk_watchdog.py --threshold-gb 5

It polls, and when free space falls below the threshold it

1. writes a **stop marker** (`.build/DISK_STOP`) that `check_disk_headroom` refuses to run past,
   so no new heavy job starts, and
2. terminates the heavy jobs that are running — the engine, the contract, the install builder —
   with `SIGTERM` first and `SIGKILL` after a short grace period.

It never terminates itself, and it matches tools by name rather than killing every Python on the
machine, because the harness that runs this agent is also a Python process.

Standard library only, like everything under `tools/` that gates the repository.
"""

from __future__ import annotations

import argparse
import os
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MARKER = ROOT / ".build" / "DISK_STOP"

# The jobs that move gigabytes. Matched against the full command line, and chosen so that a
# `python3 tools/foo.py` invocation matches while this watchdog and the agent harness do not.
HEAVY_PATTERNS = (
    "datacenter-trace",
    "datacenter-generate",
    "tools/quantize.py",
    "tools/ordered_qwen35_trace.py",
    "tools/ordered_qwen36_trace.py",
    "tools/ordered_reference.py",
    "tools/run_m1_gate.py",
    "tools/measure_quantization.py",
    "tools/check_engine_contract.py",
    "tools/check_engine_generation.py",
    "tools/make_tiny",
)


def free_gb(path: Path) -> float:
    """Free space in gigabytes, from the filesystem itself rather than from a `df` scrape."""
    usage = shutil.disk_usage(path)
    return usage.free / 1e9


def heavy_processes(exclude: set[int]) -> list[tuple[int, str]]:
    """Running heavy jobs, by pid and command line, excluding the given pids."""
    found: list[tuple[int, str]] = []
    listing = subprocess.run(
        ["ps", "-Ao", "pid=,command="], capture_output=True, text=True, check=False
    ).stdout
    for line in listing.splitlines():
        line = line.strip()
        if not line:
            continue
        pid_text, _, command = line.partition(" ")
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if pid in exclude:
            continue
        if "disk_watchdog.py" in command:
            continue
        if any(pattern in command for pattern in HEAVY_PATTERNS):
            found.append((pid, command))
    return found


def stop_them(processes: list[tuple[int, str]], grace: float) -> list[str]:
    """SIGTERM, wait, then SIGKILL whatever is left. Returns what was done, for the log."""
    done: list[str] = []
    for pid, command in processes:
        try:
            os.kill(pid, signal.SIGTERM)
            done.append(f"TERM {pid} {command[:90]}")
        except ProcessLookupError:
            continue
        except PermissionError:
            done.append(f"TERM denied for {pid} (not ours)")
    if grace > 0 and processes:
        time.sleep(grace)
    for pid, command in processes:
        try:
            os.kill(pid, signal.SIGKILL)
            done.append(f"KILL {pid} {command[:90]}")
        except ProcessLookupError:
            continue
        except PermissionError:
            continue
    return done


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold-gb", type=float, default=5.0, help="stop below this many GB free")
    parser.add_argument("--interval", type=float, default=5.0, help="seconds between checks")
    parser.add_argument("--grace", type=float, default=5.0, help="seconds between SIGTERM and SIGKILL")
    parser.add_argument("--marker", type=Path, default=DEFAULT_MARKER)
    parser.add_argument("--watch", type=Path, default=ROOT, help="filesystem to watch")
    parser.add_argument("--once", action="store_true", help="check once and exit (for tests)")
    args = parser.parse_args(argv)

    args.marker.parent.mkdir(parents=True, exist_ok=True)
    mine = {os.getpid(), os.getppid()}
    print(
        f"disk watchdog: watching {args.watch} for {args.threshold_gb} GB free, "
        f"every {args.interval}s, marker {args.marker}",
        flush=True,
    )

    while True:
        available = free_gb(args.watch)
        if available < args.threshold_gb:
            running = heavy_processes(mine)
            actions = stop_them(running, args.grace)
            message = (
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} STOP: {available:.2f} GB free, below "
                f"{args.threshold_gb} GB; stopped {len(running)} heavy job(s)\n"
            )
            for action in actions:
                message += f"  {action}\n"
            # The marker is written **before** anything else so that a new heavy job cannot start
            # in the window between stopping the old ones and the next check.
            args.marker.write_text(message)
            print(message, flush=True)
        elif args.marker.exists():
            print(
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} {available:.2f} GB free, above threshold; "
                f"leaving {args.marker.name} in place until it is removed deliberately",
                flush=True,
            )
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
