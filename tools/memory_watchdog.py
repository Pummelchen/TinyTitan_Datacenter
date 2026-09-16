"""Stop heavy jobs that outgrow this machine's real-memory budget.

`disk_watchdog.py` guards the disk, and it earned its place: on 2026-09-16 the engine's memory grew swap,
swap is disk, and the machine panicked. This guards the thing upstream of that — **resident memory** — and it
exists because the guard was missing when it was needed.

On 2026-09-17 two of my own comparison scripts grew to gigabytes on an 8 GB node. The first was killed by the
OS, and the second filled the page cache until `disk_watchdog.py` tripped the 5 GB disk floor and wrote its
stop marker. Neither script *meant* to be heavy: one wrapped an already-streaming iterator in `list()`, which
materialises a whole tensor, and the other asked for a dequantised tensor as a Python list of floats, which is
twenty-four bytes per value. The lesson is not "be careful" — it is that a budget nobody enforces is a
comment, so:

* the ceiling is **4.0 GB of real memory** by default, which is what this machine's operator states is
  available for experiment on an 8 GB node;
* three readings in a row above it, not one, for the reason `disk_watchdog.py` records: a single reading of a
  fuzzy number once killed a read-only verification that was doing nothing wrong;
* it writes a stop marker, so a run that tripped the limit cannot resume by itself.

    python3 tools/memory_watchdog.py --limit-gb 4 --interval 5
    python3 tools/memory_watchdog.py --once --limit-gb 4      # one reading, for tests and for a check
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from disk_watchdog import HEAVY_PATTERNS  # noqa: E402  the same definition of "heavy job"

DEFAULT_LIMIT_GB = 4.0
DEFAULT_MARKER = Path(__file__).resolve().parent.parent / ".build" / "MEMORY_STOP"


class MemoryStop(Exception):
    """Raised when the budget has been breached, so a caller stops instead of continuing."""


def resident_gb(patterns: tuple[str, ...] = HEAVY_PATTERNS) -> dict[int, tuple[str, float]]:
    """Heavy processes and their resident size in GB, by pid.

    Resident, not virtual: `ps`'s `rss` is the memory actually held, which is the quantity a page cache,
    swap and a panic are made of. A process's virtual size is routinely tens of gigabytes and means nothing.
    """
    listing = subprocess.run(
        ["ps", "-Ao", "pid=,rss=,command="], capture_output=True, text=True, check=False
    ).stdout
    found: dict[int, tuple[str, float]] = {}
    for line in listing.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 2)
        if len(parts) != 3:
            continue
        pid_text, rss_text, command = parts
        if "memory_watchdog.py" in command or "disk_watchdog.py" in command:
            continue
        if not any(pattern in command for pattern in patterns):
            continue
        try:
            found[int(pid_text)] = (command, int(rss_text) / 1048576)
        except ValueError:
            continue
    return found


def offenders(limit_gb: float, patterns: tuple[str, ...] = HEAVY_PATTERNS) -> dict[int, tuple[str, float]]:
    """The heavy processes already over the budget, by pid."""
    return {pid: entry for pid, entry in resident_gb(patterns).items() if entry[1] > limit_gb}


def stop_them(processes: dict[int, tuple[str, float]], grace: float) -> list[str]:
    """SIGTERM, wait, then SIGKILL whatever is left. Returns what was done, for the log."""
    done: list[str] = []
    for pid, (command, size) in processes.items():
        try:
            os.kill(pid, signal.SIGTERM)
            done.append(f"SIGTERM {pid} ({size:.2f} GB) {command[:70]}")
        except ProcessLookupError:
            continue
    if processes:
        time.sleep(grace)
    for pid, (command, size) in processes.items():
        try:
            os.kill(pid, signal.SIGKILL)
            done.append(f"SIGKILL {pid} ({size:.2f} GB) {command[:70]}")
        except ProcessLookupError:
            pass
    return done


def write_marker(marker: Path, message: str) -> None:
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit-gb", type=float, default=DEFAULT_LIMIT_GB, help="real-memory ceiling")
    parser.add_argument("--interval", type=float, default=5.0, help="seconds between checks")
    parser.add_argument("--grace", type=float, default=5.0, help="seconds between SIGTERM and SIGKILL")
    parser.add_argument(
        "--consecutive", type=int, default=3, help="readings in a row above the limit before acting"
    )
    parser.add_argument("--marker", type=Path, default=DEFAULT_MARKER)
    parser.add_argument("--once", action="store_true", help="check once and exit (for tests)")
    parser.add_argument("--verbose", action="store_true", help="print every reading")
    args = parser.parse_args(argv)

    over = 0
    while True:
        found = offenders(args.limit_gb)
        worst = max((size for _, size in found.values()), default=0.0)
        if args.verbose:
            print(f"{time.strftime('%H:%M:%S')} {len(found)} over {args.limit_gb:.1f} GB, worst {worst:.2f} GB")
        if found:
            over += 1
        else:
            over = 0
        if over >= args.consecutive:
            done = stop_them(found, args.grace)
            for line in done:
                print(line)
            write_marker(
                args.marker,
                f"MEMORY STOP: {len(found)} heavy job(s) over {args.limit_gb:.1f} GB real, "
                f"worst {worst:.2f} GB; stopping {len(done)} action(s)",
            )
            print(f"wrote {args.marker}")
            return 1
        if args.once:
            print(
                f"memory: {len(found)} heavy job(s) over {args.limit_gb:.1f} GB"
                if found
                else f"memory: no heavy job over {args.limit_gb:.1f} GB"
            )
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
