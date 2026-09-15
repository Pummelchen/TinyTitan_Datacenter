#!/usr/bin/env python3
"""Refuse to start a heavy job when the disk is short, or when the watchdog has stopped us.

The companion to `disk_watchdog.py`. The watchdog stops what is running; this stops what has not
started yet, which is the half that matters for a job that would otherwise allocate twenty
gigabytes.

    python3 tools/check_disk_headroom.py            # exits non-zero if below the threshold
    python3 tools/check_disk_headroom.py --threshold-gb 5 --quiet

`tools/quantize.py`, the contract CLIs and `tools/run_m1_gate.py` call `require_headroom()` at
startup, so a heavy run cannot begin under the threshold even if nobody is watching. Standard
library only.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_THRESHOLD_GB = 5.0
STOP_MARKER = ROOT / ".build" / "DISK_STOP"


class HeadroomError(SystemExit):
    """Raised (as a `SystemExit`) so a tool stops with a message rather than a traceback."""


def free_gb(path: Path = ROOT) -> float:
    return shutil.disk_usage(path).free / 1e9


def require_headroom(
    threshold_gb: float = DEFAULT_THRESHOLD_GB, *, purpose: str = "this job", marker: Path | None = None
) -> None:
    """Stop the caller unless there is room for it.

    Two conditions, and the second is the one a disk check alone would miss: a **stop marker**
    from the watchdog, which stays until it is removed deliberately. Clearing it is an operator
    decision — the whole point is that a run which tripped the limit does not resume by itself.
    """
    stop_marker = STOP_MARKER if marker is None else marker
    if stop_marker.exists():
        raise HeadroomError(
            f"refusing to start {purpose}: {stop_marker} exists, so the disk watchdog stopped a "
            f"run and nobody has cleared it.\n"
            f"  Read it, free space, then delete the marker deliberately:\n"
            f"    cat {stop_marker} && rm {stop_marker}"
        )
    available = free_gb()
    if available < threshold_gb:
        raise HeadroomError(
            f"refusing to start {purpose}: {available:.2f} GB free, below the {threshold_gb} GB "
            f"floor.\n"
            f"  This floor exists because exhausting disk here means exhausting swap, and on "
            f"2026-09-16 that panicked this machine twice.\n"
            f"  Free space, or raise it deliberately with --threshold-gb if you have measured "
            f"that the job needs less."
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold-gb", type=float, default=DEFAULT_THRESHOLD_GB)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    available = free_gb()
    if not args.quiet:
        print(f"disk: {available:.2f} GB free, floor {args.threshold_gb} GB")
    require_headroom(args.threshold_gb, purpose="this check")
    if not args.quiet:
        print("disk: enough room")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
