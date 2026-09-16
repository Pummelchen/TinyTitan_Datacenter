"""Refuse to start a heavy job the machine cannot carry, or a second one at all.

`check_disk_headroom.py` enforces the disk floor. This adds the two things that made the worst incident
in this project's history possible, and neither of them is a disk question:

* **Physical memory.** On 2026-09-16 a 35 B engine at ~4.5 GB resident ran on a machine with about 4.5 GB
  usable, alongside a 20 GB install build. macOS grew swap to thirteen swapfiles, the machine stopped
  responding for 90 seconds, and the hardware watchdog panicked it. A job whose **declared** peak does not
  fit in usable memory cannot be made safe by free disk, so it is refused rather than attempted.
* **Concurrency.** "One heavy job at a time, never concurrent" was a rule in a document, and the incident
  above is what a rule in a document is worth. This is a lock instead, with stale detection — because a
  lock a crashed job leaves behind would refuse every future run and be deleted by the next person to hit
  it, which is how a guard becomes a formality.

**The refusal is deterministic and the warning is not.** A job is refused when its declared need exceeds
what the machine can use — arithmetic, no estimate. Free memory right now is *reported* and, when it is
already below the declared need, warned about loudly: macOS's notion of available memory is famously
fuzzy, and a guard that acts on one fuzzy reading is the mistake `disk_watchdog.py` already made once and
recorded. What the operator does with the warning is the human-approval rule; what this enforces is that
the job fits at all, and that it is alone.

    python3 tools/heavy_job.py --needs-gb 4.2 --purpose "the M1 checkpoint path"
    python3 tools/heavy_job.py --release          # deliberately, after reading what it says

Standard library only.
"""

from __future__ import annotations

import argparse
import atexit
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_disk_headroom import HeadroomError, free_gb, require_headroom  # noqa: E402

# The measured reserve: this node reports 8.59 GB of physical memory and the project's own notes put the
# usable figure at "about 4.5 GB usable after macOS". That is a measurement, not a safety factor, and it
# is stated here so a reader can disagree with the number rather than with a mystery.
SYSTEM_RESERVE_GB = 3.5
DEFAULT_LOCK_PATH = ROOT / ".build" / "HEAVY_JOB_LOCK"


def lock_path() -> Path:
    """Where the one heavy-job slot lives, overridable through `HEAVY_JOB_LOCK`.

    Read per call rather than at import, so a test (or a second checkout on the same machine) can point it
    somewhere harmless without depending on import order. The guard stays **enabled** — only its location
    moves — because an environment variable that switches a safety check *off* is a foot-gun, and the
    version of this that put the claim inside a library function proved the point by having tests take the
    production lock and refuse each other.
    """
    override = os.environ.get("HEAVY_JOB_LOCK")
    return Path(override) if override else DEFAULT_LOCK_PATH


def physical_gb() -> float:
    """Installed physical memory in decimal GB, from the kernel rather than from `sysctl` output text."""
    out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=True)
    return int(out.stdout.strip()) / 1e9


def usable_gb(physical: float | None = None) -> float:
    """What a job may use: physical memory less the measured reserve."""
    return (physical if physical is not None else physical_gb()) - SYSTEM_RESERVE_GB


def parse_vm_stat(text: str) -> dict[str, float]:
    """`vm_stat` output to GB, using the page size from its own header.

    Parsed from the recorded output, trailing periods and all: the first version of the test for this
    invented a tidier format than the tool prints.
    """
    header = re.search(r"page size of (\d+) bytes", text)
    if header is None:
        raise ValueError("vm_stat output has no page size in its header")
    page_bytes = int(header.group(1))
    pages: dict[str, float] = {}
    for line in text.splitlines():
        match = re.match(r"Pages ([a-z ]+?):\s+(\d+)\.?$", line.strip())
        if match is not None:
            pages[match.group(1).replace(" ", "_")] = int(match.group(2)) * page_bytes / 1e9
    return pages


def parse_swapusage(text: str) -> dict[str, float]:
    """`sysctl vm.swapusage` output to GB: `total = 2048.00M  used = 1486.88M  free = 561.12M`."""
    values: dict[str, float] = {}
    for name, amount, unit in re.findall(r"(\w+)\s*=\s*([\d.]+)([MG])", text):
        values[name] = float(amount) * (1e9 if unit == "G" else 1e6) / 1e9
    return values


def memory_state() -> dict[str, float]:
    """Physical, reclaimable, swap-used and swap-total in GB, as the machine reports them now."""
    stats = parse_vm_stat(subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout)
    swap = parse_swapusage(
        subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, check=True).stdout
    )
    reclaimable = sum(stats.get(key, 0.0) for key in ("free", "inactive", "purgeable", "speculative"))
    state = {"physical_gb": physical_gb(), "reclaimable_gb": reclaimable}
    state.update({f"swap_{key}": value for key, value in swap.items()})
    return state


def describe(state: dict[str, float]) -> str:
    parts = [
        f"physical {state['physical_gb']:.2f} GB",
        f"usable {usable_gb(state['physical_gb']):.2f} GB",
        f"reclaimable now {state.get('reclaimable_gb', 0.0):.2f} GB",
    ]
    if "swap_used" in state:
        parts.append(f"swap {state['swap_used']:.2f} of {state.get('swap_total', 0.0):.2f} GB used")
    return ", ".join(parts)


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class HeavyClaim:
    """The right to run a heavy job on this machine, released deliberately or at exit."""

    def __init__(self, path: Path, purpose: str) -> None:
        self.path = path
        self.purpose = purpose

    def release(self) -> None:
        try:
            if self.path.exists() and f"pid={os.getpid()}" in self.path.read_text():
                self.path.unlink()
        except OSError:
            pass


def require_memory_headroom(needed_gb: float | None, *, purpose: str = "this job") -> str:
    """Refuse a job that cannot fit, and warn when the machine is already under pressure.

    Returns the sentence describing the machine, so a caller can print it as part of its own output.
    """
    state = memory_state()
    usable = usable_gb(state["physical_gb"])
    if needed_gb is not None and needed_gb > usable:
        raise HeadroomError(
            f"refusing to start {purpose}: it declares {needed_gb:.2f} GB and this machine can use "
            f"{usable:.2f} GB ({state['physical_gb']:.2f} GB less a {SYSTEM_RESERVE_GB} GB reserve for "
            f"macOS).\n"
            f"  Free disk cannot make this safe: on 2026-09-16 a job of this size on this machine grew "
            f"swap to thirteen swapfiles and the hardware watchdog panicked it.\n"
            f"  Run it somewhere with more memory, or reduce what it holds at once."
        )
    message = describe(state)
    if needed_gb is not None and state.get("reclaimable_gb", 0.0) < needed_gb:
        print(
            f"heavy job warning: {purpose} declares {needed_gb:.2f} GB and only "
            f"{state.get('reclaimable_gb', 0.0):.2f} GB looks reclaimable right now ({message}); "
            f"other work is using this machine",
            file=sys.stderr,
        )
    return message


def claim_heavy(
    purpose: str,
    *,
    needed_gb: float | None = None,
    lock: Path | None = None,
    release_at_exit: bool = True,
) -> HeavyClaim:
    """Take the one heavy-job slot, refusing when another live job holds it.

    `release_at_exit` is right for a job that holds the slot while it runs, which is the programmatic path.
    An operator holding the slot by hand wants the opposite: a claim that survives the command that made it,
    or `--claim` would release the moment it returned.
    """
    path = lock_path() if lock is None else lock
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        holder = path.read_text().strip()
        match = re.search(r"pid=(\d+)", holder)
        if match is not None and process_alive(int(match.group(1))):
            raise HeadroomError(
                f"refusing to start {purpose}: another heavy job holds {path}.\n"
                f"  {holder}\n"
                f"  One heavy job at a time is not a preference: running two is what panicked this "
                f"machine. Wait for it, or release the claim deliberately if you know it is gone:\n"
                f"    python3 tools/heavy_job.py --release"
            )
        print(
            f"heavy job: taking over a stale claim in {path} ({holder or 'no holder recorded'}) — its "
            f"process is gone",
            file=sys.stderr,
        )
    path.write_text(f"purpose={purpose} pid={os.getpid()} at={time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
    claim = HeavyClaim(path, purpose)
    if release_at_exit:
        atexit.register(claim.release)
    return claim


def require_heavy_headroom(
    needed_gb: float | None, *, purpose: str = "this job", take_lock: bool = True
) -> HeavyClaim | None:
    """Everything a heavy job needs to check before it starts: disk, memory, and being alone."""
    require_headroom(purpose=purpose)
    message = require_memory_headroom(needed_gb, purpose=purpose)
    print(f"headroom: {message} (disk {free_gb():.2f} GB free)")
    if not take_lock:
        return None
    return claim_heavy(purpose, needed_gb=needed_gb)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--needs-gb", type=float, default=None, help="the job's declared peak")
    parser.add_argument("--purpose", default="this check")
    parser.add_argument("--claim", action="store_true", help="take the one heavy-job slot and hold it")
    parser.add_argument("--release", action="store_true", help="release the slot deliberately")
    parser.add_argument("--lock", type=Path, default=None)
    args = parser.parse_args(argv)

    path = lock_path() if args.lock is None else args.lock
    if args.release:
        if not path.exists():
            print(f"no claim to release at {path}")
            return 0
        print(f"releasing {path}:\n  {path.read_text().strip()}")
        path.unlink()
        return 0

    if args.claim:
        try:
            claim = claim_heavy(
                args.purpose, needed_gb=args.needs_gb, lock=args.lock, release_at_exit=False
            )
        except HeadroomError as error:
            print(error, file=sys.stderr)
            return 1
        print(f"claimed for {claim.purpose}; release with --release")
        return 0

    try:
        message = require_memory_headroom(args.needs_gb, purpose=args.purpose)
    except HeadroomError as error:
        print(error, file=sys.stderr)
        return 1
    print(f"memory: {message}")
    if path.exists():
        print(f"heavy job: {path.read_text().strip()}")
    else:
        print("heavy job: none claimed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
