"""Compare a run's `metrics.json` against the recorded baselines (`tools/baselines.json`).

`DC-084`: the sister project keeps golden baselines and re-checks them, and this repository's numbers lived
only in prose — a figure in a document that nothing re-measures is a claim with a date on it, not a
baseline. This turns them into data and checks them.

Two kinds, and the difference is the point:

* a **count** is deterministic given its input, so it is asserted **exactly**. Expert requests, payload bytes
  read and cache hits do not depend on how busy the farm is, which is why they can be re-checked at any
  time — on a shared farm, in the middle of other work;
* an **observed** value depends on the machine and the moment (seconds, memory). It is **reported with its
  delta** and asserted only with `--assert-observed`, which belongs on a quiet farm. Yesterday's throughput
  is not a property of the code.

A baseline that cannot be found in the metrics given is reported **NOT CHECKED** — never assumed and never
silently skipped — together with the command that produces it.

Standard library only, like every other gate here.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASELINES = ROOT / "baselines.json"


def load_baselines(path: Path | None = None) -> tuple[list[dict], list[dict]]:
    """`(counts, observations)`, with the file's own shape validated rather than trusted."""
    document = json.loads((path or BASELINES).read_text())
    counts = list(document.get("baselines") or [])
    observations = list(document.get("observations") or [])
    for entry in counts:
        if entry.get("kind") != "count":
            raise ValueError(f"{entry.get('key')!r} is in `baselines` but is not a count")
        if "key" not in entry or "value" not in entry or "source" not in entry:
            raise ValueError(f"a count baseline is missing key/value/source: {entry!r}")
    for entry in observations:
        if entry.get("kind") != "observed":
            raise ValueError(f"{entry.get('key')!r} is in `observations` but is not observed")
    return counts, observations


def applies(entry: dict, metrics: dict) -> bool:
    """Whether a baseline applies to this run at all.

    `applies_to` says in prose what a figure was measured on, and prose cannot filter: the first version of
    this compared the two-node fixture's `exchange_reduces` against a **single-node** real run and reported
    the correct 0 as a failure. `requires` is the same statement in a form that can be checked, and a
    baseline whose requirements are not met is NOT CHECKED rather than compared.
    """
    for key, expected in (entry.get("requires") or {}).items():
        if key not in metrics or metrics[key] != expected:
            return False
    return True


def derive(key: str, metrics: dict) -> float | None:
    """Values a metrics file *implies* rather than states.

    The M1 throughput baseline is 9.25 s/token; the generation CLI records per-step seconds, so the rate is
    derivable from them and does not need its own key. Anything not derivable here is left to the caller to
    report as NOT CHECKED.
    """
    if key != "steps_per_second":
        return None

    # The generation CLI records `steps`, `step_seconds_total`, a `step_seconds` **list** and
    # `step_seconds_slowest`. The first version of this matched every key starting with `step_seconds` and
    # fed the *list* to `fmean` — and the unit tests passed, because they had invented the scalar
    # `step_seconds_0` shape rather than reading a real metrics file. The tests now use the recorded shape.
    def number(value: object) -> float | None:
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None

    steps, total = number(metrics.get("steps")), number(metrics.get("step_seconds_total"))
    if steps is not None and total is not None and steps > 0 and total > 0:
        return steps / total

    seconds: list[float] = []
    listed = metrics.get("step_seconds")
    if isinstance(listed, list):
        seconds = [value for value in (number(item) for item in listed) if value is not None]
    elif (single := number(listed)) is not None:
        seconds = [single]
    for name, value in metrics.items():
        if name.startswith("step_seconds_") and (parsed := number(value)) is not None:
            seconds.append(parsed)
    if not seconds:
        return None
    mean = statistics.fmean(seconds)
    return None if mean <= 0 else 1.0 / mean


def compare(
    counts: list[dict],
    observations: list[dict],
    metrics: dict,
    *,
    assert_observed: bool = False,
    tolerance: float = 0.25,
) -> tuple[list[str], int, list[str], list[str]]:
    """`(problems, checked, not_checked, reported)` for one metrics file."""
    problems: list[str] = []
    not_checked: list[str] = []
    reported: list[str] = []
    checked = 0

    for entry in counts:
        key = entry["key"]
        if not applies(entry, metrics):
            not_checked.append(f"{key} (does not apply to this run: {entry.get('requires')})")
            continue
        if key not in metrics:
            not_checked.append(f"{key} (a count, from {entry['source']}; {entry['produced_by']})")
            continue
        checked += 1
        observed = metrics[key]
        if isinstance(entry["value"], float):
            same = abs(float(observed) - entry["value"]) <= 1e-9
        else:
            same = observed == entry["value"]
        if same:
            reported.append(f"{key} = {observed} (baseline {entry['value']}, {entry['source']})")
        else:
            problems.append(
                f"{key} is {observed}, but {entry['source']} records {entry['value']} for {entry['applies_to']}"
            )

    for entry in observations:
        key = entry["key"]
        if not applies(entry, metrics):
            not_checked.append(f"{key} (does not apply to this run: {entry.get('requires')})")
            continue
        observed = metrics.get(key, derive(key, metrics))
        if observed is None:
            not_checked.append(
                f"{key} (an observation, from {entry['source']}; produced by {entry['produced_by']})"
            )
            continue
        checked += 1
        baseline = float(entry["value"])
        delta = 0.0 if baseline == 0 else (observed - baseline) / baseline
        line = (
            f"{key} = {observed:.4g} against a recorded {baseline:.4g} "
            f"({delta * 100:+.1f}%, {entry['source']})"
        )
        if assert_observed:
            if abs(delta) > tolerance:
                problems.append(
                    f"{line} — outside the {tolerance * 100:.0f}% band, and --assert-observed was asked for"
                )
            else:
                reported.append(line)
        else:
            reported.append(line + " — reported, not asserted")
    return problems, checked, not_checked, reported


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--metrics", action="append", default=[], help="a metrics.json to check (repeatable)")
    parser.add_argument(
        "--assert-observed",
        action="store_true",
        help="assert timing and memory baselines too; this belongs on a quiet farm",
    )
    parser.add_argument("--tolerance", type=float, default=0.25, help="the band for observations (default 0.25)")
    parser.add_argument("--baselines", type=Path, default=None)
    args = parser.parse_args(argv)

    if not args.metrics:
        parser.error("give at least one --metrics file; a baseline check with nothing to check is not a check")

    counts, observations = load_baselines(args.baselines)
    problems: list[str] = []
    checked = 0
    not_checked: list[str] = []

    for name in args.metrics:
        path = Path(name)
        if not path.exists():
            problems.append(f"{name} does not exist")
            continue
        metrics = json.loads(path.read_text())
        file_problems, file_checked, file_missing, reported = compare(
            counts, observations, metrics, assert_observed=args.assert_observed, tolerance=args.tolerance
        )
        print(f"{name}: {file_checked} baseline(s) checked")
        for line in reported:
            print(f"  {line}")
        problems.extend(file_problems)
        checked += file_checked
        not_checked.extend(file_missing)

    for line in not_checked:
        print(f"  NOT CHECKED: {line}")

    if not checked:
        problems.append("no baseline was found in any metrics file, so nothing was checked")

    for problem in problems:
        print(f"BASELINE: {problem}", file=sys.stderr)
    if problems:
        print(f"BASELINE CHECK FAILED: {len(problems)} problem(s)", file=sys.stderr)
        return 1
    print(f"BASELINE CHECK OK: {checked} baseline(s) against {len(counts) + len(observations)} recorded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
