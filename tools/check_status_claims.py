#!/usr/bin/env python3
"""Check the documentation's claims against what is actually true.

Every rule in this repository ends with "the tracker says so with the evidence that closed it", and the
numbers in the README, `AGENTS.md` and the wiki are the visible half of that. They drift — a test count
quoted after a test was added, `2 skipped` after the skips were removed, a decision described as open a
round after it was decided. I have hand-fixed that class of defect more than a dozen times in this project,
which is exactly the argument for a gate rather than another careful read.

    python3 tools/check_status_claims.py --swift-tests 184 --swift-skipped 0 --python-tests 200

What it checks, offline:

* **test counts** — every `N tests, S skipped` and `N Swift, M Python` claim in the tracked documents
  matches the numbers it was given;
* **decision ids** — every `Dnn` referenced anywhere in the repository's Markdown has a heading in a
  `docs/*-decisions.md`, so a citation cannot point at a decision that was never written;
* **release state** — if a document says there are no tags, `git tag` agrees.

It reports how many claims it *found*, and refuses to pass on zero: a regex that stopped matching would
otherwise turn this gate into a decoration. A check that cannot run must never look like a check that
passed, and neither must a check that is checking nothing.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Documents whose numbers are claims about the repository **now**. The wiki's News page is deliberately
# not here: it is a dated log, so "105 tests, 2 skipped" in an entry from an earlier milestone is a true
# statement about that day, not a stale one — and this gate's first run failed on exactly that, which is
# the difference between a claim and a record.
CLAIM_DOCUMENTS = [
    "README.md",
    "AGENTS.md",
    ".wiki/Project-Tracker.md",
    ".wiki/Roadmap.md",
    ".wiki/Testbed.md",
    ".wiki/Architecture.md",
]
# Documents whose *citations* matter, log or not: a decision cited anywhere must exist.
REFERENCE_DOCUMENTS = [".wiki/News.md"]
# The two that must be present; the wiki is a separate checkout and is gitignored here, so a missing wiki
# page is reported NOT CHECKED rather than passing — and rather than failing a run that cannot see it.
REQUIRED = ["README.md", "AGENTS.md"]
# Discovered rather than listed. The list was explicit until `D36` — the decision this gate's own
# repository-decision record introduced — turned up as "cited but undefined" because the file that defines
# it had not been added here. A gate whose configuration can go stale is a gate that will.
DECISION_GLOB = "docs/*-decisions.md"

# `**184 tests, 0 skipped, 0 failures**`, `at **184 tests, 2 skipped`, `184 tests, 0 failures`.
SWIFT_TESTS = re.compile(r"\*\*(\d+) tests?, (\d+) skipped")
# `**184 Swift, 200 Python**` — the tracker's Tests row.
PAIR = re.compile(r"\*\*(\d+) Swift, (\d+) Python\*\*")
# `with **200** standard-library Python tests`.
PYTHON_TESTS = re.compile(r"\*\*(\d+)\*\* standard-library Python tests")
# The same claims in the form a reader *copies*: `--swift-tests 184 --swift-skipped 0 --python-tests 219`.
# These matter more than the prose, not less — a stale example is a command that reports the wrong thing.
COMMAND_CLAIMS = (
    ("--swift-tests", "swift_tests"),
    ("--swift-skipped", "swift_skipped"),
    ("--python-tests", "python_tests"),
)
# A decision citation: `D17`, `D34`, but not `3D4`.
DECISION = re.compile(r"(?<![A-Za-z0-9])D(\d{1,3})(?![0-9])")
# The heading that defines one: `## D34 — ...`.
DEFINITION = re.compile(r"^##\s+D(\d+)\b", re.MULTILINE)
NO_TAGS = re.compile(r"\*\*no releases and no tags\*\*")


def lines_with(text: str, pattern: re.Pattern[str]):
    for number, line in enumerate(text.splitlines(), start=1):
        for match in pattern.finditer(line):
            yield number, match


def check(
    root: Path, swift_tests: int, swift_skipped: int, python_tests: int
) -> tuple[list[str], int, list[str]]:
    problems: list[str] = []
    checked = 0

    not_checked: list[str] = []
    for name in CLAIM_DOCUMENTS:
        path = root / name
        if not path.exists():
            (problems if name in REQUIRED else not_checked).append(name)
            if name in REQUIRED:
                problems[-1] = f"{name}: the document this gate checks is missing"
            continue
        text = path.read_text()
        for number, match in lines_with(text, SWIFT_TESTS):
            if swift_tests is None or swift_skipped is None:
                continue
            checked += 1
            tests, skipped = int(match.group(1)), int(match.group(2))
            if tests != swift_tests or skipped != swift_skipped:
                problems.append(
                    f"{name}:{number}: claims {tests} tests with {skipped} skipped; the suite reports "
                    f"{swift_tests} with {swift_skipped}"
                )
        for number, match in lines_with(text, PAIR):
            if swift_tests is None or python_tests is None:
                continue
            checked += 1
            swift, python = int(match.group(1)), int(match.group(2))
            if swift != swift_tests or python != python_tests:
                problems.append(
                    f"{name}:{number}: claims {swift} Swift and {python} Python tests; the suites report "
                    f"{swift_tests} and {python_tests}"
                )
        for number, match in lines_with(text, PYTHON_TESTS):
            if python_tests is None:
                continue
            checked += 1
            claimed = int(match.group(1))
            if claimed != python_tests:
                problems.append(
                    f"{name}:{number}: claims {claimed} Python tests; the suite reports {python_tests}"
                )

    decision_files = sorted(str(path) for path in root.glob(DECISION_GLOB))
    observed = {"swift_tests": swift_tests, "swift_skipped": swift_skipped, "python_tests": python_tests}
    for name in CLAIM_DOCUMENTS:
        path = root / name
        if not path.exists():
            continue
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            for flag, key in COMMAND_CLAIMS:
                for match in re.finditer(rf"{re.escape(flag)}\s+(\d+)", line):
                    if observed[key] is None:
                        continue
                    checked += 1
                    claimed = int(match.group(1))
                    if claimed != observed[key]:
                        problems.append(
                            f"{name}:{number}: the example says {flag} {claimed}; the suite reports "
                            f"{observed[key]}"
                        )

    defined: set[int] = set()
    for name in decision_files:
        path = root / name
        if not path.exists():
            problems.append(f"{name}: the decision record this gate reads is missing")
            continue
        defined.update(int(match.group(1)) for match in DEFINITION.finditer(path.read_text()))
    if not defined:
        problems.append("no decision headings were found at all, so citations cannot be checked")

    cited: dict[int, list[str]] = {}
    for name in CLAIM_DOCUMENTS + REFERENCE_DOCUMENTS + decision_files:
        path = root / name
        if not path.exists():
            continue
        for _, match in lines_with(path.read_text(), DECISION):
            cited.setdefault(int(match.group(1)), []).append(name)
    checked += len(cited)
    for identifier in sorted(cited):
        if identifier not in defined:
            problems.append(
                f"D{identifier} is cited in {', '.join(sorted(set(cited[identifier])))} and has no "
                f"definition in any of {', '.join(decision_files)}"
            )

    for name in CLAIM_DOCUMENTS:
        path = root / name
        if not path.exists() or not NO_TAGS.search(path.read_text()):
            continue
        checked += 1
        tags = subprocess.run(
            ["git", "tag"], cwd=root, capture_output=True, text=True, check=False
        ).stdout.strip()
        if tags:
            problems.append(f"{name}: says there are no tags, and `git tag` lists {tags!r}")

    return problems, checked, not_checked


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--swift-tests", type=int, default=None)
    parser.add_argument("--swift-skipped", type=int, default=None)
    parser.add_argument("--python-tests", type=int, default=None)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)

    problems, checked, not_checked = check(
        args.root, args.swift_tests, args.swift_skipped, args.python_tests
    )
    # Reported, never silent: a page that is absent is NOT CHECKED, which is a different thing from a
    # page that passed — the same distinction the CI job makes about the wiki's tables.
    for name in not_checked:
        print(f"NOT CHECKED: {name} is not present, so its claims were not checked", file=sys.stderr)
    if args.swift_tests is None and args.python_tests is None:
        print("STATUS CLAIMS FAILED: no test counts were supplied, so nothing could be compared", file=sys.stderr)
        return 1
    if checked == 0:
        print(
            "STATUS CLAIMS FAILED: no claims were found to check, which means this gate is checking "
            "nothing",
            file=sys.stderr,
        )
        return 1
    if problems:
        print(f"STATUS CLAIMS FAILED: {len(problems)} of {checked} claim(s) disagree", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    observed = []
    if args.swift_tests is not None and args.swift_skipped is not None:
        observed.append(f"{args.swift_tests} Swift tests with {args.swift_skipped} skipped")
    if args.python_tests is not None:
        observed.append(f"{args.python_tests} Python tests")
    print(f"STATUS CLAIMS OK: {checked} claim(s) checked against " + " and ".join(observed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
