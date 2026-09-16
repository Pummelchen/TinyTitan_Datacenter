#!/usr/bin/env python3
"""Check the provenance position this repository states — as far as it can be checked offline.

`DC-013`'s review concluded that no third-party source is included, so MIT is the whole story and no
Apache-2.0 `NOTICE` transfers. That conclusion was reached by comparing this repository against the sister
project by hand, which is not a thing a reviewer can re-run. What *can* be re-run offline is the state that
conclusion depends on:

* `THIRD_PARTY_NOTICES.md` exists — the file the review's acceptance asks for;
* it still names the relationships it describes, so it cannot quietly decay into a stub;
* **no source file has acquired a third-party copyright header**, which is the earliest visible symptom of
  code being copied in without its obligations.

    python3 tools/check_provenance.py

It reports how many files it inspected and refuses to pass on zero — a check that examined nothing must not
look like a check that passed.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTICES = "THIRD_PARTY_NOTICES.md"

# The holder of this repository's own copyright, and the phrase the notices file may use for the project.
OWN_HOLDERS = ("André Borchert", "TinyTitan Datacenter")
COPYRIGHT = re.compile(r"copyright\s*(?:\(c\)|©)?\s*\d{4}[^\n]*", re.IGNORECASE)

SOURCE_DIRS = ("sources", "tools", "tests", "plugins")
SOURCE_SUFFIXES = (".swift", ".py", ".c", ".h", ".m", ".metal", ".sh")

# Phrases the notices file has to keep: the relationship it records, and the obligations it rules out.
REQUIRED_PHRASES = (
    "No third-party source is included",
    "TinyTitan",
    "Apache",
    "NOTICE",
)


def source_files(root: Path):
    for directory in SOURCE_DIRS:
        base = root / directory
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_file() and path.suffix in SOURCE_SUFFIXES:
                yield path


def check(root: Path) -> tuple[list[str], int]:
    problems: list[str] = []
    inspected = 0

    notices = root / NOTICES
    if not notices.exists():
        problems.append(f"{NOTICES} is missing: the provenance review has nowhere to be recorded")
        text = ""
    else:
        text = notices.read_text()
        inspected += 1
        for phrase in REQUIRED_PHRASES:
            if phrase not in text:
                problems.append(f"{NOTICES} no longer mentions {phrase!r}")

    for path in source_files(root):
        inspected += 1
        for line in path.read_text(errors="ignore").splitlines():
            match = COPYRIGHT.search(line)
            if not match or not re.search(r"\d{4}", match.group(0)):
                continue
            if any(holder in line for holder in OWN_HOLDERS):
                continue
            # A licence *reference* is fine; an attribution line is the thing to notice.
            if "SPDX-License-Identifier" in line:
                continue
            problems.append(
                f"{path.relative_to(root)}: carries a copyright line that is not this repository's — "
                f"{line.strip()[:110]!r}. Copied code brings its licence and NOTICE material with it "
                f"({NOTICES} records what would then be required)"
            )
    return problems, inspected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)

    problems, inspected = check(args.root)
    if inspected == 0:
        print(
            "PROVENANCE FAILED: nothing was inspected, so this check is checking nothing", file=sys.stderr
        )
        return 1
    if problems:
        print(f"PROVENANCE FAILED: {len(problems)} problem(s)", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    print(f"PROVENANCE OK: {inspected} file(s) inspected, no third-party attribution to account for")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
