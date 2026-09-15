#!/usr/bin/env python3
"""Check that every Markdown table has a consistent number of columns.

The link gate next to this one catches a broken link. Nothing caught a **broken table**, and the
tracker has now had two task rows land in the three-column story table below it — a five-column row
in a three-column table renders as garbage, and both times it was found by a shell one-liner run by
hand rather than by the repository. A defect class that has happened twice belongs in the gate.

Rules, kept deliberately narrow so the check cannot be wrong about anything that matters:

- a **table row** is a line whose first non-space character is `|`;
- a table is a **run** of consecutive table rows, and every row in a run must have the same number
  of `|` characters;
- text inside inline code spans and escaped pipes does not count, because Markdown does not count it.

Standard library only, like everything under `tools/` that gates the repository.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

#: Inline code spans, whose contents are literal and may legitimately contain pipes.
CODE_SPAN = re.compile(r"`[^`]*`")


def effective_pipes(line: str) -> int:
    """The number of `|` characters Markdown will treat as column separators."""
    without_code = CODE_SPAN.sub(lambda match: "`" + " " * (len(match.group(0)) - 2) + "`", line)
    return without_code.replace("\\|", "").count("|")


def columns(line: str) -> int:
    """Columns, not pipes: a row is `| a | b |`, which is three pipes and **two** columns.

    The first version reported the pipe count while calling it a column count, which its own tests
    caught — a message that misleads is worse than no message, and this session has produced more
    than one of those.
    """
    return max(effective_pipes(line) - 1, 0)


def check(path: Path) -> list[str]:
    """A message per table that does not line up. Empty means the file is fine."""
    problems: list[str] = []
    run_start = 0
    expected = 0
    previous = 0
    lines = path.read_text(encoding="utf-8").splitlines()
    for number, line in enumerate(lines, start=1):
        if line.lstrip().startswith("|"):
            count = columns(line)
            if previous + 1 != number or expected == 0:
                # A new run may begin anywhere; only rows *inside* a run must agree.
                if expected == 0:
                    run_start, expected = number, count
            if count != expected:
                problems.append(
                    f"{path}:{number}: {count} columns, but the table starting at line {run_start} "
                    f"has {expected}"
                )
            previous = number
        else:
            run_start, expected, previous = 0, 0, 0
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, default=None)
    args = parser.parse_args(argv)
    paths = args.paths or sorted(
        [*Path(".").glob("*.md"), *Path("docs").glob("*.md"), *Path(".wiki").glob("*.md")]
    )
    problems: list[str] = []
    for path in paths:
        if path.exists():
            problems.extend(check(path))
    for problem in problems:
        print(problem, file=sys.stderr)
    print(f"Checked {len(paths)} file(s); {len(problems)} misaligned row(s).")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
