#!/usr/bin/env python3
"""Check the documentation's claims against what is actually true.

Every rule in this repository ends with "the tracker says so with the evidence that closed it", and the
numbers in the README, `AGENTS.md` and the wiki are the visible half of that. They drift — a test count
quoted after a test was added, `2 skipped` after the skips were removed, a decision described as open a
round after it was decided. I have hand-fixed that class of defect more than a dozen times in this project,
which is exactly the argument for a gate rather than another careful read.

    python3 tools/check_status_claims.py --swift-tests <n> --swift-skipped <n> --python-tests <n>

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
# Claim documents are **discovered**, not listed. They were a list until `.wiki/Home.md` sat outside it
# carrying "its gate is still open" and a test count from the era it was written in — the trap this module
# already names for the decision records, biting a second time in the same file. A page is a claim document
# if it is the README, `AGENTS.md`, or **any wiki page that is not the log**: `News.md` records what was true
# when each entry was written, so a stale number there is history rather than an error, and GitHub's own
# `_`-prefixed pages are furniture.
FIXED_CLAIM_DOCUMENTS = ["README.md", "AGENTS.md"]
LOG_PAGE = "News.md"


def claim_documents(root: Path) -> list[str]:
    """The documents whose numbers are claims about *today*."""
    discovered = sorted(
        f".wiki/{path.name}"
        for path in (root / ".wiki").glob("*.md")
        if path.name != LOG_PAGE and not path.name.startswith("_")
    )
    return FIXED_CLAIM_DOCUMENTS + discovered
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
# The same claims in the forms prose actually uses. The two patterns above are the house style; a document
# that phrased its number differently was not checked *at all*, which is how `.wiki/Home.md` carried "105 Swift
# tests" and "174 standard-library Python tests" for ninety rounds while this gate reported success. These are
# deliberately narrow -- a bolded count immediately before "tests", and an unbolded count before
# "standard-library Python tests" -- because widening further would start reading historical asides as claims.
SWIFT_TESTS_PROSE = re.compile(r"\*\*(\d+)\s+(?:Swift\s+)?tests?\b")
PYTHON_TESTS_PROSE = re.compile(r"(\d+)\s+standard-library Python tests?\b")
# A decision citation: `D17`, `D34`, but not `3D4`.
DECISION = re.compile(r"(?<![A-Za-z0-9])D(\d{1,3})(?![0-9])")
# The heading that defines one: `## D34 — ...`.
DEFINITION = re.compile(r"^##\s+D(\d+)\b", re.MULTILINE)
NO_TAGS = re.compile(r"\*\*no releases and no tags\*\*")
# The form a reader copies includes **paths**: `tools/run_m1_gate.py`, `docs/m1-gate.md`. The link gate checks
# `[links](…)` and the claims above check the numbers, but a path in a code block or a sentence was checked by
# nothing — the same gap the command claims closed for flags, in the other direction. A documented path that
# does not exist is a command that cannot be run.
PATH = re.compile(r"\b((?:tools|docs|sources|tests)/[A-Za-z0-9_./-]+\.[A-Za-z0-9]{1,5})\b")
# A template is not a claim about a file. The rule is a property, not a list: an angle-bracketed or
# mixed-case name cannot be a path in this repository, because every one of them is lower case.
PLACEHOLDER = re.compile(r"[<>]")


# The invariants audit: every invariant must have a section, a status from this vocabulary, and at least
# one named artifact as evidence. A status is a claim about the project, so it gets the same treatment as a
# test count — the difference being that this one cannot be computed, only checked for shape.
INVARIANTS = ("I1", "I2", "I3", "I4", "I5", "I6")
AUDIT_PATH = "docs/invariants-audit.md"
INVARIANT_STATUSES = ("`verified`", "`partly`", "`not applicable yet`", "`not yet`")
EVIDENCE = re.compile(r"`[^`]*(?:docs/|tools/|tests/)[^`]*`")


def check_invariants(root: Path) -> tuple[list[str], int]:
    """Every invariant has a section, a status, and evidence that can be looked up."""
    path = root / AUDIT_PATH
    if not path.exists():
        return [f"{AUDIT_PATH} is missing, so the invariants have no audit"], 0
    text = path.read_text()
    problems: list[str] = []
    checked = 0
    headings = list(re.finditer(r"^##\s+(I\d+)\b.*$", text, re.MULTILINE))
    for invariant in INVARIANTS:
        checked += 1
        heading = next((match for match in headings if match.group(1) == invariant), None)
        if heading is None:
            problems.append(f"{AUDIT_PATH}: {invariant} has no section")
            continue
        end = next((match.start() for match in headings if match.start() > heading.start()), len(text))
        section = text[heading.start():end]
        if not any(status in heading.group(0) for status in INVARIANT_STATUSES):
            problems.append(
                f"{AUDIT_PATH}: {invariant}'s heading states no status; expected one of "
                + ", ".join(INVARIANT_STATUSES)
            )
        if not EVIDENCE.search(section):
            problems.append(
                f"{AUDIT_PATH}: {invariant}'s section names no file under docs/, tools/ or tests/ as evidence"
            )
    return problems, checked


def lines_with(text: str, pattern: re.Pattern[str]):
    for number, line in enumerate(text.splitlines(), start=1):
        for match in pattern.finditer(line):
            yield number, match


def collapsed_matches(text: str, pattern: re.Pattern[str]):
    """Matches against the text with whitespace collapsed, reporting the line the match starts on.

    A claim wrapped across two lines is still a claim. `.wiki/Home.md` wrote "**105 Swift\ntests pass**", and a
    reader that works line by line cannot see it — which is how a count from the era the page was written in
    survived a gate that reported success. The line number is kept because the message has to point at a place.
    """
    # A blockquote's leading `>` is markup, not prose, and `.wiki/Home.md` wrapped its claim inside one:
    # "**105 Swift\n> tests pass**". Collapsing whitespace alone leaves the marker sitting between two words
    # of the same sentence, so it is removed first. Newlines are untouched, so the line numbers stay true.
    text = re.sub(r"(?m)^[ \t]*>[ \t]?", " ", text)
    pieces: list[str] = []
    origin: list[int] = []
    line = 1
    index = 0
    while index < len(text):
        if text[index].isspace():
            start = index
            while index < len(text) and text[index].isspace():
                index += 1
            pieces.append(" ")
            origin.append(line)
            line += text[start:index].count("\n")
        else:
            pieces.append(text[index])
            origin.append(line)
            index += 1
    for match in pattern.finditer("".join(pieces)):
        yield origin[match.start()], match


def check(
    root: Path, swift_tests: int, swift_skipped: int, python_tests: int
) -> tuple[list[str], int, list[str]]:
    problems: list[str] = []
    checked = 0

    claims = claim_documents(root)
    not_checked: list[str] = []
    if not (root / ".wiki").is_dir():
        # The wiki is a separate checkout and is gitignored here, so a run that cannot see it says so rather
        # than passing. Discovery cannot name the pages that are missing when the whole directory is gone --
        # it can only check what is there -- so it names the wiki itself.
        not_checked.append(".wiki (the wiki is a separate checkout and is not present here)")
    for name in claims:
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
        for number, match in collapsed_matches(text, SWIFT_TESTS_PROSE):
            if swift_tests is None:
                continue
            checked += 1
            claimed = int(match.group(1))
            if claimed != swift_tests:
                problems.append(
                    f"{name}:{number}: claims {claimed} Swift test(s) in prose; the suite reports {swift_tests}"
                )
        for number, match in collapsed_matches(text, PYTHON_TESTS_PROSE):
            if python_tests is None:
                continue
            checked += 1
            claimed = int(match.group(1))
            if claimed != python_tests:
                problems.append(
                    f"{name}:{number}: claims {claimed} Python test(s) in prose; the suite reports {python_tests}"
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
    for name in claims:
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

    invariant_problems, invariant_checked = check_invariants(root)
    problems.extend(invariant_problems)
    checked += invariant_checked

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
    for name in claims + REFERENCE_DOCUMENTS + decision_files:
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

    for name in claims:
        path = root / name
        if not path.exists():
            continue
        for number, match in lines_with(path.read_text(), PATH):
            documented = match.group(1)
            if PLACEHOLDER.search(documented) or documented != documented.lower():
                continue  # `docs/release-notes-vX.Y.md` is a template, not a claim about a file
            checked += 1
            if not (root / documented).exists():
                problems.append(f"{name}:{number}: names {documented}, which does not exist")

    for name in claims:
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
