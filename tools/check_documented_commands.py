#!/usr/bin/env python3
"""Every command the documentation gives a reader must be runnable against the tools in this repository.

The repository's own trap says it: **check the form a reader copies, not only the prose.** That check has been
extended three times — to the flags in the claims gate's own examples (`D67`), to the paths named in the claim
documents (`D67`), and to the invocations the *gates* make (`D76`). What was left is the surface a reader meets
first and copies most often: the commands in `docs/`, `README.md`, `AGENTS.md`, `RELEASE.md` and the wiki. A
documented command that names a flag its tool no longer accepts is a command that cannot be run, and nothing
checked one — which is how it looked when this was written: the audit found all thirty-two documented flags
valid, and no gate was watching to keep them that way.

This does **not** run the commands. Most of them need a checkpoint, a cluster or a GPU. It parses them out of the
markdown and asks each tool what it accepts, which is the part that goes stale when a flag is renamed, and it
reports a tool it could not ask as *not checked* rather than as a pass.

Standard library only, like the rest of the gates.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# The documents a reader copies from. The wiki is a separate checkout and is gitignored here, so a run without
# it checks the rest and says the wiki was not checked.
DOCUMENTS = ["README.md", "AGENTS.md", "RELEASE.md"]
DOCUMENT_GLOBS = ["docs/*.md", ".wiki/*.md"]

# `python3 tools/thing.py …`, or `.venv/bin/python tools/thing.py …`. A command the documentation gives a reader
# to copy is written this way; prose that names a tool without invoking it is not a command and is not checked.
COMMAND = re.compile(r"(?:\S*python3?|\.venv/bin/python)\s+(tools/[A-Za-z_][A-Za-z0-9_]*\.py)([^\n`|]*)")
FLAG = re.compile(r"(?<!\w)(--[a-z][a-z0-9-]*)")


def documents(root: Path) -> tuple[list[Path], list[str]]:
    """The documents to read, and the ones that are not present on this checkout."""
    found, missing = [], []
    for name in DOCUMENTS:
        path = root / name
        (found if path.exists() else missing).append(path)
    for pattern in DOCUMENT_GLOBS:
        matching = sorted(root.glob(pattern))
        if not matching and pattern.startswith(".wiki"):
            missing.append(root / ".wiki")
        found += matching
    return found, [str(path) for path in missing]


def commands_in(text: str):
    """`(script, flags)` for every command the text gives a reader, joining backslash continuations."""
    joined = text.replace("\\\n", " ")
    for match in COMMAND.finditer(joined):
        yield match.group(1), sorted(set(FLAG.findall(match.group(2))))


def accepted_flags(script: str, cache: dict[str, str | None]) -> str | None:
    """The tool's `--help`, or `None` when it cannot be asked — which is reported rather than assumed."""
    if script not in cache:
        done = subprocess.run(
            [sys.executable, str(ROOT / script), "--help"], cwd=ROOT, capture_output=True, text=True
        )
        cache[script] = (done.stdout + done.stderr) if done.returncode == 0 else None
    return cache[script]


def check(
    root: Path = ROOT, only: list[Path] | None = None
) -> tuple[list[str], int, list[str]]:
    """`(problems, checked, not_checked)`. `only` limits the scan to given documents, which is how it is tested.

    The tools are always resolved against `root` rather than against the document's own directory, because a
    command in a document names a path relative to the repository root.
    """
    problems: list[str] = []
    checked = 0
    cache: dict[str, str | None] = {}
    not_checked: list[str] = []

    if only is None:
        found, missing = documents(root)
        not_checked += [f"{name} (not present on this checkout)" for name in missing]
    else:
        found = list(only)

    for path in found:
        name = path.name if path.parent == root else f"{path.parent.name}/{path.name}"
        for script, flags in commands_in(path.read_text()):
            if not (root / script).exists():
                problems.append(f"{name}: names {script}, which does not exist")
                continue
            if not flags:
                continue
            help_text = accepted_flags(script, cache)
            if help_text is None:
                not_checked.append(f"{script} (its --help could not be run, so its flags were not checked)")
                continue
            for flag in flags:
                checked += 1
                if flag not in help_text:
                    problems.append(f"{name}: {script} does not accept {flag}")
    return problems, checked, not_checked


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--verbose", action="store_true", help="name every document that was read")
    args = parser.parse_args(argv)

    problems, checked, not_checked = check()
    if args.verbose:
        found, _ = documents(ROOT)
        for path in found:
            print(f"  read {path.relative_to(ROOT)}")
    for name in not_checked:
        print(f"NOT CHECKED: {name}", file=sys.stderr)
    if problems:
        for problem in problems:
            print(f"DOCUMENTED COMMAND: {problem}", file=sys.stderr)
        print(f"DOCUMENTED COMMANDS FAILED: {len(problems)} problem(s)", file=sys.stderr)
        return 1
    print(f"DOCUMENTED COMMANDS OK: {checked} flag(s) checked against the tools that take them")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
