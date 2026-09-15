#!/usr/bin/env python3
"""Compare two reference modules at the AST level, after renaming one family onto the other.

The question this answers is the one a shape check cannot: **does `qwen3_5_moe`'s attention
and Gated DeltaNet do the same arithmetic as `qwen3_5`'s**, or does it merely have the same
names and shapes? Reading 2100 lines twice is not a method; diffing the two modules'
entities after normalising the family prefix is.

What it does: parse both files, take every top-level function and class, rename the second
family's identifiers onto the first's, and compare the unparsed bodies line by line. Two
entities with identical bodies after renaming are the *same arithmetic*; anything in
`--allow-differ` is a difference that has been looked at and is expected.

    .venv/bin/python tools/compare_reference_modules.py \\
        .venv/lib/python3.14/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py \\
        .venv/lib/python3.14/site-packages/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py \\
        --rename Qwen3_5Moe=Qwen3_5

Needs the venv (it imports nothing outside the standard library, but the paths are the
installed transformers), so it is not part of the stdlib-only CI gate. It exits non-zero when
an entity differs without being listed, which is what makes it re-runnable after a
transformers upgrade rather than a one-off observation.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import sys
from pathlib import Path


def entities(path: Path) -> dict[str, str]:
    """Every top-level function and class, unparsed back to source."""
    return {
        node.name: ast.unparse(node)
        for node in ast.parse(path.read_text()).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def normalise(text: str, renames: list[tuple[str, str]]) -> str:
    for old, new in renames:
        text = text.replace(old, new)
    return text


def body(entity: str) -> list[str]:
    return [line.rstrip() for line in entity.splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path, help="the module whose names win")
    parser.add_argument("right", type=Path)
    parser.add_argument(
        "--rename", action="append", default=[], metavar="OLD=NEW",
        help="rename a family prefix in the right-hand module; repeatable",
    )
    parser.add_argument(
        "--allow-differ", default="", metavar="A,B",
        help="entities whose difference has been reviewed and is expected",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    renames = []
    for pair in args.rename:
        if "=" not in pair:
            print(f"--rename wants OLD=NEW, got {pair!r}", file=sys.stderr)
            return 2
        old, new = pair.split("=", 1)
        renames.append((old, new))
    allowed = {name.strip() for name in args.allow_differ.split(",") if name.strip()}

    left = {normalise(k, renames): normalise(v, renames) for k, v in entities(args.left).items()}
    right = {normalise(k, renames): normalise(v, renames) for k, v in entities(args.right).items()}
    shared = sorted(set(left) & set(right))

    identical, differing, missing = [], [], []
    for name in shared:
        (identical if body(left[name]) == body(right[name]) else differing).append(name)
    missing = sorted(set(left) - set(right)) + sorted(set(right) - set(left))

    if not args.quiet:
        print(f"entities: {len(left)} and {len(right)}, shared {len(shared)}")
        print(f"\nIDENTICAL after renaming ({len(identical)}) — the same arithmetic:")
        for name in identical:
            print(f"    {name}")
        print(f"\nDIFFER ({len(differing)}):")
        for name in differing:
            delta = [
                line for line in difflib.unified_diff(body(left[name]), body(right[name]), lineterm="", n=0)
                if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
            ]
            verdict = "reviewed" if name in allowed else "UNREVIEWED"
            print(f"    {name}: {len(delta)} changed line(s) — {verdict}")
            if name not in allowed or not args.quiet:
                for line in delta[:12]:
                    print(f"        {line}")
        if missing:
            print(f"\nONLY IN ONE MODULE ({len(missing)}): {', '.join(missing)}")

    unreviewed = [name for name in differing if name not in allowed]
    if unreviewed:
        print(f"\nFAIL — {len(unreviewed)} difference(s) not listed in --allow-differ: {', '.join(unreviewed)}")
        return 1
    print(f"\nOK — {len(identical)} entities share their arithmetic, {len(differing)} reviewed difference(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
