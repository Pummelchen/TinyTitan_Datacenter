#!/usr/bin/env python3
"""Check the local links in this repository's Markdown files.

Right now the documentation *is* the product: a broken link in the README or a wiki
page is a defect a reader hits immediately, and a link to a heading that no longer
exists is just as invisible to the eye. This gate finds both.

It is deliberately **offline**: external URLs are counted and skipped, so a flaky
network or a rate-limited host can never fail a build. It is also deliberately
dependency-free — the standard library only — because a gate nobody can run locally
is a gate that gets bypassed.

Usage:
    python3 tools/check_markdown_links.py [--verbose]

Exit status:
    0  every local link and anchor resolves
    1  at least one local link or anchor is broken

Tests for this tool: tools/test_check_markdown_links.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

# Directories that never contain repository Markdown worth checking. The virtual
# environments matter: coremltools ships its own README files, and a third party's
# broken relative link must not be able to fail this repository's gate.
SKIP_DIRS = {
    ".git",
    ".build",
    ".swiftpm",
    ".wiki",
    ".inspect",
    ".venv",
    "venv",
    "site-packages",
    "__pycache__",
    "DerivedData",
    "node_modules",
    "models",
    "xcuserdata",
}

# Schemes that are not this tool's business (and http/https would need a network).
SKIP_SCHEMES = {"http", "https", "mailto", "tel", "ftp", "data"}

# [text](target) / ![alt](target), where the target may be wrapped in <...> and may
# be followed by a "title". A title is ignored.
MD_LINK = re.compile(r"!?\[[^\]]*\]\(\s*(?:<([^>]*)>|([^)\s]+))")
# <img src="..."> and <a href="...">
HTML_ATTR = re.compile(r"""(?:src|href)\s*=\s*["']([^"']+)["']""")
FENCE = re.compile(r"^\s*(```|~~~)")
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
INLINE_CODE = re.compile(r"`[^`]*`")


def slugify(text: str) -> str:
    """GitHub's heading anchor: strip markup and punctuation, spaces to hyphens.

    GitHub's slugger replaces *every* space with a hyphen rather than collapsing
    runs, so ``## P0 — Foundation`` anchors as ``#p0--foundation`` (two hyphens:
    the em dash is dropped and the spaces around it are not merged). Anchors a
    reader copies out of the rendered page have to match, so this does too.
    """
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = text.strip().lower()
    text = "".join(ch for ch in text if ch.isalnum() or ch in " -_")
    return text.replace(" ", "-")


def iter_content_lines(text: str):
    """Yield (line_number, line, in_fenced_block) for every line of ``text``.

    Fenced code blocks are reported rather than skipped by the caller, because a
    link-looking string inside a fence is an example, not a link.
    """
    in_fence = False
    marker = ""
    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.lstrip()
        if FENCE.match(line):
            opening = stripped[:3]
            if not in_fence:
                in_fence, marker = True, opening
            elif opening == marker:
                in_fence, marker = False, ""
            yield number, line, True
            continue
        yield number, line, in_fence


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def headings_of(path: Path) -> set[str]:
    """The anchors a Markdown file actually offers, duplicate headings included."""
    anchors: set[str] = set()
    seen: dict[str, int] = {}
    for _, line, fenced in iter_content_lines(read_text(path)):
        if fenced:
            continue
        match = HEADING.match(line)
        if not match:
            continue
        slug = slugify(match.group(2))
        if not slug:
            continue
        count = seen.get(slug, 0)
        seen[slug] = count + 1
        anchors.add(slug if count == 0 else f"{slug}-{count}")
    return anchors


def targets_of(text: str):
    """Yield (line_number, link_target) for every link and image in ``text``."""
    for number, line, fenced in iter_content_lines(text):
        if fenced:
            continue
        line = INLINE_CODE.sub("", line)
        for match in MD_LINK.finditer(line):
            yield number, (match.group(1) or match.group(2))
        for match in HTML_ATTR.finditer(line):
            yield number, match.group(1)


def markdown_files(root: Path) -> list[Path]:
    files = [
        path
        for path in root.rglob("*.md")
        if not any(part in SKIP_DIRS for part in path.relative_to(root).parts)
    ]
    return sorted(files)


def scan(root: Path) -> tuple[int, int, int, list[str]]:
    """Check every Markdown file under ``root``.

    Returns (local links checked, anchors checked, external links skipped,
    problems). ``problems`` holds human-readable, one-per-line messages.
    """
    # Resolve the root once: link targets are resolved (symlinks and all), so on
    # macOS an unresolved /var root would make relative_to() fail against
    # /private/var paths.
    root = root.resolve()
    links = anchors = external = 0
    problems: list[str] = []
    for path in markdown_files(root):
        relative = path.relative_to(root)
        for number, raw in targets_of(read_text(path)):
            target = raw.strip()
            if not target:
                continue
            if target.startswith("#"):
                anchors += 1
                anchor = unquote(target[1:]).lower()
                if anchor and anchor not in headings_of(path):
                    problems.append(
                        f"{relative}:{number}: no anchor '#{anchor}' in {relative}"
                    )
                continue
            parts = urlsplit(target)
            if parts.scheme in SKIP_SCHEMES or target.startswith("//"):
                external += 1
                continue
            if parts.scheme or target.startswith("/"):
                continue  # an unknown scheme, or site-absolute: not ours to resolve
            cleaned = unquote(parts.path)
            if not cleaned:
                continue
            resolved = (path.parent / cleaned).resolve()
            if not resolved.exists():
                problems.append(
                    f"{relative}:{number}: broken link '{target}' — no such file or directory"
                )
                continue
            links += 1
            if parts.fragment and resolved.suffix.lower() == ".md":
                anchors += 1
                anchor = unquote(parts.fragment).lower()
                if anchor not in headings_of(resolved):
                    problems.append(
                        f"{relative}:{number}: no anchor '#{anchor}' in "
                        f"{resolved.relative_to(root)}"
                    )
    return links, anchors, external, problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="repository root to scan (default: the parent of this script)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="print every file as it is checked"
    )
    args = parser.parse_args(argv)

    root = args.root.resolve()
    if args.verbose:
        for path in markdown_files(root):
            print(f"checking {path.relative_to(root)}")

    links, anchors, external, problems = scan(root)
    for problem in problems:
        print(problem)
    print(
        f"Checked {links} local link(s) and {anchors} anchor(s) in "
        f"{len(markdown_files(root))} file(s); {external} external link(s) skipped; "
        f"{len(problems)} broken."
    )
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
