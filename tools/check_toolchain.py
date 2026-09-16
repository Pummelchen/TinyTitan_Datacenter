#!/usr/bin/env python3
"""The project's toolchain, asserted: **Xcode 27 with Swift 6.4, and nothing else**.

This is a gate rather than a sentence in a README because the failure it prevents is silent.
`swift-tools-version:6.4` is not a preference the build negotiates — a toolchain below it cannot
parse the manifest at all. The first version of this check lived inside the CI job as

    version=$(swift --version 2>&1 | tail -1)
    if echo "$version" | grep -qE "Swift version 6\\.([4-9]|[1-9][0-9])"; then ... else ::warning:: ...

which is wrong twice over: `tail -1` takes the **`Target:`** line, not the version line, so the pattern
could never match and the job skipped **even on a correct runner**; and the skip was reported as a
warning, so a green run meant either "the toolchain was right" or "the toolchain was so wrong that
nothing ran". Both halves are why the rule here is *no exceptions*: parse the output properly, refuse
anything that is not the standard, and let the job go red until the runner can satisfy it.

A red Swift job means **the runner image is below the standard**, never that the code is broken — which
is the sentence the CI job itself prints when it fails.

Standard library only, like everything under `tools/` that gates the repository.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys

#: The only supported toolchain. Changing either of these is a project decision, not a build option.
REQUIRED_SWIFT = (6, 4)
REQUIRED_XCODE = 27

#: `swift --version` prints several lines; the version is on the first, the target on the last.
SWIFT_VERSION = re.compile(r"Swift version (\d+)\.(\d+)")
XCODE_VERSION = re.compile(r"Xcode (\d+)")


def parse_swift_version(text: str) -> tuple[int, int] | None:
    """The Swift version from anywhere in `swift --version`, or `None` if it is not there."""
    match = SWIFT_VERSION.search(text)
    return (int(match.group(1)), int(match.group(2))) if match else None


def parse_xcode_version(text: str) -> int | None:
    """The major Xcode version from `xcodebuild -version`, or `None`."""
    match = XCODE_VERSION.search(text)
    return int(match.group(1)) if match else None


def verdict(swift: tuple[int, int] | None, xcode: int | None) -> list[str]:
    """Every reason the toolchain is not the project's standard. Empty means it is."""
    problems: list[str] = []
    if swift is None:
        problems.append("swift --version did not report a version at all")
    elif swift != REQUIRED_SWIFT:
        problems.append(
            f"Swift {swift[0]}.{swift[1]} is not the project standard, "
            f"Swift {REQUIRED_SWIFT[0]}.{REQUIRED_SWIFT[1]}"
        )
    if xcode is None:
        problems.append("xcodebuild -version did not report an Xcode version at all")
    elif xcode != REQUIRED_XCODE:
        problems.append(f"Xcode {xcode} is not the project standard, Xcode {REQUIRED_XCODE}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--swift", help="swift --version output, instead of running it")
    parser.add_argument("--xcode", help="xcodebuild -version output, instead of running it")
    args = parser.parse_args(argv)

    swift_text = args.swift if args.swift is not None else subprocess.run(
        ["swift", "--version"], capture_output=True, text=True
    ).stdout
    xcode_text = args.xcode if args.xcode is not None else subprocess.run(
        ["xcodebuild", "-version"], capture_output=True, text=True
    ).stdout

    print("swift --version:")
    print("  " + "\n  ".join(swift_text.strip().splitlines() or ["(no output)"]))
    print("xcodebuild -version:")
    print("  " + "\n  ".join(xcode_text.strip().splitlines() or ["(no output)"]))

    swift = parse_swift_version(swift_text)
    xcode = parse_xcode_version(xcode_text)
    problems = verdict(swift, xcode)

    if problems:
        print("", file=sys.stderr)
        for problem in problems:
            print(f"REFUSING: {problem}", file=sys.stderr)
        print(
            f"\nThis project uses Xcode {REQUIRED_XCODE} with Swift "
            f"{REQUIRED_SWIFT[0]}.{REQUIRED_SWIFT[1]} and supports no other toolchain: "
            f"swift-tools-version {REQUIRED_SWIFT[0]}.{REQUIRED_SWIFT[1]} cannot be parsed by anything "
            f"older, and there are no version conditionals in the sources to fall back on. A red run "
            f"means the machine or runner image is below the standard, not that the code is broken.",
            file=sys.stderr,
        )
        return 1

    print(f"\ntoolchain: Swift {swift[0]}.{swift[1]} on Xcode {xcode} — the project standard")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
