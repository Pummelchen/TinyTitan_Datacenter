#!/usr/bin/env python3
"""Tests for the toolchain gate.

Most of these exist because the check this replaces was wrong in a way nothing noticed: it read
`swift --version | tail -1` and looked for "Swift version 6.4" there. The last line is the **target**,
so the pattern could never match, and the job took its skip branch on every runner it ever saw — a
gate that reports success by not running. Two tests below pin that specifically: the real output's last
line does not carry the version, and the parse must find it anyway.

Nothing here shells out to `swift` or `xcodebuild`: the parsing and the verdict are pure, and the real
outputs are copied in verbatim, so the tests say what the project requires rather than what this
machine happens to have.
"""

from __future__ import annotations

import unittest

import check_toolchain
from check_toolchain import parse_swift_version, parse_xcode_version, verdict

#: Verbatim from `swift --version` on the farm, where the build and the tests pass.
REAL_SWIFT = (
    "swift-driver version: 1.168.6 Apple Swift version 6.4 "
    "(swiftlang-6.4.0.34.1 clang-2100.3.34.1)\n"
    "Target: arm64-apple-macosx27.0.0\n"
)

#: Verbatim from `xcodebuild -version`.
REAL_XCODE = "Xcode 27.0\nBuild version 27A266a\n"

#: What a `macos-26` runner image carries, which the project refuses.
OLD_SWIFT = "swift-driver version: 1.120.5 Apple Swift version 6.3.3 (swiftlang-6.3.3.1)\nTarget: arm64-apple-macosx26.0\n"
OLD_XCODE = "Xcode 26.5\nBuild version 26F74\n"


class ParsingTests(unittest.TestCase):
    def test_the_version_is_found_on_the_first_line_not_the_last(self):
        """The bug that made the old check a no-op: `tail -1` is the target line."""
        self.assertEqual(parse_swift_version(REAL_SWIFT), (6, 4))
        last_line = REAL_SWIFT.strip().splitlines()[-1]
        self.assertNotIn("Swift version", last_line)
        self.assertIsNone(
            parse_swift_version(last_line),
            "the last line carries the target, so a check that reads only it can never pass",
        )

    def test_the_xcode_major_version_is_parsed(self):
        self.assertEqual(parse_xcode_version(REAL_XCODE), 27)
        self.assertEqual(parse_xcode_version("Xcode 27.1\nBuild version 27B1\n"), 27)

    def test_output_without_a_version_is_none_rather_than_a_guess(self):
        self.assertIsNone(parse_swift_version("command not found: swift\n"))
        self.assertIsNone(parse_xcode_version("xcode-select: error: tool 'xcodebuild' requires Xcode\n"))


class VerdictTests(unittest.TestCase):
    def test_the_project_standard_passes(self):
        self.assertEqual(verdict((6, 4), 27), [])

    def test_an_older_toolchain_is_refused_on_both_counts(self):
        problems = verdict(parse_swift_version(OLD_SWIFT), parse_xcode_version(OLD_XCODE))
        self.assertEqual(len(problems), 2)
        self.assertIn("Swift 6.3", problems[0])
        self.assertIn("Xcode 26", problems[1])

    def test_a_newer_swift_is_also_refused(self):
        """No exceptions means no exceptions: a newer toolchain is a decision, not a drift."""
        problems = verdict((6, 5), 27)
        self.assertEqual(len(problems), 1)
        self.assertIn("is not the project standard", problems[0])

    def test_a_newer_xcode_is_also_refused(self):
        self.assertEqual(len(verdict((6, 4), 28)), 1)

    def test_missing_output_is_a_refusal_not_a_pass(self):
        problems = verdict(None, None)
        self.assertEqual(len(problems), 2)
        self.assertTrue(all("did not report" in problem for problem in problems))


class CommandLineTests(unittest.TestCase):
    def test_the_standard_toolchain_exits_zero(self):
        self.assertEqual(
            check_toolchain.main(["--swift", REAL_SWIFT, "--xcode", REAL_XCODE]), 0
        )

    def test_an_unsupported_toolchain_exits_non_zero(self):
        self.assertEqual(
            check_toolchain.main(["--swift", OLD_SWIFT, "--xcode", OLD_XCODE]), 1
        )

    def test_the_constants_are_the_project_standard(self):
        """If either moves, the manifest moves with it — this is the single place that says so."""
        self.assertEqual(check_toolchain.REQUIRED_SWIFT, (6, 4))
        self.assertEqual(check_toolchain.REQUIRED_XCODE, 27)


if __name__ == "__main__":
    unittest.main()
