"""Tests for `tools/run_all_gates.py`.

The parsers are tested against **recorded output**, not against an idea of it: the Swift summary that a real
`swift test --no-parallel` prints has one `Executed …` line per suite and then the total, twice, and the first
version of this took the first match — a suite's six tests — and reported 13 of 186.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_all_gates import (  # noqa: E402
    counts_completeness,
    parse_swift_summary,
    python_test_files,
    run_python_tests,
)

# Copied from a real run, per-suite lines and all.
REAL_SWIFT_TAIL = """Test Suite 'Selected tests' started at 2026-09-16 01:20:31.000
Test Suite 'All tests' started at 2026-09-16 01:20:31.000
\t Executed 6 tests, with 0 failures (0 unexpected) in 0.002 (0.002) seconds
\t Executed 7 tests, with 0 failures (0 unexpected) in 0.054 (0.054) seconds
\t Executed 186 tests, with 0 failures (0 unexpected) in 20.183 (20.199) seconds
\t Executed 186 tests, with 0 failures (0 unexpected) in 20.183 (20.200) seconds
"""


class SwiftSummaryTests(unittest.TestCase):
    def test_the_overall_line_is_the_last_one_and_not_a_suite(self) -> None:
        self.assertEqual(parse_swift_summary(REAL_SWIFT_TAIL), (186, 0, 0))

    def test_a_skipped_test_is_counted(self) -> None:
        line = "Executed 186 tests, with 1 test skipped and 0 failures (0 unexpected) in 20.1 (20.2) seconds"
        self.assertEqual(parse_swift_summary(line), (186, 1, 0))

    def test_failures_are_read_from_the_overall_line(self) -> None:
        line = "Executed 186 tests, with 3 failures (1 unexpected) in 20.1 (20.2) seconds"
        self.assertEqual(parse_swift_summary(line), (186, 0, 3))

    def test_output_with_no_summary_is_not_guessed_at(self) -> None:
        self.assertIsNone(parse_swift_summary("error: no such module 'DatacenterEngine'"))


class CompletenessTests(unittest.TestCase):
    """When the documentation's counts can be checked, and when the *machine* is what is short."""

    def test_a_complete_run_can_check_the_counts(self) -> None:
        self.assertIsNone(counts_completeness(276, [], 186))

    def test_an_incomplete_python_run_reports_the_reason_instead_of_comparing(self) -> None:
        reason = counts_completeness(168, ["test_quantize.py (needs numpy)"], 186)
        self.assertIsNotNone(reason)
        self.assertIn("test_quantize.py", reason)
        self.assertIn("incomplete", reason)

    def test_no_python_tests_ran_is_a_reason(self) -> None:
        self.assertIsNotNone(counts_completeness(0, [], 186))

    def test_no_swift_count_is_a_reason(self) -> None:
        self.assertIsNotNone(counts_completeness(276, [], None))


class TestDiscoveryTests(unittest.TestCase):
    def test_the_repository_has_test_files_and_they_are_sorted(self) -> None:
        found = python_test_files()
        self.assertGreater(len(found), 10)
        self.assertEqual(found, sorted(found))

    def test_a_file_that_cannot_be_imported_is_not_checked_rather_than_fatal(self) -> None:
        """A machine without a pinned package is a machine with a gap, not a repository that is broken."""
        with tempfile.TemporaryDirectory() as directory:
            tools = Path(directory)
            (tools / "test_needs_a_missing_package.py").write_text(
                "import definitely_not_installed_anywhere\n"
                "import unittest\n\n"
                "class T(unittest.TestCase):\n"
                "    def test_x(self):\n        self.assertTrue(True)\n"
            )
            total, problems, not_checked, _ = run_python_tests(tools=tools)
        self.assertEqual(problems, [], "a missing package must not be reported as a failure")
        self.assertEqual(total, 0)
        self.assertEqual(len(not_checked), 1)
        self.assertIn("definitely_not_installed_anywhere", not_checked[0])

    def test_a_real_test_file_runs_and_its_count_is_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tools = Path(directory)
            (tools / "test_two.py").write_text(
                "import unittest\n\n"
                "class T(unittest.TestCase):\n"
                "    def test_a(self):\n        self.assertTrue(True)\n"
                "    def test_b(self):\n        self.assertTrue(True)\n"
            )
            total, problems, not_checked, detail = run_python_tests(tools=tools)
        self.assertEqual((total, problems, not_checked), (2, [], []))
        self.assertIn("2 test(s), OK", detail[0])

    def test_a_failing_test_file_is_reported_with_its_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tools = Path(directory)
            (tools / "test_fails.py").write_text(
                "import unittest\n\n"
                "class T(unittest.TestCase):\n"
                "    def test_a(self):\n        self.assertEqual(1, 2)\n"
            )
            total, problems, _, _ = run_python_tests(tools=tools)
        self.assertEqual(total, 1)
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("test_fails.py", problems[0])


if __name__ == "__main__":
    unittest.main()
