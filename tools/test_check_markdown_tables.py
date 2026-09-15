#!/usr/bin/env python3
"""The table gate, tested where it is cheap: does it actually fail?

A checker that cannot fail is a checker that lies. This one exists because two tracker rows landed in
a three-column table below them, and both times a human ran a shell one-liner instead of the
repository — so the tests here are mostly about the **misaligned** case, plus the two ways a
legitimate table could be reported wrongly (escaped pipes and inline code spans, which Markdown does
not treat as separators).
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from check_markdown_tables import check, effective_pipes


class TableCheckTests(unittest.TestCase):
    def check_text(self, body: str) -> list[str]:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "page.md"
            path.write_text(body, encoding="utf-8")
            return check(path)

    def test_an_aligned_table_passes(self):
        self.assertEqual(
            self.check_text("| a | b |\n| --- | --- |\n| 1 | 2 |\n"), []
        )

    def test_a_row_with_too_many_columns_fails(self):
        """The exact defect: a five-column task row inside a three-column story table."""
        problems = self.check_text("| a | b | c |\n| --- | --- | --- |\n| 1 | 2 | 3 | 4 |\n| 4 | 5 | 6 |\n")
        self.assertEqual(len(problems), 1)
        self.assertIn("4 columns", problems[0])
        self.assertIn("has 3", problems[0])

    def test_a_row_with_too_few_columns_fails(self):
        problems = self.check_text("| a | b |\n| --- | --- |\n| 1 |\n")
        self.assertEqual(len(problems), 1)
        self.assertIn("1 columns", problems[0])

    def test_a_second_table_is_measured_on_its_own(self):
        body = "| a | b |\n| --- | --- |\n| 1 | 2 |\n\ntext between tables\n\n| a | b | c |\n| --- | --- | --- |\n| 1 | 2 | 3 |\n"
        self.assertEqual(self.check_text(body), [])

    def test_an_escaped_pipe_is_not_a_separator(self):
        self.assertEqual(effective_pipes(r"| a \| b | c |"), 3)
        self.assertEqual(self.check_text("| a \\| b | c |\n| --- | --- |\n| 1 | 2 |\n"), [])

    def test_a_pipe_inside_an_inline_code_span_is_not_a_separator(self):
        self.assertEqual(effective_pipes("| `a | b` | c |"), 3)
        self.assertEqual(self.check_text("| `a | b` | c |\n| --- | --- |\n| 1 | 2 |\n"), [])

    def test_a_table_outside_a_run_does_not_inherit_the_previous_width(self):
        problems = self.check_text("| a |\n| --- |\n| 1 |\n\nprose\n\n| a | b |\n| --- | --- |\n| 1 | 2 |\n")
        self.assertEqual(problems, [])


if __name__ == "__main__":
    unittest.main()
