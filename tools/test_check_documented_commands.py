#!/usr/bin/env python3
"""Check the documented-command gate itself, so it is never untested code.

The gate reads the commands the documentation gives a reader and asks each tool what it accepts. Its judgement
is the thing worth pinning here: a flag a tool does not take is a problem, a tool that cannot be asked is *not
checked* rather than a pass, and a command naming a script that does not exist is a problem of its own.

The tools are resolved against the real repository, so these tests point the scan at a temporary document and
leave the repository alone.

Standard library only, like the rest of the CI suite.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_documented_commands import ROOT, check  # noqa: E402


class DocumentedCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def run_gate(self, text: str) -> tuple[list[str], int, list[str]]:
        document = self.root / "EXAMPLE.md"
        document.write_text(text)
        return check(ROOT, only=[document])

    def test_a_flag_the_tool_takes_is_accepted_and_counted(self) -> None:
        problems, checked, _ = self.run_gate("```bash\npython3 tools/check_markdown_links.py --verbose\n```\n")
        self.assertEqual(problems, [])
        self.assertEqual(checked, 1)

    def test_a_flag_the_tool_does_not_take_is_reported(self) -> None:
        problems, _, _ = self.run_gate("```bash\npython3 tools/check_markdown_links.py --invented\n```\n")
        self.assertTrue(any("--invented" in problem for problem in problems), problems)

    def test_a_command_naming_a_missing_script_is_reported(self) -> None:
        problems, _, _ = self.run_gate("```bash\npython3 tools/gone.py --verbose\n```\n")
        self.assertTrue(any("gone.py" in problem for problem in problems), problems)

    def test_a_command_with_no_flags_is_not_a_claim_about_flags(self) -> None:
        problems, checked, _ = self.run_gate("```bash\npython3 tools/check_markdown_links.py\n```\n")
        self.assertEqual(problems, [])
        self.assertEqual(checked, 0)

    def test_prose_that_names_a_tool_without_invoking_it_is_not_a_command(self) -> None:
        """A sentence is not a command, and treating one as a command would invent flags to check."""
        problems, checked, _ = self.run_gate("See `tools/check_markdown_links.py` for the anchor rules.\n")
        self.assertEqual(problems, [])
        self.assertEqual(checked, 0)

    def test_a_backslash_continuation_does_not_hide_the_rest_of_the_command(self) -> None:
        """The commands in the gate docs wrap, and a wrapped flag is still a flag the reader copies."""
        problems, checked, _ = self.run_gate(
            "```bash\npython3 tools/check_markdown_links.py \\\n    --verbose\n```\n"
        )
        self.assertEqual(problems, [])
        self.assertEqual(checked, 1)

    def test_a_tool_that_cannot_be_asked_is_not_checked_rather_than_passed(self) -> None:
        """The one outcome this must never produce is silence dressed as success.

        `install_reader.py` is a library with a self-description rather than an `argparse` CLI, so it exits
        non-zero for `--help`; the gate has to say it could not ask rather than count its flags as fine.
        """
        problems, _, not_checked = self.run_gate(
            "```bash\npython3 tools/install_reader.py --uncached\n```\n"
        )
        self.assertEqual(problems, [])
        self.assertTrue(not_checked, "a tool that could not be asked must be reported")


if __name__ == "__main__":
    unittest.main()
