#!/usr/bin/env python3
"""Tests for tools/check_markdown_links.py.

Each case builds a throwaway tree and asserts on what ``scan()`` reports, so the
gate's own behaviour — including the false positives it must not produce — is
pinned rather than assumed. The last case runs the gate over this repository.

Run: python3 -m unittest discover -s tools
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import check_markdown_links as gate

REPO_ROOT = Path(__file__).resolve().parent.parent


class LinkGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def write(self, relative: str, text: str) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def problems(self) -> list[str]:
        return gate.scan(self.root)[3]

    # --- relative links -----------------------------------------------------

    def test_existing_relative_file_passes(self) -> None:
        self.write("LICENSE", "MIT")
        self.write("README.md", "See [the licence](LICENSE).\n")
        self.assertEqual(self.problems(), [])

    def test_missing_relative_file_is_reported(self) -> None:
        self.write("README.md", "See [the licence](LICENSE).\n")
        problems = self.problems()
        self.assertEqual(len(problems), 1)
        self.assertIn("broken link 'LICENSE'", problems[0])
        self.assertIn("README.md:1", problems[0])

    def test_link_resolves_relative_to_the_containing_file(self) -> None:
        self.write("docs/deep/note.md", "x")
        self.write("docs/README.md", "See [the note](deep/note.md).\n")
        self.assertEqual(self.problems(), [])

    def test_url_encoded_and_angle_bracket_paths(self) -> None:
        self.write("a file.md", "x")
        self.write(
            "README.md",
            "See [one](a%20file.md) and [two](<a file.md>).\n",
        )
        self.assertEqual(self.problems(), [])

    # --- anchors ------------------------------------------------------------

    def test_anchor_in_another_file_passes(self) -> None:
        self.write("doc.md", "# The thesis\n\ntext\n")
        self.write("README.md", "See [thesis](doc.md#the-thesis).\n")
        self.assertEqual(self.problems(), [])

    def test_missing_anchor_is_reported(self) -> None:
        self.write("doc.md", "# The thesis\n\ntext\n")
        self.write("README.md", "See [nope](doc.md#not-there).\n")
        problems = self.problems()
        self.assertEqual(len(problems), 1)
        self.assertIn("no anchor '#not-there'", problems[0])

    def test_same_file_anchor(self) -> None:
        self.write("README.md", "# Top\n\ntext\n\n[up](#top)\n[bad](#bottom)\n")
        problems = self.problems()
        self.assertEqual(len(problems), 1)
        self.assertIn("no anchor '#bottom'", problems[0])

    def test_duplicate_headings_get_numbered_anchors(self) -> None:
        self.write("doc.md", "# Open questions\n\na\n\n# Open questions\n\nb\n")
        self.write("README.md", "[first](doc.md#open-questions)\n[second](doc.md#open-questions-1)\n")
        self.assertEqual(self.problems(), [])

    def test_github_slug_rules_applied(self) -> None:
        # Punctuation is dropped, spaces become hyphens, case is folded.
        self.write("doc.md", "# P0 — Foundation: gates & risks!\n\ntext\n")
        self.write("README.md", "[x](doc.md#p0--foundation-gates--risks)\n")
        self.assertEqual(self.problems(), [])

    # --- things that must not be treated as links ---------------------------

    def test_external_and_mailto_links_are_skipped(self) -> None:
        self.write(
            "README.md",
            "[web](https://example.invalid/nope) [mail](mailto:x@example.invalid)\n",
        )
        self.assertEqual(self.problems(), [])
        self.assertEqual(gate.scan(self.root)[2], 2)

    def test_links_inside_code_fences_are_ignored(self) -> None:
        self.write("README.md", "```markdown\n[not a link](missing.md)\n```\n")
        self.assertEqual(self.problems(), [])
        self.assertEqual(gate.scan(self.root)[0], 0)

    def test_links_inside_inline_code_are_ignored(self) -> None:
        self.write("README.md", "Write `[text](missing.md)` to link.\n")
        self.assertEqual(self.problems(), [])

    def test_headings_inside_code_fences_are_not_anchors(self) -> None:
        self.write("doc.md", "```\n# Not a heading\n```\n")
        self.write("README.md", "[x](doc.md#not-a-heading)\n")
        self.assertEqual(len(self.problems()), 1)

    def test_html_attributes_are_checked(self) -> None:
        self.write("logo.png", "x")
        self.write("README.md", '<img src="logo.png"> and <a href="gone.png">x</a>\n')
        problems = self.problems()
        self.assertEqual(len(problems), 1)
        self.assertIn("broken link 'gone.png'", problems[0])

    def test_local_wiki_clone_is_skipped(self) -> None:
        # The wiki is a separate git repository, cloned into .wiki/ for editing.
        # Its links are GitHub wiki page names, not file paths, so scanning it
        # would report every one of them as broken.
        self.write(".wiki/Home.md", "See [Roadmap](Roadmap) and [x](missing.md).\n")
        self.assertEqual(self.problems(), [])
        self.assertEqual(gate.markdown_files(self.root), [])

    def test_virtual_environment_is_skipped(self) -> None:
        # coremltools ships README files inside site-packages. A third party's
        # broken relative link must not be able to fail this repository's gate.
        self.write(
            ".venv/lib/python3.13/site-packages/coremltools/README.md",
            "See [x](missing.md).\n",
        )
        self.assertEqual(self.problems(), [])
        self.assertEqual(gate.markdown_files(self.root), [])

    # --- the repository itself ---------------------------------------------

    def test_this_repository_is_clean(self) -> None:
        problems = gate.scan(REPO_ROOT)[3]
        self.assertEqual(problems, [], "broken links in this repository")

    def test_this_repository_is_scanned_at_all(self) -> None:
        links, _anchors, _external, _problems = gate.scan(REPO_ROOT)
        self.assertGreater(links, 3, "the gate found almost no links to check")


if __name__ == "__main__":
    unittest.main()
