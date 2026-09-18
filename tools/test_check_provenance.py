"""Tests for `tools/check_provenance.py`.

The provenance conclusion is that nothing here is copied, so MIT is the whole story and no Apache-2.0
`NOTICE` transfers. These tests pin the state that conclusion rests on: the file that records it, the
phrases that stop it decaying into a stub, and the earliest visible symptom of copied code arriving
without its obligations.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_provenance import NOTICES, check  # noqa: E402

GOOD_NOTICES = """# Third-party software and model terms

## No third-party source is included

Nothing here came from the sister project TinyTitan, which is Apache-2.0 and carries its own NOTICE.
"""


class ProvenanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / "sources").mkdir()
        # The licence and notice material a taken file must travel with is now **required** by the gate
        # (`D124`), whether or not a file has been taken. The fixture has to carry it, or these tests are
        # asserting a repository state the gate no longer accepts.
        material = self.root / "third_party" / "TinyTitan"
        material.mkdir(parents=True)
        (material / "LICENSE").write_text("Apache License, Version 2.0\n")
        (material / "NOTICE").write_text("TinyTitan\n")
        (self.root / NOTICES).write_text(GOOD_NOTICES)
        (self.root / "sources" / "Engine.swift").write_text(
            "// Copyright (c) 2026 André Borchert\n// MIT\nstruct Engine {}\n"
        )

    def test_the_repository_as_it_stands_passes(self) -> None:
        problems, inspected = check(self.root)
        self.assertEqual(problems, [])
        self.assertEqual(inspected, 2, "the notices file and the one source file")

    def test_a_missing_notices_file_fails(self) -> None:
        (self.root / NOTICES).unlink()
        problems, _ = check(self.root)
        self.assertTrue(any("is missing" in problem for problem in problems), problems)

    def test_a_notices_file_that_lost_its_content_fails(self) -> None:
        """A stub is worse than nothing: it looks like a review happened."""
        (self.root / NOTICES).write_text("# Third-party notices\n\nNone.\n")
        problems, _ = check(self.root)
        self.assertGreaterEqual(len(problems), 3, problems)
        self.assertTrue(all("no longer mentions" in problem for problem in problems), problems)

    def test_the_licence_and_notice_material_is_required(self) -> None:
        """Staged before the first file is taken, and required from then on (`D124`).

        The review this replaced could only observe that nothing had been taken yet; the gate now demands the
        material that makes taking safe. Removing it must therefore fail even when no source file is derived.
        """
        (self.root / "third_party" / "TinyTitan" / "NOTICE").unlink()
        problems, _ = check(self.root)
        self.assertTrue(any("NOTICE is missing" in problem for problem in problems), problems)

    def test_a_third_party_copyright_line_is_noticed(self) -> None:
        # The line is assembled at runtime on purpose. Written literally, this fixture is a file under
        # `tools/` containing a third-party copyright notice — and `check_provenance.py` flagged it, which
        # is the gate working: the alternative was a special case exempting the checker's own tests, and a
        # rule with an exemption for the file that tests it is a rule that stops being true quietly.
        notice = "Copyright " + "(c) 2025 Some Other Project"
        (self.root / "sources" / "Port.swift").write_text(
            f"// Ported from elsewhere: {notice}\nstruct Port {{}}\n"
        )
        problems, _ = check(self.root)
        self.assertTrue(any("Port.swift" in problem for problem in problems), problems)
        self.assertTrue(any("not this repository's" in problem for problem in problems), problems)

    def test_this_repositorys_own_copyright_is_allowed(self) -> None:
        problems, _ = check(self.root)
        self.assertEqual(problems, [], "the repository's own header must not be flagged")

    def test_an_spdx_identifier_is_allowed(self) -> None:
        (self.root / "sources" / "Spdx.swift").write_text("// SPDX-License-Identifier: MIT\nstruct S {}\n")
        problems, _ = check(self.root)
        self.assertTrue(all("Spdx.swift" not in problem for problem in problems), problems)

    def test_nothing_inspected_fails(self) -> None:
        """A check that examined nothing must not look like a check that passed."""
        (self.root / NOTICES).unlink()
        (self.root / "sources" / "Engine.swift").unlink()
        problems, inspected = check(self.root)
        self.assertEqual(inspected, 0)
        self.assertTrue(problems, "an empty tree must not pass quietly")


if __name__ == "__main__":
    unittest.main()
