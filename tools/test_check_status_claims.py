"""Tests for `tools/check_status_claims.py`.

The gate exists because the numbers in the documentation drift — a test count quoted after a test was
added, `2 skipped` after the skips were removed, a decision described as open after it was decided. So the
tests here are about *the gate's own* judgement: what counts as a claim, what is only a record, and what
must fail rather than pass quietly.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_status_claims import check, check_invariants  # noqa: E402

DEFINITIONS = """# Decisions

## D1 — the first decision

Something about `D1`.

## D2 — the second decision

Something that cites `D1`.
"""


def audit_text(statuses: dict[str, str] | None = None, evidence: bool = True) -> str:
    """A minimal valid invariants audit, so a fixture repository is a *valid* one."""
    statuses = statuses or {f"I{n}": "`verified`" for n in range(1, 7)}
    body = ["# Audit", ""]
    for invariant, status in statuses.items():
        body.append(f"## {invariant} — a claim — {status}")
        body.append("")
        body.append("See `tools/run_m2_gate.py` for the evidence." if evidence else "Nothing named.")
        body.append("")
    return "\n".join(body)


def build(
    root: Path,
    *,
    readme: str | None = "**184 tests, 0 skipped, 0 failures**\n",
    agents: str | None = "at **184 tests, 0 skipped, 0 failures**\n",
    tracker: str | None = "| Tests | **184 Swift, 200 Python** (all of them) |\n",
    news: str | None = "| 2026-01-01 | then it was **105 tests, 2 skipped, 0 failures** |\n",
    decisions: str | None = DEFINITIONS,
    wiki: bool = True,
    audit: bool = True,
) -> Path:
    (root / "docs").mkdir(parents=True, exist_ok=True)
    if wiki:
        (root / ".wiki").mkdir(parents=True, exist_ok=True)
    # All three decision records, because the tool is right to complain when one is absent: they are part
    # of the repository, not optional context. (The fixture only defined D1/D2 in one of them, which is
    # exactly the sort of shortcut the gate is supposed to notice.)
    (root / "docs").mkdir(parents=True, exist_ok=True)
    if audit:
        (root / "docs" / "invariants-audit.md").write_text(audit_text())
    for name, text in (
        ("README.md", readme),
        ("AGENTS.md", agents),
        ("docs/m0-decisions.md", decisions),
        ("docs/m1-decisions.md", "# M1 decisions\n" if decisions else None),
        ("docs/m2-decisions.md", "# M2 decisions\n" if decisions else None),
        (".wiki/Project-Tracker.md", tracker if wiki else None),
        (".wiki/News.md", news if wiki else None),
        (".wiki/Roadmap.md", "# Roadmap\n" if wiki else None),
        (".wiki/Testbed.md", "# Testbed\n" if wiki else None),
        (".wiki/Architecture.md", "# Architecture\n" if wiki else None),
    ):
        if text is None:
            continue
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


class StatusClaimTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def run_gate(self, **kwargs) -> tuple[list[str], int, list[str]]:
        return check(self.root, **{"swift_tests": 184, "swift_skipped": 0, "python_tests": 200, **kwargs})

    def test_a_current_repository_passes(self) -> None:
        build(self.root)
        problems, checked, not_checked = self.run_gate()
        self.assertEqual(problems, [])
        self.assertGreaterEqual(checked, 3, "the counts in three documents should all be checked")
        self.assertEqual(not_checked, [])

    def test_a_documented_path_that_does_not_exist_fails(self) -> None:
        # The form a reader copies includes paths. A command that cannot be run is worse than a stale
        # number, because the number at least reports something.
        build(self.root, readme="see `tools/gone.py` for the detail\n")
        problems, _, _ = self.run_gate()
        self.assertTrue(any("tools/gone.py" in problem for problem in problems), problems)

    def test_a_real_documented_path_is_checked_and_passes(self) -> None:
        (self.root / "tools").mkdir(parents=True, exist_ok=True)
        (self.root / "tools" / "here.py").write_text("")
        build(self.root, readme="see `tools/here.py`\n")
        problems, checked, _ = self.run_gate()
        self.assertEqual(problems, [])
        self.assertGreaterEqual(checked, 4, "the counts in three documents plus the path")

    def test_a_template_path_is_not_a_claim_about_a_file(self) -> None:
        # `docs/release-notes-vX.Y.md` and `tools/NAME.py` are shapes, not files. The rule is a property —
        # a path in this repository is lower case — rather than a list of exceptions to maintain.
        build(self.root, readme="`docs/release-notes-vX.Y.md` and `tools/NAME.py` are templates\n")
        problems, _, _ = self.run_gate()
        self.assertEqual(problems, [])

    def test_a_stale_test_count_fails(self) -> None:
        build(self.root, readme="**183 tests, 0 skipped, 0 failures**\n")
        problems, _, _ = self.run_gate()
        self.assertTrue(any("183 tests with 0 skipped" in problem for problem in problems), problems)

    def test_a_stale_skip_count_fails(self) -> None:
        """The exact drift this gate was written after: the skips were removed and a document was not."""
        build(self.root, agents="at **184 tests, 2 skipped, 0 failures**\n")
        problems, _, _ = self.run_gate()
        self.assertTrue(any("2 skipped" in problem for problem in problems), problems)

    def test_the_tracker_pair_is_checked_on_both_numbers(self) -> None:
        build(self.root, tracker="| Tests | **184 Swift, 199 Python** |\n")
        problems, _, _ = self.run_gate()
        self.assertTrue(any("199 Python" in problem for problem in problems), problems)

    def test_a_historical_count_in_the_news_is_not_a_claim(self) -> None:
        """News is a dated log: "105 tests" in an old entry was true when it was written."""
        build(self.root)  # the fixture's news entry says 105
        problems, _, _ = self.run_gate()
        self.assertEqual(problems, [], "a record must not be judged as a claim")

    def test_a_dangling_decision_citation_fails(self) -> None:
        build(self.root, agents="the rule in **D34** settles it\n")
        problems, _, _ = self.run_gate()
        self.assertTrue(any("D34" in problem for problem in problems), problems)

    def test_a_citation_of_a_defined_decision_passes(self) -> None:
        build(self.root, agents="the rule in **D2** settles it, and **D1** is why\n")
        problems, _, _ = self.run_gate()
        self.assertEqual(problems, [])

    def test_a_repository_with_no_claims_fails(self) -> None:
        """A regex that stopped matching must not turn the gate into a decoration."""
        # No audit either, so nothing is checked at all — which is the state the guard exists for.
        build(
            self.root,
            readme="nothing to see\n",
            agents="nothing either\n",
            tracker="| Tests | none |\n",
            decisions="",
            audit=False,
        )
        problems, checked, _ = self.run_gate()
        self.assertEqual(checked, 0, "nothing was checked, so the gate must not pass")
        self.assertTrue(any("no decision headings" in problem for problem in problems), problems)

    def test_a_missing_wiki_page_is_reported_rather_than_failing(self) -> None:
        """The wiki is a separate checkout; a run that cannot see it must say so, not fail or pass."""
        build(self.root, wiki=False)
        problems, _, not_checked = self.run_gate()
        self.assertEqual(problems, [])
        self.assertIn(".wiki/Project-Tracker.md", not_checked)

    def test_a_missing_readme_fails(self) -> None:
        build(self.root, readme=None)
        problems, _, _ = self.run_gate()
        self.assertTrue(any("README.md" in problem for problem in problems), problems)

    def test_a_decision_in_any_record_is_found(self) -> None:
        """The record that defines the citation is discovered, so adding one cannot leave a dangling id."""
        build(self.root, agents="the rule in **D7** settles it\n")
        (self.root / "docs" / "repository-decisions.md").write_text("# Repository\n\n## D7 — later\n")
        problems, _, _ = self.run_gate()
        self.assertEqual(problems, [], "a decision defined in a new record must be found")

    def test_a_stale_example_command_is_a_claim_too(self) -> None:
        """The form a reader copies matters more than the prose, not less."""
        build(self.root, agents="python3 tools/check_status_claims.py --swift-tests 183 --python-tests 200\n")
        problems, _, _ = self.run_gate()
        self.assertTrue(any("--swift-tests 183" in problem for problem in problems), problems)

    def test_a_current_example_command_passes(self) -> None:
        build(self.root, agents="--swift-tests 184 --swift-skipped 0 --python-tests 200\n")
        self.assertEqual(self.run_gate()[0], [])

    def test_a_family_without_its_number_is_not_compared(self) -> None:
        build(self.root, readme="**999 tests, 9 skipped, 0 failures**\n")
        problems, _, _ = check(self.root, swift_tests=None, swift_skipped=None, python_tests=200)
        self.assertEqual(problems, [], "the Swift claim cannot be judged without the Swift number")


if __name__ == "__main__":
    unittest.main()


class InvariantAuditTests(unittest.TestCase):
    """The invariants audit has to keep its shape, because a status is a claim about the project."""

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / "docs").mkdir()

    def write(self, text: str) -> None:
        (self.root / "docs" / "invariants-audit.md").write_text(text)

    def audit(self, statuses: dict[str, str] | None = None, evidence: bool = True) -> str:
        return audit_text(statuses, evidence)

    def test_a_missing_audit_is_reported(self) -> None:
        problems, checked = check_invariants(self.root)
        self.assertEqual(checked, 0)
        self.assertTrue(any("no audit" in problem for problem in problems), problems)

    def test_a_complete_audit_passes(self) -> None:
        self.write(self.audit())
        problems, checked = check_invariants(self.root)
        self.assertEqual(problems, [])
        self.assertEqual(checked, 6)

    def test_an_invariant_without_a_section_is_reported(self) -> None:
        self.write(self.audit({f"I{n}": "`verified`" for n in range(1, 6)}))
        problems, _ = check_invariants(self.root)
        self.assertTrue(any("I6 has no section" in problem for problem in problems), problems)

    def test_a_section_without_a_status_is_reported(self) -> None:
        text = self.audit().replace("## I3 — a claim — `verified`", "## I3 — a claim")
        self.write(text)
        problems, _ = check_invariants(self.root)
        self.assertTrue(any("I3's heading states no status" in problem for problem in problems), problems)

    def test_a_status_outside_the_vocabulary_is_reported(self) -> None:
        text = self.audit().replace("## I2 — a claim — `verified`", "## I2 — a claim — `probably fine`")
        self.write(text)
        problems, _ = check_invariants(self.root)
        self.assertTrue(any("I2's heading states no status" in problem for problem in problems), problems)

    def test_a_section_without_named_evidence_is_reported(self) -> None:
        self.write(self.audit(evidence=False))
        problems, _ = check_invariants(self.root)
        self.assertEqual(len(problems), 6, problems)
        self.assertTrue(all("names no file" in problem for problem in problems), problems)
