"""Tests for `tools/check_milestones.py` and for the claims data it reads.

The distinction these pin is the one that went wrong: **the engine reproducing its own recorded digest** and
**the engine agreeing with the reference** are different claims, and a stale agreement is allowed only if it
is declared and tracked.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_milestones import DEFAULT_DATA, evaluate, load_milestones, main, read_digest  # noqa: E402

DOCUMENT = {
    "prompt": {"id": "capital", "tokens": [1, 2, 3], "length": 3},
    "milestones": [
        {"id": "M1", "claim": "matches", "agreement": "stale", "known_mismatch": "DC-111",
         "engine_digest": "b0d382dbabf36df0…", "source": "docs/m1-gate.md"},
        {"id": "M2", "claim": "two machines match one", "agreement": "current",
         "engine_digest": "b0d382dbabf36df0…", "source": "docs/m2-decisions.md"},
        {"id": "M0", "claim": "two-b model matches", "agreement": "not checked here",
         "engine_digest": None, "recheck": "needs a checkpoint that is not cached", "source": "docs/m0-gate.md"},
    ],
}


class LoadTests(unittest.TestCase):
    def test_the_committed_data_loads(self) -> None:
        document = load_milestones()
        self.assertTrue(document["milestones"])
        self.assertTrue(document["prompt"]["tokens"])

    def test_every_milestone_names_a_document_that_exists(self) -> None:
        for entry in load_milestones()["milestones"]:
            with self.subTest(milestone=entry["id"]):
                self.assertTrue((DEFAULT_DATA.parent.parent / entry["source"]).exists(), entry["source"])

    def test_a_stale_agreement_must_declare_its_task(self) -> None:
        """A stale agreement is a fact somebody owns, or the tool would carry it forever."""
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        path = directory / "bad.json"
        path.write_text(json.dumps({
            "prompt": {"tokens": [1]},
            "milestones": [{"id": "M1", "claim": "c", "agreement": "stale", "source": "docs/m1-gate.md"}],
        }))
        with self.assertRaises(ValueError) as caught:
            load_milestones(path)
        self.assertIn("known_mismatch", str(caught.exception))

    def test_a_milestone_missing_a_field_is_refused(self) -> None:
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        path = directory / "bad.json"
        path.write_text(json.dumps({"prompt": {"tokens": [1]}, "milestones": [{"id": "M1"}]}))
        with self.assertRaises(ValueError):
            load_milestones(path)

    def test_data_with_no_milestones_is_refused(self) -> None:
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        path = directory / "empty.json"
        path.write_text(json.dumps({"prompt": {"tokens": [1]}, "milestones": []}))
        with self.assertRaises(ValueError):
            load_milestones(path)


class EvaluateTests(unittest.TestCase):
    def test_a_matching_digest_produces_no_problems(self) -> None:
        problems, rows = evaluate(DOCUMENT, "b0d382dbabf36df0aabbcc")
        self.assertEqual(problems, [])
        self.assertEqual([row[0] for row in rows], ["M1", "M2", "M0"])
        self.assertEqual(rows[0][1], "reproduces")

    def test_a_declared_divergence_is_reported_but_not_a_problem(self) -> None:
        """M1's divergence is declared and tracked; M2's, in this fixture, is not — which is the whole point."""
        problems, rows = evaluate(DOCUMENT, "ffffffff00000000")
        self.assertIn("DC-111", rows[0][1], "a declared divergence names its task")
        self.assertIn("UNDECLARED", rows[1][1], "an undeclared one is called out")
        self.assertEqual(len(problems), 1, "exactly one divergence here has no declaration")
        self.assertIn("M2", problems[0])

    def test_a_milestone_that_cannot_be_checked_says_why(self) -> None:
        _, rows = evaluate(DOCUMENT, "b0d382dbabf36df0")
        self.assertEqual(rows[2][1], "not checkable here")
        self.assertIn("not cached", rows[2][3])

    def test_a_missing_digest_is_a_divergence_not_a_pass(self) -> None:
        """No digest is a divergence: the declared one is reported, the undeclared one fails."""
        problems, rows = evaluate(DOCUMENT, None)
        self.assertEqual([row[0] for row in rows if "UNDECLARED" in row[1]], ["M2"])
        self.assertEqual(len(problems), 1, problems)

    def test_the_prefix_match_ignores_the_ellipsis_in_the_recorded_digest(self) -> None:
        problems, _ = evaluate(
            {"milestones": [{"id": "M", "claim": "c", "agreement": "current", "source": "s",
                             "engine_digest": "abc123…"}]},
            "abc123def456",
        )
        self.assertEqual(problems, [])


class TraceDirTests(unittest.TestCase):
    def test_the_digest_is_read_from_a_manifest(self) -> None:
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        (directory / "manifest.json").write_text(json.dumps({"digest": "deadbeef"}))
        self.assertEqual(read_digest(directory), "deadbeef")

    def test_a_directory_without_a_manifest_has_no_digest(self) -> None:
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.assertIsNone(read_digest(directory))

    def test_the_command_line_uses_an_existing_trace(self) -> None:
        """The tests must not run a real trace: they point the tool at one that already exists."""
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        (directory / "manifest.json").write_text(json.dumps({"digest": "b0d382dbabf36df0zz"}))
        data = directory / "data.json"
        data.write_text(json.dumps(DOCUMENT))
        self.assertEqual(main(["--trace-dir", str(directory), "--data", str(data)]), 0)

    def test_the_command_line_fails_on_an_undeclared_divergence(self) -> None:
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        (directory / "manifest.json").write_text(json.dumps({"digest": "0000000000000000"}))
        data = directory / "data.json"
        data.write_text(json.dumps(DOCUMENT))
        self.assertEqual(main(["--trace-dir", str(directory), "--data", str(data)]), 1)


if __name__ == "__main__":
    unittest.main()
