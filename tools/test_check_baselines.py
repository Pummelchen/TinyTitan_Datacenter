"""Tests for `tools/check_baselines.py`, and for the baselines file it reads.

The last class is the one that keeps `baselines.json` honest: every recorded figure must name a document
that exists, because a baseline whose source has been deleted is a number with nothing behind it.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_baselines import BASELINES, ROOT, compare, derive, load_baselines, main  # noqa: E402

COUNTS = [
    {"key": "expert_requests", "value": 2218, "kind": "count", "source": "docs/m2-decisions.md",
     "produced_by": "datacenter-trace", "applies_to": "the M1 install", "requires": {"layers": 40}},
    {"key": "expert_hit_rate", "value": 0.0, "kind": "count", "source": "docs/m2-decisions.md",
     "produced_by": "datacenter-trace", "applies_to": "the M1 install", "requires": {"layers": 40}},
]
OBSERVATIONS = [
    {"key": "steps_per_second", "value": 0.108, "kind": "observed", "source": "docs/m1-gate.md",
     "produced_by": "datacenter-generate --cached", "applies_to": "the M1 install", "requires": {"steps": 3}},
]


class CountTests(unittest.TestCase):
    def test_a_matching_count_passes_and_names_its_source(self) -> None:
        problems, checked, missing, reported = compare(COUNTS, [], {"layers": 40, "expert_requests": 2218})
        self.assertEqual(problems, [])
        self.assertEqual(checked, 1)
        self.assertEqual(missing, ["expert_hit_rate (a count, from docs/m2-decisions.md; datacenter-trace)"])
        self.assertTrue(any("docs/m2-decisions.md" in line for line in reported), reported)

    def test_a_differing_count_fails_with_both_numbers(self) -> None:
        problems, _, _, _ = compare(COUNTS, [], {"layers": 40, "expert_requests": 2217, "expert_hit_rate": 0.0})
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("2217", problems[0])
        self.assertIn("2218", problems[0])
        self.assertIn("docs/m2-decisions.md", problems[0])

    def test_a_float_count_is_compared_exactly(self) -> None:
        """The hit rate is 0.0 at every bank size, and `0.0000001` is not 0."""
        problems, _, _, _ = compare(COUNTS, [], {"layers": 40, "expert_hit_rate": 1e-7})
        self.assertEqual(len(problems), 1, problems)

    def test_a_count_whose_requirements_are_not_met_is_not_compared(self) -> None:
        """The two-node fixture's `exchange_reduces` is not a statement about a single-node run."""
        fixture = [{"key": "exchange_reduces", "value": 10, "kind": "count", "source": "docs/x.md",
                    "produced_by": "datacenter-generate", "applies_to": "the fixture", "requires": {"nodes": 2}}]
        problems, checked, missing, _ = compare(fixture, [], {"nodes": 1, "exchange_reduces": 0})
        self.assertEqual(problems, [], "a correct single-node run must not fail a two-node baseline")
        self.assertEqual(checked, 0)
        self.assertTrue(any("does not apply" in line for line in missing), missing)

    def test_a_count_whose_requirements_are_met_is_compared(self) -> None:
        fixture = [{"key": "exchange_reduces", "value": 10, "kind": "count", "source": "docs/x.md",
                    "produced_by": "datacenter-generate", "applies_to": "the fixture", "requires": {"nodes": 2}}]
        problems, checked, _, _ = compare(fixture, [], {"nodes": 2, "exchange_reduces": 10})
        self.assertEqual(problems, [])
        self.assertEqual(checked, 1)

    def test_a_baseline_without_requirements_applies_to_any_run(self) -> None:
        """`peak_rss_bytes` comes from `/usr/bin/time`, so no metrics file can be asked for it — and a
        requirement it cannot satisfy would have left it permanently uncheckable."""
        entry = [{"key": "peak_rss_bytes", "value": 1, "kind": "observed", "source": "docs/x.md",
                  "produced_by": "/usr/bin/time -l"}]
        _, checked, missing, _ = compare([], entry, {"anything": 1})
        self.assertEqual(checked, 0)
        self.assertTrue(any("produced by /usr/bin/time" in line for line in missing), missing)

    def test_a_required_key_the_run_does_not_report_is_not_checked(self) -> None:
        fixture = [{"key": "k", "value": 1, "kind": "count", "source": "docs/x.md",
                    "produced_by": "c", "applies_to": "a", "requires": {"nodes": 2}}]
        problems, checked, missing, _ = compare(fixture, [], {"k": 1})
        self.assertEqual(problems, [])
        self.assertEqual(checked, 0)
        self.assertEqual(len(missing), 1)

    def test_a_missing_count_is_not_checked_rather_than_assumed(self) -> None:
        problems, checked, missing, _ = compare(COUNTS, [], {"layers": 40})
        self.assertEqual(problems, [])
        self.assertEqual(checked, 0)
        self.assertEqual(len(missing), 2, missing)


class ObservationTests(unittest.TestCase):
    def test_an_observation_is_reported_and_not_asserted_by_default(self) -> None:
        problems, checked, _, reported = compare([], OBSERVATIONS, {"steps": 3, "steps_per_second": 0.5})
        self.assertEqual(problems, [], "a shared farm's throughput is reported, not asserted")
        self.assertEqual(checked, 1)
        self.assertTrue(any("not asserted" in line for line in reported), reported)

    def test_asserting_observations_fails_outside_the_band(self) -> None:
        problems, _, _, _ = compare([], OBSERVATIONS, {"steps": 3, "steps_per_second": 0.5}, assert_observed=True)
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("band", problems[0])

    def test_asserting_observations_passes_inside_the_band(self) -> None:
        problems, _, _, _ = compare([], OBSERVATIONS, {"steps": 3, "steps_per_second": 0.11}, assert_observed=True)
        self.assertEqual(problems, [])

    def test_the_rate_is_derived_from_the_shape_a_real_run_writes(self) -> None:
        """**The recorded shape**, copied from a real metrics.json: a list, and the stated aggregates.

        The first version of this test invented `step_seconds_0` scalars, so it passed while the tool
        crashed on the artifact — the same mistake as a diagnostic narrower than its gate.
        """
        real = {
            "generated_tokens": 1,
            "install_bytes_read_total": 2928600000,
            "step_seconds": [5.680495023727417, 5.681842923164368, 5.672945976257324],
            "step_seconds_slowest": 5.681842923164368,
            "step_seconds_total": 17.03528392314911,
            "steps": 3,
        }
        self.assertAlmostEqual(derive("steps_per_second", real), 3 / 17.03528392314911, places=6)

    def test_the_aggregates_are_preferred_and_a_bare_list_still_works(self) -> None:
        self.assertAlmostEqual(derive("steps_per_second", {"steps": 2, "step_seconds_total": 4.0}), 0.5)
        self.assertAlmostEqual(derive("steps_per_second", {"step_seconds": [2.0, 2.0]}), 0.5)

    def test_the_scalar_per_step_form_still_works(self) -> None:
        self.assertAlmostEqual(derive("steps_per_second", {"step_seconds_0": 9.25}), 0.108108, places=5)

    def test_an_unexpected_shape_returns_nothing_rather_than_crashing(self) -> None:
        """A metrics file this tool does not understand is NOT CHECKED, not a traceback."""
        for metrics in (
            {"step_seconds": [[1.0], [2.0]]},
            {"step_seconds": "5.5"},
            {"step_seconds": [], "steps": 0, "step_seconds_total": 0},
            {"steps": True, "step_seconds_total": True},
        ):
            with self.subTest(metrics=metrics):
                self.assertIsNone(derive("steps_per_second", metrics))

    def test_a_derivation_that_cannot_be_made_returns_nothing(self) -> None:
        self.assertIsNone(derive("steps_per_second", {"steps": 3}))
        self.assertIsNone(derive("something_else", {"step_seconds_0": 1.0}))

    def test_a_zero_mean_is_not_divided_by(self) -> None:
        self.assertIsNone(derive("steps_per_second", {"step_seconds_0": 0.0}))


class CommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "metrics.json"

    def test_a_run_with_nothing_to_check_fails(self) -> None:
        """A gate that passes having checked nothing is decoration."""
        self.path.write_text(json.dumps({"generated_tokens": 3}))
        self.assertEqual(main(["--metrics", str(self.path)]), 1)

    def test_a_missing_metrics_file_fails(self) -> None:
        self.assertEqual(main(["--metrics", str(self.path)]), 1)

    def test_a_matching_run_passes(self) -> None:
        self.path.write_text(json.dumps({"layers": 40, "expert_requests": 2218, "expert_hit_rate": 0.0}))
        self.assertEqual(main(["--metrics", str(self.path)]), 0)

    def test_no_metrics_at_all_is_a_usage_error(self) -> None:
        with self.assertRaises(SystemExit):
            main([])


class BaselinesFileTests(unittest.TestCase):
    """The committed data has to be well formed, and every figure has to name a document that exists."""

    def test_the_committed_file_loads_and_is_well_formed(self) -> None:
        counts, observations = load_baselines()
        self.assertTrue(counts, "no count baselines were recorded")
        self.assertTrue(observations, "no observations were recorded")
        self.assertTrue(BASELINES.exists())

    def test_every_baseline_names_a_source_that_exists(self) -> None:
        counts, observations = load_baselines()
        for entry in counts + observations:
            with self.subTest(key=entry["key"]):
                source = ROOT.parent / entry["source"]
                self.assertTrue(source.exists(), f"{entry['key']} names {entry['source']}, which is gone")
                self.assertIn("produced_by", entry, "a baseline must say what produces it")
                self.assertIn("applies_to", entry, "a baseline must say what it applies to")

    def test_the_two_kinds_are_separated(self) -> None:
        counts, observations = load_baselines()
        self.assertTrue(all(entry["kind"] == "count" for entry in counts))
        self.assertTrue(all(entry["kind"] == "observed" for entry in observations))

    def test_a_count_in_the_observations_section_is_refused(self) -> None:
        path = Path(self.enterContext(tempfile.TemporaryDirectory())) / "bad.json"
        path.write_text(json.dumps({"baselines": [{"key": "k", "value": 1, "kind": "observed"}]}))
        with self.assertRaises(ValueError):
            load_baselines(path)

    def test_a_baseline_missing_its_source_is_refused(self) -> None:
        path = Path(self.enterContext(tempfile.TemporaryDirectory())) / "bad.json"
        path.write_text(json.dumps({"baselines": [{"key": "k", "value": 1, "kind": "count"}]}))
        with self.assertRaises(ValueError):
            load_baselines(path)


if __name__ == "__main__":
    unittest.main()
