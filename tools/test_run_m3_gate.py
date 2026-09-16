"""Tests for `tools/run_m3_gate.py`.

The gate's *judgement* is what is tested here — which node's time decides the ratio, and what counts as a
farm quiet enough to measure on. The measurement itself needs four idle machines and a 21.7 GB install, so it
is deliberately not in the test suite.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_m3_gate import LOAD, farm_state, load_average, speedup  # noqa: E402


class SpeedupTests(unittest.TestCase):
    def test_the_slowest_node_decides(self) -> None:
        """A cluster step finishes when its slowest member does; an average reports a speed nobody gets."""
        self.assertAlmostEqual(speedup(10.0, [2.0, 5.0, 25.0]), 0.4)

    def test_an_empty_measurement_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            speedup(10.0, [])

    def test_a_zero_measurement_is_refused_rather_than_dividing_by_it(self) -> None:
        with self.assertRaises(ValueError):
            speedup(10.0, [0.0])
        # A zero *alongside* a positive measurement is not a problem: the slowest node decides, and the
        # first version of this test asserted an error for `[1.0, 0.0]`, which the gate is right not to
        # raise. The test was wrong, not the rule.
        self.assertAlmostEqual(speedup(10.0, [1.0, 0.0]), 10.0)

    def test_a_slower_cluster_than_one_node_is_reported_as_such(self) -> None:
        self.assertAlmostEqual(speedup(1.0, [4.0]), 0.25)


class FarmStateTests(unittest.TestCase):
    def test_a_quiet_farm_has_nothing_to_report(self) -> None:
        busy, unknown = farm_state({"a": 0.2, "b": 0.9}, 1.0)
        self.assertEqual((busy, unknown), ([], []))

    def test_a_busy_node_is_named(self) -> None:
        busy, unknown = farm_state({"a": 0.2, "b": 3.5, "c": 1.5}, 1.0)
        self.assertEqual(sorted(busy), ["b", "c"])
        self.assertEqual(unknown, [])

    def test_a_node_that_could_not_be_asked_is_unknown_not_quiet(self) -> None:
        """`None` is not zero. A node nobody could ask must never look like a quiet one."""
        busy, unknown = farm_state({"a": 0.2, "b": None}, 1.0)
        self.assertEqual(busy, [])
        self.assertEqual(unknown, ["b"])

    def test_the_threshold_is_a_boundary_not_an_estimate(self) -> None:
        busy, _ = farm_state({"a": 1.0, "b": 1.000001}, 1.0)
        self.assertEqual(busy, ["b"], "exactly at the threshold is quiet; above it is not")


class LoadParsingTests(unittest.TestCase):
    def test_the_one_minute_average_is_the_first_number(self) -> None:
        line = "{ 23:51:27 } up 1 day, 7:07, 2 users, load averages: 5.58 3.38 3.36"
        self.assertEqual(LOAD.search(line).group(1), "5.58")

    def test_a_line_without_a_load_average_is_not_guessed_at(self) -> None:
        self.assertIsNone(LOAD.search("uptime: command not found"))

    def test_the_local_node_reads_its_own_load_average(self) -> None:
        self.assertAlmostEqual(load_average("ignored", local=True), os.getloadavg()[0], places=2)


if __name__ == "__main__":
    unittest.main()
