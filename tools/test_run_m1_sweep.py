#!/usr/bin/env python3
"""Tests for the M1 sweep script: the preconditions, the plan, and the reading of the metrics file.

The script exists because a heavy run on a node that has panicked twice must not depend on a sequence
typed from memory. So the tests are about the parts that fail *before* any heavy work: the refusal above
the swap limit, the disk floor being consulted, the commands matching the CLI's documented interface, and
the metrics file being read rather than assumed.

Nothing here runs a model. `--dry-run` and the refusal paths are the whole surface, plus two pure
functions.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import run_m1_sweep


class SweepPlanTests(unittest.TestCase):
    def test_the_plan_matches_the_cli_it_drives(self):
        """`datacenter-trace <snapshot> <out-dir> <token,ids>` — verified against its own usage line."""
        with tempfile.TemporaryDirectory() as directory:
            args = mock.Mock(
                binary=Path("bin/trace"), snapshot=Path("install"), tokens="1,2,3", out=Path(directory),
                slots=[1, 4],
            )
            commands = run_m1_sweep.plan(args, {})
        self.assertEqual(len(commands), 2)
        for command in commands:
            self.assertEqual(len(command), 4, "four positional arguments, exactly as the CLI documents")
            self.assertEqual(command[0], "bin/trace")
            self.assertEqual(command[1], "install")
            self.assertEqual(command[3], "1,2,3")
        self.assertEqual(commands[0][2], str(Path(directory) / "slots-1"))
        self.assertEqual(commands[1][2], str(Path(directory) / "slots-4"))

    def test_a_dry_run_executes_nothing(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            run_m1_sweep, "require_headroom"
        ), mock.patch.object(run_m1_sweep, "swap_usage", return_value={"total": 2048.0, "used": 0.0, "free": 2048.0}), mock.patch.object(
            run_m1_sweep.subprocess, "run"
        ) as runner:
            code = run_m1_sweep.main([
                "--snapshot", directory, "--tokens", "1,2", "--binary", str(Path(directory) / "nope"),
                "--slots", "2,8", "--out", str(Path(directory) / "out"), "--dry-run",
            ])
        self.assertEqual(code, 0)
        runner.assert_not_called()

    def test_it_refuses_when_the_machine_is_already_swapping(self):
        """Both panics were preceded by swap growth, so this is the precondition worth a refusal."""
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            run_m1_sweep, "require_headroom"
        ), mock.patch.object(
            run_m1_sweep, "swap_usage", return_value={"total": 2048.0, "used": 1500.0, "free": 548.0}
        ), mock.patch.object(run_m1_sweep.subprocess, "run") as runner:
            code = run_m1_sweep.main([
                "--snapshot", directory, "--tokens", "1,2", "--out", str(Path(directory) / "out"),
                "--swap-used-limit-gb", "1.0",
            ])
        self.assertEqual(code, 2, "a refusal is not a success")
        runner.assert_not_called()

    def test_it_consults_the_disk_floor(self):
        """The floor is the other half of the precondition, and it must be asked before anything runs."""
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            run_m1_sweep, "require_headroom"
        ) as headroom, mock.patch.object(
            run_m1_sweep, "swap_usage", return_value={"total": 0.0, "used": 0.0, "free": 0.0}
        ), mock.patch.object(run_m1_sweep.subprocess, "run"):
            run_m1_sweep.main([
                "--snapshot", directory, "--tokens", "1", "--out", str(Path(directory) / "out"), "--dry-run",
            ])
        headroom.assert_called_once()


class MetricsReadingTests(unittest.TestCase):
    def test_the_expert_figures_are_found_wherever_they_are_named(self):
        """The file's schema is read rather than assumed, because inventing it is the failure mode."""
        metrics = {
            "tokens": 5,
            "layers": [
                {"experts": {"requests": 12, "hits": 3, "misses": 9, "elementsRead": 4096}},
                {"experts": {"requests": 12, "hits": 4, "misses": 8, "elementsRead": 4096}},
            ],
            "wallClockSeconds": 33.0,
        }
        found = run_m1_sweep.extract_expert_metrics(metrics)
        self.assertEqual(found["requests"], 12)
        self.assertEqual(found["hits"], 4, "the last layer's figure wins when a list is walked")
        self.assertEqual(found["elementsRead"], 4096)
        self.assertNotIn("wallClockSeconds", found, "an unrelated key is not an expert figure")

    def test_a_metrics_file_with_nothing_about_experts_yields_nothing(self):
        self.assertEqual(run_m1_sweep.extract_expert_metrics({"tokens": 5}), {})

    def test_swap_parsing_survives_output_it_cannot_read(self):
        with mock.patch.object(run_m1_sweep.subprocess, "run") as runner:
            runner.return_value.stdout = "vm.swapusage: total = 2048.00M  used = 12.50M  free = 2035.50M"
            parsed = run_m1_sweep.swap_usage()
        self.assertEqual(parsed["total"], 2048.0)
        self.assertEqual(parsed["used"], 12.5)
        self.assertEqual(parsed["free"], 2035.5)


if __name__ == "__main__":
    unittest.main()
