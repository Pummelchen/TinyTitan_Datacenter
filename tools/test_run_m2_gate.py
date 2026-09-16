"""Tests for the M2 two-process harness's own arithmetic.

The harness spawns processes, so its orchestration needs the binaries; these pin the parts that decide
*what* it runs — the plan it writes and the facts it reads from the install — because a wrong plan is the
failure that looks like an engine bug. It did, the first time: the family was hardcoded, the node refused
the plan, and the refusal arrived as a Swift trap.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_m2_gate as harness  # noqa: E402


def fake_install(root: Path, family: str = "qwen3_5_moe", experts: int = 8) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": 1,
        "family": family,
        "tensors": [
            {"name": "model.layers.0.mlp.experts.gate_up_proj", "role": "expert.stack_gate_up",
             "shape": [experts, 4, 8], "nbytes": experts * 34, "offset": 0},
        ],
    }
    (root / "install.json").write_text(json.dumps(manifest))
    return root


class PlanTests(unittest.TestCase):
    def test_the_family_comes_from_the_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            install = fake_install(Path(directory) / "install", family="tiny-qwen36")
            self.assertEqual(harness.install_family(install), "tiny-qwen36")

    def test_the_expert_count_comes_from_the_stacked_tensor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            install = fake_install(Path(directory) / "install", experts=16)
            self.assertEqual(harness.expert_count(install), 16)

    def test_an_install_without_experts_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            install = Path(directory) / "install"
            install.mkdir(parents=True)
            (install / "install.json").write_text(json.dumps({"family": "x", "tensors": []}))
            with self.assertRaises(SystemExit):
                harness.expert_count(install)

    def test_the_plan_is_total_balanced_and_names_the_install_family(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            install = fake_install(Path(directory) / "install", experts=10)
            path = Path(directory) / "plan.json"
            plan = harness.write_plan(install, 3, path)

            self.assertEqual(len(plan["owners"]), 10, "every expert must be owned")
            self.assertLessEqual(
                max(plan["owners"].count(node) for node in range(3))
                - min(plan["owners"].count(node) for node in range(3)),
                1,
                "the blocks may differ by at most one",
            )
            self.assertEqual(sorted(set(plan["owners"])), [0, 1, 2])
            self.assertEqual(plan["family"], "qwen3_5_moe")
            # Written, so the nodes read the same bytes rather than rebuilding the dict.
            self.assertEqual(json.loads(path.read_text()), plan)

    def test_fewer_experts_than_nodes_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            install = fake_install(Path(directory) / "install", experts=2)
            with self.assertRaises(SystemExit):
                harness.write_plan(install, 4, Path(directory) / "plan.json")


if __name__ == "__main__":
    unittest.main()
