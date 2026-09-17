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


class ForwardedEnvironmentTests(unittest.TestCase):
    """A gate must tell every peer the same measurement switches it told the local node.

    `SHARD_PROFILE` and `SHARD_LAYER_CACHE_MB` are read by the engine from its own environment. A gate that set
    them locally only would profile one node of four and report the rest as unprofiled — or hand one node a
    different cache budget and call the difference a cluster result (`D90`).
    """

    def test_nothing_is_forwarded_when_nothing_was_asked_for(self) -> None:
        self.assertEqual(harness.forwarded_environment({}), {})
        self.assertEqual(harness.environment_prefix({}), "")

    def test_the_measurement_switches_are_forwarded(self) -> None:
        forwarded = harness.forwarded_environment(
            {"SHARD_PROFILE": "1", "SHARD_LAYER_CACHE_MB": "256", "PATH": "/usr/bin"}
        )
        self.assertEqual(forwarded, {"SHARD_PROFILE": "1", "SHARD_LAYER_CACHE_MB": "256"})

    def test_the_prefix_is_sorted_so_a_command_is_a_stable_string(self) -> None:
        prefix = harness.environment_prefix({"SHARD_PROFILE": "1", "SHARD_LAYER_CACHE_MB": "256"})
        self.assertEqual(prefix, "SHARD_LAYER_CACHE_MB=256 SHARD_PROFILE=1 ")

    def test_an_empty_value_is_not_forwarded_as_a_switch(self) -> None:
        # `SHARD_PROFILE=` means "not set" to the engine, so passing it on would say nothing either way; it is
        # omitted rather than sent as an empty switch.
        self.assertEqual(harness.forwarded_environment({"SHARD_PROFILE": ""}), {})


class RemoteInstallPathTests(unittest.TestCase):
    """The path a peer is launched with must be the path that was staged.

    It was not. Both gates copied the install into the remote directory under its own name and then launched
    the node with `./install` — a name that only matches when the local install happens to be called
    `install`. The default is the one the gate docstrings show a reader, so the documented command copied
    21.7 GB to every peer and had every peer fail on a missing `install/install.json` (`D83`). These tests
    pin the two halves together, because the defect was the two halves disagreeing.
    """

    def test_the_default_is_the_name_that_was_staged(self) -> None:
        self.assertEqual(harness.remote_install_path(Path(".build/m1-install"), None), "./m1-install")

    def test_an_explicit_remote_install_is_used_exactly_as_given(self) -> None:
        named = "~/Downloads/m1-install"
        self.assertEqual(harness.remote_install_path(Path(".build/m1-install"), named), named)

    def test_a_differently_named_install_does_not_inherit_the_old_default(self) -> None:
        """The old default returned `./install` whatever the install was called, which is the whole bug."""
        self.assertEqual(harness.remote_install_path(Path("/tmp/x/35b-install"), None), "./35b-install")

    def test_neither_gate_spells_the_install_directory_name_itself(self) -> None:
        """A hardcoded name is how the two halves drifted apart; the name belongs in exactly one place."""
        for name in ("run_m2_gate.py", "run_m3_gate.py"):
            source = (Path(__file__).resolve().parent / name).read_text()
            self.assertNotIn(
                '"./install"', source,
                f"{name} names the remote install directory itself; it must ask remote_install_path",
            )


class DistributionTests(unittest.TestCase):
    """The plan's two splits, because the choice is now a measurement rather than a default.

    `D92` measured the cluster's exchange as 99.6% **receive** — nodes waiting for peers to have something to
    send — so whether a contiguous block of expert ids carries a skewed share of the routing is a question with
    a cheap test: interleave them and look.
    """

    def plan(self, distribution: str, nodes: int = 4, experts: int = 8) -> dict:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        install = Path(directory.name) / "install"
        install.mkdir()
        (install / "install.json").write_text(
            # The manifest is the authority on the expert count: `expert_count` reads it from the
            # stacked tensor's leading dimension rather than trusting the caller.
            json.dumps({
                "family": "tiny",
                "tensors": [{"role": "expert.stack_gate_up", "shape": [experts, 2, 4]}],
            })
        )
        return harness.write_plan(install, nodes, install / "plan.json", distribution)

    def test_contiguous_hands_each_node_a_block(self) -> None:
        self.assertEqual(self.plan("contiguous")["owners"], [0, 0, 1, 1, 2, 2, 3, 3])

    def test_round_robin_interleaves_them(self) -> None:
        self.assertEqual(self.plan("round-robin")["owners"], [0, 1, 2, 3, 0, 1, 2, 3])

    def test_both_record_what_they_are(self) -> None:
        # The plan is data a node validates, so a split that did not name itself would be unusable.
        self.assertEqual(self.plan("contiguous")["distribution"], "contiguous")
        self.assertEqual(self.plan("round-robin")["distribution"], "round-robin")

    def test_an_unknown_split_is_refused_rather_than_defaulted(self) -> None:
        with self.assertRaises(SystemExit):
            self.plan("interleaved")


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


class PerNodeReportTests(unittest.TestCase):
    """`DC-081`: each node's own numbers, and NOT REPORTED when a node wrote none."""

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.original = harness.OUT
        harness.OUT = self.root
        self.addCleanup(lambda: setattr(harness, "OUT", self.original))

    def test_a_node_without_metrics_is_reported_not_silently_zero(self) -> None:
        import contextlib
        import io

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            harness.report_per_node(range(2))
        printed = output.getvalue()
        self.assertIn("node 0: NOT REPORTED", printed)
        self.assertIn("node 1: NOT REPORTED", printed)

    def test_a_node_with_metrics_reports_what_it_measured(self) -> None:
        import contextlib
        import io
        import json

        directory = self.root / "node-0"
        directory.mkdir(parents=True)
        (directory / "metrics.json").write_text(
            json.dumps(
                {
                    "expert_requests": 2218,
                    "install_bytes_read_this_forward": 3_060_562_432,
                    "dense_payload_bytes_read": 1_043_708_416,
                    "dense_payload_cache_hits": 4277,
                    "exchange_reduces": 40,
                    "exchange_terms_sent": 120,
                    "exchange_terms_received": 118,
                    "exchange_seconds": 0.25,
                }
            )
        )
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            harness.report_per_node([0])
        printed = output.getvalue()
        self.assertIn("2,218 request(s)", printed)
        self.assertIn("1,043,708,416 B dense", printed)
        self.assertIn("4,277 cache hit(s)", printed)
        self.assertIn("120/118 terms", printed)
        self.assertIn("0.250 s in the all-reduce", printed)
