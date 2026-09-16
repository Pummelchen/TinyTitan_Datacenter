#!/usr/bin/env python3
"""The source-selection rule has one home, and every contract CLI offers the flag that goes with it.

This is the check that two rounds of fixing did not have. `D69` added `--stream-experts` to one invocation that
had omitted it and `D73` added `--uncached` to the same one, so the M1 gate's checkpoint runs mapped 67 GB while
the gate's own document recorded both flags as the survivable pair. Both times the question *which reader, with
which flag* had been answered wherever it came up, and one of the answers was wrong. These tests make the
answers converge: one function decides, and every contract CLI must expose the flag for it.

Standard library only, like the rest of the CI suite; the byte-identity case runs the real CLI on a fixture.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

FIXTURE = ROOT / "tests" / "DatacenterEngineTests" / "Fixtures" / "tiny-qwen35"
TOKENS = "3,17,5,42,8"

# Every CLI that reads a checkpoint on the contract side, and whether its model family has routed experts.
# `--stream-experts` is a property of the family rather than of the reader: `qwen3_5` is dense and its forward
# has no such parameter, so declaring the flag there would be a no-op that reads like a capability.
CONTRACT_CLIS = (
    ("ordered_qwen35_trace.py", False),
    ("ordered_qwen36_trace.py", True),
)


def help_text(script: str) -> str:
    done = subprocess.run(
        ["python3", str(ROOT / "tools" / script), "--help"], capture_output=True, text=True
    )
    return done.stdout + done.stderr


class SourceArgumentTests(unittest.TestCase):
    def test_every_contract_cli_can_read_a_checkpoint_without_mapping_it(self) -> None:
        for script, _ in CONTRACT_CLIS:
            with self.subTest(script=script):
                self.assertIn(
                    "--uncached", help_text(script),
                    f"{script} cannot be told to read through pread, so a run of it maps the checkpoint",
                )

    def test_stream_experts_is_declared_exactly_where_the_family_has_experts(self) -> None:
        for script, has_experts in CONTRACT_CLIS:
            with self.subTest(script=script):
                declared = "--stream-experts" in help_text(script)
                self.assertEqual(
                    declared, has_experts,
                    f"{script}: --stream-experts declared={declared}, family has experts={has_experts}",
                )


class SelectionTests(unittest.TestCase):
    """The rule itself, against the fixture's real artifacts.

    The readers parse their file at construction, so a placeholder directory is not enough: these use the
    fixture's checkpoint and the fixture's own install, which is what the branches are for.
    """

    def test_an_install_is_read_through_the_install_reader(self) -> None:
        from contract_source import open_source
        from install_source import InstallSource

        self.assertIsInstance(open_source(FIXTURE / "install"), InstallSource)

    def test_the_uncached_flag_selects_the_pread_reader(self) -> None:
        from contract_source import open_source
        from uncached_safetensors import UncachedSafetensorsSource

        self.assertIsInstance(open_source(FIXTURE, uncached=True), UncachedSafetensorsSource)

    def test_the_default_is_still_the_mapped_reader(self) -> None:
        from contract_source import open_source
        from safetensors_source import SafetensorsSource

        self.assertIsInstance(open_source(FIXTURE), SafetensorsSource)

    def test_an_install_wins_over_the_uncached_flag(self) -> None:
        """The install branch is chosen first, so passing both must not change which reader is used.

        `run_m1_gate.py` passes `--uncached` only for a checkpoint and relies on this; if the order were the
        other way round, the flag would silently redirect an install run to a checkpoint reader.
        """
        from contract_source import open_source
        from install_source import InstallSource

        self.assertIsInstance(open_source(FIXTURE / "install", uncached=True), InstallSource)


class ByteIdentityTests(unittest.TestCase):
    def test_pread_and_mmap_produce_the_same_trace(self) -> None:
        """The claim `--uncached` rests on, on a fixture: the reader changes, the numbers do not.

        `tools/test_uncached_safetensors.py` checks the same property on a real 4 GB shard of the checkpoint;
        this runs the whole contract CLI twice, which is what the gates actually invoke.
        """
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            traces = {}
            for flag, label in (([], "mapped"), (["--uncached"], "pread")):
                out = work / label
                done = subprocess.run(
                    [
                        "python3", str(ROOT / "tools" / "ordered_qwen35_trace.py"), str(FIXTURE), str(out),
                        "--spec", str(FIXTURE / "spec.json"), "--tokens", TOKENS, *flag,
                    ],
                    capture_output=True, text=True,
                )
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                traces[label] = {
                    entry["name"]: entry["sha256"]
                    for entry in json.loads((out / "manifest.json").read_text())["tensors"]
                }
            self.assertEqual(traces["mapped"], traces["pread"])


if __name__ == "__main__":
    unittest.main()
