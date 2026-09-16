#!/usr/bin/env python3
"""Check M1's gate instrument on the tiny fixture, so the gate is never untested code.

The gate itself runs on the 67 GB checkpoint; this runs the *same script* on the 236 KB one.
The frozen prompt set cannot be used here — its token ids come from the real tokenizer and are
outside the tiny model's 128-token vocabulary, which is a property of the fixture and not a
defect — so the test writes a fixture-sized prompt file and drives the gate with it.

Standard library only, like the rest of the CI suite: it builds the engine and runs it, so it
needs a Swift toolchain and skips cleanly without one.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import os
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "DatacenterEngineTests" / "Fixtures" / "tiny-qwen36"

def swift_version() -> tuple[int, ...] | None:
    """The toolchain's version, or None when there is no toolchain.

    The package is `swift-tools-version:6.4`, so a runner with 6.3 cannot build it at all
    (`DC-036`: GitHub's `macos-26` image ships Xcode 26.x). The gate script itself is right to
    fail there — it cannot measure anything without an engine — so the *test* skips instead,
    the same way the Swift CI job prints its toolchain and gates only at 6.4 or above.
    """
    if shutil.which("swift") is None:
        return None
    try:
        output = subprocess.run(["swift", "--version"], capture_output=True, text=True).stdout
    except OSError:
        return None
    for word in output.replace("(", " ").split():
        parts = word.split(".")
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            return tuple(int(part) for part in parts if part.isdigit())
    return None


VERSION = swift_version()
HAVE_SWIFT = VERSION is not None and VERSION >= (6, 4)
swift_required = unittest.skipUnless(
    HAVE_SWIFT,
    f"the package needs swift-tools-version 6.4; this runner has {VERSION} (DC-036)",
)


@swift_required
class GateInstrumentTests(unittest.TestCase):
    def build(self) -> tuple[Path, Path]:
        """A fixture-sized prompt set, and a built engine."""
        golden = json.loads((FIXTURE / "golden.json").read_text())
        first = golden["tokens"]
        second = list(reversed(golden["tokens"]))
        work = Path(tempfile.mkdtemp(prefix="m1-gate-test-"))
        prompts = work / "prompts.json"
        prompts.write_text(json.dumps({
            "note": "Fixture-sized prompts for testing the gate instrument; not the gate's set.",
            "schema": 1,
            "prompts": [
                {"id": "short", "text": "", "decoded": "", "tokens": first, "length": len(first)},
                {"id": "reversed", "text": "", "decoded": "", "tokens": second, "length": len(second)},
            ],
        }, indent=1) + "\n")
        built = subprocess.run(
            ["swift", "build", "-c", "release"], cwd=ROOT, capture_output=True, text=True
        )
        self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
        return work, prompts

    def testTheGatePassesOnAFixtureThatMatchesItself(self):
        work, prompts = self.build()
        result = subprocess.run(
            [
                "python3", str(ROOT / "tools" / "run_m1_gate.py"),
                "--snapshot", str(FIXTURE), "--work", str(work / "gate"),
                "--prompts", str(prompts), "--model", "tiny-fixture", "--revision", "generated",
                "--skip-generation",
            ],
            cwd=ROOT, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads((work / "gate" / "report.json").read_text())
        self.assertEqual(report["spec_family"], "qwen3_5_moe")
        self.assertTrue(report["identical"], "the engine must reproduce the contract on every prompt")
        self.assertEqual(len(report["results"]), 2)
        for entry in report["results"]:
            self.assertTrue(entry["identical"], entry["id"])
            # The traffic counters are the third part of M1's gate, so they must be there.
            self.assertIsNotNone(entry["expert_requests"], f"{entry['id']}: expert requests")
            self.assertIsNotNone(entry["expert_hit_rate"], f"{entry['id']}: hit rate")

    # This case found DC-113 and then confirmed its fix: the install source mapped one expert to 16 install
    # rows where the quantiser writes 32, which the real model's geometry hides. It is the only test that
    # drives the install path, and it is ordinary now rather than an expected failure.
    def testTheGateRunsAgainstATinyInstallAndWouldHaveCaughtTheMissingFlag(self):
        """The install is the source that refuses to materialise an expert stack.

        The gate's contract half must therefore stream experts (`--stream-experts`), because a single expert
        layer is gigabytes; without the flag the contract dies with an `InstallSourceError` and the gate
        cannot run against an install at all. Nothing asked it to until `D69`, which is how the flag came to
        be missing while the install source's own docstring said the contract used it. The checkpoint case
        above cannot catch that: a checkpoint source does not refuse stacks. This case is the one that has to
        exist, and it is the reason the gate can now run on the 8 GB node without mapping a 70 GB file.
        """
        work, prompts = self.build()
        spec = work / "spec.json"
        emitted = subprocess.run(
            [str(ROOT / ".build" / "release" / "datacenter-trace"), "--emit-spec", str(spec), str(FIXTURE)],
            cwd=ROOT, capture_output=True, text=True,
        )
        self.assertEqual(emitted.returncode, 0, emitted.stdout + emitted.stderr)
        install = work / "install"
        built = subprocess.run(
            [
                "python3", str(ROOT / "tools" / "quantize.py"), "build",
                "--spec", str(spec), str(FIXTURE), str(install),
            ],
            cwd=ROOT, capture_output=True, text=True,
        )
        self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
        result = subprocess.run(
            [
                "python3", str(ROOT / "tools" / "run_m1_gate.py"),
                "--snapshot", str(install), "--work", str(work / "gate-install"),
                "--prompts", str(prompts), "--model", "tiny-fixture", "--revision", "generated",
                "--skip-generation",
            ],
            cwd=ROOT, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads((work / "gate-install" / "report.json").read_text())
        self.assertTrue(report["identical"], "the engine must reproduce the contract reading the install")

    def testTheGateRefusesAFamilyItIsNotFor(self):
        """M1's gate is for M1's model. Running it on a `qwen3_5` checkpoint must stop rather
        than measure something else and label it M1."""
        work = Path(tempfile.mkdtemp(prefix="m1-gate-refuse-"))
        snapshot = work / "checkpoint"
        snapshot.mkdir()
        (snapshot / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
        result = subprocess.run(
            [
                "python3", str(ROOT / "tools" / "run_m1_gate.py"),
                "--snapshot", str(snapshot), "--work", str(work / "gate"), "--skip-generation",
            ],
            cwd=ROOT, capture_output=True, text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("refusing", result.stderr)


# A test that runs a heavy entry point must not take the **production** lock: point it at a temporary
# file for the whole module. The guard stays enabled; only its location moves.
_LOCK_DIR = tempfile.TemporaryDirectory()
os.environ["HEAVY_JOB_LOCK"] = str(Path(_LOCK_DIR.name) / "HEAVY_JOB_LOCK")


if __name__ == "__main__":
    unittest.main()
