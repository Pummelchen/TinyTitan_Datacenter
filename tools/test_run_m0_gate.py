#!/usr/bin/env python3
"""Check M0's gate instrument on the tiny fixture, so the gate is never untested code.

M1, M2 and M3's gate scripts each have one of these; M0's did not, which mattered more than a coverage
percentage when `D74` changed the invocation this gate makes — adding `--uncached` to the contract it runs — and
nothing was watching it. A gate that cannot be tested is a gate whose invocation can drift, and the invocation is
exactly what went wrong twice in the M1 gate (`D69`, `D73`).

The gate itself runs on the real 2B checkpoint; this runs the **same script** on the 236 KB one, so both halves of
it are exercised: byte equality against the contract, and the discrete decisions against the reference
implementation through `tools/trace_capture.py`. The frozen prompt set cannot be used here — its token ids are
outside the tiny model's vocabulary, which is a property of the fixture and not a defect — so the test writes a
fixture-sized prompt file and drives the gate with it.

Standard library only, like the rest of the CI suite: it builds the engine and runs it, so it needs a Swift
toolchain, and the oracle half needs the venv, so it skips cleanly without either.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "DatacenterEngineTests" / "Fixtures" / "tiny-qwen35"


def swift_version() -> tuple[int, ...] | None:
    """The toolchain's version, or None when there is no toolchain.

    The package is `swift-tools-version:6.4`, so a runner with 6.3 cannot build it at all (`DC-036`). The gate
    script itself is right to fail there — it cannot measure anything without an engine — so the *test* skips
    instead, the same way the Swift CI job gates only at 6.4 or above.
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
        tokens = golden["tokens"]
        work = Path(tempfile.mkdtemp(prefix="m0-gate-test-"))
        prompts = work / "prompts.json"
        prompts.write_text(json.dumps({
            "note": "Fixture-sized prompts for testing the gate instrument; not the gate's set.",
            "schema": 1,
            "prompts": [
                {"id": "short", "text": "", "decoded": "", "tokens": tokens, "length": len(tokens)},
                {"id": "reversed", "text": "", "decoded": "", "tokens": list(reversed(tokens)),
                 "length": len(tokens)},
            ],
        }, indent=1) + "\n")
        built = subprocess.run(
            ["swift", "build", "-c", "release", "--product", "datacenter-trace"],
            cwd=ROOT, capture_output=True, text=True,
        )
        self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
        return work, prompts

    def testTheGatePassesOnAFixtureThatMatchesItself(self) -> None:
        work, prompts = self.build()
        result = subprocess.run(
            [
                "python3", str(ROOT / "tools" / "run_m0_gate.py"),
                "--snapshot", str(FIXTURE), "--work", str(work / "gate"),
                "--prompts", str(prompts), "--model", "tiny-fixture", "--revision", "generated",
            ],
            cwd=ROOT, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads((work / "gate" / "report.json").read_text())
        self.assertTrue(report["passed"], "the gate must pass on a fixture that matches itself")
        self.assertEqual(len(report["prompts"]), 2)
        for entry in report["prompts"]:
            self.assertTrue(entry["contract_identical"], f"{entry['id']}: engine against the contract")
            self.assertTrue(entry["oracle_discrete_match"], f"{entry['id']}: against the reference")
            # The oracle half is a *numeric* comparison with a discrete assertion on top, so both are checked:
            # a margin of zero would mean the argmax is a coin toss that happened to land right.
            self.assertGreater(entry["oracle_smallest_margin"], 0.0, f"{entry['id']}: margin")

    def test_the_report_records_the_flags_the_contract_was_given(self) -> None:
        """The invocation is evidence, and this is the assertion that keeps it that way.

        `D74` gave the 2B contract a `--uncached` flag and made this gate pass it, because without it the run
        goes through `safe_open`, which maps the checkpoint. Nothing pinned that: the fixture is small enough
        that a mapped read changes no number, so a byte-identity test cannot see the flag disappear. The report
        records the flags, so this can.
        """
        work, prompts = self.build()
        result = subprocess.run(
            [
                "python3", str(ROOT / "tools" / "run_m0_gate.py"),
                "--snapshot", str(FIXTURE), "--work", str(work / "gate"),
                "--prompts", str(prompts), "--model", "tiny-fixture", "--revision", "generated",
                "--only", "short",
            ],
            cwd=ROOT, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads((work / "gate" / "report.json").read_text())
        self.assertIn("--uncached", report["contract_flags"])


if __name__ == "__main__":
    unittest.main()
