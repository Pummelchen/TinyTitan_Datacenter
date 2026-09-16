"""Tests for `tools/verify_install.py` (`DC-108`).

The verifier's own claims need checking, and the two that matter are the ones a per-tensor reader
cannot make: that the tensors **tile** the payload (so no tensor is reading another's bytes), and that
each role's quantisation is the one the policy requires. Both are tested against a payload that is wrong
on purpose.
"""

from __future__ import annotations

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from install_reader import Install  # noqa: E402
from test_install_reader import InstallFixture  # noqa: E402
from verify_install import coverage, main, policy_for  # noqa: E402


def two_tensors(root: Path) -> InstallFixture:
    fixture = InstallFixture(root)
    fixture.add(
        "a",
        {"role": "embed", "dtype": "fp32", "quant": "fp32", "group": 0,
         "padded_columns": 2, "shape": [1, 2]},
        struct.pack("<2f", 1.0, 2.0),
    )
    fixture.add(
        "b",
        {"role": "head.lm", "dtype": "fp32", "quant": "fp32", "group": 0,
         "padded_columns": 2, "shape": [1, 2]},
        struct.pack("<2f", 3.0, 4.0),
    )
    return fixture


class CoverageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_a_tiling_payload_has_no_problems(self) -> None:
        install = two_tensors(self.root).write()
        problems, count = coverage(install)
        self.assertEqual(problems, [])
        self.assertEqual(count, 2)

    def test_a_gap_is_reported(self) -> None:
        # Eight filler bytes between the two tensors, so `b` still fits: shifting `b` without growing
        # the payload trips the container's bounds guard instead, which is a different check. The
        # fixture is built here rather than with `two_tensors` because that helper already adds `b`,
        # and adding it twice is (correctly) refused as a duplicate name.
        fixture = InstallFixture(self.root)
        fixture.add(
            "a",
            {"role": "embed", "dtype": "fp32", "quant": "fp32", "group": 0,
             "padded_columns": 2, "shape": [1, 2]},
            struct.pack("<2f", 1.0, 2.0),
        )
        fixture.payload.extend(b"\x00" * 8)
        fixture.add(
            "b",
            {"role": "head.lm", "dtype": "fp32", "quant": "fp32", "group": 0,
             "padded_columns": 2, "shape": [1, 2]},
            struct.pack("<2f", 3.0, 4.0),
        )
        install = fixture.write()
        problems, _ = coverage(install)
        self.assertTrue(any("starts at" in problem for problem in problems), problems)

    def test_an_overlap_is_reported(self) -> None:
        fixture = two_tensors(self.root)
        fixture.tensors[1]["offset"] = 4  # b starts inside a
        install = fixture.write()
        problems, _ = coverage(install)
        self.assertTrue(any("overlaps" in problem for problem in problems), problems)

    def test_a_payload_longer_than_its_tensors_is_reported(self) -> None:
        fixture = two_tensors(self.root)
        install = fixture.write()
        with open(install.data_path, "ab") as handle:
            handle.write(b"\x00\x00")
        install = Install(self.root, uncached=False)
        problems, _ = coverage(install)
        self.assertTrue(any("the payload is" in problem for problem in problems), problems)


class PolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_the_policy_is_read_from_the_manifest_or_the_project_default(self) -> None:
        fixture = two_tensors(self.root)
        fixture.write()
        policy = policy_for(self.root)
        self.assertTrue(policy, "the project's quant policy should have roles")
        self.assertTrue(all(isinstance(role, str) for role in policy))

    def test_the_policy_is_found_beside_a_copied_tool(self) -> None:
        """A copied install travels without the policy's absolute path, so the sibling file must win.

        This is the defect the first remote verification hit: the tool ran on another node, the manifest's
        `/Users/<builder>/tools/quant_policy.json` did not exist there, and the fallback looked in a
        *repository* layout that was not there either.
        """
        fixture = two_tensors(self.root)
        fixture.write()
        policy = json.loads((Path(__file__).resolve().parent / "quant_policy.json").read_text())
        self.assertTrue(policy.get("quant"))
        # `script_dir` is injected so the lookup can be tested where the script actually sits.
        self.assertEqual(policy_for(self.root, script_dir=Path(__file__).resolve().parent), policy["quant"])

    def test_a_role_missing_from_the_policy_fails_the_run(self) -> None:
        fixture = two_tensors(self.root)
        fixture.tensors[0]["role"] = "role.that.does.not.exist"
        fixture.write()
        self.assertEqual(main([str(self.root)]), 1)

    def test_a_quantisation_the_policy_forbids_fails_the_run(self) -> None:
        """The same role the policy knows, given a different quantisation than it requires."""
        fixture = two_tensors(self.root)
        fixture.write()
        policy = policy_for(self.root)
        role = sorted(policy)[0]
        required = policy[role]
        wrong = next(candidate for candidate in ("fp32", "bf16", "int4-affine")
                     if candidate != required)
        fixture.tensors[0]["role"] = role
        fixture.tensors[0]["quant"] = wrong
        fixture.tensors[0]["dtype"] = "fp32" if wrong == "fp32" else "bf16"
        fixture.write()
        self.assertEqual(main([str(self.root)]), 1)

    def test_a_good_install_passes_and_writes_a_report(self) -> None:
        fixture = two_tensors(self.root)
        fixture.write()
        policy = policy_for(self.root)
        # A role the policy pins to a dense quantisation, so the fixture's payload stays simple.
        role = next(
            (name for name in sorted(policy) if policy[name] in ("fp32", "bf16")), None
        )
        if role is None:
            self.skipTest("the policy has no dense role to build a passing fixture from")
        for entry in fixture.tensors:
            entry["role"] = role
            entry["quant"] = policy[role]
            entry["dtype"] = "fp32" if policy[role] == "fp32" else "bf16"
        # A bf16 tensor needs its payload rebuilt at half the width, so keep the payload fp32 and the
        # declaration honest by only using an fp32 role here.
        if policy[role] != "fp32":
            self.skipTest(f"{role} is {policy[role]}, and this fixture's payload is fp32")
        fixture.write()
        report = self.root / "report.json"
        self.assertEqual(main([str(self.root), "--digests", "--json", str(report)]), 0)
        written = json.loads(report.read_text())
        self.assertEqual(written["problem_count"], 0)
        self.assertEqual(written["hashed"], 2)
        self.assertEqual(written["tensors"], 2)


if __name__ == "__main__":
    unittest.main()
