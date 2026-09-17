"""Tests for the version's single source of truth (`RELEASE.md` §1.3).

The tool is the thing that stops a mirror drifting from the authority, so its *refusals* are what matter
and they are what is asserted here. Every case runs against a temporary root, because a test that edits the
real `VERSION` to see what happens is a test that can ship a wrong version.
"""

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
VERSION_TOOL = TOOLS / "version.py"


def run(*arguments: str, root: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(VERSION_TOOL), *arguments, "--root", str(root)],
        capture_output=True, text=True,
    )


class VersionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="tt-version-"))
        (self.root / "VERSION").write_text("1.0.0\n", encoding="utf-8")
        run("--write", root=self.root)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def mirror(self) -> Path:
        return self.root / "sources/DatacenterEngine/Version.swift"

    def testWriteThenCheckAgrees(self) -> None:
        result = run("--check", root=self.root)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("1.0.0", result.stdout)

    def testAMangledMirrorIsRefused(self) -> None:
        # The whole point: editing the number where it is *generated* is how a wrong version would ship.
        self.mirror().write_text(
            self.mirror().read_text().replace('"1.0.0"', '"9.9.9"'), encoding="utf-8"
        )
        result = run("--check", root=self.root)
        self.assertEqual(result.returncode, 1)
        self.assertIn("VERSION MISMATCH", result.stderr)

    def testAMissingMirrorIsRefused(self) -> None:
        self.mirror().unlink()
        self.assertEqual(run("--check", root=self.root).returncode, 1)

    def testAMalformedAuthorityIsRefused(self) -> None:
        for malformed in ["1.0", "v1.0.0", "1.0.0-rc1", "", "one.two.three"]:
            with self.subTest(malformed=malformed):
                (self.root / "VERSION").write_text(malformed + "\n", encoding="utf-8")
                result = run("--check", root=self.root)
                self.assertEqual(result.returncode, 1, f"{malformed!r} must not be a version")

    def testAMissingAuthorityIsRefused(self) -> None:
        (self.root / "VERSION").unlink()
        self.assertEqual(run("--check", root=self.root).returncode, 1)

    def testWritePropagatesABumpWithoutTouchingAnythingElse(self) -> None:
        (self.root / "VERSION").write_text("2.5.1\n", encoding="utf-8")
        self.assertNotIn('"2.5.1"', self.mirror().read_text(), "the mirror is stale until it is written")
        self.assertEqual(run("--write", root=self.root).returncode, 0)
        self.assertIn('"2.5.1"', self.mirror().read_text())
        self.assertEqual(run("--check", root=self.root).returncode, 0)

    def testTheRealTreeIsConsistent(self) -> None:
        # The gate, asked the same question the CI gate asks — and this one reads the real root, which is
        # the only test here that does.
        result = subprocess.run(
            [sys.executable, str(VERSION_TOOL), "--check"], capture_output=True, text=True, cwd=ROOT
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
