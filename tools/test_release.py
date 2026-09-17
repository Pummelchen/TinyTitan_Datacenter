"""Tests for the release script's refusals (`RELEASE.md` §1.8).

§1.8's rule is that publishing **refuses** unless the notes carry the checksum placeholder or quote the real
value, "because a release quoting the wrong digest is worse than one quoting none". A refusal that has never
been seen to refuse is not yet trusted, so every branch of it is asserted here — with the real `CHANGELOG.md`
left alone, because a test that edits it is a test that can ship wrong notes.
"""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import release


class NotesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = Path(tempfile.mkdtemp(prefix="tt-release-"))
        self.archive = self.directory / release.archive_name("1.0.0")
        self.archive.write_bytes(b"an archive, standing in for the real one")
        self.digest = hashlib.sha256(self.archive.read_bytes()).hexdigest()
        self.size = self.archive.stat().st_size
        self.changelog = self.directory / "CHANGELOG.md"
        self.original = release.CHANGELOG
        release.CHANGELOG = self.changelog
        self.notes = self.directory / "notes.md"

    def tearDown(self) -> None:
        release.CHANGELOG = self.original
        import shutil

        shutil.rmtree(self.directory, ignore_errors=True)

    def changelog_with(self, body: str) -> None:
        self.changelog.write_text(body, encoding="utf-8")

    def testPlaceholdersAreSubstitutedWithTheRealDigestAndSize(self) -> None:
        self.changelog_with(
            "## [1.0.0] — 2026-09-17\n\n### Checksums\n\n```\nSHA256_PENDING  x.tar.gz\n"
            "ARCHIVE_BYTES_PENDING  bytes\n```\n"
        )
        digest = release.notes(self.archive, "1.0.0", self.notes)
        written = self.notes.read_text(encoding="utf-8")
        self.assertEqual(digest, self.digest)
        self.assertIn(self.digest, written)
        self.assertIn(str(self.size), written)
        self.assertNotIn("PENDING", written)

    def testNotesQuotingTheRightDigestAreAcceptedWithoutAPlaceholder(self) -> None:
        self.changelog_with(f"## [1.0.0]\n\nChecksum: {self.digest}\n")
        self.assertEqual(release.notes(self.archive, "1.0.0", self.notes), self.digest)

    def testNotesQuotingNoDigestAndNoPlaceholderAreRefused(self) -> None:
        self.changelog_with("## [1.0.0]\n\nNo checksum here at all.\n")
        with self.assertRaises(release.Refused):
            release.notes(self.archive, "1.0.0", self.notes)

    def testNotesQuotingTheWrongDigestAreRefused(self) -> None:
        self.changelog_with(f"## [1.0.0]\n\nChecksum: {'0' * 64}\n")
        with self.assertRaises(release.Refused):
            release.notes(self.archive, "1.0.0", self.notes)

    def testAMissingVersionSectionIsRefused(self) -> None:
        self.changelog_with("## [0.9.0]\n\nSHA256_PENDING\n")
        with self.assertRaises(release.Refused):
            release.notes(self.archive, "1.0.0", self.notes)

    def testOneVersionsSectionStopsAtTheNext(self) -> None:
        self.changelog_with(
            "## [1.0.0]\n\nSHA256_PENDING\nARCHIVE_BYTES_PENDING\n\n## [0.9.0]\n\nolder, and not in these notes\n"
        )
        release.notes(self.archive, "1.0.0", self.notes)
        written = self.notes.read_text(encoding="utf-8")
        self.assertNotIn("older, and not in these notes", written)

    def testBundlesComeFromThePlanAndExcludeTestTargets(self) -> None:
        # The first published release carried the two *test* bundles: 2 MB of fixtures in `bin/`, and a wrong
        # story about what the executables need. Reachability from an executable is what decides it.
        plan = json.dumps({
            "name": "TinyTitanDatacenter",
            "targets": [
                {"name": "datacenter-generate", "type": "executable",
                 "dependencies": [{"byName": ["DatacenterEngine"]}], "resources": []},
                {"name": "DatacenterEngine", "type": "library",
                 "dependencies": [{"byName": ["DatacenterIR"]}], "resources": []},
                {"name": "DatacenterIR", "type": "library", "dependencies": [],
                 "resources": [{"path": "Shaders", "rule": {"copy": {}}}]},
                {"name": "DatacenterEngineTests", "type": "test",
                 "dependencies": [{"byName": ["DatacenterEngine"]}],
                 "resources": [{"path": "Fixtures", "rule": {"copy": {}}}]},
            ],
        })
        self.assertEqual(release.declared_bundles(plan), ["TinyTitanDatacenter_DatacenterIR.bundle"])

    def testAnExecutableThatDeclaresNoResourcesNeedsNoBundle(self) -> None:
        plan = json.dumps({"name": "P", "targets": [
            {"name": "datacenter-generate", "type": "executable", "dependencies": [], "resources": []},
        ]})
        self.assertEqual(release.declared_bundles(plan), [])

    def testTheArchiveNameFollowsTheConvention(self) -> None:
        # §1.6's shape, asserted rather than eyeballed: `<project>-<version>-macos-arm64.tar.gz`.
        self.assertEqual(release.archive_name("1.0.0"), "TinyTitan_Datacenter-1.0.0-macos-arm64.tar.gz")

    def testTheRealChangelogHasA1_0_0SectionWithThePlaceholders(self) -> None:
        # The one test that reads the real notes: they exist, they name this version, and they carry the
        # placeholders the release substitutes.
        text = self.original.read_text(encoding="utf-8")
        self.assertIn("## [1.0.0]", text)
        self.assertIn(release.SHA256_PLACEHOLDER, text)
        self.assertIn(release.BYTES_PLACEHOLDER, text)


if __name__ == "__main__":
    unittest.main()
