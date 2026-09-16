"""Tests for the provenance repair.

The digests are computed here with `hashlib`, independently of the tool, so the assertion is about the value
that lands in the artifact rather than about the tool agreeing with itself. The rest of the manifest is
compared canonical-JSON-equal before and after, which is the property the repair claims: `source` and
`passes` move, nothing else does.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from repair_install_provenance import (  # noqa: E402
    PASS_NAME,
    RepairError,
    discover_snapshot,
    main,
    repo_from_cache,
    repair,
)

REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"


class Fixture:
    """A cache layout and an install manifest, small enough to reason about by hand."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.snapshot = root / "hf-cache" / "models--Qwen--Qwen3.6-35B-A3B" / "snapshots" / REVISION
        self.snapshot.mkdir(parents=True)
        (self.snapshot / "config.json").write_text("{}")
        self.shards = {}
        for name, content in (("model-00001-of-00002.safetensors", b"first"), ("model-00002-of-00002.safetensors", b"second")):
            (self.snapshot / name).write_bytes(content)
            self.shards[name] = hashlib.sha256(content).hexdigest()
        self.install = root / "m1-install"
        self.install.mkdir()
        self.manifest_path = self.install / "install.json"
        self.manifest = {
            "schema": 1,
            "family": "qwen3_5_moe",
            "source": {"files": {}, "repo": REVISION, "revision": "local", "spec_family": "qwen3_5_moe"},
            "passes": ["quantize-group-affine-int4"],
            "tensors": [{"name": "a", "sha256": "0" * 64}],
        }
        self.manifest_path.write_text(json.dumps(self.manifest))

    def read(self) -> dict:
        return json.loads(self.manifest_path.read_text())

    def untouched(self, before: dict, after: dict) -> bool:
        keys = (set(before) | set(after)) - {"source", "passes"}
        return all(json.dumps(before.get(k), sort_keys=True) == json.dumps(after.get(k), sort_keys=True) for k in keys)


class RepairTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_repairs_a_placeholder_source_and_touches_nothing_else(self) -> None:
        fixture = Fixture(self.root)
        before = fixture.read()
        result = repair(fixture.install, fixture.snapshot, None, dry_run=False)
        after = fixture.read()
        self.assertTrue(result["changed"])
        self.assertEqual(after["source"]["files"], fixture.shards)
        self.assertEqual(after["source"]["repo"], "Qwen/Qwen3.6-35B-A3B")
        self.assertEqual(after["source"]["revision"], REVISION)
        self.assertEqual(after["source"]["spec_family"], "qwen3_5_moe")
        self.assertIn(PASS_NAME, after["passes"])
        self.assertEqual(after["passes"][0], "quantize-group-affine-int4")
        self.assertTrue(fixture.untouched(before, after))
        self.assertTrue((self.root / "reverify" / f"m1-install.install.json.before-{PASS_NAME}").exists())

    def test_a_second_run_is_a_no_op(self) -> None:
        fixture = Fixture(self.root)
        repair(fixture.install, fixture.snapshot, None, dry_run=False)
        once = fixture.read()
        result = repair(fixture.install, fixture.snapshot, None, dry_run=False)
        self.assertFalse(result["changed"])
        self.assertEqual(fixture.read(), once)

    def test_refuses_when_the_recorded_digests_are_not_the_source_on_disk(self) -> None:
        fixture = Fixture(self.root)
        manifest = fixture.read()
        manifest["source"]["files"] = {"model-00001-of-00002.safetensors": "f" * 64}
        fixture.manifest_path.write_text(json.dumps(manifest))
        with self.assertRaises(RepairError) as caught:
            repair(fixture.install, fixture.snapshot, None, dry_run=False)
        self.assertIn("not the same pair", str(caught.exception))
        self.assertEqual(fixture.read()["source"]["files"], {"model-00001-of-00002.safetensors": "f" * 64})

    def test_dry_run_writes_nothing(self) -> None:
        fixture = Fixture(self.root)
        before = fixture.read()
        result = repair(fixture.install, fixture.snapshot, None, dry_run=True)
        self.assertFalse(result["changed"])
        self.assertEqual(fixture.read(), before)
        self.assertFalse((self.root / "reverify").exists())

    def test_refuses_a_snapshot_without_weights(self) -> None:
        fixture = Fixture(self.root)
        for path in fixture.snapshot.glob("*.safetensors"):
            path.unlink()
        with self.assertRaises(RepairError) as caught:
            repair(fixture.install, fixture.snapshot, None, dry_run=False)
        self.assertIn("no *.safetensors", str(caught.exception))

    def test_the_cli_refuses_rather_than_traces_an_unknown_report(self) -> None:
        # `--install` naming a directory with no manifest: exit 1 and a message, not a traceback.
        fixture = Fixture(self.root)
        empty = self.root / "empty"
        empty.mkdir()
        code = main(["--install", str(empty), "--snapshot", str(fixture.snapshot), "--dry-run"])
        self.assertEqual(code, 1)


class DiscoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_repo_comes_from_the_cache_directory_name(self) -> None:
        fixture = Fixture(self.root)
        self.assertEqual(repo_from_cache(fixture.snapshot), "Qwen/Qwen3.6-35B-A3B")

    def test_repo_from_cache_refuses_a_path_that_is_not_a_cache(self) -> None:
        with self.assertRaises(RepairError):
            repo_from_cache(self.root / "elsewhere" / "snapshots" / REVISION)

    def test_discovers_the_single_snapshot(self) -> None:
        fixture = Fixture(self.root)
        self.assertEqual(discover_snapshot(self.root / "hf-cache", None), fixture.snapshot)

    def test_refuses_when_there_is_no_snapshot_to_find(self) -> None:
        with self.assertRaises(RepairError):
            discover_snapshot(self.root / "nothing-here", None)


if __name__ == "__main__":
    unittest.main()
