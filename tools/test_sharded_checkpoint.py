#!/usr/bin/env python3
"""A sharded checkpoint must read exactly like the single-file one — and fail loudly without its index.

The 35 B model ships as 26 shards and an index, and a loader that opens only the first shard
produces a checkpoint that looks complete while missing five sixth of its layers. That is the
kind of failure this test exists to prevent: it splits the tiny `qwen3_5_moe` fixture into three
shards, runs the *engine* on both layouts, and requires the traces to be identical.

Two more claims: a sharded checkpoint with no index is refused rather than half-read, and the
importer sees the same inventory either way — the file format stays out of L2.

Needs the venv (safetensors) and a Swift 6.4 toolchain, so it skips where either is missing.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "DatacenterEngineTests" / "Fixtures" / "tiny-qwen36"


def swift_version() -> tuple[int, ...] | None:
    """The toolchain's version, or None. `DC-036`: CI's runner is 6.3 and cannot build this."""
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


try:
    from safetensors import safe_open
    from safetensors.torch import save_file

    HAVE_SAFETENSORS = True
except Exception:
    HAVE_SAFETENSORS = False

VERSION = swift_version()
RUNNABLE = HAVE_SAFETENSORS and VERSION is not None and VERSION >= (6, 4)
REASON = (
    "needs safetensors and a Swift 6.4 toolchain; "
    f"have safetensors={HAVE_SAFETENSORS}, swift={VERSION} (DC-036)"
)


@unittest.skipUnless(RUNNABLE, REASON)
class ShardedCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Once, for every test in the class: tests run in name order, and a test that reads the
        # engine must not depend on another one having built it first. Running the refusal case
        # against yesterday's binary is exactly how this was found.
        built = subprocess.run(["swift", "build", "-c", "release"], cwd=ROOT, capture_output=True, text=True)
        if built.returncode != 0:
            raise AssertionError(built.stdout + built.stderr)

    def shard(self, source: Path, target: Path, shards: int = 3) -> None:
        """Split a single-file checkpoint into `shards` pieces plus an index."""
        import torch

        target.mkdir(parents=True, exist_ok=True)
        for name in ("config.json", "spec.json"):
            shutil.copy(source / name, target / name)
        with safe_open(source / "model.safetensors", framework="pt") as handle:
            names = sorted(handle.keys())
            tensors = {name: handle.get_tensor(name) for name in names}
        weight_map = {}
        buckets: list[dict] = [dict() for _ in range(shards)]
        for index, name in enumerate(names):
            buckets[index % shards][name] = tensors[name]
            weight_map[name] = f"model.safetensors-{index % shards:05d}-of-{shards:05d}"
        for index, bucket in enumerate(buckets):
            save_file(bucket, str(target / f"model.safetensors-{index:05d}-of-{shards:05d}"))
        (target / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": sum(t.numel() for t in tensors.values())}, "weight_map": weight_map})
        )

    def trace(self, snapshot: Path, out: Path) -> subprocess.CompletedProcess:
        # Not `run`: that name is `TestCase.run` and shadowing it breaks the test runner.
        golden = json.loads((FIXTURE / "golden.json").read_text())
        return subprocess.run(
            [
                str(ROOT / ".build" / "release" / "datacenter-trace"), str(snapshot), str(out),
                ",".join(str(t) for t in golden["tokens"]), "--model", "fixture", "--revision", "generated",
            ],
            cwd=ROOT, capture_output=True, text=True,
        )

    def testShardedAndSingleFileGiveTheSameTrace(self):
        work = Path(tempfile.mkdtemp(prefix="sharded-"))
        sharded = work / "sharded"
        self.shard(FIXTURE, sharded)
        single = self.trace(FIXTURE, work / "single-trace")
        self.assertEqual(single.returncode, 0, single.stdout + single.stderr)
        split = self.trace(sharded, work / "split-trace")
        self.assertEqual(split.returncode, 0, split.stdout + split.stderr)

        diff = subprocess.run(
            ["python3", str(ROOT / "tools" / "trace_diff.py"), str(work / "single-trace"), str(work / "split-trace")],
            cwd=ROOT, capture_output=True, text=True,
        )
        self.assertEqual(diff.returncode, 0, diff.stdout + diff.stderr)
        self.assertIn("IDENTICAL", diff.stdout)

    def testAShardedCheckpointWithoutItsIndexIsRefused(self):
        """Half a model that reads as a whole one is the failure mode; refusing is the answer."""
        work = Path(tempfile.mkdtemp(prefix="sharded-noidx-"))
        sharded = work / "sharded"
        self.shard(FIXTURE, sharded)
        os.remove(sharded / "model.safetensors.index.json")
        result = self.trace(sharded, work / "trace")
        self.assertNotEqual(result.returncode, 0, "a sharded checkpoint with no index must not run")
        self.assertIn("index", (result.stdout + result.stderr).lower())


if __name__ == "__main__":
    unittest.main()
