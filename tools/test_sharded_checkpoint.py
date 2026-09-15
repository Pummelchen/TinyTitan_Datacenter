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
import sys
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


@unittest.skipUnless(HAVE_SAFETENSORS, "safetensors is needed to build a sharded checkpoint")
class PythonReaderShardingTests(unittest.TestCase):
    """The Python side must agree with the Swift side about what a sharded checkpoint is.

    Both readers used to take `sorted(glob(...))[0]`: the contract would have computed a partial
    model's forward, and the install builder would have written an install missing five sixth of a
    sharded model's layers and reported success. These tests need no Swift, so they run wherever
    the venv does.
    """

    def sharded(self, work: Path) -> Path:
        sys.path.insert(0, str(ROOT / "tools"))
        import test_sharded_checkpoint as module  # the helper above

        target = work / "sharded"
        ShardedCheckpointTests.shard(self, FIXTURE, target)
        return target

    def testTheContractReadsShardedAndSingleFileTheSame(self):
        import sys as _sys

        _sys.path.insert(0, str(ROOT / "tools"))
        import ordered_qwen36 as q36
        from safetensors_source import SafetensorsSource

        work = Path(tempfile.mkdtemp(prefix="sharded-py-"))
        sharded = self.sharded(work)
        spec = json.loads((FIXTURE / "spec.json").read_text())
        tokens = json.loads((FIXTURE / "golden.json").read_text())["tokens"]

        single_capture: dict = {}
        single_decisions: dict = {}
        q36.streamed_text_forward(
            spec, SafetensorsSource(FIXTURE), tokens,
            capture=single_capture, discrete=single_decisions,
        )
        split_capture: dict = {}
        split_decisions: dict = {}
        q36.streamed_text_forward(
            spec, SafetensorsSource(sharded), tokens,
            capture=split_capture, discrete=split_decisions,
        )
        self.assertEqual(sorted(single_capture), sorted(split_capture))
        for name in single_capture:
            self.assertEqual(
                single_capture[name].tobytes(), split_capture[name].tobytes(),
                f"{name}: a sharded checkpoint must read as the same model",
            )
        for name in single_decisions:
            self.assertEqual(
                single_decisions[name].tolist(), split_decisions[name].tolist(), f"{name}: decisions"
            )

    def testAShardedCheckpointWithoutAnIndexIsRefusedByThePythonReader(self):
        sys.path.insert(0, str(ROOT / "tools"))
        from safetensors_source import SafetensorsSource

        work = Path(tempfile.mkdtemp(prefix="sharded-py-noidx-"))
        sharded = self.sharded(work)
        (sharded / "model.safetensors.index.json").unlink()
        with self.assertRaises(SystemExit) as raised:
            SafetensorsSource(sharded)
        self.assertIn("refusing", str(raised.exception))

    def testTheInstallBuilderUsesEveryShard(self):
        """An install built from the sharded checkpoint must hold every tensor in the spec, not
        the ones that happened to live in the first shard."""
        sys.path.insert(0, str(ROOT / "tools"))
        import quantize

        work = Path(tempfile.mkdtemp(prefix="sharded-install-"))
        sharded = self.sharded(work)
        spec = json.loads((FIXTURE / "spec.json").read_text())
        policy_path = ROOT / "tools" / "quant_policy.json"
        policy = json.loads(policy_path.read_text())
        manifest = quantize.build_install(sharded, work / "install", spec, policy, str(policy_path))
        # `tensors` is every tensor written, each with the precision it got; `skipped` names the
        # ones kept above int4, which is a *subset* — adding them double-counts, which is how this
        # assertion first failed.
        self.assertEqual(len(manifest["tensors"]), len(spec["tensors"]), "every spec tensor must be written")
        for entry in manifest["tensors"]:
            self.assertIn(entry["quant"], {"int4-affine", "bf16", "fp32", "fp16"})

        single = quantize.build_install(
            FIXTURE, work / "install-single", spec, policy, str(policy_path)
        )
        self.assertEqual(
            (work / "install" / "data.bin").read_bytes(), (work / "install-single" / "data.bin").read_bytes(),
            "the payload must not depend on how the checkpoint was split",
        )
        self.assertEqual(len(single["tensors"]), len(spec["tensors"]))


if __name__ == "__main__":
    unittest.main()
