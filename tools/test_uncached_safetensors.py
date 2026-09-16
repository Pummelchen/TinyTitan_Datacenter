"""Tests for `tools/uncached_safetensors.py`.

The claim that matters is **byte-identity with `safe_open`**, because this reader exists to replace it on a
machine where mapping 67 GB is not survivable. Identity is checked on a file built here, where every dtype
and edge case can be constructed, and then on a **real shard** of the checkpoint, where the data is not mine
to choose. The second one is the evidence; the first is what makes a failure diagnosable.
"""

from __future__ import annotations

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from uncached_safetensors import DTYPES, UncachedSafetensorsSource, UnsupportedDtype  # noqa: E402

try:
    from safetensors import safe_open

    HAVE_SAFETENSORS = True
except Exception:  # pragma: no cover - the stdlib-only path
    HAVE_SAFETENSORS = False

SNAPSHOT = Path(__file__).resolve().parent.parent / ".build" / "hf-cache"


def bf16_bits(values: np.ndarray) -> bytes:
    """Round the fp32 values to bf16 the way the format stores them: the top sixteen bits."""
    return (values.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16).tobytes()


def write_safetensors(path: Path, tensors: list[tuple[str, str, tuple, bytes]]) -> None:
    """A minimal but valid safetensors file, written here so the tests own the bytes."""
    header: dict[str, dict] = {}
    offset = 0
    body = b""
    for name, dtype, shape, raw in tensors:
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + len(raw)]}
        offset += len(raw)
        body += raw
    encoded = json.dumps(header).encode()
    padded = encoded + b" " * ((8 - (len(encoded) % 8)) % 8)
    path.write_bytes(struct.pack("<Q", len(padded)) + padded + body)


class FixtureFileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        rng = np.random.default_rng(7)
        self.bf16 = rng.standard_normal((6, 4)).astype(np.float32)
        self.f32 = rng.standard_normal((3, 5)).astype(np.float32)
        specials = np.array(
            [[0.0, -0.0, np.inf, -np.inf], [np.nan, 1e-38, -1e-38, 3.5]], dtype=np.float32
        )
        self.path = self.directory / "model-00001-of-00001.safetensors"
        write_safetensors(
            self.path,
            [
                ("bf16", "BF16", self.bf16.shape, bf16_bits(self.bf16)),
                ("f32", "F32", self.f32.shape, self.f32.tobytes()),
                ("specials", "BF16", specials.shape, bf16_bits(specials)),
            ],
        )

    def source(self, **kwargs) -> UncachedSafetensorsSource:
        return UncachedSafetensorsSource(self.directory, **kwargs)

    def test_a_whole_bf16_tensor_is_widened_exactly(self) -> None:
        """The reader must apply the same widening as `.float()`: a shift, not a rounding."""
        source = self.source()
        self.addCleanup(source.close)
        got = source.tensor("bf16")
        expected = np.frombuffer(bf16_bits(self.bf16), dtype=np.uint16).astype(np.uint32) << 16
        self.assertEqual(got.tobytes(), expected.view(np.float32).tobytes())
        self.assertEqual(got.shape, (6, 4))

    def test_an_f32_tensor_is_returned_byte_for_byte(self) -> None:
        source = self.source()
        self.addCleanup(source.close)
        self.assertEqual(source.tensor("f32").tobytes(), self.f32.tobytes())

    def test_a_row_range_reads_only_those_rows(self) -> None:
        source = self.source()
        self.addCleanup(source.close)
        got = source.rows("bf16", 2, 5)
        self.assertEqual(got.shape, (3, 4))
        self.assertEqual(got.tobytes(), source.tensor("bf16")[2:5].tobytes())

    def test_an_empty_range_is_empty_rather_than_an_error(self) -> None:
        source = self.source()
        self.addCleanup(source.close)
        self.assertEqual(source.rows("bf16", 4, 4).shape[0], 0)

    def test_infinities_nans_and_signed_zero_survive(self) -> None:
        """The values this project has already been bitten by, carried through the widening."""
        source = self.source()
        self.addCleanup(source.close)
        got = source.tensor("specials")
        self.assertTrue(np.isposinf(got[0, 2]))
        self.assertTrue(np.isneginf(got[0, 3]))
        self.assertTrue(np.isnan(got[1, 0]))
        self.assertEqual(np.signbit(got[0, 1]), True, "-0.0 must not lose its sign")

    def test_a_missing_tensor_is_a_key_error(self) -> None:
        source = self.source()
        self.addCleanup(source.close)
        with self.assertRaises(KeyError):
            source.tensor("not-a-tensor")

    def test_an_unknown_dtype_is_refused_by_name(self) -> None:
        path = self.directory / "odd.safetensors"
        write_safetensors(path, [("odd", "I8", (2,), b"\x01\x02")])
        source = UncachedSafetensorsSource(self.directory)
        self.addCleanup(source.close)
        with self.assertRaises(UnsupportedDtype) as caught:
            source.tensor("odd")
        self.assertIn("I8", str(caught.exception))
        self.assertIn("add it deliberately", str(caught.exception))

    def test_the_index_is_used_when_one_exists(self) -> None:
        (self.directory / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"bf16": self.path.name, "f32": self.path.name}})
        )
        source = UncachedSafetensorsSource(self.directory)
        self.addCleanup(source.close)
        self.assertEqual(source.names, {"bf16", "f32"})
        self.assertEqual(source.shard_count, 1)


@unittest.skipUnless(HAVE_SAFETENSORS, "safetensors is not installed")
class RealShardTests(unittest.TestCase):
    """The evidence: a real 4 GB shard of the 35 B checkpoint, read both ways and compared."""

    @classmethod
    def setUpClass(cls) -> None:
        snapshots = sorted((SNAPSHOT / "models--Qwen--Qwen3.6-35B-A3B" / "snapshots").glob("*/"))
        if not snapshots:
            raise unittest.SkipTest("the 35 B checkpoint is not cached on this machine")
        cls.snapshot = snapshots[0]

    def test_a_row_range_of_the_largest_tensor_matches_safe_open(self) -> None:
        """Only a few rows are read: the point is that the bytes agree, not that the test is big."""
        name = "model.language_model.layers.0.mlp.experts.gate_up_proj"
        source = UncachedSafetensorsSource(self.snapshot)
        self.addCleanup(source.close)
        if name not in source.names:
            raise unittest.SkipTest(f"{name} is not in this checkpoint")

        ours = source.rows(name, 0, 2)
        shard = source._owner[name]
        with safe_open(str(self.snapshot / shard), framework="pt") as handle:
            theirs = handle.get_slice(name)[0:2].float().numpy().astype(np.float32)
        self.assertEqual(ours.shape, theirs.shape)
        self.assertEqual(ours.tobytes(), theirs.tobytes(), "the uncached reader must not change a byte")

    def test_a_one_dimensional_tensor_matches_safe_open(self) -> None:
        source = UncachedSafetensorsSource(self.snapshot)
        self.addCleanup(source.close)
        name = next((n for n in sorted(source.names) if "." in n and "norm" in n), None)
        if name is None:
            raise unittest.SkipTest("no norm tensor found")
        ours = source.tensor(name)
        with safe_open(str(self.snapshot / source._owner[name]), framework="pt") as handle:
            theirs = handle.get_tensor(name).float().numpy().astype(np.float32)
        self.assertEqual(ours.tobytes(), theirs.tobytes())

    def test_the_reader_knows_every_dtype_the_checkpoint_uses(self) -> None:
        source = UncachedSafetensorsSource(self.snapshot)
        self.addCleanup(source.close)
        seen: set[str] = set()
        for shard in sorted(set(source._owner.values()))[:3]:
            seen |= {entry["dtype"] for entry in source._header(self.snapshot / shard).values()}
        self.assertTrue(seen, "no dtypes were seen at all")
        self.assertEqual(seen - set(DTYPES), set(), f"unhandled dtypes: {seen - set(DTYPES)}")


if __name__ == "__main__":
    unittest.main()
