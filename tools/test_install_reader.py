"""Tests for `tools/install_reader.py` (`DC-108`).

The expectations here are computed by hand and written out, not recovered by running the reader: a test
that asserts whatever the implementation happens to produce cannot find a layout mistake, and a layout
mistake is the whole risk. The int4 rows below are packed byte by byte and their values are worked
through in the comments.
"""

from __future__ import annotations

import hashlib
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from install_reader import DENSE_WIDTH, Install, InstallError  # noqa: E402

PADDED = 8
GROUP = 4


def pack_nibbles(nibbles: list[int]) -> bytes:
    """Two codes per byte, low nibble first — the order is the format."""
    out = bytearray()
    for index in range(0, len(nibbles), 2):
        out.append((nibbles[index] & 0x0F) | ((nibbles[index + 1] & 0x0F) << 4))
    return bytes(out)


def int4_payload(rows: list[list[int]], scales: list[list[float]], zeros: list[list[int]]) -> bytes:
    """`[all codes][all scales][all zeros]` for the whole tensor, row major."""
    codes = b"".join(pack_nibbles(row) for row in rows)
    scale_bytes = b"".join(struct.pack(f"<{len(row)}f", *row) for row in scales)
    zero_bytes = b"".join(struct.pack(f"<{len(row)}b", *row) for row in zeros)
    return codes + scale_bytes + zero_bytes


def bf16(values: list[float]) -> bytes:
    """bfloat16 is the top half of the fp32, so the widening is exact in both directions."""
    return b"".join(struct.pack("<f", value)[2:] for value in values)


class InstallFixture:
    """Builds a small install on disk, with whatever the test wants to be wrong about it."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.tensors: list[dict] = []
        self.payload = bytearray()

    def add(self, name: str, entry: dict, payload: bytes) -> None:
        entry = dict(entry)
        entry["name"] = name
        entry["offset"] = len(self.payload)
        entry["nbytes"] = len(payload)
        entry["sha256"] = hashlib.sha256(payload).hexdigest()
        self.payload.extend(payload)
        self.tensors.append(entry)

    def write(self, schema: int = 1, tensors: list[dict] | None = None) -> Install:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "data.bin").write_bytes(bytes(self.payload))
        (self.root / "install.json").write_text(
            json.dumps(
                {
                    "schema": schema,
                    "family": "tiny-qwen36",
                    "source": {"repo": "example/model", "revision": "0" * 40},
                    "spec": {"family": "tiny-qwen36"},
                    "passes": ["quantize-group-affine-int4"],
                    "tensors": tensors if tensors is not None else self.tensors,
                }
            )
        )
        return Install(self.root, uncached=False)


class Int4LayoutTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def fixture(self) -> InstallFixture:
        return InstallFixture(self.root)

    def test_dequantizes_the_codes_the_packer_wrote(self) -> None:
        # row 0: nibbles 0x0,0x1,0xF,0x8,0x7,0xE,0x2,0x9 -> signed 0, 1, -1, -8, 7, -2, 2, -7
        # group 0 (scale 0.5, zero -3):  (0+3)*0.5=1.5  (1+3)*0.5=2.0  (-1+3)*0.5=1.0  (-8+3)*0.5=-2.5
        # group 1 (scale 0.25, zero 2):  (7-2)*0.25=1.25 (-2-2)*0.25=-1.0 (2-2)*0.25=0.0 (-7-2)*0.25=-2.25
        row0 = [0x0, 0x1, 0xF, 0x8, 0x7, 0xE, 0x2, 0x9]
        # row 1: group 0's scale is denormal, so `D11` reads it as zero and the first four values vanish.
        # group 1 (scale 2.0, zero 0): 7*2=14  -8*2=-16  -7*2=-14  -6*2=-12
        row1 = [0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xA]
        fixture = self.fixture()
        fixture.add(
            "experts.gate",
            {"role": "expert.gate", "dtype": "int4", "quant": "int4-affine", "group": GROUP,
             "padded_columns": PADDED, "shape": [2, PADDED]},
            int4_payload([row0, row1], [[0.5, 0.25], [1e-40, 2.0]], [[-3, 2], [0, 0]]),
        )
        install = fixture.write()
        values = install.dequantize(install.tensors["experts.gate"])
        self.assertEqual(
            values,
            [1.5, 2.0, 1.0, -2.5, 1.25, -1.0, 0.0, -2.25, 0.0, 0.0, 0.0, 0.0, 14.0, -16.0, -14.0, -12.0],
        )

    def test_row_blocks_agree_with_the_whole_tensor(self) -> None:
        """The bounded-memory path is the one a 21.7 GB install uses, so it must agree exactly."""
        fixture = self.fixture()
        fixture.add(
            "experts.gate",
            {"role": "expert.gate", "dtype": "int4", "quant": "int4-affine", "group": GROUP,
             "padded_columns": PADDED, "shape": [2, PADDED]},
            int4_payload(
                [[0x0, 0x1, 0xF, 0x8, 0x7, 0xE, 0x2, 0x9], [0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xA]],
                [[0.5, 0.25], [1e-40, 2.0]],
                [[-3, 2], [0, 0]],
            ),
        )
        install = fixture.write()
        tensor = install.tensors["experts.gate"]
        whole = install.dequantize(tensor, row_block=64)
        for block in (1, 2, 3):
            pieces: list[float] = []
            for row in install.rows(tensor, row_block=block):
                pieces.extend(row)
            self.assertEqual(pieces, whole, f"row_block={block}")

    def test_a_stacked_expert_tensor_counts_its_leading_axes_as_rows(self) -> None:
        fixture = self.fixture()
        rows = 4
        fixture.add(
            "experts.stacked",
            {"role": "expert.up", "dtype": "int4", "quant": "int4-affine", "group": GROUP,
             "padded_columns": PADDED, "shape": [2, 2, PADDED]},
            int4_payload(
                [[0] * PADDED for _ in range(rows)],
                [[1.0, 1.0] for _ in range(rows)],
                [[0, 0] for _ in range(rows)],
            ),
        )
        install = fixture.write()
        tensor = install.tensors["experts.stacked"]
        self.assertEqual(tensor.geometry(), (4, PADDED))
        self.assertEqual(install.dequantize(tensor), [0.0] * (4 * PADDED))

    def test_layout_guards_name_the_invariant_that_failed(self) -> None:
        cases = {
            "group": ({"group": 0, "padded_columns": PADDED}, "cannot divide anything"),
            "not dividing": ({"group": 3, "padded_columns": PADDED}, "does not divide"),
            "odd width": ({"group": 7, "padded_columns": 7}, "share a byte"),
        }
        for label, (overrides, expected) in cases.items():
            with self.subTest(label):
                fixture = self.fixture()
                entry = {"role": "expert.gate", "dtype": "int4", "quant": "int4-affine",
                         "group": GROUP, "padded_columns": PADDED, "shape": [2, PADDED]}
                entry.update(overrides)
                fixture.add("t", entry, int4_payload([[0] * 8, [0] * 8], [[1.0, 1.0]] * 2, [[0, 0]] * 2))
                install = fixture.write()
                with self.assertRaises(InstallError) as caught:
                    install.tensors["t"].geometry()
                self.assertIn(expected, str(caught.exception))

    def test_a_payload_that_is_not_whole_rows_is_refused(self) -> None:
        fixture = self.fixture()
        fixture.add(
            "t",
            {"role": "expert.gate", "dtype": "int4", "quant": "int4-affine", "group": GROUP,
             "padded_columns": PADDED, "shape": [2, PADDED]},
            int4_payload([[0] * 8, [0] * 8], [[1.0, 1.0]] * 2, [[0, 0]] * 2) + b"\x00",
        )
        install = fixture.write()
        with self.assertRaises(InstallError) as caught:
            install.tensors["t"].geometry()
        self.assertIn("whole number of", str(caught.exception))

    def test_a_shape_that_disagrees_with_the_payload_is_refused(self) -> None:
        fixture = self.fixture()
        fixture.add(
            "t",
            {"role": "expert.gate", "dtype": "int4", "quant": "int4-affine", "group": GROUP,
             "padded_columns": PADDED, "shape": [9, PADDED]},
            int4_payload([[0] * 8, [0] * 8], [[1.0, 1.0]] * 2, [[0, 0]] * 2),
        )
        install = fixture.write()
        with self.assertRaises(InstallError) as caught:
            install.tensors["t"].geometry()
        self.assertIn("the payload holds", str(caught.exception))


class DenseAndContainerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_bfloat16_widens_exactly_and_rows_come_back_whole(self) -> None:
        fixture = InstallFixture(self.root)
        values = [1.0, -2.5, 0.0, 3.25]
        fixture.add(
            "embed.weight",
            {"role": "embed", "dtype": "bf16", "quant": "bf16", "group": 0,
             "padded_columns": 2, "shape": [2, 2]},
            bf16(values),
        )
        install = fixture.write()
        tensor = install.tensors["embed.weight"]
        self.assertEqual(tensor.geometry(), (2, 2))
        self.assertEqual(install.dequantize(tensor), values)
        # One row at a time, which is how a caller with no memory to spare would read it.
        rows = list(install.rows(tensor, row_block=1))
        self.assertEqual(rows, [[1.0, -2.5], [0.0, 3.25]])

    def test_float32_round_trips_exactly(self) -> None:
        fixture = InstallFixture(self.root)
        # `0.1` is not representable in fp32, so the expectation is the fp32 value — asserting the
        # double 0.1 would be asserting that the format is something it is not.
        values = [0.1, -1e30, 3.0]
        expected = [struct.unpack("<f", struct.pack("<f", value))[0] for value in values]
        fixture.add(
            "head.weight",
            {"role": "head.lm", "dtype": "fp32", "quant": "fp32", "group": 0,
             "padded_columns": 3, "shape": [1, 3]},
            struct.pack("<3f", *values),
        )
        install = fixture.write()
        self.assertEqual(install.dequantize(install.tensors["head.weight"]), expected)

    def test_a_denormal_scale_is_read_as_zero(self) -> None:
        """`D11` is a contract, not an accident: both readers flush, so a comparison stays exact."""
        fixture = InstallFixture(self.root)
        fixture.add(
            "t",
            {"role": "expert.gate", "dtype": "int4", "quant": "int4-affine", "group": GROUP,
             "padded_columns": PADDED, "shape": [1, PADDED]},
            int4_payload([[0x7] * PADDED], [[1e-40, 1e-40]], [[0, 0]]),
        )
        install = fixture.write()
        self.assertEqual(install.dequantize(install.tensors["t"]), [0.0] * PADDED)

    def test_digests_are_checked_by_streaming_and_a_mismatch_is_reported(self) -> None:
        fixture = InstallFixture(self.root)
        fixture.add(
            "t",
            {"role": "embed", "dtype": "fp32", "quant": "fp32", "group": 0,
             "padded_columns": 2, "shape": [1, 2]},
            struct.pack("<2f", 1.0, 2.0),
        )
        install = fixture.write()
        self.assertTrue(install.verified(install.tensors["t"]))

        # Rewrite one byte of the payload: the manifest's digest must stop matching. The byte has to
        # be *different*: the first byte of `1.0f` is already 0x00, and the first version of this test
        # "corrupted" it to 0x00 and then wondered why the digest still matched.
        original = install.data_path.read_bytes()[:1]
        with open(install.data_path, "r+b") as handle:
            handle.seek(0)
            handle.write(b"\xff" if original != b"\xff" else b"\x00")
        self.assertFalse(install.verified(install.tensors["t"]))

    def test_a_tensor_without_a_digest_is_not_a_pass(self) -> None:
        fixture = InstallFixture(self.root)
        fixture.add(
            "t",
            {"role": "embed", "dtype": "fp32", "quant": "fp32", "group": 0,
             "padded_columns": 2, "shape": [1, 2]},
            struct.pack("<2f", 1.0, 2.0),
        )
        entries = [dict(entry) for entry in fixture.tensors]
        del entries[0]["sha256"]
        install = fixture.write(tensors=entries)
        with self.assertRaises(InstallError) as caught:
            install.verified(install.tensors["t"])
        self.assertIn("no digest", str(caught.exception))

    def test_container_guards(self) -> None:
        fixture = InstallFixture(self.root)
        fixture.add(
            "t",
            {"role": "embed", "dtype": "fp32", "quant": "fp32", "group": 0,
             "padded_columns": 2, "shape": [1, 2]},
            struct.pack("<2f", 1.0, 2.0),
        )
        entries = [dict(entry) for entry in fixture.tensors]

        with self.subTest("schema"):
            with self.assertRaises(InstallError) as caught:
                fixture.write(schema=2)
            self.assertIn("schema", str(caught.exception))

        with self.subTest("duplicate name"):
            with self.assertRaises(InstallError) as caught:
                fixture.write(tensors=entries + entries)
            self.assertIn("twice", str(caught.exception))

        with self.subTest("outside the payload"):
            broken = [dict(entries[0])]
            broken[0]["nbytes"] = 1 << 20
            with self.assertRaises(InstallError) as caught:
                fixture.write(tensors=broken)
            self.assertIn("outside", str(caught.exception))

        with self.subTest("read past the end"):
            install = fixture.write()
            with self.assertRaises(InstallError) as caught:
                install.read(install.tensors["t"], 4, 1024)
            self.assertIn("outside it", str(caught.exception))


class RealInstallTests(unittest.TestCase):
    """The real install is 21.7 GB: this only touches it when it is there, and only structurally."""

    def test_the_real_install_agrees_with_its_manifest(self) -> None:
        root = Path(__file__).resolve().parent.parent / ".build" / "m1-install"
        if not (root / "install.json").exists():
            self.skipTest("the M1 install is not present in this checkout")
        with Install(root) as install:
            self.assertEqual(install.family, "qwen3_5_moe")
            self.assertGreater(len(install.tensors), 100)
            for tensor in install.tensors.values():
                tensor.geometry()
            quantised = [t for t in install.tensors.values() if t.is_int4]
            self.assertTrue(quantised, "the real install should have quantised experts")
            # One tensor's digest, streamed: the whole install would be a heavy job on this host.
            sample = sorted(quantised, key=lambda t: (t.nbytes, t.name))[0]
            self.assertTrue(install.verified(sample), sample.name)
            widths = {t.dtype for t in install.tensors.values()}
            self.assertTrue(widths <= set(DENSE_WIDTH) | {"int4"}, widths)


if __name__ == "__main__":
    unittest.main()
