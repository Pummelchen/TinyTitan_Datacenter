"""Tests for the install-backed contract source.

The point of `install_source.py` is that a contract run and an engine run can be given **the same weights**,
so the tests are about exactly that: the source returns what the install holds, trimmed to the shape the
checkpoint would have had, and refuses the two things it must not do — materialise an expert stack, and read
past the end of a tensor.

`row_range` is new beside it and gets the equivalence test that keeps it honest: `row_range(a, b)` must equal
`rows()[a:b]`, or the fast path and the scan path would be two implementations of one thing, which is how the
last two of those disagreements were found.
"""

from __future__ import annotations

import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from install_reader import InstallError  # noqa: E402
from install_source import InstallSource, InstallSourceError  # noqa: E402
from test_install_reader import InstallFixture, bf16, int4_payload  # noqa: E402

GROUP = 4
PADDED = 8

# The fixtures' own installs, checked in rather than built here: the dense one carries int4 matrices and
# the MoE one carries the expert stacks the block dequantiser exists for.
FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "DatacenterEngineTests" / "Fixtures"


class InstallSourceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def _bf16_fixture(self) -> InstallFixture:
        fixture = InstallFixture(self.root)
        fixture.add(
            "layer.0.linear.in_qkv",
            {"role": "linear.in_qkv", "dtype": "bf16", "quant": "bf16", "group": 0,
             "padded_columns": 3, "shape": [2, 3]},
            bf16([1.0, -2.5, 0.0, 0.5, 2.0, -0.25]),
        )
        fixture.add(
            "layer.0.linear.norm",
            {"role": "linear.norm", "dtype": "fp32", "quant": "fp32", "group": 0,
             "padded_columns": 0, "shape": [3]},
            struct.pack("<3f", 0.5, 1.0, 2.0),
        )
        return fixture

    def test_tensor_returns_what_the_install_holds(self) -> None:
        install = self._bf16_fixture().write()
        with InstallSource(self.root) as source:
            tensor = source.tensor("layer.0.linear.in_qkv")
        self.assertEqual(tensor.shape, (2, 3))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertEqual(
            tensor.reshape(-1).tolist(), install.dequantize(install.tensors["layer.0.linear.in_qkv"])
        )

    def test_a_one_dimensional_tensor_keeps_its_shape(self) -> None:
        self._bf16_fixture().write()
        with InstallSource(self.root) as source:
            tensor = source.tensor("layer.0.linear.norm")
        self.assertEqual(tensor.shape, (3,))
        self.assertEqual(tensor.tolist(), [0.5, 1.0, 2.0])

    def test_rows_are_shaped_like_the_checkpoint(self) -> None:
        install = self._bf16_fixture().write()
        with InstallSource(self.root) as source:
            first = source.rows("layer.0.linear.in_qkv", 0, 1)
            second = source.rows("layer.0.linear.in_qkv", 1, 2)
        self.assertEqual(first.shape, (1, 3))
        self.assertEqual(second.shape, (1, 3))
        self.assertEqual(first.tolist(), [[1.0, -2.5, 0.0]])
        self.assertEqual(second.tolist(), [[0.5, 2.0, -0.25]])
        # and the same rows the scan would give
        self.assertEqual(install.row_range(install.tensors["layer.0.linear.in_qkv"], 0, 2)[1],
                         second.reshape(-1).tolist())

    def test_padding_is_trimmed_to_the_true_shape(self) -> None:
        fixture = InstallFixture(self.root)
        # Four stored columns per row, two of them real: the shape decides the row length, not the payload.
        fixture.add(
            "padded.weight",
            {"role": "padded", "dtype": "bf16", "quant": "bf16", "group": 0,
             "padded_columns": 4, "shape": [1, 2]},
            bf16([7.0, 8.0, 9.0, 10.0]),
        )
        fixture.write()
        with InstallSource(self.root) as source:
            self.assertEqual(source.tensor("padded.weight").tolist(), [[7.0, 8.0]])

    def test_int4_tensors_come_back_through_the_same_dequantiser(self) -> None:
        fixture = InstallFixture(self.root)
        row0 = [0x0, 0x1, 0xF, 0x8, 0x7, 0xE, 0x2, 0x9]
        fixture.add(
            "layer.0.linear.out",
            {"role": "linear.out", "dtype": "int4", "quant": "int4-affine", "group": GROUP,
             "padded_columns": PADDED, "shape": [1, PADDED]},
            int4_payload([row0], [[0.5, 0.25]], [[-3, 2]]),
        )
        install = fixture.write()
        tensor = install.tensors["layer.0.linear.out"]
        with InstallSource(self.root) as source:
            self.assertEqual(source.tensor("layer.0.linear.out").reshape(-1).tolist(),
                             install.dequantize(tensor))

    def test_refuses_the_expert_stacks(self) -> None:
        fixture = InstallFixture(self.root)
        # Deliberately the install's real layout: shape (2, 2, 4) stored as four install rows of four, so
        # one expert is *two* consecutive install rows and a source that ignored that would silently return
        # half an expert -- which is exactly what happened on the real install.
        fixture.add(
            "layer.0.expert.stack_gate_up",
            {"role": "expert.stack_gate_up", "dtype": "bf16", "quant": "bf16", "group": 0,
             "padded_columns": 4, "shape": [2, 2, 4]},
            bf16([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                  9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]),
        )
        fixture.write()
        with InstallSource(self.root) as source:
            with self.assertRaises(InstallSourceError) as caught:
                source.tensor("layer.0.expert.stack_gate_up")
            self.assertIn("rows()", str(caught.exception))
            # ... but one expert at a time is exactly what it is for.
            self.assertEqual(source.rows("layer.0.expert.stack_gate_up", 1, 2).shape, (1, 2, 4))
            self.assertEqual(source.rows("layer.0.expert.stack_gate_up", 1, 2).reshape(-1).tolist(),
                             [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0])
            provider = source.expert_provider("layer.0.expert.stack_gate_up")
            # One expert is one row and keeps the stack's trailing shape, exactly as the checkpoint-backed
            # source returns it -- so the contract's provider sees the same thing from either source.
            self.assertEqual(provider(0).shape, (2, 4))
            self.assertEqual(provider(0).reshape(-1).tolist(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

    def test_every_int4_tensor_matches_the_scalar_definition(self) -> None:
        """The block dequantiser against the row-at-a-time definition, on both fixtures' checked-in installs.

        `_dequantize_row` says what the layout means, one value at a time; `_materialise` reads the same bytes a
        block at a time with numpy, because the scalar path costs about a quarter of a second per real expert
        and is why a contract run against an install took hours where the same run against the checkpoint took
        a minute. Two implementations agreeing today is exactly the situation in which the second one gets
        forgotten (`D34`'s lesson), so the agreement is asserted rather than argued — over the dense fixture's
        int4 matrices and the MoE fixture's expert stacks, which are the case the fast path exists for.
        """
        checked = 0
        for fixture in (FIXTURES / "tiny-qwen35", FIXTURES / "tiny-qwen36"):
            install = fixture / "install"
            if not (install / "install.json").exists():
                continue
            with InstallSource(install) as source:
                int4 = [t for t in source.install.tensors.values() if t.is_int4]
                self.assertTrue(int4, f"{fixture} has no int4 tensor, so nothing was checked")
                for tensor in int4:
                    shape = tuple(tensor.shape or ())
                    leading = shape[0] if len(shape) >= 2 else 1
                    fast = source.rows(tensor.name, 0, leading)
                    slow = self._scalar_rows(source, tensor.name, 0, leading)
                    self.assertEqual(fast.shape, slow.shape, tensor.name)
                    np.testing.assert_array_equal(
                        fast.view(np.uint32), slow.view(np.uint32),
                        err_msg=f"{tensor.name} differs between the block and scalar dequantisers",
                    )
                    checked += 1
        self.assertGreaterEqual(checked, 25, f"only {checked} int4 tensor(s) were checked")

    @staticmethod
    def _scalar_rows(source, name: str, start: int, end: int):
        """The same rows through the per-row path, which is the definition the block path must reproduce."""
        tensor = source._tensor(name)
        shape = tuple(tensor.shape or ())
        rows_total, padded = tensor.geometry()
        columns = padded if len(shape) <= 1 else source._width(tensor)
        factor = source._rows_per_index(tensor)
        flat: list[float] = []
        for row in source.install.row_range(tensor, start * factor, end * factor, row_block=8):
            flat.extend(row[:columns])
        return np.asarray(flat, dtype=np.float32).reshape((end - start,) + shape[1:])

    def test_the_index_mapping_is_checked_and_the_formula_it_replaced_is_refused(self) -> None:
        """`D71`'s defect, as a test rather than as a bisection two rounds long.

        A stack whose `shape.last` is not its `padded_columns` is where the index mapping and the layout can
        disagree: the install holds `prod(shape[:-1])` rows of `shape.last` (padded), so one checkpoint index
        is `prod(shape[1:-1])` of them. The formula this replaced divided the flattened trailing width by the
        padded row, which is the same number whenever the last axis and the padding agree -- always true for
        the real model, and never for the fixture.
        """
        fixture = InstallFixture(self.root)
        fixture.add(
            "layer.0.expert.stack_gate_up",
            {"role": "expert.stack_gate_up", "dtype": "bf16", "quant": "bf16", "group": 0,
             "padded_columns": 8, "shape": [2, 2, 4]},
            bf16([float(v) for v in range(1, 33)]),
        )
        fixture.write()
        # Two indices, two install rows each, four real values per row: the padding is dropped, not read.
        with InstallSource(self.root) as source:
            first = source.rows("layer.0.expert.stack_gate_up", 0, 1)
            self.assertEqual(first.shape, (1, 2, 4))
            self.assertEqual(first.reshape(-1).tolist(), [1.0, 2.0, 3.0, 4.0, 9.0, 10.0, 11.0, 12.0])
            second = source.rows("layer.0.expert.stack_gate_up", 1, 2)
            self.assertEqual(second.reshape(-1).tolist(), [17.0, 18.0, 19.0, 20.0, 25.0, 26.0, 27.0, 28.0])
            source.check_index_mapping(source._tensor("layer.0.expert.stack_gate_up"))

        # And the mapping that was wrong is refused rather than trusted: this is the check that fails at
        # open, so no caller can read half an expert and only notice hours later.
        original = InstallSource._rows_per_index
        InstallSource._rows_per_index = lambda self, tensor: self._width(tensor) // max(tensor.padded_columns, 1)
        try:
            with self.assertRaises(InstallSourceError) as caught:
                with InstallSource(self.root):
                    pass
            self.assertIn("index mapping and", str(caught.exception))
        finally:
            InstallSource._rows_per_index = original

    def test_a_three_dimensional_tensor_that_is_not_a_stack_is_read_whole(self) -> None:
        # The convolution is (8192, 1, 4) in the real model: three dimensions, and small. The refusal is
        # about the expert stacks, not about the rank, which the first version of this got wrong.
        fixture = InstallFixture(self.root)
        fixture.add(
            "layer.0.linear_attn.conv1d.weight",
            {"role": "linear.conv", "dtype": "bf16", "quant": "bf16", "group": 0,
             "padded_columns": 4, "shape": [2, 1, 4]},
            bf16([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        )
        fixture.write()
        with InstallSource(self.root) as source:
            values = source.tensor("layer.0.linear_attn.conv1d.weight")
        self.assertEqual(values.shape, (2, 1, 4))
        self.assertEqual(values.reshape(-1).tolist(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

    def test_an_absent_name_is_an_error(self) -> None:
        self._bf16_fixture().write()
        with InstallSource(self.root) as source:
            with self.assertRaises(InstallSourceError):
                source.tensor("not.in.this.install")


class RowRangeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_row_range_equals_a_slice_of_rows(self) -> None:
        fixture = InstallFixture(self.root)
        fixture.add(
            "stack",
            {"role": "stack", "dtype": "bf16", "quant": "bf16", "group": 0,
             "padded_columns": 2, "shape": [4, 2]},
            bf16([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
        )
        install = fixture.write()
        tensor = install.tensors["stack"]
        everything = list(install.rows(tensor, row_block=3))
        self.assertEqual(install.row_range(tensor, 1, 3), everything[1:3])
        self.assertEqual(install.row_range(tensor, 3, 4), everything[3:4])
        self.assertEqual(install.row_range(tensor, 4, 4), [])

    def test_row_range_refuses_a_range_outside_the_tensor(self) -> None:
        fixture = InstallFixture(self.root)
        fixture.add(
            "stack",
            {"role": "stack", "dtype": "bf16", "quant": "bf16", "group": 0,
             "padded_columns": 2, "shape": [2, 2]},
            bf16([0.0, 1.0, 2.0, 3.0]),
        )
        install = fixture.write()
        tensor = install.tensors["stack"]
        for start, end in ((0, 3), (2, 1), (-1, 1)):
            with self.assertRaises(InstallError):
                install.row_range(tensor, start, end)


if __name__ == "__main__":
    unittest.main()
