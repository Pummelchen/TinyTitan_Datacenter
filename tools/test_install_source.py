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
