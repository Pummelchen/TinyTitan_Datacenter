#!/usr/bin/env python3
"""The gather probe's arithmetic is the whole argument, so it is pinned here.

`gather/dense` is `budget * headDim / total` per head-query, and the probe's
verdict rests on it being ~64x for the shipped 3.8 geometry and ~103 GB at the
real chunk. A silent change to the geometry or the budget would move that number
without failing anything, so it is asserted rather than computed in prose.
"""
from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmark"))

try:
    import ane_gather_probe as probe
    IMPORT_ERROR = ""
except Exception as exc:  # coremltools or numpy absent
    probe = None
    IMPORT_ERROR = str(exc)


@unittest.skipIf(probe is None, f"probe needs coremltools: {IMPORT_ERROR}")
class GatherGeometryTests(unittest.TestCase):
    def test_the_gathered_keys_are_64x_the_dense_score_matrix(self):
        geom = probe.Geometry(chunk=32)
        self.assertEqual(geom.dense_values, 24 * 32 * 8_192)
        self.assertEqual(geom.gather_values, 24 * 32 * 2_051 * 256)
        self.assertAlmostEqual(geom.gather_values / geom.dense_values, 64.09,
                               places=2)

    def test_the_real_chunk_would_need_103_gb_of_gathered_keys(self):
        # 24 heads x 4096 x 2051 x 256 fp16, against a 1.6 GB dense score matrix.
        real = probe.Geometry(chunk=32).at_chunk(4_096)
        self.assertAlmostEqual(real.gather_values * 2 / 1e9, 103.2, places=1)
        self.assertAlmostEqual(real.dense_values * 2 / 1e6, 1_611, places=0)
        self.assertGreater(real.gather_values / real.dense_values, 64)

    def test_the_probe_chunk_does_not_change_the_geometry(self):
        small = probe.Geometry(chunk=16)
        self.assertEqual((small.total, small.heads, small.head_dim,
                          small.budget), (8_192, 24, 256, 2_051))


@unittest.skipIf(probe is None, f"probe needs coremltools: {IMPORT_ERROR}")
class GatherGraphTests(unittest.TestCase):
    """Both graphs must stay buildable: a probe that cannot build the thing it
    measures reports nothing, and a broken probe looks like a refusal."""

    def geometry(self):
        return probe.Geometry(chunk=8, total=64, budget=8)

    def test_the_dense_graph_takes_a_mask_and_the_gather_graph_an_index_list(self):
        geom = self.geometry()
        dense = probe.build_dense(geom).functions["main"]
        gather = probe.build_gather(geom).functions["main"]
        self.assertEqual(set(dense.inputs.keys()), {"q", "k", "v", "mask"})
        self.assertEqual(set(gather.inputs.keys()), {"q", "k", "v", "idx"})

    def test_the_gather_graph_carries_no_mask(self):
        # The selection is the gather; folding a mask back in would be today's
        # graph wearing the gather's inputs.
        program = probe.build_gather(self.geometry()).functions["main"]
        operations = [op.op_type for op in program.operations]
        self.assertIn("gather", operations)
        self.assertNotIn("add", operations)


if __name__ == "__main__":
    unittest.main()
