#!/usr/bin/env python3
"""Tests for the NumPy reference the sidecar verifier judges the graph against.

`selection_mask` builds the QSA-shaped mask that the folded check feeds both the
graph and the reference. It is the yardstick, so a bug in it would not fail —
it would make a wrong graph pass. The rule it mirrors lives in
`QSAIndexer.selectKeysPrefill`.

It imports the verifier, which imports the exporter, which imports coremltools,
so it skips where that is not installed (the CI python does not have it;
`~/.venvs/coreml-py311` does):

    cd benchmark && ~/.venvs/coreml-py311/bin/python -m unittest \
        test_verify_ane_sidecar -v
"""
from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

try:
    import verify_ane_sidecar as verify
    IMPORT_ERROR = ""
except Exception as exc:  # coremltools or numpy absent
    verify = None
    IMPORT_ERROR = str(exc)


@unittest.skipIf(verify is None, f"verifier needs coremltools: {IMPORT_ERROR}")
class SelectionMaskTests(unittest.TestCase):
    def test_rows_inside_the_dense_window_keep_everything_visible(self):
        # budget 4, ratio 2 -> the dense window is 5 visible keys.
        mask, kept = verify.selection_mask(16, 4, 2, seed=1)
        for row in range(5):
            visible = row + 1
            self.assertEqual(kept[row], visible)
            self.assertTrue((mask[0, 0, row, :visible] == 0.0).all())

    def test_a_row_past_the_window_keeps_the_budget_and_its_tail(self):
        mask, kept = verify.selection_mask(16, 4, 2, seed=1)
        row = 15
        visible = row + 1
        self.assertEqual(kept[row], 5)          # budget + ratio - 1
        # Dropped keys are masked, not merely absent.
        self.assertEqual(int((mask[0, 0, row, :visible] == verify.ex.NEG).sum()),
                         visible - 5)
        # Nothing past the query's own position is ever visible.
        self.assertTrue((mask[0, 0, row, visible:] == verify.ex.NEG).all())

    def test_the_ragged_tail_of_the_query_block_is_always_kept(self):
        mask, _ = verify.selection_mask(16, 4, 2, seed=3)
        for row in range(16):
            visible = row + 1
            complete = (visible // 2) * 2
            tail = mask[0, 0, row, complete:visible]
            self.assertTrue((tail == 0.0).all(),
                            f"row {row} dropped its own block's tail")

    def test_the_kept_count_matches_the_mask(self):
        mask, kept = verify.selection_mask(32, 6, 4, seed=5)
        for row in range(32):
            visible = row + 1
            row_mask = mask[0, 0, row, :visible]
            self.assertEqual(int((row_mask == 0.0).sum()), kept[row])
            self.assertEqual(int((row_mask == verify.ex.NEG).sum()),
                             visible - kept[row])

    def test_the_budget_is_never_exceeded(self):
        budget, ratio = 8, 4
        selection_width = budget + ratio - 1
        _, kept = verify.selection_mask(64, budget, ratio, seed=7)
        self.assertLessEqual(int(kept.max()), selection_width)
        self.assertGreater(int(kept.max()), 0)

    def test_it_is_deterministic_for_a_seed(self):
        first, first_kept = verify.selection_mask(16, 4, 2, seed=11)
        second, second_kept = verify.selection_mask(16, 4, 2, seed=11)
        self.assertTrue((first == second).all())
        self.assertTrue((first_kept == second_kept).all())
        other, _ = verify.selection_mask(16, 4, 2, seed=12)
        self.assertFalse((first == other).all())


if __name__ == "__main__":
    unittest.main()
