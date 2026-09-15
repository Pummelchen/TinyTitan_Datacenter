#!/usr/bin/env python3
"""Tests for the controlled-order reference.

The first test is the finding that made this module necessary, kept as a test so it
cannot quietly stop being true: a matmul accumulated in ascending ``k`` does not
reproduce PyTorch's, and neither is more correct than the other. The rest pin the
properties the engine is required to reproduce — an order, not an approximation.

Run: .venv/bin/python -m unittest discover -s tools   (torch tests skip without it)
"""

from __future__ import annotations

import unittest

try:
    import numpy as np

    import ordered_reference as ref

    HAVE_NUMPY = True
except ImportError:  # the repository gates run without packages installed
    HAVE_NUMPY = False

try:
    import torch  # noqa: F401

    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False

NUMPY_REASON = "numpy is not installed in this interpreter; the contract needs it, so run with .venv/bin/python"
SKIP_REASON = "torch is not installed in this interpreter; run with .venv/bin/python"

numpy_required = unittest.skipUnless(HAVE_NUMPY, NUMPY_REASON)


@numpy_required
class OrderTests(unittest.TestCase):
    def test_ordered_matmul_matches_a_naive_triple_loop(self):
        """The contract is a specific sequence of fp32 additions, so it must equal the
        most explicit possible statement of it."""
        rng = np.random.default_rng(0)
        x = ref.f32(rng.standard_normal((3, 7)) * 0.5)
        w = ref.f32(rng.standard_normal((5, 7)) * 0.5)
        expected = np.zeros((3, 5), dtype=np.float32)
        for row in range(3):
            for out in range(5):
                acc = np.float32(0.0)
                for k in range(7):
                    acc = np.float32(acc + np.float32(x[row, k] * w[out, k]))
                expected[row, out] = acc
        self.assertTrue(np.array_equal(ref.ordered_matmul(x, w), expected))

    def test_ordered_sum_is_left_to_right(self):
        values = ref.f32([1e8, 1.0, -1e8, 1.0])
        expected = np.float32(0.0)
        for value in values:
            expected = np.float32(expected + value)
        self.assertEqual(float(ref.ordered_sum(values)), float(expected))
        self.assertEqual(ref.ordered_sum(values).ndim, 0, "a 1-D sum is a scalar, not a 1-element array")

    def test_rms_norm_matches_its_definition(self):
        rng = np.random.default_rng(1)
        x = ref.f32(rng.standard_normal((2, 4)))
        weight = ref.f32(rng.standard_normal(4))
        variance = ref.f32(ref.ordered_sum(ref.f32(x * x), axis=-1) / np.float32(4))
        inverse = ref.f32(np.float32(1.0) / np.sqrt(ref.f32(variance + np.float32(1e-6))))
        expected = ref.f32(weight * ref.f32(x * inverse[:, None]))
        self.assertTrue(np.array_equal(ref.rms_norm(x, weight, 1e-6), expected))

    def test_silu_does_not_overflow_at_the_extremes(self):
        """A stable sigmoid is part of the contract: the naive form warns and produces
        inf-inf on the way to a correct zero."""
        extremes = ref.f32([-200.0, -1.0, 0.0, 1.0, 200.0])
        with np.errstate(over="raise"):
            result = ref.silu(extremes)
        self.assertTrue(np.isfinite(result).all())
        self.assertEqual(result[2], np.float32(0.0))
        self.assertAlmostEqual(float(result[4]), 200.0, places=3)

    def test_softmax_sums_to_one_and_is_order_explicit(self):
        rng = np.random.default_rng(2)
        x = ref.f32(rng.standard_normal((2, 9)) * 3)
        probabilities = ref.softmax(x)
        for row in probabilities:
            self.assertAlmostEqual(float(ref.ordered_sum(row)), 1.0, places=6)


@numpy_required
@unittest.skipUnless(HAVE_TORCH, SKIP_REASON)
class TorchComparisonTests(unittest.TestCase):
    def test_ordered_matmul_differs_from_torch_and_neither_is_more_accurate(self):
        """The measurement that defined the two-reference model (D3, R13).

        The shapes are a real layer's. The claim is *not* that one is wrong: both sit the
        same distance from an fp64 computation, which is why the gate compares the engine
        to the ordered contract and uses torch as the semantic oracle instead.
        """
        torch.manual_seed(0)
        torch.set_num_threads(1)
        row, k_dim, out = 8, 1024, 512
        x = (torch.randn(row, k_dim) * 0.05).to(torch.float32)
        w = (torch.randn(out, k_dim) * 0.05).to(torch.float32)

        from_torch = (x @ w.t()).numpy()
        from_us = ref.ordered_matmul(x.numpy(), w.numpy())

        differing = int((from_torch != from_us).sum())
        self.assertGreater(
            differing, 0, "if this ever passes, the ordered reference is unnecessary and D3 should be revisited"
        )

        exact = (x.double() @ w.double().t()).numpy()
        torch_error = float(np.abs(from_torch - exact).max())
        ordered_error = float(np.abs(from_us - exact).max())
        # Neither is meaningfully more accurate than the other: same order of error.
        self.assertLess(abs(torch_error - ordered_error) / max(torch_error, 1e-12), 0.5)


if __name__ == "__main__":
    unittest.main()
