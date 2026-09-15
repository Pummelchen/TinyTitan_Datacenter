#!/usr/bin/env python3
"""Check the transcribed chunked delta rule against the reference's own implementation.

This is the test the working agreement asks for before a kernel exists: the reference is
the authority, so the transcription is checked against `transformers`' actual function
rather than against a second reading of it. Agreement is semantic — closeness, not bit
equality — because torch's blocked triangular solve and BLAS matmuls accumulate in a
different order (D3, R13).

Skipped when torch is absent, like the other venv-run tests.

    .venv/bin/python -m unittest discover -s tools
"""

from __future__ import annotations

import unittest

try:
    import numpy as np

    HAVE_NUMPY = True
except ImportError:  # the repository gates run without packages installed
    HAVE_NUMPY = False
    np = None

MODULE_ERROR = ""

# Imported separately from numpy so a path problem cannot masquerade as a missing
# package: the first run of these tests reported "numpy is not installed" when the real
# cause was `tools/` not being on sys.path.
if HAVE_NUMPY:
    try:
        import ordered_gdn as gdn

        HAVE_GDN = True
    except ImportError as error:
        HAVE_GDN = False
        MODULE_ERROR = str(error)
else:
    HAVE_GDN = False
    MODULE_ERROR = "numpy is not installed"


try:
    import torch
    from transformers.models.qwen3_5.modeling_qwen3_5 import l2norm as torch_l2norm
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        torch_chunk_gated_delta_rule as reference_chunk_rule,
    )

    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False

NUMPY_REASON = "numpy is not installed; run with .venv/bin/python"
TORCH_REASON = "torch/transformers are not installed; run with .venv/bin/python"

numpy_required = unittest.skipUnless(HAVE_GDN, NUMPY_REASON + ": " + MODULE_ERROR)
torch_required = unittest.skipUnless(HAVE_NUMPY and HAVE_TORCH, TORCH_REASON)


def random_inputs(seed: int, length: int, heads: int = 2, key_dim: int = 8, value_dim: int = 8):
    """Inputs shaped and scaled like the real layer's, so the comparison is not easier
    than the model: decay is in log space and must be <= 0."""
    rng = np.random.default_rng(seed)
    query = (rng.standard_normal((1, length, heads, key_dim)) * 0.5).astype(np.float32)
    key = (rng.standard_normal((1, length, heads, key_dim)) * 0.5).astype(np.float32)
    value = (rng.standard_normal((1, length, heads, value_dim)) * 0.5).astype(np.float32)
    decay = (-np.abs(rng.standard_normal((1, length, heads))) * 0.1).astype(np.float32)
    beta = (1.0 / (1.0 + np.exp(-rng.standard_normal((1, length, heads))))).astype(np.float32)
    return query, key, value, decay, beta


@numpy_required
class ShapeTests(unittest.TestCase):
    def test_shapes_survive_padding_and_chunking(self):
        query, key, value, decay, beta = random_inputs(0, length=100)
        output, state = gdn.chunk_gated_delta_rule(query, key, value, decay, beta, chunk_size=64)
        self.assertEqual(output.shape, value.shape)
        self.assertEqual(state.shape, (1, 2, 8, 8))

    def test_an_exact_multiple_of_the_chunk_size_needs_no_padding(self):
        for length in (64, 128):
            query, key, value, decay, beta = random_inputs(1, length=length)
            output, _ = gdn.chunk_gated_delta_rule(query, key, value, decay, beta, chunk_size=64)
            self.assertEqual(output.shape, value.shape)

    def test_causality_prefix_stability(self):
        """Changing a later token must not change an earlier output: the rule is causal,
        and a chunked implementation that leaked across chunks would fail this."""
        query, key, value, decay, beta = random_inputs(2, length=40)
        output, _ = gdn.chunk_gated_delta_rule(query, key, value, decay, beta, chunk_size=16)

        query2, key2, value2, decay2, beta2 = (x.copy() for x in (query, key, value, decay, beta))
        for array in (query2, key2, value2, decay2, beta2):
            array[:, 30:] = np.float32(0.5)
        output2, _ = gdn.chunk_gated_delta_rule(query2, key2, value2, decay2, beta2, chunk_size=16)

        np.testing.assert_array_equal(output[:, :30], output2[:, :30])


@torch_required
class ReferenceComparisonTests(unittest.TestCase):
    def reference(self, query, key, value, decay, beta, chunk_size):
        out, state = reference_chunk_rule(
            torch.from_numpy(query), torch.from_numpy(key), torch.from_numpy(value),
            torch.from_numpy(decay), torch.from_numpy(beta),
            chunk_size=chunk_size, initial_state=None, output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        return out.numpy(), state.numpy()

    def assert_close(self, ours, theirs, label):
        scale = float(np.abs(theirs).max())
        error = float(np.abs(ours - theirs).max())
        # A relative bound against the output's own scale: this is a semantic check on the
        # transcription, and the two implementations accumulate in different orders.
        self.assertLess(error, max(1e-5, scale * 1e-4), f"{label}: max |Δ| {error:.3e} against scale {scale:.3e}")

    def test_matches_the_reference_on_a_single_chunk(self):
        query, key, value, decay, beta = random_inputs(3, length=32)
        ours, _ = gdn.chunk_gated_delta_rule(query, key, value, decay, beta, chunk_size=64)
        theirs, _ = self.reference(query, key, value, decay, beta, chunk_size=64)
        self.assert_close(ours, theirs, "single chunk")

    def test_matches_the_reference_across_several_chunks(self):
        """The sequential scan over chunks is where a transcription error would hide: with
        one chunk it never runs."""
        for length in (100, 192):
            query, key, value, decay, beta = random_inputs(4, length=length)
            ours, _ = gdn.chunk_gated_delta_rule(query, key, value, decay, beta, chunk_size=64)
            theirs, _ = self.reference(query, key, value, decay, beta, chunk_size=64)
            self.assert_close(ours, theirs, f"length {length}")

    def test_matches_the_reference_on_a_small_chunk_size(self):
        query, key, value, decay, beta = random_inputs(5, length=70)
        ours, _ = gdn.chunk_gated_delta_rule(query, key, value, decay, beta, chunk_size=16)
        theirs, _ = self.reference(query, key, value, decay, beta, chunk_size=16)
        self.assert_close(ours, theirs, "chunk 16")

    def test_final_state_matches_the_reference(self):
        query, key, value, decay, beta = random_inputs(6, length=100)
        ours_out, ours_state = gdn.chunk_gated_delta_rule(query, key, value, decay, beta, chunk_size=64)
        theirs_out, theirs_state = self.reference(query, key, value, decay, beta, chunk_size=64)
        self.assert_close(ours_out, theirs_out, "output")
        self.assert_close(ours_state, theirs_state, "final state")

    def test_an_initial_state_is_threaded_through(self):
        query, key, value, decay, beta = random_inputs(7, length=100)
        moved, state = gdn.chunk_gated_delta_rule(query, key, value, decay, beta, chunk_size=64)
        del moved
        query2, key2, value2, decay2, beta2 = random_inputs(8, length=64)
        ours, _ = gdn.chunk_gated_delta_rule(
            query2, key2, value2, decay2, beta2, chunk_size=64, initial_state=state
        )
        theirs, _ = reference_chunk_rule(
            torch.from_numpy(query2), torch.from_numpy(key2), torch.from_numpy(value2),
            torch.from_numpy(decay2), torch.from_numpy(beta2),
            chunk_size=64, initial_state=torch.from_numpy(state), output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        self.assert_close(ours, theirs.numpy(), "with an initial state")

    def test_l2norm_matches_the_reference(self):
        rng = np.random.default_rng(9)
        x = (rng.standard_normal((4, 5, 8)) * 3).astype(np.float32)
        ours = gdn.l2norm(x)
        theirs = torch_l2norm(torch.from_numpy(x), dim=-1, eps=1e-6).numpy()
        # rsqrt and 1/sqrt may differ in the last bits; the contract states 1/sqrt.
        np.testing.assert_allclose(ours, theirs, rtol=1e-5, atol=1e-6)

    def test_l2norm_produces_unit_vectors(self):
        rng = np.random.default_rng(10)
        x = (rng.standard_normal((7, 8)) * 5).astype(np.float32)
        norms = np.sqrt((gdn.l2norm(x) ** 2).sum(-1))
        np.testing.assert_allclose(norms, np.ones(7), rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
