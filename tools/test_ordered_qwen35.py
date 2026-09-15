#!/usr/bin/env python3
"""Check the Gated DeltaNet *layer* against the reference's own module.

`test_ordered_gdn.py` checks the chunked rule; this checks what surrounds it — the
projection, the causal conv's boundaries, the decay's construction, the gate and the gated
norm — against `transformers`' `Qwen3_5GatedDeltaNet`, on a tiny configuration with random
weights. A tiny model is the point: the wiring is what is being tested, and it can be tested
without an 8 GB checkpoint.

Skipped without torch, like the other venv-run tests.
"""

from __future__ import annotations

import unittest

try:
    import numpy as np

    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False
    np = None

MODULE_ERROR = ""
if HAVE_NUMPY:
    try:
        import ordered_qwen35 as q35

        HAVE_MODULE = True
    except ImportError as error:
        HAVE_MODULE = False
        MODULE_ERROR = str(error)
else:
    HAVE_MODULE = False
    MODULE_ERROR = "numpy is not installed"

try:
    import torch

    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False

numpy_required = unittest.skipUnless(HAVE_MODULE, "numpy and the contract are needed: " + MODULE_ERROR)
torch_required = unittest.skipUnless(HAVE_MODULE and HAVE_TORCH, "torch is not installed; run with .venv/bin/python")

# A configuration small enough to run anywhere and shaped like the real one: three Gated
# DeltaNet layers then one full-attention layer, key and value heads of equal width, and a
# convolution kernel of 4.
TINY = dict(
    vocab_size=128,
    hidden_size=64,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=32,
    intermediate_size=128,
    full_attention_interval=4,
    linear_num_key_heads=2,
    linear_num_value_heads=2,
    linear_key_head_dim=16,
    linear_value_head_dim=16,
    linear_conv_kernel_dim=4,
    rms_norm_eps=1e-6,
    hidden_act="silu",
    rope_parameters={"rope_theta": 1e7, "rope_type": "default", "partial_rotary_factor": 0.25,
                     "mrope_section": [11, 11, 10], "mrope_interleaved": True},
)


@torch_required
class GatedDeltaNetLayerTests(unittest.TestCase):
    def build(self, seed: int = 0, length: int = 20):
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

        torch.manual_seed(seed)
        config = Qwen3_5TextConfig(**TINY)
        layer = Qwen3_5GatedDeltaNet(config, layer_idx=0).eval()
        hidden = torch.randn(1, length, config.hidden_size) * 0.5

        weights = {
            "in_proj_qkv": layer.in_proj_qkv.weight.detach().numpy().astype(np.float32),
            "in_proj_z": layer.in_proj_z.weight.detach().numpy().astype(np.float32),
            "in_proj_b": layer.in_proj_b.weight.detach().numpy().astype(np.float32),
            "in_proj_a": layer.in_proj_a.weight.detach().numpy().astype(np.float32),
            "conv1d": layer.conv1d.weight.detach().numpy().astype(np.float32),
            "A_log": layer.A_log.detach().numpy().astype(np.float32),
            "dt_bias": layer.dt_bias.detach().numpy().astype(np.float32),
            "norm": layer.norm.weight.detach().numpy().astype(np.float32),
            "out_proj": layer.out_proj.weight.detach().numpy().astype(np.float32),
        }
        return config, layer, hidden, weights

    def assert_close(self, ours, theirs, label):
        theirs = theirs.detach().numpy() if hasattr(theirs, "detach") else theirs
        scale = float(np.abs(theirs).max())
        error = float(np.abs(ours - theirs).max())
        self.assertLess(error, max(1e-5, scale * 1e-4), f"{label}: max |Δ| {error:.3e} vs scale {scale:.3e}")

    def test_the_layer_matches_the_reference(self):
        config, layer, hidden, weights = self.build()
        with torch.no_grad():
            theirs = layer(hidden)[0]
        ours = q35.gated_delta_net_layer(hidden.numpy(), weights, config)
        self.assert_close(ours, theirs, "gated delta net layer")

    def test_a_second_seed_agrees(self):
        config, layer, hidden, weights = self.build(seed=7, length=37)
        with torch.no_grad():
            theirs = layer(hidden)[0]
        ours = q35.gated_delta_net_layer(hidden.numpy(), weights, config)
        self.assert_close(ours, theirs, "gated delta net layer, seed 7")

    def test_a_length_beyond_one_chunk_agrees(self):
        """The chunk scan only runs with more than 64 positions, which is exactly where a
        padding or scan error would live."""
        config, layer, hidden, weights = self.build(seed=3, length=70)
        with torch.no_grad():
            theirs = layer(hidden)[0]
        ours = q35.gated_delta_net_layer(hidden.numpy(), weights, config)
        self.assert_close(ours, theirs, "length 70")


    def test_every_batch_element_is_convolved_independently(self):
        """The first version of the conv indexed batch 0 inside the loop, so a batch of 2
        returned the first sequence twice. Shapes hid it; this does not."""
        # With left padding of 3, output s reads x[s + k - 3] through tap k, so a weight of
        # [1, 0, 0, 0] delays the input by three positions.
        weights = {"conv1d": np.array([[[1.0, 0.0, 0.0, 0.0]]], dtype=np.float32)}
        x = np.zeros((2, 1, 6), dtype=np.float32)
        x[0, 0, 0] = 1.0
        x[1, 0, 1] = 3.0
        out = q35.depthwise_causal_conv(x, weights["conv1d"], activation=None)
        self.assertEqual(float(out[0, 0, 3]), 1.0)
        self.assertEqual(float(out[1, 0, 4]), 3.0)
        # Each batch element carries only its own input.
        self.assertEqual(float(out[1, 0, 3]), 0.0)
        self.assertEqual(float(out[0, 0, 4]), 0.0)

    def test_the_conv_is_causal(self):
        """Changing a later position must not change an earlier conv output; a
        centred convolution would pass every shape check and be wrong."""
        weights = {
            "conv1d": np.arange(2 * 1 * 4, dtype=np.float32).reshape(2, 1, 4) / 10.0,
        }
        x = np.zeros((1, 2, 10), dtype=np.float32)
        x[0, :, 3] = 1.0
        out = q35.depthwise_causal_conv(x, weights["conv1d"], activation=None)
        # The impulse at position 3 can only reach positions 3, 4, 5, 6 (kernel 4) and must
        # not appear before it.
        self.assertTrue(np.allclose(out[0, :, :3], 0.0))
        self.assertNotEqual(float(out[0, 0, 3]), 0.0)
        self.assertTrue(np.allclose(out[0, :, 7:], 0.0))

    def test_softplus_uses_the_threshold(self):
        """Above the threshold the reference returns the input unchanged, which is a
        different number from log1p(exp(x)) in the last bits."""
        large = np.array([25.0, 30.0], dtype=np.float32)
        self.assertTrue(np.array_equal(q35.softplus(large), large))
        small = np.array([0.0, 1.0, -1.0], dtype=np.float32)
        expected = np.log1p(np.exp(small.astype(np.float64))).astype(np.float32)
        np.testing.assert_allclose(q35.softplus(small), expected, rtol=1e-6)

    def test_the_decay_is_negative_and_the_gate_is_bounded(self):
        """g must be <= 0 (it is a log-space decay) and beta in (0, 1): a sign slip in
        either would produce a plausible, wrong model."""
        config, _, _, weights = self.build(seed=5)
        rng = np.random.default_rng(0)
        a = (rng.standard_normal((1, 12, 2)) * 2).astype(np.float32)
        b = (rng.standard_normal((1, 12, 2)) * 2).astype(np.float32)
        gate = -q35.exp32(weights["A_log"]) * q35.softplus(a + weights["dt_bias"])
        self.assertLessEqual(float(gate.max()), 0.0)
        beta = q35.sigmoid(b)
        self.assertGreater(float(beta.min()), 0.0)
        self.assertLess(float(beta.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
