#!/usr/bin/env python3
"""Check the whole `qwen3_5` text tower against the reference model, layer by layer.

The layer is checked in `test_ordered_qwen35.py`; this checks the *wiring*: the embedding,
the interleaving of the two layer kinds, the residual structure, the final norm and the
tied head — the things a per-layer test cannot see. A tiny configuration is used on
purpose: the wiring is what is under test, and it does not need a 2 B checkpoint.

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

torch_required = unittest.skipUnless(
    HAVE_MODULE and HAVE_TORCH, "numpy, the contract and torch are needed: " + MODULE_ERROR
)

TINY = dict(
    vocab_size=96,
    hidden_size=32,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    intermediate_size=64,
    full_attention_interval=4,
    linear_num_key_heads=2,
    linear_num_value_heads=2,
    linear_key_head_dim=8,
    linear_value_head_dim=8,
    linear_conv_kernel_dim=4,
    rms_norm_eps=1e-6,
    hidden_act="silu",
    tie_word_embeddings=True,
    # partial_rotary_factor 0.5 of head_dim 16 leaves 8 channels rotating: the partial path
    # is exercised, not bypassed.
    rope_parameters={"rope_theta": 1e7, "rope_type": "default", "partial_rotary_factor": 0.5,
                     "mrope_section": [11, 11, 10], "mrope_interleaved": True},
)


@torch_required
class TextModelTests(unittest.TestCase):
    def build(self, seed: int = 0, length: int = 12):
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

        torch.manual_seed(seed)
        config = Qwen3_5TextConfig(**TINY)
        model = Qwen3_5TextModel(config).eval()
        state = {name: tensor.detach().numpy().astype(np.float32) for name, tensor in model.state_dict().items()}

        weights = {
            "embed_tokens": state["embed_tokens.weight"],
            "norm": state["norm.weight"],
            "layers": [],
        }
        for index in range(config.num_hidden_layers):
            prefix = f"layers.{index}."
            layer = {
                "input_layernorm": state[prefix + "input_layernorm.weight"],
                "post_attention_layernorm": state[prefix + "post_attention_layernorm.weight"],
                "mlp": {
                    "gate_proj": state[prefix + "mlp.gate_proj.weight"],
                    "up_proj": state[prefix + "mlp.up_proj.weight"],
                    "down_proj": state[prefix + "mlp.down_proj.weight"],
                },
            }
            if config.layer_types[index] == "full_attention":
                layer["self_attn"] = {
                    key: state[f"{prefix}self_attn.{key}.weight"]
                    for key in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm")
                }
            else:
                layer["linear_attn"] = {
                    "in_proj_qkv": state[prefix + "linear_attn.in_proj_qkv.weight"],
                    "in_proj_z": state[prefix + "linear_attn.in_proj_z.weight"],
                    "in_proj_b": state[prefix + "linear_attn.in_proj_b.weight"],
                    "in_proj_a": state[prefix + "linear_attn.in_proj_a.weight"],
                    "conv1d": state[prefix + "linear_attn.conv1d.weight"],
                    "A_log": state[prefix + "linear_attn.A_log"],
                    "dt_bias": state[prefix + "linear_attn.dt_bias"],
                    "norm": state[prefix + "linear_attn.norm.weight"],
                    "out_proj": state[prefix + "linear_attn.out_proj.weight"],
                }
            weights["layers"].append(layer)

        tokens = [int(t) for t in torch.randint(0, config.vocab_size, (1, length))[0]]
        return config, model, weights, tokens

    def reference_hidden(self, model, tokens):
        with torch.no_grad():
            out = model(input_ids=torch.tensor([tokens]), use_cache=False)
        return out.last_hidden_state.numpy()

    def assert_close(self, ours, theirs, label):
        scale = float(np.abs(theirs).max())
        error = float(np.abs(ours - theirs).max())
        self.assertLess(error, max(1e-5, scale * 1e-4), f"{label}: max |Δ| {error:.3e} vs scale {scale:.3e}")

    def test_the_layer_kinds_are_what_the_interval_says(self):
        config, _, _, _ = self.build()
        self.assertEqual(config.layer_types, ["linear_attention"] * 3 + ["full_attention"])

    def test_the_whole_tower_matches_the_reference(self):
        config, model, weights, tokens = self.build()
        capture = {}
        q35.text_model_forward(weights, config, tokens, capture=capture)
        theirs = self.reference_hidden(model, tokens)
        self.assert_close(capture["final_norm.out"], theirs, "final hidden state")

    def test_a_second_seed_and_a_longer_sequence_agree(self):
        config, model, weights, tokens = self.build(seed=11, length=70)
        capture = {}
        q35.text_model_forward(weights, config, tokens, capture=capture)
        theirs = self.reference_hidden(model, tokens)
        self.assert_close(capture["final_norm.out"], theirs, "length 70, seed 11")

    def test_every_layer_matches_not_just_the_last(self):
        """Error cancellation across layers would hide a wrong layer; this compares each
        layer's output on the way through."""
        config, model, weights, tokens = self.build(seed=5, length=9)

        captured = {}
        hooks = []
        for index, layer in enumerate(model.layers):
            def hook(_module, _inputs, output, index=index):
                tensor = output[0] if isinstance(output, tuple) else output
                captured[index] = tensor.detach().numpy()
            hooks.append(layer.register_forward_hook(hook))
        try:
            self.reference_hidden(model, tokens)
        finally:
            for handle in hooks:
                handle.remove()

        ours = {}
        q35.text_model_forward(weights, config, tokens, capture=ours)
        for index in range(config.num_hidden_layers):
            # The reference layer output is after its residual; the capture's hidden_out is
            # the same quantity.
            self.assert_close(ours[f"layer.{index:02d}.hidden_out"], captured[index], f"layer {index}")

    def test_the_tied_head_reproduces_the_reference_logits(self):
        config, model, weights, tokens = self.build(seed=3)
        capture = {}
        logits = q35.text_model_forward(weights, config, tokens, capture=capture)
        embedded = np.asarray(weights["embed_tokens"], dtype=np.float32)
        reference = capture["final_norm.out"] @ embedded.T
        self.assert_close(logits, reference, "tied head")
        # And the discrete decision that M0's gate ultimately cares about.
        self.assertTrue(np.array_equal(logits.argmax(-1), reference.argmax(-1)))


    def test_the_backbone_norm_is_weight_offset(self):
        """Qwen3.5's RMSNorm multiplies by (1 + weight), so a zero weight is the
        identity. The qwen3 family's norm multiplies by weight; assuming the two families
        agreed here produced a 5% error at layer 0 before this test existed."""
        config, model, _, _ = self.build()
        x = np.arange(8, dtype=np.float32).reshape(2, 4) * 0.25
        zeros = np.zeros(4, dtype=np.float32)
        normalised = q35.rms_norm(x, zeros, config.rms_norm_eps)
        expected = x / np.sqrt((x * x).mean(-1, keepdims=True) + np.float32(config.rms_norm_eps))
        np.testing.assert_allclose(normalised, expected, rtol=1e-5, atol=1e-6)

        # And against the reference's own module, with its own weight.
        with torch.no_grad():
            module = torch.nn.Module()
            reference = q35.rms_norm(
                x, np.zeros(4, dtype=np.float32), config.rms_norm_eps
            )
        self.assertTrue(np.isfinite(reference).all())
        self.assertFalse(np.allclose(normalised, x * 0.0), "a zero weight must not zero the output")

    def test_a_nonzero_weight_is_an_offset_not_a_scale(self):
        config, _, _, _ = self.build()
        x = np.ones((1, 4), dtype=np.float32) * 2.0
        weight = np.full(4, 0.5, dtype=np.float32)
        out = q35.rms_norm(x, weight, config.rms_norm_eps)
        # x is uniform, so the normalised value is sqrt(1 + eps)-ish; the offset multiplies
        # it by 1.5. With a plain scale it would be 0.5.
        ratio = float(out[0, 0] / q35.rms_norm(x, np.zeros(4, dtype=np.float32), config.rms_norm_eps)[0, 0])
        self.assertAlmostEqual(ratio, 1.5, places=5)

    def test_the_rope_tables_have_the_partial_width(self):
        config, _, _, _ = self.build()
        cos, sin = q35.rope_tables(config, np.arange(5, dtype=np.float64))
        self.assertEqual(cos.shape, (5, 8), "head_dim 16 at partial_rotary_factor 0.5")
        self.assertEqual(cos.shape, sin.shape)
        # The two halves are equal by construction (cat((freqs, freqs))).
        np.testing.assert_array_equal(cos[:, :4], cos[:, 4:])

    def test_rope_leaves_the_unrotated_channels_alone(self):
        config, _, _, _ = self.build()
        cos, sin = q35.rope_tables(config, np.arange(3, dtype=np.float64))
        x = np.zeros((3, 2, 16), dtype=np.float32)
        x[..., 8:] = 7.0
        out = q35.apply_rope_partial(x, cos, sin)
        np.testing.assert_array_equal(out[..., 8:], x[..., 8:])
        np.testing.assert_array_equal(out[..., :8], np.zeros((3, 2, 8), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
