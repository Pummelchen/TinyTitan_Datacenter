#!/usr/bin/env python3
"""Check the M1 text tower against the reference's own `Qwen3_5MoeTextModel`.

The other half of M1's arithmetic, after `test_ordered_moe` checked the mixture in isolation.
This one checks the *composition* — the mixture behind the layer's norms and residual, the
Gated DeltaNet and full attention in their proper places — and, above all, that the router's
**discrete decisions match the reference exactly at every layer**, which I3 requires to be an
assertion of its own rather than a consequence of the numbers being close.

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
        import ordered_qwen36 as q36

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

TINY = dict(
    vocab_size=128,
    hidden_size=32,
    num_hidden_layers=2,          # one Gated DeltaNet, one full attention
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=8,
    moe_intermediate_size=16,
    shared_expert_intermediate_size=16,
    num_experts=8,
    num_experts_per_tok=2,
    rms_norm_eps=1e-6,
    hidden_act="silu",
    full_attention_interval=2,
    attn_output_gate=True,
    tie_word_embeddings=False,    # this family ships a separate head
    linear_num_key_heads=2,
    linear_num_value_heads=4,
    linear_key_head_dim=8,
    linear_value_head_dim=8,
    linear_conv_kernel_dim=4,
    rope_parameters={"rope_theta": 1e7, "rope_type": "default", "partial_rotary_factor": 0.5},
)


@torch_required
class TextTowerTests(unittest.TestCase):
    def build(self, seed: int = 0, length: int = 9):
        from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextModel

        torch.manual_seed(seed)
        config = Qwen3_5MoeTextConfig(**TINY)
        model = Qwen3_5MoeTextModel(config).eval()
        # `Qwen3_5MoeExperts` allocates with `torch.empty`: a freshly built block is
        # uninitialised, so a tiny model has to fill the experts before anything is compared.
        with torch.no_grad():
            for layer in model.layers:
                layer.mlp.experts.gate_up_proj.normal_(0.0, 0.3)
                layer.mlp.experts.down_proj.normal_(0.0, 0.3)
                layer.mlp.gate.weight.normal_(0.0, 0.4)
        head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        with torch.no_grad():
            head.weight.normal_(0.0, 0.3)

        state = {name: tensor.detach().numpy().astype(np.float32) for name, tensor in model.state_dict().items()}
        weights = {
            "embed_tokens": state["embed_tokens.weight"],
            "norm": state["norm.weight"],
            "lm_head": head.weight.detach().numpy().astype(np.float32),
            "layers": [],
        }
        for index in range(config.num_hidden_layers):
            prefix = f"layers.{index}."
            layer = {
                "input_layernorm": state[prefix + "input_layernorm.weight"],
                "post_attention_layernorm": state[prefix + "post_attention_layernorm.weight"],
                "mlp": {
                    "router_weight": state[prefix + "mlp.gate.weight"],
                    "gate_up": state[prefix + "mlp.experts.gate_up_proj"],
                    "down": state[prefix + "mlp.experts.down_proj"],
                    "shared_gate": state[prefix + "mlp.shared_expert.gate_proj.weight"],
                    "shared_up": state[prefix + "mlp.shared_expert.up_proj.weight"],
                    "shared_down": state[prefix + "mlp.shared_expert.down_proj.weight"],
                    "shared_scalar_gate": state[prefix + "mlp.shared_expert_gate.weight"],
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

    def spec_config(self, config):
        """The contract reads a *spec*; here the fields come straight from the reference.

        `full_attention_interval` is consumed by the reference's constructor and is not an
        attribute afterwards — `layer_types` is the authoritative list — so the interval here
        is recovered from that list, and `test_the_layer_kinds_follow_the_same_pattern` checks
        that the contract derives the same list back.
        """
        interval = next(index for index, kind in enumerate(config.layer_types) if kind == "full_attention") + 1
        return q36.SpecConfig(
            {
                "hiddenSize": config.hidden_size,
                "numLayers": config.num_hidden_layers,
                "numAttentionHeads": config.num_attention_heads,
                "numKeyValueHeads": config.num_key_value_heads,
                "headDim": config.head_dim,
                "intermediateSize": config.moe_intermediate_size,
                "vocabSize": config.vocab_size,
                "rmsNormEps": config.rms_norm_eps,
                "tieWordEmbeddings": False,
                "attnOutputGate": config.attn_output_gate,
                "fullAttentionInterval": interval,
                "numExperts": config.num_experts,
                "numExpertsPerTok": config.num_experts_per_tok,
                "moeIntermediateSize": config.moe_intermediate_size,
                "sharedExpertIntermediateSize": config.shared_expert_intermediate_size,
                "linearKeyHeads": config.linear_num_key_heads,
                "linearValueHeads": config.linear_num_value_heads,
                "linearKeyDim": config.linear_num_key_heads * config.linear_key_head_dim,
                "linearValueHeadDim": config.linear_value_head_dim,
                "linearConvKernelDim": config.linear_conv_kernel_dim,
                "ropeTheta": 1e7,
                "partialRotaryFactor": 0.5,
            }
        )

    def reference(self, model, tokens, capture=None):
        """The reference tower, with the router's own decisions captured as it runs."""
        decisions = {}
        hooks = []
        for index, layer in enumerate(model.layers):
            def hook(module, args, output, index=index):
                decisions[f"layer.{index:02d}.router.topk"] = output[2].detach().numpy()
            hooks.append(layer.mlp.gate.register_forward_hook(hook))
            if capture is not None:
                def layer_hook(module, args, output, index=index):
                    capture[f"layer.{index:02d}.hidden_out"] = output.detach().numpy()[0]
                hooks.append(layer.register_forward_hook(layer_hook))
        try:
            with torch.no_grad():
                out = model(input_ids=torch.tensor([tokens]), use_cache=False)
        finally:
            for hook in hooks:
                hook.remove()
        return out.last_hidden_state.numpy()[0], decisions

    def assert_close(self, ours, theirs, label):
        scale = float(np.abs(theirs).max())
        error = float(np.abs(ours - theirs).max())
        self.assertLess(error, max(1e-5, scale * 1e-4), f"{label}: max |Δ| {error:.3e} vs scale {scale:.3e}")

    def test_the_layer_kinds_are_what_the_interval_says(self):
        config, _, _, _ = self.build()
        self.assertEqual(config.layer_types, ["linear_attention", "full_attention"])

    def test_the_layer_kinds_follow_the_same_pattern(self):
        """The contract derives the layer kinds from an interval; the reference uses a list.
        They have to agree, and the interval is not even an attribute on the reference's
        configuration object — it is consumed by its constructor."""
        config, _, _, _ = self.build()
        self.assertEqual(self.spec_config(config).layer_types, config.layer_types)

    def test_the_tower_matches_the_reference(self):
        """The reference's `TextModel` returns the *normed* hidden state, so that is what the
        contract's `final_norm.out` is compared against — not the logits, which are a
        different shape and a different thing."""
        config, model, weights, tokens = self.build()
        reference, _ = self.reference(model, tokens)
        capture = {}
        q36.text_model_forward(weights, self.spec_config(config), tokens, capture=capture)
        self.assert_close(capture["final_norm.out"], reference, "final hidden state")

    def test_every_layer_matches_the_reference(self):
        """Layer by layer, so a divergence is located rather than just detected."""
        config, model, weights, tokens = self.build(seed=3)
        their_capture = {}
        reference, _ = self.reference(model, tokens, capture=their_capture)
        our_capture = {}
        q36.text_model_forward(weights, self.spec_config(config), tokens, capture=our_capture)
        for index in range(config.num_hidden_layers):
            key = f"layer.{index:02d}.hidden_out"
            self.assert_close(our_capture[key], their_capture[key], key)

    def test_the_router_decisions_match_the_reference_at_every_layer(self):
        """I3, as its own assertion: the same experts, in the same order, at every layer."""
        config, model, weights, tokens = self.build(seed=5)
        _, theirs = self.reference(model, tokens)
        ours = {}
        q36.text_model_forward(weights, self.spec_config(config), tokens, discrete=ours)
        self.assertEqual(sorted(ours), sorted(theirs), "a decision per layer, and no extras")
        for key in sorted(theirs):
            self.assertTrue(
                np.array_equal(ours[key], theirs[key]),
                f"{key}: ours {ours[key].tolist()} vs reference {theirs[key].tolist()}",
            )

    def test_a_second_seed_agrees_on_both_numbers_and_decisions(self):
        config, model, weights, tokens = self.build(seed=11, length=14)
        reference, their_decisions = self.reference(model, tokens)
        our_decisions = {}
        capture = {}
        q36.text_model_forward(
            weights, self.spec_config(config), tokens, capture=capture, discrete=our_decisions
        )
        self.assert_close(capture["final_norm.out"], reference, "seed 11")
        for key in their_decisions:
            self.assertTrue(np.array_equal(our_decisions[key], their_decisions[key]), key)

    def test_the_head_is_its_own_tensor(self):
        """This family is not tied: the logits come from `lm_head`, not the embedding."""
        config, _, weights, tokens = self.build(seed=7)
        with np.testing.assert_raises(AssertionError):
            np.testing.assert_array_equal(weights["embed_tokens"], weights["lm_head"])
        logits = q36.text_model_forward(weights, self.spec_config(config), tokens)
        self.assertEqual(logits.shape, (len(tokens), config.vocab_size))


if __name__ == "__main__":
    unittest.main()
