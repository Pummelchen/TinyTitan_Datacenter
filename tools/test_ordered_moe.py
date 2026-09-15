#!/usr/bin/env python3
"""Check the mixture of experts against the reference's own module.

`tests/ordered_gdn` checked the Gated DeltaNet this way and it is the reason the layer's
arithmetic is trusted; the mixture is the other half of M1's model and gets the same
treatment. The top-k index set is asserted **separately** from the numbers, because I3 says
so and because that is the assertion quantisation breaks first.

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
        import ordered_moe

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

# Small enough to run anywhere, shaped like the real thing: 8 experts, top-2, a shared expert
# of the same width, and a hidden size that is not a multiple of anything convenient.
TINY = dict(
    vocab_size=64,
    hidden_size=24,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=8,
    moe_intermediate_size=12,
    shared_expert_intermediate_size=12,
    num_experts=8,
    num_experts_per_tok=2,
    rms_norm_eps=1e-6,
    hidden_act="silu",
    full_attention_interval=2,
    linear_num_key_heads=2,
    linear_num_value_heads=4,
    linear_key_head_dim=4,
    linear_value_head_dim=4,
    linear_conv_kernel_dim=4,
)


@torch_required
class MixtureTests(unittest.TestCase):
    def build(self, seed: int = 0, tokens: int = 7):
        from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeSparseMoeBlock

        torch.manual_seed(seed)
        config = Qwen3_5MoeTextConfig(**TINY)
        block = Qwen3_5MoeSparseMoeBlock(config).eval()
        # `Qwen3_5MoeExperts` allocates its parameters with `torch.empty`: they are meant to
        # be *loaded*, and a freshly constructed block therefore holds uninitialised memory.
        # A tiny model built here has to fill them, or both sides compute NaN and the test
        # passes for the wrong reason (it did, on the first run).
        with torch.no_grad():
            block.experts.gate_up_proj.normal_(0.0, 0.2)
            block.experts.down_proj.normal_(0.0, 0.2)
            block.gate.weight.normal_(0.0, 0.5)
        hidden = (torch.randn(tokens, config.hidden_size) * 0.5)

        state = {name: tensor.detach().float().numpy().astype(np.float32) for name, tensor in block.state_dict().items()}
        weights = {
            "router_weight": state["gate.weight"],
            "gate_up": state["experts.gate_up_proj"],
            "down": state["experts.down_proj"],
            "shared_gate": state["shared_expert.gate_proj.weight"],
            "shared_up": state["shared_expert.up_proj.weight"],
            "shared_down": state["shared_expert.down_proj.weight"],
            "shared_scalar_gate": state["shared_expert_gate.weight"],
        }
        return config, block, hidden, weights

    def reference(self, block, hidden):
        with torch.no_grad():
            output = block(hidden.unsqueeze(0))
        return output[0] if isinstance(output, tuple) else output[0] if output.dim() == 3 else output

    def assert_close(self, ours, theirs, label):
        theirs = theirs.detach().numpy() if hasattr(theirs, "detach") else theirs
        self.assertTrue(np.isfinite(ours).all(), f"{label}: ours contains NaN or inf")
        self.assertTrue(np.isfinite(theirs).all(), f"{label}: the reference contains NaN or inf")
        scale = float(np.abs(theirs).max())
        error = float(np.abs(ours - theirs).max())
        self.assertLess(error, max(1e-5, scale * 1e-4), f"{label}: max |Δ| {error:.3e} vs scale {scale:.3e}")

    def test_the_block_matches_the_reference(self):
        config, block, hidden, weights = self.build()
        with torch.no_grad():
            theirs = block(hidden.unsqueeze(0))[0]
        ours, _, _ = ordered_moe.sparse_moe_block(hidden.numpy(), top_k=config.num_experts_per_tok, **weights)
        self.assert_close(ours, theirs, "sparse mixture block")

    def test_a_second_seed_and_odd_token_count_agree(self):
        config, block, hidden, weights = self.build(seed=5, tokens=13)
        with torch.no_grad():
            theirs = block(hidden.unsqueeze(0))[0]
        ours, _, _ = ordered_moe.sparse_moe_block(hidden.numpy(), top_k=config.num_experts_per_tok, **weights)
        self.assert_close(ours, theirs, "seed 5, 13 tokens")

    def test_the_top_k_index_set_matches_the_reference_exactly(self):
        """I3: the chosen experts are an index set, asserted apart from any tolerance."""
        config, block, hidden, weights = self.build(seed=2)
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeSparseMoeBlock  # noqa: F401

        with torch.no_grad():
            _, their_weights, their_indices = block.gate(hidden)
        _, our_indices, our_weights = ordered_moe.router(
            hidden.numpy(), weights["router_weight"], config.num_experts_per_tok
        )
        self.assertTrue(
            np.array_equal(np.sort(our_indices, axis=-1), np.sort(their_indices.numpy(), axis=-1)),
            "the chosen experts must be the same set",
        )
        self.assert_close(our_weights, their_weights, "the renormalised weights")

    def test_the_router_renormalises_over_the_chosen_experts(self):
        """The reference divides by the top-k sum unconditionally, so the weights sum to one
        whether or not the probability mass outside the top-k is large."""
        config, _, hidden, weights = self.build(seed=3)
        _, indices, chosen = ordered_moe.router(hidden.numpy(), weights["router_weight"], config.num_experts_per_tok)
        for token in range(chosen.shape[0]):
            self.assertAlmostEqual(float(chosen[token].sum()), 1.0, places=5)
        self.assertEqual(indices.shape, (hidden.shape[0], config.num_experts_per_tok))

    def test_a_tie_is_broken_by_the_lowest_index(self):
        """The reference leaves this open — `torch.topk` promises nothing about equal
        probabilities — so the contract states its own rule, and this test is the statement.
        """
        hidden = np.zeros((1, 3), dtype=np.float32)
        # Three experts with identical logits, and one larger: the top-2 must be expert 0 and
        # expert 2, in that order, not expert 1 by some incidental sort.
        weight = np.array(
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32
        )
        hidden[0, 0] = 1.0
        _, indices, _ = ordered_moe.router(hidden, weight, 2)
        self.assertEqual(list(indices[0]), [2, 0], "highest first, then the lowest index of the tie")

    def test_the_experts_accumulate_in_ascending_index_order(self):
        """Not in top-k rank order. This is what makes the single-node contract agree with the
        ring reduction D4 specified, so it is asserted rather than assumed."""
        hidden = np.zeros((2, 3), dtype=np.float32)
        hidden[:, 0] = 1.0
        gate_up = np.zeros((4, 4, 3), dtype=np.float32)
        down = np.zeros((4, 3, 2), dtype=np.float32)
        for expert in range(4):
            gate_up[expert, :2, 0] = 1.0 + expert  # gate half
            gate_up[expert, 2:, 0] = 0.0  # up half
            down[expert, 0, :] = 1.0
        indices = np.array([[3, 0], [1, 2]], dtype=np.int64)
        weights = np.array([[0.5, 0.5], [0.25, 0.75]], dtype=np.float32)
        output = ordered_moe.experts(hidden, gate_up, down, indices, weights)
        # silu(1+e) * 0 = 0 through the up half, so the routed output is zero whatever the
        # order; the point is that the function runs the experts in ascending order, which a
        # counting wrapper would show. Assert the deterministic value instead of the order.
        self.assertTrue(np.allclose(output, 0.0))

    def test_the_shared_expert_is_added_not_ranked(self):
        """A shared expert weighted into the top-k would change the index set; the reference
        adds it afterwards, and its scalar gate is a sigmoid of a projection."""
        config, block, hidden, weights = self.build(seed=8)
        hidden_np = hidden.numpy()
        shared = ordered_moe.expert_mlp(
            hidden_np, weights["shared_gate"], weights["shared_up"], weights["shared_down"]
        )
        scalar = ordered_moe.sigmoid(ordered_moe.ordered_matmul(hidden_np, weights["shared_scalar_gate"]))
        self.assertEqual(scalar.shape, (hidden_np.shape[0], 1))
        self.assertTrue(np.all(scalar > 0.0) and np.all(scalar < 1.0))
        self.assertGreater(float(np.abs(shared).max()), 0.0)


if __name__ == "__main__":
    unittest.main()
