#!/usr/bin/env python3
"""The streaming forward must equal the resident one, tensor for tensor.

Two paths through the same contract is exactly how a project gets a divergence nobody
notices: the resident `text_model_forward` is what the oracle tests use, and the streaming
`streamed_text_forward` is what runs the real 2 B model on a node that cannot hold it. This
checks them against each other on the committed tiny checkpoint, so the path that only runs
where a 5 GB file exists is still covered by CI.

Skipped without numpy, like the other venv-run tests.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

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

FIXTURE = Path(__file__).resolve().parent.parent / "tests" / "DatacenterEngineTests" / "Fixtures" / "tiny-qwen35"
TOKENS = [3, 17, 5, 42, 8]

numpy_required = unittest.skipUnless(HAVE_MODULE, "numpy and the contract are needed: " + MODULE_ERROR)


class SafetensorsSource:
    """A whole-tensor reader with row ranges, over the fixture's checkpoint."""

    def __init__(self, path: Path):
        from safetensors import safe_open

        self._handle = safe_open(str(path), framework="pt")

    def tensor(self, name: str):
        return self._handle.get_tensor(name).float().numpy().astype(np.float32)

    def rows(self, name: str, start: int, end: int):
        return self._handle.get_slice(name)[start:end].float().numpy().astype(np.float32)


@numpy_required
class StreamingTests(unittest.TestCase):
    def setUp(self):
        spec_path = FIXTURE / "spec.json"
        if not spec_path.exists():
            self.skipTest("no spec.json; run tools/make_tiny_qwen35_checkpoint.py after building the engine")
        self.spec = json.loads(spec_path.read_text())
        self.source = SafetensorsSource(FIXTURE / "model.safetensors")

    def test_streamed_forward_equals_the_resident_one(self):
        golden = json.loads((FIXTURE / "golden.json").read_text())
        captured = {}
        q35.streamed_text_forward(self.spec, self.source, TOKENS, capture=captured)

        self.assertEqual(sorted(captured), sorted(golden["tensors"]), "same captured tensors")
        for name, expected in golden["tensors"].items():
            want = np.asarray(expected["bits"], dtype=np.uint32).view(np.float32).reshape(expected["shape"])
            got = captured[name]
            self.assertEqual(list(got.shape), list(expected["shape"]), name)
            np.testing.assert_array_equal(
                got.reshape(-1).view(np.uint32), want.reshape(-1).view(np.uint32),
                err_msg=f"{name}: the streaming path and the resident path disagree",
            )

    def test_the_spec_carries_every_role_the_layer_needs(self):
        blocks = q35.roles_by_block(self.spec)
        self.assertIn("embed", blocks)
        self.assertIn("norm.final", blocks["final"])
        for index in range(self.spec["config"]["numLayers"]):
            names = blocks[f"layer.{index:02d}"]
            self.assertIn("norm.attn", names)
            self.assertIn("mlp.gate", names)
            self.assertTrue("linear.in_qkv" in names or "attn.q" in names, names)

    def test_the_spec_config_is_enough_to_configure_the_contract(self):
        config = q35.SpecConfig(self.spec["config"])
        self.assertEqual(config.hidden_size, 32)
        self.assertEqual(config.layer_types, ["linear_attention"] * 3 + ["full_attention"])
        self.assertEqual(config.rope_parameters["partial_rotary_factor"], 0.5)
        self.assertEqual(config.linear_num_value_heads, 2)
        self.assertEqual(config.linear_key_head_dim, 8)


if __name__ == "__main__":
    unittest.main()
