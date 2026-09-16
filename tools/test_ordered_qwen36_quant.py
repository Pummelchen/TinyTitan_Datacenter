#!/usr/bin/env python3
"""M1c: what 4-bit experts cost the mixture, measured on the tiny `qwen3_5_moe` checkpoint.

The M0c equivalent asked the same question of a dense model and found 28 of 29 discrete
decisions preserved. For a mixture the question has a sharper form, because I3 says the
router's top-k must survive *exactly*: the policy keeps `router.logits` at bf16 precisely so
that it does, and this test is where that claim is checked rather than asserted.

Two measurements, deliberately separated:

- **the discrete decisions**, which must be identical — not close, identical;
- **the numbers**, which are allowed to move, with the observed movement recorded.

Needs numpy, safetensors and the venv, like the other quantization tests.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np

    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False
    np = None

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "DatacenterEngineTests" / "Fixtures" / "tiny-qwen36"

MODULE_ERROR = ""
if HAVE_NUMPY:
    try:
        from contract_source import open_source
        import ordered_qwen36 as q36
        import quantize

        HAVE_MODULE = True
    except ImportError as error:
        HAVE_MODULE = False
        MODULE_ERROR = str(error)
else:
    HAVE_MODULE = False
    MODULE_ERROR = "numpy is not installed"

numpy_required = unittest.skipUnless(HAVE_MODULE, "numpy and the tools are needed: " + MODULE_ERROR)


@numpy_required
class MixtureQuantizationTests(unittest.TestCase):
    def build(self) -> tuple[dict, Path, dict, dict]:
        """The tiny checkpoint's contract output, and the same on a fresh int4 install."""
        spec = json.loads((FIXTURE / "spec.json").read_text())
        policy_path = ROOT / "tools" / "quant_policy.json"
        policy = json.loads(policy_path.read_text())
        work = Path(tempfile.mkdtemp(prefix="m1c-"))
        install = work / "install"
        quantize.build_install(FIXTURE, install, spec, policy, str(policy_path))

        tokens = json.loads((FIXTURE / "golden.json").read_text())["tokens"]
        reference_capture: dict = {}
        reference_decisions: dict = {}
        q36.streamed_text_forward(
            spec, open_source(FIXTURE), tokens,
            capture=reference_capture, discrete=reference_decisions,
        )
        install_capture: dict = {}
        install_decisions: dict = {}
        q36.streamed_text_forward(
            spec, quantize.InstallSource(install), tokens,
            capture=install_capture, discrete=install_decisions,
        )
        return spec, install, {"capture": reference_capture, "decisions": reference_decisions}, {
            "capture": install_capture, "decisions": install_decisions
        }

    def testTheRouterDecisionsSurviveQuantizationExactly(self):
        """I3: the policy keeps the router at bf16, and this is the check that it worked.

        A tolerance cannot express this. If the decisions moved, every numeric check below
        could still be green while the continuations were unrelated.
        """
        _, _, reference, quantized = self.build()
        self.assertTrue(reference["decisions"], "the fixture must have a mixture")
        self.assertEqual(
            sorted(reference["decisions"]), sorted(quantized["decisions"]),
            "one decision entry per layer, and no extras",
        )
        for key in sorted(reference["decisions"]):
            np.testing.assert_array_equal(
                reference["decisions"][key], quantized["decisions"][key],
                err_msg=f"{key}: the chosen experts must be identical, not close",
            )

    def testTheNumbersMoveButStayBounded(self):
        """The measured cost, with the number recorded rather than a bound invented for it.

        The bound is loose on purpose: this is a tiny random model, whose logits are close to
        noise, so the relative divergence here says nothing about the 35 B model. What it does
        say is that the install does not produce garbage — and the exact figure belongs in the
        tracker, not in an assertion that would have to be renegotiated by every fixture change.
        """
        _, _, reference, quantized = self.build()
        a = reference["capture"]["logits"]
        b = quantized["capture"]["logits"]
        relative = float(np.abs(a - b).max()) / float(np.abs(a).max())
        self.assertLess(relative, 1.0, f"relative divergence {relative:.3e} is not a bounded cost")
        self.assertGreater(relative, 0.0, "4-bit experts that changed nothing would be suspicious")

    def testTheStackedExpertPayloadRoundTripsThroughTheInstall(self):
        """The format's core invariant at rank 3: a stacked expert tensor reads back as itself.

        The payload flattens the leading axis, so `experts` rows where there are
        `experts x rows` is the mistake this guards — and the reconstruction would still look
        like weights.
        """
        spec, install, _, _ = self.build()
        source = quantize.InstallSource(install)
        names = {tensor["role"]: tensor["name"] for tensor in spec["tensors"]}
        for role in ("expert.stack_gate_up", "expert.stack_down"):
            name = names[role]
            decoded = source.tensor(name)
            self.assertEqual(list(decoded.shape), spec_shape(spec, name), f"{role}: shape")
            # One row is one expert: the provider's fetch must return a whole expert's slice.
            one = source.rows(name, 0, 1)
            self.assertEqual(one.size, decoded[0].size, f"{role}: one row is one expert")
            np.testing.assert_allclose(one.reshape(decoded[0].shape), decoded[0], rtol=0, atol=0)


def spec_shape(spec: dict, name: str) -> list:
    for tensor in spec["tensors"]:
        if tensor["name"] == name:
            return list(tensor["shape"])
    raise AssertionError(f"{name} is not in the spec")


if __name__ == "__main__":
    unittest.main()
