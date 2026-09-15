#!/usr/bin/env python3
"""Tests for the reference-side capture tool.

These need torch, which lives in the project venv and not in CI, so they skip with an
explicit reason when it is missing. Run them with::

    .venv/bin/python -m unittest discover -s tools

The measurements they pin are the foundation of the gate (D3):

- two captures of the same input are **bit-identical** (the reference itself obeys I1);
- thread count did not change the result on this model — measured, not assumed, and
  re-measured on the real checkpoint before it is trusted;
- bf16 and fp32 differ in **every element**, and the divergence grows with depth —
  which is why the gate is bit-exactness and not a tolerance.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import trace_capture as tc
import trace_diff as td
import trace_format as tf

HAVE_TORCH = tc.torch_available()
SKIP_REASON = "torch is not installed in this interpreter; run with .venv/bin/python"


@unittest.skipUnless(HAVE_TORCH, SKIP_REASON)
class CaptureTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def capture(self, name, **kwargs):
        out = self.root / name
        manifest = tc.capture(out, tiny=True, **kwargs)
        return manifest, tf.read_trace(out)

    def test_capture_records_the_reference_configuration(self):
        manifest, trace = self.capture("c")
        reference = trace.manifest["reference"]
        self.assertEqual(reference["compute_dtype"], "f32")
        self.assertEqual(reference["attn_implementation"], "eager")
        self.assertTrue(reference["determinism"]["deterministic_algorithms"])
        self.assertEqual(reference["determinism"]["threads"], 1)
        self.assertEqual(reference["determinism"]["seed"], 1234)
        self.assertIn("transformers", reference)
        self.assertIn("torch", reference)
        self.assertEqual(trace.manifest["prompt"]["token_ids"], [1, 2, 3, 4, 5, 6, 7, 8])

    def test_capture_writes_a_tensor_at_every_layer_boundary(self):
        _manifest, trace = self.capture("c")
        names = trace.tensor_names
        self.assertEqual(names[0], "embed.out")
        self.assertEqual(names[-1], "logits")
        self.assertIn("final_norm.out", names)
        for index in range(4):
            for suffix in ("hidden_in", "mixer_out", "mlp_out"):
                self.assertIn(f"layer.{index:02d}.{suffix}", names)
        self.assertEqual(len(names), 1 + 4 * 3 + 2)

    def test_dense_capture_has_no_discrete_decisions(self):
        """A dense model has no router, and the trace says so by carrying none —
        rather than carrying a placeholder that could be mistaken for a checked
        decision."""
        _manifest, trace = self.capture("c")
        self.assertEqual(trace.discrete_names, [])

    def test_two_captures_are_bit_identical(self):
        """I1 on the reference side: same input, same configuration, same bytes."""
        _, first = self.capture("a")
        _, second = self.capture("b")
        self.assertEqual(first.digest, second.digest)
        self.assertTrue(td.compare(first, second).identical)

    def test_thread_count_did_not_change_this_model(self):
        """Measured on a 4-layer, 64-wide model: deterministic algorithms plus the
        same seed gave the same bytes at one thread and at four. That is a fact about
        this model, not a licence to stop pinning the thread count — the real
        checkpoint re-measures it, and the trace records whatever it was."""
        _, one = self.capture("one", threads=1)
        _, four = self.capture("four", threads=4)
        self.assertEqual(one.manifest["reference"]["determinism"]["threads"], 1)
        self.assertEqual(four.manifest["reference"]["determinism"]["threads"], 4)
        self.assertEqual(one.digest, four.digest, "thread count changed the result; pin it in the gate")

    def test_bf16_diverges_from_fp32_everywhere_and_grows_with_depth(self):
        """D3's evidence. bf16 and fp32 differ in every element, and the absolute
        divergence grows through the stack; relative error is not usable as a gate
        because it explodes on near-zero values (13285% on a 1e-6 activation)."""
        _, fp32 = self.capture("fp32")
        _, bf16 = self.capture("bf16", dtype_name="bf16")

        self.assertNotEqual(fp32.digest, bf16.digest)
        differing = 0
        for name in fp32.tensor_names:
            a, b = fp32.tensor(name).values(), bf16.tensor(name).values()
            if any(tf.ulp_distance_f32(x, y) for x, y in zip(a, b)):
                differing += 1
        self.assertEqual(differing, len(fp32.tensor_names), "every tensor should differ under bf16")

        def worst_absolute(name):
            a, b = fp32.tensor(name).values(), bf16.tensor(name).values()
            return max(abs(x - y) for x, y in zip(a, b))

        self.assertGreater(
            worst_absolute("final_norm.out"),
            worst_absolute("embed.out"),
            "the divergence should grow with depth",
        )

    def test_comparing_across_reference_configurations_is_refused(self):
        """An fp32 trace and a bf16 trace are two different reference configurations,
        not a candidate and a reference. The differ refuses rather than reporting a
        difference that belongs to the configuration; the *size* of that divergence is
        measured directly by test_bf16_diverges_from_fp32_everywhere_and_grows_with_depth."""
        _, fp32 = self.capture("fp32")
        _, bf16 = self.capture("bf16", dtype_name="bf16")
        report = td.compare(fp32, bf16)
        self.assertEqual([f.kind for f in report.findings], ["provenance"])
        self.assertIn("not comparable", report.findings[0].detail)

    def test_mixed_reference_stacks_are_refused(self):
        """Two traces from different reference builds are not two measurements of the
        same thing. Editing the recorded stack is enough to make the comparison
        refuse, even with identical bytes underneath — which is the point: the gate
        must not silently compare across a version change."""
        import json

        _, trace = self.capture("pinned")
        path = self.root / "restacked"
        import shutil

        shutil.copytree(self.root / "pinned", path)
        manifest = json.loads((path / tf.MANIFEST_FILE).read_text())
        manifest["reference"]["transformers"] = "0.0.0-other"
        (path / tf.MANIFEST_FILE).write_text(json.dumps(manifest))
        # The recorded stack is not part of the digest, so the edit is legal for the
        # container and must be caught by the comparison instead.
        other = tf.read_trace(path)
        report = td.compare(trace, other)
        self.assertEqual([f.kind for f in report.findings], ["provenance"])
        self.assertIn("not comparable", report.findings[0].detail)


if __name__ == "__main__":
    unittest.main()
