#!/usr/bin/env python3
"""Tests for the golden-trace harness: the format, the differ, and the fixtures.

The point of this file is the *seeded failures*. A differ that reports "identical"
is worthless unless it has been shown to report the opposite for each way a trace can
be wrong, so every defect class below is planted deliberately and then located:

- a one-ULP value change, located to the tensor and the element,
- a second change after the first, which must **not** be reported (the first
  divergence is the finding; the rest is noise),
- a transposed tensor, which must be a shape finding rather than a byte diff,
- a flipped discrete decision **with every float identical** — I3's failure mode,
- a missing or extra tensor,
- a reordered discrete set, which is a mismatch even though the values match,
- a trace edited after capture, which must be refused rather than compared.

Run: python3 -m unittest discover -s tools
"""

from __future__ import annotations

import contextlib
import io
import json
import struct
import tempfile
import unittest
from pathlib import Path

import make_synthetic_trace as syn
import trace_diff as td
import trace_format as tf


def bump_f32(value: float, ulps: int = 1) -> float:
    """Move an fp32 value by whole representable steps, in either direction."""
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    key = 0x80000000 - bits if bits & 0x80000000 else bits
    key += ulps
    bits = 0x80000000 - key if key & 0x80000000 else key
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


def f32_values(payload: bytes):
    return list(struct.unpack(f"<{len(payload) // 4}f", payload))


def pack_f32(values) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def replace_tensor(tensors, name, *, payload=None, shape=None, drop=False):
    out = []
    for entry in tensors:
        entry_name, dtype, entry_shape, entry_payload = entry
        if entry_name != name:
            out.append(entry)
            continue
        if drop:
            continue
        out.append((entry_name, dtype, shape or entry_shape, payload or entry_payload))
    return out


class HarnessCase(unittest.TestCase):
    """Shared plumbing: a reference trace and a candidate built from it."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.tensors, self.discrete = syn.synthetic_tensors(seed=1)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def write(self, name, tensors=None, discrete=None):
        path = self.root / name
        tf.write_trace(
            path,
            tensors=self.tensors if tensors is None else tensors,
            discrete=self.discrete if discrete is None else discrete,
            model={"repo": "synthetic", "revision": "seed-1"},
            reference={"compute_dtype": "f32"},
            producer="test",
        )
        return path

    def compare(self, reference, candidate):
        return td.compare(tf.read_trace(reference), tf.read_trace(candidate))

    def kinds(self, report):
        return [finding.kind for finding in report.findings]


class FormatTests(HarnessCase):
    def test_round_trip_and_digest_stability(self):
        first = self.write("a")
        second = self.write("b")
        a, b = tf.read_trace(first), tf.read_trace(second)
        self.assertEqual(a.digest, b.digest, "same input must produce the same digest")
        self.assertEqual(a.tensor("logits").payload, b.tensor("logits").payload)
        # f32 -> double -> f32 is exact, so a decode/re-encode round trip must be
        # byte-identical; anything else means the container lost precision.
        self.assertEqual(pack_f32(a.tensor("logits").values()), self.tensors[-1][3])

    def test_tensor_values_decode(self):
        trace = tf.read_trace(self.write("a"))
        values = trace.tensor("embed.out").values()
        self.assertEqual(len(values), syn.TOKENS * syn.HIDDEN)
        self.assertTrue(all(isinstance(v, float) for v in values))

    def test_edited_data_is_refused(self):
        path = self.write("a")
        data = bytearray((path / tf.DATA_FILE).read_bytes())
        data[0] ^= 0x01
        (path / tf.DATA_FILE).write_bytes(bytes(data))
        with self.assertRaises(tf.TraceError) as caught:
            tf.read_trace(path)
        self.assertIn("edited after capture", str(caught.exception))

    def test_edited_manifest_is_refused(self):
        path = self.write("a")
        manifest = json.loads((path / tf.MANIFEST_FILE).read_text())
        manifest["tensors"][0]["sha256"] = "0" * 64
        (path / tf.MANIFEST_FILE).write_text(json.dumps(manifest))
        with self.assertRaises(tf.TraceError):
            tf.read_trace(path)

    def test_write_refuses_a_payload_that_does_not_match_its_shape(self):
        with self.assertRaises(tf.TraceError) as caught:
            tf.write_trace(
                self.root / "bad",
                tensors=[("x", "f32", [4], pack_f32([1.0, 2.0]))],
            )
        self.assertIn("needs 16", str(caught.exception))

    def test_unknown_dtype_is_refused(self):
        with self.assertRaises(tf.TraceError):
            tf.write_trace(self.root / "bad", tensors=[("x", "f64", [1], b"\0" * 8)])

    def test_missing_manifest_is_refused(self):
        with self.assertRaises(tf.TraceError):
            tf.read_trace(self.root / "nothing-here")

    def test_ulp_distance_is_one_step_apart(self):
        value = 1.0
        nudged = bump_f32(value, 1)
        self.assertNotEqual(struct.pack("<f", value), struct.pack("<f", nudged))
        self.assertEqual(tf.ulp_distance_f32(value, nudged), 1)
        self.assertEqual(tf.ulp_distance_f32(value, value), 0)

    def test_bf16_rounding_reproduces_the_router_hazard(self):
        """R11, kept as an executable reminder: two distinct fp32 logits collapse to
        the same bf16 value, which is how a top-k set flips. Measured at 922 of
        20,000 random 256-way routers (4.61%) under round-to-nearest-even."""
        low, high = 5.363673687, 5.385571480
        self.assertNotEqual(low, high)
        self.assertEqual(tf.bf16_to_f32(tf.f32_to_bf16(low)), 5.375)
        self.assertEqual(tf.bf16_to_f32(tf.f32_to_bf16(high)), 5.375)


class DiffTests(HarnessCase):
    def test_identical_traces_pass(self):
        report = self.compare(self.write("ref"), self.write("cand"))
        self.assertTrue(report.identical, report.summary())
        self.assertGreater(report.checked_tensors, 5)
        self.assertGreater(report.discrete_checked, 0)

    def test_one_ulp_change_is_located_to_the_element(self):
        values = f32_values(dict((t[0], t[3]) for t in self.tensors)["layer.01.mlp_out"])
        values[3] = bump_f32(values[3], 1)
        candidate = self.write(
            "cand",
            replace_tensor(self.tensors, "layer.01.mlp_out", payload=pack_f32(values)),
        )
        report = self.compare(self.write("ref"), candidate)
        self.assertFalse(report.identical)
        self.assertEqual(self.kinds(report), ["float"])
        finding = report.findings[0]
        self.assertEqual(finding.where, "layer.01.mlp_out")
        self.assertIn("element 3", finding.detail)
        self.assertIn("1 ULP apart", finding.detail)

    def test_the_first_divergence_is_the_one_reported(self):
        early = f32_values(dict((t[0], t[3]) for t in self.tensors)["embed.out"])
        early[0] = bump_f32(early[0], 2)
        late = f32_values(dict((t[0], t[3]) for t in self.tensors)["logits"])
        late[0] = bump_f32(late[0], 2)
        tensors = replace_tensor(self.tensors, "embed.out", payload=pack_f32(early))
        tensors = replace_tensor(tensors, "logits", payload=pack_f32(late))
        report = self.compare(self.write("ref"), self.write("cand", tensors))
        self.assertEqual(self.kinds(report), ["float"])
        self.assertEqual(report.findings[0].where, "embed.out")

    def test_transposed_tensor_is_a_shape_finding(self):
        entry = dict((t[0], t) for t in self.tensors)["layer.00.hidden_in"]
        flipped = list(reversed(entry[2]))
        candidate = self.write(
            "cand", replace_tensor(self.tensors, "layer.00.hidden_in", shape=flipped)
        )
        report = self.compare(self.write("ref"), candidate)
        self.assertEqual(self.kinds(report), ["shape"])
        self.assertEqual(report.findings[0].where, "layer.00.hidden_in")

    def test_flipped_discrete_decision_with_identical_floats(self):
        """I3's failure mode: every number matches, the decision does not."""
        discrete = []
        for name, shape, values in self.discrete:
            mutated = list(values)
            if name == "layer.01.router.topk":
                mutated[0] = (mutated[0] + 1) % syn.HIDDEN
            discrete.append((name, shape, mutated))
        report = self.compare(self.write("ref"), self.write("cand", discrete=discrete))
        self.assertNotIn("float", self.kinds(report))
        self.assertIn("discrete", self.kinds(report))
        finding = [f for f in report.findings if f.kind == "discrete"][0]
        self.assertEqual(finding.where, "layer.01.router.topk")

    def test_reordered_discrete_set_is_reported(self):
        discrete = [
            (name, shape, list(reversed(values))) if name == "layer.00.router.topk" else (name, shape, values)
            for name, shape, values in self.discrete
        ]
        report = self.compare(self.write("ref"), self.write("cand", discrete=discrete))
        self.assertEqual(self.kinds(report), ["discrete"])
        self.assertIn("different order", report.findings[0].detail)

    def test_missing_and_extra_tensors_are_reported(self):
        missing = replace_tensor(self.tensors, "layer.01.mlp_out", drop=True)
        report = self.compare(self.write("ref"), self.write("cand", missing))
        self.assertEqual(self.kinds(report), ["missing_tensor"])

        extra = list(self.tensors) + [("layer.99.mlp_out", "f32", [1], pack_f32([0.0]))]
        report = self.compare(self.write("ref"), self.write("cand", extra))
        self.assertEqual(self.kinds(report), ["extra_tensor"])

    def test_cli_exit_codes(self):
        ref, cand = self.write("ref"), self.write("cand")
        changed = self.write(
            "changed",
            replace_tensor(
                self.tensors,
                "logits",
                payload=pack_f32([bump_f32(v, 1) for v in f32_values(dict((t[0], t[3]) for t in self.tensors)["logits"])]),
            ),
        )
        # The CLI prints a summary and writes errors to stderr; keep the suite's own
        # output readable.
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            self.assertEqual(td.main([str(ref), str(cand), "--quiet"]), 0)
            self.assertEqual(td.main([str(ref), str(changed), "--quiet"]), 1)
            self.assertEqual(td.main([str(ref), str(self.root / "absent")]), 2)
        self.assertIn("IDENTICAL", out.getvalue())
        self.assertIn("not a trace", err.getvalue())


class SyntheticTests(HarnessCase):
    def test_fixture_is_deterministic_and_seed_sensitive(self):
        first = self.write("first")
        second = self.write("second")
        self.assertEqual(tf.read_trace(first).digest, tf.read_trace(second).digest)

        other_tensors, other_discrete = syn.synthetic_tensors(seed=2)
        third = self.write("third", tensors=other_tensors, discrete=other_discrete)
        self.assertNotEqual(tf.read_trace(first).digest, tf.read_trace(third).digest)

    def test_fixture_carries_a_router_decision_per_layer(self):
        self.assertEqual(len(self.discrete), syn.LAYERS)
        for _name, shape, values in self.discrete:
            self.assertEqual(len(values), shape[0])
            self.assertEqual(len(set(values)), len(values), "a top-k selects distinct experts")
            self.assertTrue(all(0 <= v < syn.HIDDEN for v in values))


if __name__ == "__main__":
    unittest.main()
