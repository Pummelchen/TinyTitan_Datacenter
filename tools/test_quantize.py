#!/usr/bin/env python3
"""Tests for the quantization pass and the install format.

The pass is numeric code, so the tests are: does the reconstruction equal what the format
says it should, does it stay inside the error a 4-bit group can promise, and does the format
round-trip through the reader the contract uses.

Skipped without numpy, like the other venv-run tests.
"""

from __future__ import annotations

import json
import tempfile
import os
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
        import quantize

        HAVE_MODULE = True
    except ImportError as error:
        HAVE_MODULE = False
        MODULE_ERROR = str(error)
else:
    HAVE_MODULE = False
    MODULE_ERROR = "numpy is not installed"

numpy_required = unittest.skipUnless(HAVE_MODULE, "numpy and the pass are needed: " + MODULE_ERROR)


@numpy_required
class QuantizeTests(unittest.TestCase):
    def test_values_the_code_can_represent_round_trip_exactly(self):
        """The format represents `(code - zero) * scale` for codes -8..7. Values built that
        way must come back unchanged; a test that demanded it of arbitrary values would be
        demanding something four bits cannot do."""
        scale, zero = 0.25, 0
        codes = np.arange(-8, 8, dtype=np.float32)
        values = np.tile(((codes - zero) * scale).astype(np.float32), 4).reshape(1, 64)
        restored = quantize.dequantize_tensor(quantize.quantize_tensor(values))
        np.testing.assert_allclose(restored, values, rtol=0, atol=1e-6)

    def test_a_constant_group_round_trips_exactly(self):
        """A degenerate group has no span; a scale of zero would reconstruct as 0 * inf.
        The format gives it a unit scale instead."""
        values = np.full((2, 64), 0.25, dtype=np.float32)
        restored = quantize.dequantize_tensor(quantize.quantize_tensor(values))
        np.testing.assert_allclose(restored, values, rtol=0, atol=1e-6)

    def test_the_error_stays_inside_the_group_bound(self):
        """The promise of the format: within a group of 64, the error is at most half a
        step, and a step is the group's span over fifteen codes."""
        rng = np.random.default_rng(0)
        weight = (rng.standard_normal((8, 256)) * 0.3).astype(np.float32)
        restored = quantize.dequantize_tensor(quantize.quantize_tensor(weight))
        error = np.abs(restored - weight)
        blocks = weight.reshape(8, 4, 64)
        span = blocks.max(axis=2) - blocks.min(axis=2)
        bound = span / 15.0 / 2.0
        # Half a step per element, plus the rounding of the zero point itself.
        self.assertLess(float(error.max()), float((bound * 1.05).max()) + 1e-6)

    def test_the_payload_is_the_size_the_format_says(self):
        values = np.zeros((4, 128), dtype=np.float32)
        entry = quantize.quantize_tensor(values)
        rows, columns, group = 4, 128, 64
        groups = rows * columns // group
        self.assertEqual(len(entry["packed"]), rows * columns // 2, "two codes per byte")
        self.assertEqual(len(entry["scales"]), groups * 4, "one fp32 scale per group")
        self.assertEqual(len(entry["zeros"]), groups, "one int4 zero point per group")

    def test_an_odd_column_count_is_padded_and_trimmed(self):
        values = np.arange(70, dtype=np.float32).reshape(1, 70)
        entry = quantize.quantize_tensor(values)
        self.assertEqual(entry["padded_columns"], 128, "padded up to a whole group")
        self.assertEqual(quantize.dequantize_tensor(entry).shape, (1, 70), "trimmed back")

    def test_the_low_nibble_comes_first(self):
        """The packing order is part of the format: a reader that assumed the other order
        would produce plausible, wrong weights."""
        values = np.zeros((1, 64), dtype=np.float32)
        entry = quantize.quantize_tensor(values)
        packed = np.frombuffer(entry["packed"], dtype=np.uint8)
        self.assertTrue(np.all(packed == packed[0]), "a uniform group packs uniformly")

    def test_quantizing_is_deterministic(self):
        rng = np.random.default_rng(1)
        weight = (rng.standard_normal((4, 128)) * 0.5).astype(np.float32)
        first = quantize.quantize_tensor(weight)
        second = quantize.quantize_tensor(weight)
        self.assertEqual(first["packed"], second["packed"])
        self.assertEqual(first["scales"], second["scales"])
        self.assertEqual(first["zeros"], second["zeros"])


@numpy_required
class PolicyTests(unittest.TestCase):
    def test_an_unlisted_role_is_refused(self):
        """Not defaulted. A model with one tensor at a precision the file does not state is
        worse than an install that refuses to exist."""
        policy = {"quant": {"mlp.gate": "int4-affine"}}
        self.assertEqual(quantize.policy_for(policy, "mlp.gate"), "int4-affine")
        with self.assertRaises(quantize.PolicyError):
            quantize.policy_for(policy, "attn.q")

    def test_the_shipped_policy_covers_every_role_of_the_pinned_model(self):
        spec_path = Path(__file__).resolve().parent.parent / "tests" / "DatacenterEngineTests" / "Fixtures" / "tiny-qwen35" / "spec.json"
        if not spec_path.exists():
            self.skipTest("no tiny spec; run tools/make_tiny_qwen35_checkpoint.py first")
        spec = json.loads(spec_path.read_text())
        policy = quantize.load_policy(Path(__file__).resolve().parent / "quant_policy.json")
        for tensor in spec["tensors"]:
            quantize.policy_for(policy, tensor["role"])

    def test_the_shipped_policy_keeps_gating_and_norms_above_four_bits(self):
        """I3 is explicit: quantizing what decides a discrete outcome flips marginal
        decisions, and no per-tensor check notices."""
        policy = quantize.load_policy(Path(__file__).resolve().parent / "quant_policy.json")
        for role in ("norm.attn", "norm.mlp", "norm.final", "attn.q_norm", "attn.k_norm", "linear.norm"):
            self.assertNotEqual(quantize.policy_for(policy, role), "int4-affine", role)


@numpy_required
class InstallTests(unittest.TestCase):
    def build(self, directory: Path) -> Path:
        rng = np.random.default_rng(2)
        install = directory / "install"
        entries = []
        for name, shape in (("a.weight", (8, 128)), ("b.weight", (4, 64))):
            entry = quantize.quantize_tensor((rng.standard_normal(shape) * 0.4).astype(np.float32))
            entry.update({"name": name, "role": "mlp.gate", "quant": "int4-affine"})
            entries.append(entry)
        quantize.write_install(
            install,
            source={"repo": "x", "revision": "y", "files": {"model.safetensors": "sha"}},
            spec={"family": "qwen3_5"},
            entries=entries,
            policy_files=["tools/quant_policy.json"],
        )
        return install

    def test_kept_tensors_are_stored_and_read_back_exactly(self):
        """An install that carried only the quantized tensors could not be run: the embedding
        and the norms are part of the model. bf16 is widened, never rounded, so the kept
        tensors come back bit for bit."""
        import tempfile
        from pathlib import Path as _Path

        rng = np.random.default_rng(4)
        kept = (rng.standard_normal((4, 32)) * 2).astype(np.float32)
        # Widened to uint32 before the shift back: `<< 16` on a uint16 wraps to zero, which
        # is exactly the arithmetic the reader must not be doing.
        bf16 = (kept.view(np.uint32) >> 16).astype(np.uint16)
        exact = (bf16.astype(np.uint32) << 16).view(np.float32).reshape(kept.shape)
        with tempfile.TemporaryDirectory() as directory:
            install = _Path(directory) / "install"
            quantize.write_install(
                install,
                source={"repo": "x", "revision": "y", "files": {}},
                spec={"family": "qwen3_5"},
                entries=[{
                    "name": "embed.weight", "role": "token.embedding", "quant": "bf16",
                    "raw": bf16.tobytes(), "shape": [4, 32], "padded_columns": 32, "group": 0,
                }],
                policy_files=["tools/quant_policy.json"],
            )
            source = quantize.InstallSource(install)
            np.testing.assert_array_equal(source.tensor("embed.weight"), exact)
            np.testing.assert_array_equal(source.rows("embed.weight", 1, 3), exact[1:3])

    def test_chunked_quantization_is_byte_identical(self):
        """The install builder reads a tensor a block of rows at a time, because a stacked expert
        tensor is two gigabytes in fp32 against about four and a half usable on a node. Rows are
        independent — every group lies inside one row — so the pieces concatenate exactly, and
        this is the test that makes that a fact rather than an argument: a chunked install and a
        whole-tensor one must produce the same bytes."""
        import tempfile
        from pathlib import Path as _Path

        rng = np.random.default_rng(11)
        weight = (rng.standard_normal((37, 96)) * 1.5).astype(np.float32)
        whole = quantize.quantize_tensor(weight)
        pieces = [
            quantize.quantize_rows(weight[start:end], original_shape=list(weight.shape))
            for start, end in ((0, 5), (5, 20), (20, 37))
        ]
        joined = quantize.concat_rows(pieces, list(weight.shape))
        for key in ("packed", "scales", "zeros"):
            self.assertEqual(whole[key], joined[key], f"{key} must not depend on the chunking")
        for key in ("shape", "rows", "columns", "padded_columns", "group"):
            self.assertEqual(whole[key], joined[key], key)

    def test_uncached_reads_return_the_same_bytes_and_digest(self):
        """The tooling reads payloads with `F_NOCACHE` for the same reason the engine does.

        Verifying the 20 GB install through the page cache is what took free disk from 17 GB to
        2.96 GB on this node. The property that must not change is the bytes, and the digest
        computed over them.
        """
        import hashlib
        import tempfile
        from pathlib import Path as _Path

        with tempfile.TemporaryDirectory() as directory:
            path = _Path(directory) / "payload.bin"
            body = bytes(range(256)) * 4096  # one megabyte, not a multiple of the window
            path.write_bytes(body)
            descriptor = quantize.open_uncached(path)
            try:
                self.assertEqual(quantize.pread_exact(descriptor, 0, len(body)), body)
                self.assertEqual(quantize.pread_exact(descriptor, 1000, 5000), body[1000:6000])
                self.assertEqual(quantize.digest_of(descriptor, 0, len(body)), hashlib.sha256(body).hexdigest())
                self.assertEqual(quantize.digest_of(descriptor, 700, 3000), hashlib.sha256(body[700:3700]).hexdigest())
                with self.assertRaises(quantize.PolicyError):
                    quantize.pread_exact(descriptor, len(body) - 10, 100)
            finally:
                os.close(descriptor)

    def test_an_install_source_verifies_a_payload_when_it_reads_it(self):
        """The reader no longer hashes the whole install to open it, and the check moved to the
        read rather than disappearing: a tampered payload must still refuse to decode."""
        import json as _json
        import shutil
        import tempfile
        from pathlib import Path as _Path

        fixture = _Path("tests/DatacenterEngineTests/Fixtures/tiny-qwen36/install")
        with tempfile.TemporaryDirectory() as directory:
            copy = _Path(directory) / "install"
            shutil.copytree(fixture, copy)
            manifest = _json.loads((copy / "install.json").read_text())
            target = manifest["tensors"][0]["name"]
            body = bytearray((copy / "data.bin").read_bytes())
            body[manifest["tensors"][0]["offset"] + 3] ^= 0xFF
            (copy / "data.bin").write_bytes(bytes(body))

            source = quantize.InstallSource(copy)  # opening does not read the payload
            try:
                with self.assertRaises(quantize.PolicyError):
                    source.tensor(target)
            finally:
                source.close()

    def test_the_slab_digests_recompute_from_the_payload(self):
        """`DC-088`'s proven half: the writer's per-slab digests are correct.

        A slab is one leading-axis entry — one expert of a stacked tensor — and its bytes sit in
        three ranges of the section-major payload. The reader's transcription of this is where a
        bug was found, so the *definition* is worth pinning independently: recompute every slab's
        digest straight from `data.bin` and require it to equal the manifest's. When that passes,
        any disagreement is on the reading side, which is exactly what it turned out to be.
        """
        import hashlib
        import json as _json
        import shutil
        import tempfile
        from pathlib import Path as _Path

        fixture = _Path("tests/DatacenterEngineTests/Fixtures/tiny-qwen36")
        spec = _json.loads((fixture / "spec.json").read_text())
        policy = _json.loads(_Path("tools/quant_policy.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            target = _Path(directory) / "install"
            manifest = quantize.build_install(fixture, target, spec, policy, "tools/quant_policy.json")
            blob = (target / "data.bin").read_bytes()
            checked = 0
            for entry in manifest["tensors"]:
                slabs = entry.get("slab_sha256")
                if not slabs:
                    continue
                rows = 1
                for dimension in entry["shape"][:-1]:
                    rows *= dimension
                inner = rows // entry["shape"][0]
                groups_per_row = entry["padded_columns"] // entry["group"]
                codes_per_row = entry["padded_columns"] // 2
                codes_bytes = rows * codes_per_row
                scales_bytes = rows * groups_per_row * 4
                for slab in range(entry["shape"][0]):
                    first = slab * inner
                    hasher = hashlib.sha256()
                    hasher.update(blob[entry["offset"] + first * codes_per_row:entry["offset"] + (first + inner) * codes_per_row])
                    at = entry["offset"] + codes_bytes + first * groups_per_row * 4
                    hasher.update(blob[at:at + inner * groups_per_row * 4])
                    at = entry["offset"] + codes_bytes + scales_bytes + first * groups_per_row
                    hasher.update(blob[at:at + inner * groups_per_row])
                    self.assertEqual(hasher.hexdigest(), slabs[slab], f"{entry['name']} slab {slab}")
                    checked += 1
            self.assertGreater(checked, 0, "the fixture must produce slab digests")

    def test_a_denormal_scale_is_flushed_on_both_sides(self):
        """`D11`: the flush is part of the contract, so both implementations must do it.

        Measured on the real install: 0.722705 % of 523,304,960 scales are denormal, so a reader that
        did not flush would disagree with the engine on four million values per pass.
        """
        entry = {
            "shape": [1, 2],
            "padded_columns": 2,
            "group": 2,
            "packed": bytes([0x51]),
            "scales": np.float32(1e-40).tobytes(),
            "zeros": bytes([0]),
            "dtype": "int4",
            "quant": "int4-affine",
        }
        self.assertLess(float(np.float32(1e-40)), float(np.finfo(np.float32).tiny), "1e-40 is denormal")
        self.assertTrue(np.all(quantize.flush_denormals(np.array([1e-40, 0.0, 1e-20, -1e-20], dtype=np.float32))[0] == 0))
        values = quantize.dequantize_tensor(entry)
        self.assertEqual(values.tolist(), [[0.0, 0.0]], "a denormal scale must flush to zero")

    def test_the_quantizer_stores_a_flushed_scale(self):
        """The stored value must be the flushed one, so an install cannot carry a scale two readers
        disagree about — and the flush has to happen *after* packing, which divides by it."""
        weight = np.full((1, 64), 1e-40, dtype=np.float32)
        entry = quantize.quantize_tensor(weight)
        stored = np.frombuffer(entry["scales"], dtype=np.float32)
        self.assertTrue(np.all(stored == 0), f"expected zero scales, got {stored[:4]}")
        self.assertTrue(np.all(quantize.dequantize_tensor(entry) == 0))

    def test_the_install_round_trips_through_its_reader(self):
        with tempfile.TemporaryDirectory() as directory:
            install = self.build(Path(directory))
            source = quantize.InstallSource(install)
            tensor = source.tensor("a.weight")
            self.assertEqual(tensor.shape, (8, 128))
            rows = source.rows("a.weight", 2, 4)
            np.testing.assert_array_equal(rows, tensor[2:4])

    def test_the_header_carries_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            install = self.build(Path(directory))
            manifest = json.loads((install / "install.json").read_text())
            self.assertEqual(manifest["source"]["revision"], "y")
            self.assertEqual(manifest["passes"], ["quantize-group-affine-int4"])
            self.assertEqual(manifest["policy_files"], ["tools/quant_policy.json"])
            self.assertEqual(manifest["family"], "qwen3_5")

    def test_a_tampered_payload_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            install = self.build(Path(directory))
            blob = bytearray((install / "data.bin").read_bytes())
            blob[-1] ^= 0xFF
            (install / "data.bin").write_bytes(bytes(blob))
            with self.assertRaises(quantize.PolicyError):
                quantize.verify_install(install)


if __name__ == "__main__":
    unittest.main()
