#!/usr/bin/env python3
"""The golden-trace container: one directory, one JSON index, one data blob.

A trace is what a gate compares against. It has to be readable by the Swift engine
without a Python dependency, small enough to keep, and precise enough that "the
outputs differ" always resolves to *which tensor, which element, how far*.

Layout::

    <trace>/
      manifest.json   schema, provenance, and an ordered index of every tensor
      data.bin        the tensors, concatenated, each aligned to 64 bytes

Design rules, each of which exists because of a way a gate can lie:

- **Every tensor carries its own sha256.** A trace that was edited after capture is
  refused rather than compared, so a "passing" gate cannot be the result of a
  doctored reference.
- **The manifest carries a whole-trace digest** over the canonical index, so
  comparing two traces for I1 (same input, same shard count, same bytes) is one
  string comparison.
- **Floats are stored as raw little-endian words, never as text.** Decimal
  round-tripping is exactly the class of silent lossy step this project cannot
  afford, and it would make ULP distances meaningless.
- **Discrete decisions are a separate section, not tensors.** Router top-k index
  sets are compared for exact equality, never with a tolerance (I3), and keeping
  them out of the float stream means a differ cannot "almost match" them.
- **The indexing is explicit: name, offset, length, shape, dtype.** No implicit
  order, so a reordered or truncated file is a validation failure rather than a
  silent mis-comparison.

Standard library only, on purpose: the gates that run in CI must not need torch.
The capture tool may import it; this module may not.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path

SCHEMA_VERSION = 1
ALIGNMENT = 64
DATA_FILE = "data.bin"
MANIFEST_FILE = "manifest.json"

DTYPE_SIZE = {"f32": 4, "f16": 2, "bf16": 2, "i32": 4, "i64": 8, "u8": 1}
FLOAT_DTYPES = ("f32", "f16", "bf16")


class TraceError(Exception):
    """A trace is malformed, inconsistent, or fails its own integrity checks."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def dtype_size(dtype: str) -> int:
    try:
        return DTYPE_SIZE[dtype]
    except KeyError:
        raise TraceError(f"unknown dtype {dtype!r}; known: {sorted(DTYPE_SIZE)}") from None


def element_count(shape) -> int:
    count = 1
    for dim in shape:
        if not isinstance(dim, int) or dim < 0:
            raise TraceError(f"bad shape {shape!r}")
        count *= dim
    return count


def _pad(offset: int) -> int:
    return (ALIGNMENT - offset % ALIGNMENT) % ALIGNMENT


def canonical_digest(manifest: dict) -> str:
    """The whole-trace digest: every tensor's identity and hash, plus the discrete
    decisions, in index order. Deliberately excludes timestamps and producer
    strings, so two runs that produced the same numbers digest identically."""
    canonical = {
        "schema": manifest["schema"],
        "tensors": [
            {
                "name": t["name"],
                "shape": t["shape"],
                "dtype": t["dtype"],
                "sha256": t["sha256"],
            }
            for t in manifest["tensors"]
        ],
        "discrete": [
            {"name": d["name"], "shape": d["shape"], "values": d["values"]}
            for d in manifest.get("discrete", [])
        ],
    }
    blob = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    return sha256_bytes(blob)


def write_trace(
    root: Path,
    *,
    tensors,
    discrete=None,
    model: dict | None = None,
    reference: dict | None = None,
    prompt: dict | None = None,
    producer: str = "unknown",
    extra: dict | None = None,
) -> dict:
    """Write a trace directory and return its manifest.

    ``tensors`` is an iterable of ``(name, dtype, shape, payload)`` in capture
    order. ``discrete`` is an iterable of ``(name, shape, values)``.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    index = []
    blob = bytearray()
    for name, dtype, shape, payload in tensors:
        size = dtype_size(dtype)
        expected = element_count(shape) * size
        if len(payload) != expected:
            raise TraceError(
                f"{name}: payload is {len(payload)} bytes, shape {shape} at {dtype} "
                f"needs {expected}"
            )
        padding = _pad(len(blob))
        blob.extend(b"\0" * padding)
        offset = len(blob)
        blob.extend(payload)
        index.append(
            {
                "name": name,
                "file": DATA_FILE,
                "offset": offset,
                "nbytes": len(payload),
                "shape": list(shape),
                "dtype": dtype,
                "sha256": sha256_bytes(payload),
            }
        )

    discrete_entries = [
        {"name": name, "shape": list(shape), "values": list(values)}
        for name, shape, values in (discrete or [])
    ]

    manifest = {
        "schema": SCHEMA_VERSION,
        "producer": producer,
        "model": model or {},
        "reference": reference or {},
        "prompt": prompt or {},
        "tensors": index,
        "discrete": discrete_entries,
    }
    if extra:
        manifest["extra"] = extra
    manifest["digest"] = canonical_digest(manifest)

    (root / DATA_FILE).write_bytes(bytes(blob))
    (root / MANIFEST_FILE).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


@dataclass(frozen=True)
class Tensor:
    name: str
    dtype: str
    shape: tuple
    payload: bytes

    def values(self):
        """Decode to Python numbers, for reporting rather than comparison."""
        if self.dtype == "f32":
            return list(struct.unpack(f"<{len(self.payload) // 4}f", self.payload))
        if self.dtype == "f16":
            return list(struct.unpack(f"<{len(self.payload) // 2}e", self.payload))
        if self.dtype == "bf16":
            return [bf16_to_f32(w) for w in struct.unpack(f"<{len(self.payload) // 2}H", self.payload)]
        if self.dtype == "i32":
            return list(struct.unpack(f"<{len(self.payload) // 4}i", self.payload))
        if self.dtype == "i64":
            return list(struct.unpack(f"<{len(self.payload) // 8}q", self.payload))
        if self.dtype == "u8":
            return list(self.payload)
        raise TraceError(f"unknown dtype {self.dtype!r}")


class Trace:
    """A trace directory, validated on open."""

    def __init__(self, root: Path, manifest: dict, data: bytes):
        self.root = Path(root)
        self.manifest = manifest
        self._data = data
        self._by_name = {t["name"]: t for t in manifest["tensors"]}
        self._discrete = {d["name"]: d for d in manifest.get("discrete", [])}

    @property
    def digest(self) -> str:
        return self.manifest["digest"]

    @property
    def tensor_names(self) -> list[str]:
        return [t["name"] for t in self.manifest["tensors"]]

    @property
    def discrete_names(self) -> list[str]:
        return [d["name"] for d in self.manifest.get("discrete", [])]

    def entry(self, name: str) -> dict:
        try:
            return self._by_name[name]
        except KeyError:
            raise TraceError(f"no tensor named {name!r} in {self.root}") from None

    def tensor(self, name: str) -> Tensor:
        entry = self.entry(name)
        start = entry["offset"]
        payload = self._data[start : start + entry["nbytes"]]
        return Tensor(entry["name"], entry["dtype"], tuple(entry["shape"]), payload)

    def discrete(self, name: str):
        try:
            return self._discrete[name]
        except KeyError:
            raise TraceError(f"no discrete decision named {name!r} in {self.root}") from None

    def verify(self) -> None:
        """Refuse a trace that is internally inconsistent or has been edited."""
        for entry in self.manifest["tensors"]:
            size = dtype_size(entry["dtype"])
            expected = element_count(entry["shape"]) * size
            if entry["nbytes"] != expected:
                raise TraceError(
                    f"{entry['name']}: index says {entry['nbytes']} bytes, "
                    f"shape {entry['shape']} at {entry['dtype']} needs {expected}"
                )
            payload = self._data[entry["offset"] : entry["offset"] + entry["nbytes"]]
            if len(payload) != entry["nbytes"]:
                raise TraceError(f"{entry['name']}: data file is truncated")
            actual = sha256_bytes(payload)
            if actual != entry["sha256"]:
                raise TraceError(
                    f"{entry['name']}: content hash {actual[:16]}… does not match the "
                    f"index {entry['sha256'][:16]}… — the trace was edited after capture"
                )
        if canonical_digest(self.manifest) != self.manifest.get("digest"):
            raise TraceError("whole-trace digest does not match the index")


def read_trace(root: Path, verify: bool = True) -> Trace:
    root = Path(root)
    manifest_path = root / MANIFEST_FILE
    if not manifest_path.is_file():
        raise TraceError(f"{root} is not a trace: no {MANIFEST_FILE}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TraceError(f"{manifest_path}: {exc}") from None
    if manifest.get("schema") != SCHEMA_VERSION:
        raise TraceError(
            f"{manifest_path}: schema {manifest.get('schema')!r}, this tool reads "
            f"{SCHEMA_VERSION}"
        )
    for key in ("tensors", "digest"):
        if key not in manifest:
            raise TraceError(f"{manifest_path}: missing {key!r}")
    data_path = root / DATA_FILE
    if not data_path.is_file():
        raise TraceError(f"{root} is not a trace: no {DATA_FILE}")
    trace = Trace(root, manifest, data_path.read_bytes())
    if verify:
        trace.verify()
    return trace


# --- float interpretation, for reporting a difference honestly -------------


def bf16_to_f32(word: int) -> float:
    return struct.unpack("<f", struct.pack("<I", word << 16))[0]


def f32_to_bf16(value: float) -> int:
    """Round-to-nearest-even, which is what a cast to bfloat16 means."""
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    if (bits & 0x7F800000) == 0x7F800000:  # inf / nan: keep the top bits as they are
        return bits >> 16
    rounded = bits + 0x7FFF + ((bits >> 16) & 1)
    return (rounded >> 16) & 0xFFFF


def _monotonic_key_f32(value: float) -> int:
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    return 0x80000000 - bits if bits & 0x80000000 else bits


def ulp_distance_f32(a: float, b: float) -> int:
    """Distance in representable fp32 steps. 0 only when the values are identical,
    and for the same value with different zero signs the answer is 1, not 0,
    because the bytes differ."""
    if struct.pack("<f", a) == struct.pack("<f", b):
        return 0
    if a != a or b != b:  # NaN: no meaningful distance, but not identical
        return -1
    return abs(_monotonic_key_f32(a) - _monotonic_key_f32(b))
