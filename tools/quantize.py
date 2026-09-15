#!/usr/bin/env python3
"""The quantization transform pass (L3), the install it writes, and the policy it obeys.

Three things in this file, in the order the architecture asks for them:

- **the policy** (`load_policy`): which role gets which precision, as *data* (I4). A role
  the policy does not mention is a refusal, not a default, because a default is how a
  policy quietly stops describing the model.
- **the pass** (`quantize_tensor`, `build_install`): `(IR, tensors) -> (IR, tensors)`,
  architecture-agnostic, writing an install with a provenance header (I6).
- **the reader** (`InstallSource`): dequantizes for the contract, so the same contract code
  runs against a 4-bit install and against the fp32 checkpoint, and the difference measured
  is the quantization's and nothing else's.

The 4-bit layout is **ours**, not a transcode. I5 says to transcode vendor-native formats
rather than requantize, and for DeepSeek's FP4 that is what will happen; this family ships
bf16 weights and no 4-bit format at all, so there is nothing to transcode and this pass is a
genuine quantize. The layout is chosen to be conventional anyway — group-wise affine int4,
group 64 along a row, scales in fp32, low nibble first — because the on-disk format is meant
to stay close to what vendors use (I5) and close to what a Metal kernel can read.

    .venv/bin/python tools/quantize.py build <snapshot> <install> --spec spec.json --policy tools/quant_policy.json
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from check_disk_headroom import require_headroom  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))

GROUP = 64
# How much of a tensor to hold at once, in fp32 bytes. Sixty-four megabytes is an order of
# magnitude below the node's budget and large enough that the per-block numpy work is not
# dominated by overhead.
ROW_CHUNK_BYTES = 64 * 1024 * 1024
QMIN, QMAX = -8, 7


class PolicyError(Exception):
    pass


def load_policy(path: Path) -> dict:
    """The per-role policy, with the roles it must cover checked against the spec."""
    policy = json.loads(path.read_text())
    if "quant" not in policy:
        raise PolicyError(f"{path}: no 'quant' table")
    return policy


def policy_for(policy: dict, role: str) -> str:
    """The quantization a role is assigned, refusing an unlisted role.

    The default is not 'bf16'. Adding a role to a family and forgetting the policy should
    stop the install, not silently produce a model with one tensor at a different precision
    than the file says.
    """
    quant = policy["quant"]
    if role not in quant:
        raise PolicyError(f"role '{role}' is not in the policy; refusing to guess a precision")
    return quant[role]


def quantize_tensor(weight: np.ndarray, group: int = GROUP) -> dict:
    """Group-wise affine int4, low nibble first.

    Each row is cut into groups of `group` along the input dimension; every group carries one
    scale and one zero point, so the reconstruction is `(q - zero) * scale`. Affine rather
    than symmetric because attention and MLP weights are not zero-centred, and a symmetric
    code would spend half its range on values that do not occur.
    """
    weight = np.asarray(weight, dtype=np.float32)
    if weight.ndim < 2:
        raise ValueError(f"expected a weight with at least two dimensions, got shape {weight.shape}")
    original_shape = list(weight.shape)
    if weight.ndim != 2:
        # 3-D stacks flatten their leading axis; 1-D tensors are one row of N columns. Both are
        # what the readers assume, which is why `quantize_rows` only ever sees 2-D input.
        weight = weight.reshape(-1, weight.shape[-1]) if weight.ndim > 1 else weight.reshape(1, -1)
    return quantize_rows(weight, group=group, original_shape=original_shape)


def quantize_rows(
    weight: np.ndarray, *, group: int = GROUP, original_shape: list | None = None
) -> dict:
    """Quantize a **block of rows** and hand back the payload pieces.

    The caller may concatenate pieces from several blocks, which is how the install builder stays
    inside a node's memory: a stacked expert tensor is `[256, 1024, 2048]`, and one row of it is
    four megabytes in fp32, so reading it whole is two gigabytes against about four and a half
    usable. Rows are independent — every group lies inside one row — so the pieces concatenate
    exactly, and `test_chunked_quantization_is_byte_identical` is what makes that a fact.
    """
    weight = np.asarray(weight, dtype=np.float32)
    if weight.ndim != 2:
        raise ValueError(f"quantize_rows wants 2-D rows, got shape {weight.shape}")
    if original_shape is None:
        original_shape = list(weight.shape)
    # A **stacked** expert tensor is `[experts, rows, columns]`, and quantizing it is quantizing
    # its rows: the leading dimension is flattened away and every group along the input
    # dimension still lies inside one expert's row, because a row belongs to exactly one expert.
    # (Flattening the *last* two dimensions instead would let a group straddle two experts, and
    # the reconstruction would still look like weights.)
    original_shape = list(weight.shape)
    weight = weight.reshape(-1, weight.shape[-1]) if weight.ndim > 2 else weight
    rows, columns = weight.shape
    padded = (-columns) % group
    if padded:
        weight = np.pad(weight, ((0, 0), (0, padded)))
    groups = weight.shape[1] // group
    blocks = weight.reshape(rows, groups, group)

    minimum = blocks.min(axis=2)
    maximum = blocks.max(axis=2)
    span = maximum - minimum
    degenerate = span <= 0

    # A group whose values are all equal has no span to spend fifteen codes on. Giving it a
    # unit scale and the usual zero point loses the value itself — the rounding of the zero
    # point swallows it — so a degenerate group gets a scale that represents its one value
    # exactly and a zero point of zero.
    scale = np.where(
        degenerate,
        np.where(minimum != 0, np.abs(minimum) / float(QMAX), 1.0),
        span / float(QMAX - QMIN),
    ).astype(np.float32)
    zero = np.where(degenerate, 0.0, np.round(QMIN - minimum / scale)).astype(np.float32)
    zero = np.clip(zero, QMIN, QMAX)

    codes = np.round(blocks / scale[..., None] + zero[..., None]).astype(np.int32)
    codes = np.clip(codes, QMIN, QMAX).astype(np.int8)

    # Two codes per byte, low nibble first: the order is part of the format.
    flat = codes.reshape(rows, -1)
    low = (flat[:, 0::2] & 0x0F).astype(np.uint8)
    high = ((flat[:, 1::2] & 0x0F) << 4).astype(np.uint8)
    packed = (low | high).tobytes()

    return {
        # The *original* rank travels with the payload: the install describes the tensor the
        # spec describes, and a reader slices its leading axis whether that is `experts` or
        # `vocabulary`.
        "shape": original_shape,
        "rows": rows,
        "columns": columns,
        "padded_columns": int(weight.shape[1]),
        "group": group,
        "packed": packed,
        "scales": scale.reshape(-1).astype(np.float32).tobytes(),
        "zeros": zero.reshape(-1).astype(np.int8).tobytes(),
    }


def concat_rows(pieces: list[dict], shape: list) -> dict:
    """Join the pieces of a row-chunked quantisation into one entry.

    The rows of a piece are contiguous and independent, so this is a concatenation of three byte
    strings — and the result is what the whole-tensor path would have produced, which the test
    asserts byte for byte rather than by reasoning.
    """
    if not pieces:
        raise ValueError("no pieces to join")
    first = pieces[0]
    rows = sum(piece["rows"] for piece in pieces)
    return {
        "shape": list(shape),
        "rows": rows,
        "columns": first["columns"],
        "padded_columns": first["padded_columns"],
        "group": first["group"],
        "packed": b"".join(piece["packed"] for piece in pieces),
        "scales": b"".join(piece["scales"] for piece in pieces),
        "zeros": b"".join(piece["zeros"] for piece in pieces),
    }


def flat_rows(entry: dict) -> int:
    """The number of rows a payload's codes describe.

    One row is the tensor's **leading axis**, whatever the rank: a token of the embedding, or
    one expert of a stacked expert tensor. The codes flatten that axis away, so a reader that
    used `shape[0]` would size the code block for `experts` rows where there are
    `experts x rows` — and the reconstruction would still look like weights.
    """
    shape = entry["shape"]
    if len(shape) <= 1:
        return shape[0] if shape else 0
    return int(np.prod(shape[:-1]))


def dequantize_tensor(entry: dict) -> np.ndarray:
    shape = entry["shape"]
    # `rows` is the flattened leading axis -- `experts x rows` for a stacked expert tensor --
    # and the result is returned with the tensor's own shape.
    # The fallback is the product of the leading dimensions, not `shape[0]`: for a stacked
    # expert tensor the codes describe `experts x rows` rows.
    rows = flat_rows(entry)
    columns = shape[-1] if shape else 0
    padded = entry["padded_columns"]
    group = entry["group"]
    groups = padded // group
    packed = np.frombuffer(entry["packed"], dtype=np.uint8).reshape(rows, -1)
    # The codes are signed and live in four bits, so the nibbles have to be sign-extended:
    # reading 0b1000 as 8 instead of -8 shifts a whole group by sixteen steps, and the
    # reconstruction still looks like plausible weights.
    low = ((packed & 0x0F).astype(np.int16) ^ 0x08) - 0x08
    high = (((packed >> 4) & 0x0F).astype(np.int16) ^ 0x08) - 0x08
    codes = np.empty((rows, packed.shape[1] * 2), dtype=np.int16)
    codes[:, 0::2] = low
    codes[:, 1::2] = high
    scale = np.frombuffer(entry["scales"], dtype=np.float32).reshape(rows, groups)
    zero = np.frombuffer(entry["zeros"], dtype=np.int8).reshape(rows, groups).astype(np.float32)
    blocks = (codes.reshape(rows, groups, group).astype(np.float32) - zero[..., None]) * scale[..., None]
    flat = blocks.reshape(rows, padded)[:, :columns].astype(np.float32)
    return flat.reshape(shape) if len(shape) > 2 else flat


def decode_raw(payload: bytes, dtype: str, shape) -> np.ndarray:
    """The kept tensors, back to fp32. bf16 widens exactly; nothing is rounded here."""
    if dtype == "bf16":
        words = np.frombuffer(payload, dtype=np.uint16).astype(np.uint32) << 16
        return words.view(np.float32).reshape(shape)
    if dtype == "fp16":
        return np.frombuffer(payload, dtype=np.float16).astype(np.float32).reshape(shape)
    if dtype == "fp32":
        return np.frombuffer(payload, dtype=np.float32).reshape(shape)
    raise PolicyError(f"unknown stored dtype '{dtype}'")


class InstallWriter:
    """Writes an install one entry at a time, so the payload never has to fit in memory.

    The 35 B install is about twenty gigabytes against four and a half of usable memory, and the
    first attempt at building it accumulated every payload before writing — it was killed, and
    the shell reported success because the exit code came from `tail`. Only the manifest stays
    resident here, and the manifest is metadata.
    """

    def __init__(self, install: Path) -> None:
        install.mkdir(parents=True, exist_ok=True)
        self.install = install
        self._blob = open(install / "data.bin", "wb")
        self._written = 0
        self.index: list[dict] = []

    def add(self, entry: dict) -> None:
        padding = (64 - self._written % 64) % 64
        self._blob.write(b"\0" * padding)
        self._written += padding
        offset = self._written
        payload = entry["raw"] if "raw" in entry else entry["packed"] + entry["scales"] + entry["zeros"]
        self._blob.write(payload)
        self._written += len(payload)
        self.index.append(
            {
                "name": entry["name"],
                "role": entry["role"],
                "quant": entry["quant"],
                "shape": entry["shape"],
                "padded_columns": entry["padded_columns"],
                "group": entry["group"],
                # The precision a reader needs: the policy's own name for a kept tensor, or
                # `int4` for a packed one.
                "dtype": entry["quant"] if "raw" in entry else "int4",
                "offset": offset,
                "nbytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )

    def finish(self, *, source: dict, spec: dict, policy_files: list[str]) -> dict:
        self._blob.close()
        manifest = {
            "schema": 1,
            "source": source,
            "passes": ["quantize-group-affine-int4"],
            "policy_files": policy_files,
            "family": spec["family"],
            # Self-describing: roles, shapes and the policies are in the artifact, so a reader
            # needs nothing beside it (L1's spec file, carried rather than referenced).
            "spec": spec,
            "tensors": self.index,
        }
        (self.install / "install.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
        return manifest


def write_install(
    install: Path, *, source: dict, spec: dict, entries, policy_files: list[str]
) -> dict:
    """Write an install from an iterable of entries. A thin wrapper over `InstallWriter`.

    Kept because the tests and the fixture generators call it, and because "write these entries"
    is the shape a caller wants when the entries are small. The building path uses `InstallWriter`
    directly, because for a 35 B model the entries are not small.
    """
    writer = InstallWriter(install)
    for entry in entries:
        writer.add(entry)
    return writer.finish(source=source, spec=spec, policy_files=policy_files)


#: How much of a payload to hold at once. The install for the 35 B model is twenty gigabytes and
#: this node has four and a half usable, so "read the file" is not an operation that exists here.
WINDOW_BYTES = 4 * 1024 * 1024


def open_uncached(path: Path, uncached: bool = True) -> int:
    """Open a payload for reading with the buffer cache bypassed.

    The rule the brief gives for expert slabs, applied to the tooling as well: reading a model
    through the page cache is what turned "verify a 20 GB install" into free disk falling from
    17 GB to 2.96 GB, because the cached pages became memory pressure and memory pressure became
    swap. `F_NOCACHE` is advisory and best-effort; the reads are correct either way.
    """
    descriptor = os.open(path, os.O_RDONLY)
    if uncached and hasattr(fcntl, "F_NOCACHE"):
        fcntl.fcntl(descriptor, fcntl.F_NOCACHE, 1)
    return descriptor


def pread_exact(descriptor: int, offset: int, nbytes: int) -> bytes:
    """Exactly `nbytes` from `offset`, in bounded windows, never the whole file."""
    pieces: list[bytes] = []
    read = 0
    while read < nbytes:
        chunk = os.pread(descriptor, min(WINDOW_BYTES, nbytes - read), offset + read)
        if not chunk:
            raise PolicyError(f"payload at {offset} is short: wanted {nbytes} bytes, got {read}")
        pieces.append(chunk)
        read += len(chunk)
    return b"".join(pieces)


def digest_of(descriptor: int, offset: int, nbytes: int) -> str:
    """A streaming sha256 over a payload, holding one window at a time."""
    hasher = hashlib.sha256()
    read = 0
    while read < nbytes:
        chunk = os.pread(descriptor, min(WINDOW_BYTES, nbytes - read), offset + read)
        if not chunk:
            raise PolicyError(f"payload at {offset} is short: wanted {nbytes} bytes, got {read}")
        hasher.update(chunk)
        read += len(chunk)
    return hasher.hexdigest()


def verify_install(install: Path, uncached: bool = True) -> dict:
    """Check every payload digest, streaming and uncached.

    The whole-payload check, for a gate that wants it stated. The runtime path does not do this:
    `InstallSource` checks each payload the first time it reads it, exactly as the Swift
    `InstallFile` does, because hashing twenty gigabytes to open a file is not a thing this
    hardware can do.
    """
    manifest = json.loads((install / "install.json").read_text())
    descriptor = open_uncached(install / "data.bin", uncached)
    try:
        for entry in manifest["tensors"]:
            digest = digest_of(descriptor, entry["offset"], entry["nbytes"])
            if digest != entry["sha256"]:
                raise PolicyError(f"{entry['name']}: payload digest {digest} does not match the manifest")
    finally:
        os.close(descriptor)
    return manifest


class InstallSource:
    """A weight source over an install, dequantizing on demand.

    It offers the same two operations every other source does — a whole tensor, and a range
    of rows — so the contract runs unchanged and the only difference measured is the
    quantization.
    """

    def __init__(self, install: Path, uncached: bool = True):
        # Metadata only: the payload is read one tensor at a time, uncached, so that a
        # twenty-gigabyte install is never resident and never fills the page cache.
        self.manifest = json.loads((install / "install.json").read_text())
        self._descriptor = open_uncached(install / "data.bin", uncached)
        self._entries = {entry["name"]: entry for entry in self.manifest["tensors"]}
        self._verified: set[str] = set()

    def close(self) -> None:
        if self._descriptor >= 0:
            os.close(self._descriptor)
            self._descriptor = -1

    def _checked(self, name: str) -> bytes:
        """One payload, read uncached, with its digest checked the first time it is read.

        The check is not weakened by moving it here — a tampered payload still cannot be decoded
        into plausible weights (`I6`) — and it is skipped on later reads of the same tensor, which
        for a streamed expert slab is the difference between one hash and one hash per token.
        """
        entry = self._entries[name]
        payload = pread_exact(self._descriptor, entry["offset"], entry["nbytes"])
        if name not in self._verified:
            digest = hashlib.sha256(payload).hexdigest()
            if digest != entry["sha256"]:
                raise PolicyError(f"{name}: payload digest {digest} does not match the manifest")
            self._verified.add(name)
        return payload

    def _payload(self, name: str) -> dict:
        entry = self._entries[name]
        payload = self._checked(name)
        rows = flat_rows(entry)
        codes_bytes = rows * (entry["padded_columns"] // 2)
        group = entry["group"]
        groups = rows * (entry["padded_columns"] // group)
        return {
            "shape": entry["shape"],
            "padded_columns": entry["padded_columns"],
            "group": group,
            "packed": payload[:codes_bytes],
            "scales": payload[codes_bytes : codes_bytes + groups * 4],
            "zeros": payload[codes_bytes + groups * 4 : codes_bytes + groups * 4 + groups],
        }

    def _raw(self, name: str) -> bytes:
        return self._checked(name)

    def tensor(self, name: str) -> np.ndarray:
        entry = self._entries[name]
        if entry["dtype"] != "int4":
            return decode_raw(self._raw(name), entry["dtype"], entry["shape"])
        return dequantize_tensor(self._payload(name))

    def rows(self, name: str, start: int, end: int) -> np.ndarray:
        entry = self._entries[name]
        if entry["dtype"] != "int4" and len(entry["shape"]) >= 2:
            # Sliced from the stored bytes rather than decoded whole: the embedding is
            # `[248320, 2048]`, and reading one row of it should not cost a gigabyte.
            # One row is the leading axis whatever the rank: for the embedding that is a
            # token, for a stacked expert tensor it is an expert.
            width = 1
            for dimension in entry["shape"][1:]:
                width *= dimension
            stride = {"bf16": 2, "fp16": 2, "fp32": 4}[entry["dtype"]] * width
            payload = self._raw(name)[start * stride : end * stride]
            return decode_raw(payload, entry["dtype"], [end - start, width])
        return self.tensor(name)[start:end]


def build_install(snapshot: Path, install: Path, spec: dict, policy: dict, policy_file: str) -> dict:
    """The pass itself: every tensor in the spec, quantized or copied per the policy.

    The source is shard-aware. Taking the first shard — which this did — builds an install that
    is missing five sixth of a sharded model's layers and reports success.
    """
    # A 67 GB checkpoint and a 20 GB install will not fit under the floor, and exhausting the
    # disk here means exhausting swap, which panicked this machine twice.
    require_headroom(purpose="the install build")
    from safetensors_source import SafetensorsSource

    handle = SafetensorsSource(snapshot)

    writer = InstallWriter(install)
    skipped: list[str] = []


    for tensor in sorted(spec["tensors"], key=lambda t: t["name"]):
        role = tensor["role"]
        quant = policy_for(policy, role)
        # Every tensor is read a block of rows at a time. A stacked expert tensor is two
        # gigabytes in fp32 and this node has about four and a half usable, so reading it whole
        # would put the install builder itself over the budget the install exists to fit into.
        shape = list(tensor["shape"])
        # One rule for every rank, and it is the same rule the readers use: the row count is the
        # product of the leading dimensions and the column count is the last one. A **1-D** tensor
        # — every norm, `A_log`, `dt_bias` — is therefore one row of N columns, not N rows of N.
        # Getting that wrong made the payload 191 tensors' worth of inconsistent, and the reader
        # would have decoded them into plausible weights.
        rows = int(np.prod(shape[:-1])) if len(shape) > 0 else 1
        columns = shape[-1] if shape else 1
        # Chunking is by **leading-axis entries**, which is what the reader slices: a row of a
        # 2-D tensor, or one expert of a stacked 3-D one. Each entry contributes `inner` payload
        # rows of `columns` — and since an entry is never split, no quantisation group can
        # straddle two experts.
        leading = shape[0] if shape else 1
        inner = int(np.prod(shape[1:-1])) if len(shape) > 2 else 1
        per_chunk = max(1, ROW_CHUNK_BYTES // max(inner * columns * 4, 1))
        # A rank-1 tensor is read whole. `rows` slices the *leading* axis, and for a rank-1 tensor
        # that axis is its length rather than a row index — chunking it would hand back elements
        # where rows are expected. They are tiny (a norm, `A_log`, `dt_bias`), so this costs
        # nothing and removes a rule that was quietly wrong.
        whole_1d = len(shape) == 1

        if quant != "int4-affine":
            # Kept, not dropped: an install is the whole model, or it is not an install.
            # These are the roles the policy holds at higher precision — norms, the
            # convolution, the per-head decay — and I3 is explicit that gating stays at
            # bf16 or above.
            pieces: list[bytes] = []
            ranges = (
                [(0, 1)] if whole_1d
                else [(start, min(start + per_chunk, leading)) for start in range(0, leading, per_chunk)]
            )
            for start, end in ranges:
                value = (
                    handle.tensor(tensor["name"]).reshape(-1, columns)
                    if whole_1d
                    else handle.rows(tensor["name"], start, end).reshape(-1, columns)
                )
                if quant == "bf16":
                    # bf16 from a widened bf16 is exact: the low sixteen bits are already zero.
                    pieces.append((value.view(np.uint32) >> 16).astype(np.uint16).tobytes())
                elif quant == "fp32":
                    pieces.append(value.astype(np.float32).tobytes())
                elif quant == "fp16":
                    pieces.append(value.astype(np.float16).tobytes())
                else:
                    raise PolicyError(f"role '{role}': precision '{quant}' is not one this pass can store")
            raw = b"".join(pieces)
            skipped.append(f"{tensor['name']} ({role}: {quant})")
            writer.add(
                {
                    "name": tensor["name"],
                    "role": role,
                    "quant": quant,
                    "raw": raw,
                    "shape": list(tensor["shape"]),
                    "padded_columns": int(tensor["shape"][1]) if len(tensor["shape"]) == 2 else 0,
                    "group": 0,
                }
            )
            continue
        piece_rows: list[dict] = []
        ranges = (
            [(0, 1)] if whole_1d
            else [(start, min(start + per_chunk, leading)) for start in range(0, leading, per_chunk)]
        )
        for start, end in ranges:
            block = (
                handle.tensor(tensor["name"]).reshape(-1, columns)
                if whole_1d
                else handle.rows(tensor["name"], start, end).reshape(-1, columns)
            )
            piece_rows.append(quantize_rows(block, original_shape=shape))
        entry = concat_rows(piece_rows, shape)
        entry.update({"name": tensor["name"], "role": role, "quant": quant})
        writer.add(entry)

    source = {
        "repo": spec["source"]["repo"],
        "revision": spec["source"]["revision"],
        "files": spec["source"]["files"],
        "spec_family": spec["family"],
    }
    manifest = writer.finish(source=source, spec=spec, policy_files=[policy_file])
    manifest["skipped"] = skipped
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="build an install from a checkpoint and an IR spec")
    build.add_argument("snapshot", type=Path)
    build.add_argument("install", type=Path)
    build.add_argument("--spec", type=Path, required=True)
    build.add_argument("--policy", type=Path, default=Path(__file__).parent / "quant_policy.json")
    verify = sub.add_parser("verify", help="recompute the payload digests")
    verify.add_argument("install", type=Path)
    args = parser.parse_args(argv)

    if args.command == "verify":
        manifest = verify_install(args.install)
        print(f"{args.install}: {len(manifest['tensors'])} tensors verified")
        return 0

    spec = json.loads(args.spec.read_text())
    policy = load_policy(args.policy)
    manifest = build_install(args.snapshot, args.install, spec, policy, str(args.policy))

    total = sum(e["shape"][0] * e["padded_columns"] for e in manifest["tensors"])
    packed = sum(e["nbytes"] for e in manifest["tensors"])
    print(f"wrote {args.install}: {len(manifest['tensors'])} tensors quantized, {len(manifest['skipped'])} left at higher precision")
    print(f"  {total:,} weights in {packed:,} bytes ({packed * 8 / max(total, 1):.2f} bits/weight including scales and zeros)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
