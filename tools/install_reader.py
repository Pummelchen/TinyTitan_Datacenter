#!/usr/bin/env python3
"""Read a model install from Python — the contract `DC-108` was missing.

M1's gate compares the engine's trace to `tools/ordered_reference.py`, but the *install* — the artifact
the engine actually reads, produced by `tools/quantize.py` — had no reader on the Python side. Nothing in
Python could check that the weight the engine dequantised is the weight the packer wrote; the claim
rested on one implementation reading its own output.

This is that reader, and it is deliberately the same arithmetic, guard for guard:

* a tensor's payload is `[codes][scales][zeros]` for the **whole** tensor, row-major, with two signed
  four-bit codes per byte, **low nibble first**;
* group size comes from the manifest and every row is padded to `padded_columns`;
* a denormal scale is read as zero (`D11`) — the packer does the same, so a comparison stays exact;
* the layout guards say *which* invariant failed, because "group does not divide padded" once sent a
  reader looking at the wrong number.

Standard library only, on purpose: this gates the repository, so it has to run on any `python3`. Reads go
through `F_NOCACHE` where the platform offers it, and everything streams, because a page-cached
multi-gigabyte read on this host is a hazard rather than a neutral operation.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import struct
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

SCHEMA = 1

# Float32's least normal magnitude, from its bit pattern rather than from a decimal nobody can check.
LEAST_NORMAL_FP32 = struct.unpack("<f", struct.pack("<I", 0x00800000))[0]

# Bytes per value, and the `struct` code that reads one: bfloat16 is the top half of an fp32.
DENSE_WIDTH = {"bf16": 2, "fp16": 2, "fp32": 4}
DENSE_FORMAT = {"fp16": "e", "fp32": "f"}


class InstallError(Exception):
    """A malformed install, or a tensor that does not fit its declared layout."""


@dataclass(frozen=True)
class Tensor:
    name: str
    role: str
    dtype: str
    quant: str
    offset: int
    nbytes: int
    padded_columns: int
    group: int
    sha256: str | None
    shape: tuple[int, ...] | None

    @property
    def is_int4(self) -> bool:
        return self.quant == "int4-affine"

    def geometry(self) -> tuple[int, int]:
        """`(rows, columns)` — the second is the **padded** width, which is what the payload holds.

        The manifest carries a `shape` for every tensor and a `padded_columns` that is **zero** when
        nothing was padded (all 221 one-dimensional norms, for instance, against 2048 for the head).
        So the shape decides the row count and the padded width only overrides the row length: a
        reader that demanded `padded_columns` first refused every norm in the real install, which is
        how this rule was learned.
        """
        shape = self.shape
        rows: int | None = None
        if shape:
            rows = 1
            for dimension in shape[:-1]:
                rows *= dimension
        padded = self.padded_columns if self.padded_columns > 0 else (shape[-1] if shape else 0)
        if padded <= 0:
            raise InstallError(f"{self.name}: neither a shape nor a padded width to derive one from")

        if self.is_int4:
            group = self.group
            if group <= 0:
                raise InstallError(f"{self.name}: group is {group}, which cannot divide anything")
            if padded % group != 0:
                raise InstallError(
                    f"{self.name}: group {group} does not divide padded width {padded}"
                )
            if padded % 2 != 0:
                raise InstallError(
                    f"{self.name}: padded width {padded} is odd, and two codes share a byte"
                )
            groups = padded // group
            per_row = padded // 2 + groups * 4 + groups
            if per_row == 0 or self.nbytes % per_row != 0:
                raise InstallError(
                    f"{self.name}: payload is {self.nbytes} bytes, which is not a whole number of "
                    f"{per_row}-byte rows"
                )
            from_payload = self.nbytes // per_row
        else:
            width = DENSE_WIDTH.get(self.dtype)
            if width is None:
                raise InstallError(f"{self.name}: unknown dtype {self.dtype!r}")
            if self.nbytes % width != 0:
                raise InstallError(
                    f"{self.name}: {self.nbytes} bytes is not a whole number of {width}-byte values"
                )
            values = self.nbytes // width
            if values % padded != 0:
                raise InstallError(
                    f"{self.name}: {values} values is not a whole number of {padded}-value rows"
                )
            from_payload = values // padded

        if rows is None:
            rows = from_payload
        elif rows != from_payload:
            raise InstallError(
                f"{self.name}: shape {shape} needs {rows} row(s), the payload holds {from_payload}"
            )
        return rows, padded


def _flush_denormal(scale: float) -> float:
    """`D11`: the packer writes denormal scales and both readers take them as zero."""
    return 0.0 if scale != 0.0 and abs(scale) < LEAST_NORMAL_FP32 else scale


def _signed(nibble: int) -> int:
    """A four-bit code as a signed integer: two's complement in four bits."""
    value = nibble & 0x0F
    return value - 16 if value >= 8 else value


class Install:
    """An install directory: `install.json` plus its payload."""

    def __init__(self, root: Path, uncached: bool = True) -> None:
        self.root = Path(root)
        manifest_path = self.root / "install.json"
        if not manifest_path.exists():
            raise InstallError(f"{manifest_path} is missing")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema") != SCHEMA:
            raise InstallError(
                f"install schema is {manifest.get('schema')!r}, this reader knows {SCHEMA}"
            )
        self.family = manifest["family"]
        self.source = manifest.get("source", {})
        self.revision = self.source.get("revision")
        self.spec = manifest.get("spec", {})
        self.passes = manifest.get("passes", [])
        self.tensors: dict[str, Tensor] = {}
        for raw in manifest["tensors"]:
            shape = raw.get("shape")
            tensor = Tensor(
                name=raw["name"],
                role=raw.get("role", ""),
                dtype=raw.get("dtype", ""),
                quant=raw.get("quant", ""),
                offset=raw["offset"],
                nbytes=raw["nbytes"],
                padded_columns=raw.get("padded_columns", 0),
                group=raw.get("group", 0),
                sha256=raw.get("sha256"),
                shape=tuple(shape) if shape else None,
            )
            if tensor.name in self.tensors:
                raise InstallError(f"{tensor.name} appears twice in the manifest")
            self.tensors[tensor.name] = tensor
        self.data_path = self.root / "data.bin"
        size = self.data_path.stat().st_size
        for tensor in self.tensors.values():
            if tensor.offset < 0 or tensor.offset + tensor.nbytes > size:
                raise InstallError(
                    f"{tensor.name}: [{tensor.offset}, {tensor.offset + tensor.nbytes}) is outside "
                    f"a {size}-byte payload"
                )
        self._uncached = uncached
        self._handle: int | None = None

    def __enter__(self) -> Install:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def handle(self) -> int:
        if self._handle is None:
            self._handle = os.open(self.data_path, os.O_RDONLY)
            if self._uncached and hasattr(fcntl, "F_NOCACHE"):
                try:
                    fcntl.fcntl(self._handle, fcntl.F_NOCACHE, 1)
                except OSError:
                    pass
        return self._handle

    def close(self) -> None:
        if self._handle is not None:
            os.close(self._handle)
            self._handle = None

    def read(self, tensor: Tensor, offset: int = 0, length: int | None = None) -> bytes:
        """A bounded `pread`, so a tensor can be walked without ever holding all of it."""
        length = tensor.nbytes if length is None else length
        if offset < 0 or length < 0 or offset + length > tensor.nbytes:
            raise InstallError(f"{tensor.name}: read [{offset}, {offset + length}) is outside it")
        piece = os.pread(self.handle, length, tensor.offset + offset)
        if len(piece) != length:
            raise InstallError(
                f"{tensor.name}: payload ended at {len(piece)} of {length} bytes from {offset}"
            )
        return piece

    def digest(self, tensor: Tensor, chunk: int = 1 << 20) -> str:
        """The payload's SHA-256, streamed."""
        hasher = hashlib.sha256()
        done = 0
        while done < tensor.nbytes:
            hasher.update(self.read(tensor, done, min(chunk, tensor.nbytes - done)))
            done += min(chunk, tensor.nbytes - done)
        return hasher.hexdigest()

    def verified(self, tensor: Tensor) -> bool:
        """`None` in the manifest means the packer recorded no digest — which is not a pass."""
        if not tensor.sha256:
            raise InstallError(f"{tensor.name}: the manifest records no digest")
        return self.digest(tensor) == tensor.sha256

    # --- reading values -------------------------------------------------------------------------

    def rows(self, tensor: Tensor, row_block: int = 64) -> Iterator[list[float]]:
        """Dequantise in row blocks, so peak memory is one block rather than one tensor."""
        rows_total, padded = tensor.geometry()
        for start in range(0, rows_total, row_block):
            count = min(row_block, rows_total - start)
            if tensor.is_int4:
                yield from self._int4_block(tensor, rows_total, padded, start, count)
            else:
                yield from self._dense_block(tensor, padded, start, count)

    def int4_block_bytes(self, tensor: Tensor, start: int, count: int) -> tuple[bytes, bytes, bytes]:
        """The `(codes, scales, zeros)` for `count` rows from `start`, in three reads.

        The offsets are the layout's, and they are computed **here** rather than by the caller: a second copy
        of them is a second thing to get wrong, which is `D72`'s lesson applied before it bites. The vectorised
        dequantiser in `install_source.py` consumes these bytes with numpy, because this module is
        standard-library only on purpose -- it gates the repository, so it has to run on any `python3` -- and
        the arithmetic it does per element is what makes a real-model contract slow.
        """
        if not tensor.is_int4:
            raise InstallError(f"{tensor.name} is {tensor.quant}, not int4-affine")
        rows_total, padded = tensor.geometry()
        if not 0 <= start <= start + count <= rows_total:
            raise InstallError(f"rows {start}..{start + count} outside 0..{rows_total} for {tensor.name}")
        groups = padded // tensor.group
        code_row = padded // 2
        codes = self.read(tensor, start * code_row, count * code_row)
        scales_at = rows_total * code_row + start * groups * 4
        zeros_at = rows_total * code_row + rows_total * groups * 4 + start * groups
        scales = self.read(tensor, scales_at, count * groups * 4)
        zeros = self.read(tensor, zeros_at, count * groups)
        return codes, scales, zeros

    def _int4_block(
        self, tensor: Tensor, rows_total: int, padded: int, start: int, count: int
    ) -> Iterator[list[float]]:
        groups = padded // tensor.group
        code_row = padded // 2
        codes, scales, zeros = self.int4_block_bytes(tensor, start, count)
        for row in range(count):
            base = row * code_row
            row_scales = struct.unpack_from(f"<{groups}f", scales, row * groups * 4)
            row_zeros = zeros[row * groups : (row + 1) * groups]
            yield _dequantize_row(
                codes[base : base + code_row], padded, tensor.group, row_scales, row_zeros
            )

    def _dense_block(
        self, tensor: Tensor, padded: int, start: int, count: int
    ) -> Iterator[list[float]]:
        width = DENSE_WIDTH[tensor.dtype]
        block = self.read(tensor, start * padded * width, count * padded * width)
        size = count * padded
        if tensor.dtype == "bf16":
            # The top half of an fp32, so it widens exactly: no rounding to argue about.
            widened = [
                struct.unpack("<f", struct.pack("<I", word << 16))[0]
                for word in struct.unpack(f"<{size}H", block)
            ]
        else:
            widened = list(struct.unpack(f"<{size}{DENSE_FORMAT[tensor.dtype]}", block))
        for row in range(count):
            yield widened[row * padded : (row + 1) * padded]

    def row_range(self, tensor: Tensor, start: int, end: int, row_block: int = 32) -> list[list[float]]:
        """Rows `[start, end)` as the engine would read them, without walking the rows before `start`.

        `rows()` is sequential, which is right for a scan and wrong for a shard: fetching expert 255 should
        not dequantise the 254 experts in front of it. The block helpers already take a `start`, so this is
        the same work with the offset passed through rather than iterated to.
        """
        rows_total, padded = tensor.geometry()
        if not 0 <= start <= end <= rows_total:
            raise InstallError(f"rows {start}..{end} outside 0..{rows_total} for {tensor.name}")
        out: list[list[float]] = []
        for block_start in range(start, end, row_block):
            count = min(row_block, end - block_start)
            if tensor.is_int4:
                out.extend(self._int4_block(tensor, rows_total, padded, block_start, count))
            else:
                out.extend(self._dense_block(tensor, padded, block_start, count))
        return out

    def dequantize(self, tensor: Tensor, row_block: int = 64) -> list[float]:
        """The whole tensor, as the engine would read it. Only for tensors that fit in memory."""
        out: list[float] = []
        for row in self.rows(tensor, row_block=row_block):
            out.extend(row)
        return out


def _dequantize_row(
    codes: bytes, padded: int, group: int, scales: tuple[float, ...], zeros: bytes
) -> list[float]:
    values: list[float] = []
    for index in range(padded):
        byte = codes[index // 2]
        nibble = (byte & 0x0F) if index % 2 == 0 else (byte >> 4)
        group_index = index // group
        raw_zero = zeros[group_index]
        zero = raw_zero - 256 if raw_zero >= 128 else raw_zero
        values.append(float(_signed(nibble) - zero) * _flush_denormal(scales[group_index]))
    return values


def main(argv: list[str] | None = None) -> int:
    """A small self-description, so the reader is usable without reading it."""
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print("Read a model install and describe its tensors.", file=sys.stderr)
        print("usage: install_reader.py <install-dir> [tensor-name ...]", file=sys.stderr)
        return 2
    with Install(Path(argv[0])) as install:
        print(
            f"{install.root}: family {install.family}, revision {install.revision}, "
            f"{len(install.tensors)} tensor(s), passes {install.passes}"
        )
        for name in argv[1:] or sorted(install.tensors)[:5]:
            tensor = install.tensors.get(name)
            if tensor is None:
                print(f"  {name}: not in the install", file=sys.stderr)
                return 1
            shape = tensor.shape or f"({tensor.nbytes} bytes)"
            print(
                f"  {tensor.role:22s} {name:44s} {tensor.dtype:5s} {tensor.quant:12s} "
                f"{str(shape):22s} {tensor.nbytes:>12,} B"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
