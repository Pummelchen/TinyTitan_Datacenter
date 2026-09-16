"""Read safetensors through `pread`, never `mmap`, so a contract run does not fill the page cache.

`safetensors.safe_open` memory-maps the shard. On the 35 B checkpoint that means a **67 GB** mapping, and
this project has already learned what a large page-cached read does to an 8 GB node: verifying a 20 GB
install took free disk from 17 GB to 2.96 GB in half a minute, because the page cache filled memory, memory
pressure grew swap, and swap is disk. That is the mechanism behind the two panics, and it is why the M1 gate
has been "not re-runnable here" since the checkpoint was first read.

This source has the same interface as `SafetensorsSource` — `tensor(name)` and `rows(name, start, end)` —
and serves both by reading **exactly the bytes asked for**, through the same `F_NOCACHE` + `pread` path the
install reader uses. Nothing is mapped, so nothing accumulates in the page cache, and the peak cost of a
read is the array it returns rather than the file it came from.

It is deliberately not clever: the header is parsed once per shard, `data_offsets` are trusted only after
being bounds-checked against the file, and a dtype it does not know is **refused by name** rather than
guessed at.

    python3 -c "from tools.uncached_safetensors import UncachedSafetensorsSource as S; s = S(Path(p))"

Standard library plus `numpy`, like the rest of the reference.
"""

from __future__ import annotations

import json
import math
import os
import struct
from pathlib import Path

import numpy as np

from quantize import open_uncached

# Bytes per element, and how to turn the raw buffer into fp32. `BF16` is the top sixteen bits of an fp32,
# so widening it is a shift and a bit-cast — exact, and the same thing `.float()` does in torch.
DTYPES: dict[str, tuple[int, str]] = {"BF16": (2, "bf16"), "F32": (4, "f32"), "F16": (2, "f16")}


class UnsupportedDtype(SystemExit):
    """A dtype the reader will not guess at, named so the reader can be extended deliberately."""


class UncachedSafetensorsSource:
    """`tensor` for a whole tensor and `rows` for a row range, from one shard or many, without mapping."""

    def __init__(self, snapshot: Path, *, uncached: bool = True):
        self.snapshot = Path(snapshot)
        # The caches are set up **before** the index is read, because the no-index path looks at headers to
        # build the owner map — and the first version of this assigned them afterwards, so that path died
        # with an AttributeError on its first shard.
        self._uncached = uncached
        self._descriptors: dict[str, int] = {}
        self._headers: dict[str, tuple[int, dict]] = {}

        index = self.snapshot / "model.safetensors.index.json"
        self._owner: dict[str, str] = {}
        if index.exists():
            weight_map = json.loads(index.read_text())["weight_map"]
            self._owner = dict(weight_map)
        else:
            for shard in sorted(self.snapshot.glob("*.safetensors")):
                for name in self._header(shard):
                    self._owner[name] = shard.name
        self.names = set(self._owner)
        self.shard_count = len(set(self._owner.values()))

    # -- plumbing ----------------------------------------------------------------

    def _descriptor(self, shard: str) -> int:
        if shard not in self._descriptors:
            self._descriptors[shard] = open_uncached(self.snapshot / shard, uncached=self._uncached)
        return self._descriptors[shard]

    def _header(self, shard: Path) -> dict[str, dict]:
        """The shard's tensor table: `name -> {dtype, shape, data_offsets}`, parsed with one `pread`."""
        key = shard.name
        if key not in self._headers:
            descriptor = open_uncached(shard, uncached=self._uncached)
            try:
                length = struct.unpack("<Q", os.pread(descriptor, 8, 0))[0]
                size = shard.stat().st_size
                if not 0 < length or 8 + length > size:
                    raise SystemExit(f"{shard}: header length {length} does not fit a {size}-byte file")
                header = json.loads(os.pread(descriptor, length, 8))
            finally:
                os.close(descriptor)
            self._headers[key] = (8 + length, {n: e for n, e in header.items() if n != "__metadata__"})
        return self._headers[key][1]

    def __enter__(self) -> "UncachedSafetensorsSource":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        for descriptor in self._descriptors.values():
            os.close(descriptor)
        self._descriptors.clear()

    # -- reading -----------------------------------------------------------------

    def _read(self, name: str, first_row: int | None = None, last_row: int | None = None) -> np.ndarray:
        shard = self._owner.get(name)
        if shard is None:
            raise KeyError(f"no tensor named '{name}' in the checkpoint")
        # `_header` caches per shard, so this is one `pread` per shard for the life of the reader.
        header = self._header(self.snapshot / shard)
        data_start = self._headers[shard][0]
        entry = header.get(name)
        if entry is None:
            raise KeyError(f"no tensor named '{name}' in {shard}")

        dtype = entry["dtype"]
        if dtype not in DTYPES:
            raise UnsupportedDtype(
                f"{name} is {dtype}, which this reader does not decode. It decodes "
                f"{', '.join(sorted(DTYPES))} — add it deliberately rather than guessing."
            )
        width, kind = DTYPES[dtype]
        shape = list(entry["shape"])
        start, end = entry["data_offsets"]
        count = math.prod(shape) if shape else 1
        if end - start != count * width:
            raise SystemExit(f"{name}: {end - start} bytes for {count} {dtype} values")

        rows = math.prod(shape[1:]) if len(shape) > 1 else 1
        first = 0 if first_row is None else max(0, first_row)
        last = (shape[0] if shape else 1) if last_row is None else min(shape[0] if shape else 1, last_row)
        if last <= first:
            return np.zeros([0] + shape[1:], dtype=np.float32)

        offset = start + first * rows * width
        byte_count = (last - first) * rows * width
        raw = os.pread(self._descriptor(shard), byte_count, data_start + offset)
        if len(raw) != byte_count:
            raise SystemExit(f"{name}: read {len(raw)} of {byte_count} bytes")
        return _to_fp32(raw, kind).reshape([last - first] + shape[1:])

    def tensor(self, name: str) -> np.ndarray:
        return self._read(name)

    def rows(self, name: str, start: int, end: int) -> np.ndarray:
        return self._read(name, start, end)


def _to_fp32(raw: bytes, kind: str) -> np.ndarray:
    """The raw bytes as fp32, exactly, with no rounding on the way."""
    if kind == "bf16":
        widened = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
        return widened.view(np.float32).copy()
    if kind == "f16":
        return np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    return np.frombuffer(raw, dtype=np.float32).copy()
