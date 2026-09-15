#!/usr/bin/env python3
"""Dump a checkpoint's tensor inventory (names and shapes) as a test fixture.

The same idea as `make_contract_vectors.py`: an importer tested against names we invented
is a mapping of our own assumptions. This reads a real checkpoint's header — no weights —
and writes what the importer has to account for.

    .venv/bin/python tools/make_qwen35_fixture.py <snapshot> <out.json>
"""

from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    snapshot = Path(argv[1])
    out = Path(argv[2])
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())
    weights = sorted(set(index["weight_map"].values()))
    tensors: dict[str, list[int]] = {}
    digest = hashlib.sha256()
    for name in weights:
        path = snapshot / name
        digest.update(path.read_bytes()[: 1 << 20])
        with open(path, "rb") as handle:
            header_length = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(header_length))
        for key, value in header.items():
            if key == "__metadata__":
                continue
            tensors[key] = list(value["shape"])

    config = json.loads((snapshot / "config.json").read_text())
    payload = {
        "note": (
            "Real tensor inventory (names and shapes) of a checkpoint, dumped from the "
            "safetensors header by tools/make_qwen35_fixture.py. No weights are included."
        ),
        "source": {
            "repo": "Qwen/Qwen3.5-2B",
            "revision": snapshot.name,
            "weights": weights,
        },
        "config": config,
        "tensors": tensors,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(f"wrote {out}: {len(tensors)} tensors, {out.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
