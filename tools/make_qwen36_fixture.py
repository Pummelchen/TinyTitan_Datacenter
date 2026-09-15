#!/usr/bin/env python3
"""Dump a checkpoint's tensor inventory (names, shapes, dtypes) by reading only the headers.

The same idea as `make_qwen35_fixture.py`, for a model whose weights are 67 GiB: a sharded
checkpoint's index lists the names, and each shard's `safetensors` header — a few hundred
kilobytes at the start of the file — carries the shapes. Reading those through
`HfFileSystem` range requests gives the **real** inventory, which is what an importer has to
account for, without downloading the weights.

    .venv/bin/python tools/make_qwen36_fixture.py Qwen/Qwen3.6-35B-A3B <out.json>
"""

from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path

from huggingface_hub import HfFileSystem, hf_hub_download


def main(argv: list[str]) -> int:
    repo = argv[1] if len(argv) > 1 else "Qwen/Qwen3.6-35B-A3B"
    out = Path(argv[2]) if len(argv) > 2 else Path("tests/DatacenterIRTests/Fixtures/qwen36-35b-a3b-tensors.json")
    cache = Path(".build/hf-cache")

    filesystem = HfFileSystem()
    config_path = hf_hub_download(repo, "config.json", cache_dir=cache)
    config = json.loads(Path(config_path).read_text())
    index_path = hf_hub_download(repo, "model.safetensors.index.json", cache_dir=cache)
    index = json.loads(Path(index_path).read_text())
    weight_map = index["weight_map"]

    revision = filesystem.info(repo)["name"].split("/")[-1] if False else None
    tensors: dict[str, dict] = {}
    digest = hashlib.sha256()
    for shard in sorted(set(weight_map.values())):
        with filesystem.open(f"{repo}/{shard}", "rb") as handle:
            length = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(length))
        digest.update(json.dumps(sorted(header.items()), sort_keys=True).encode())
        for name, description in header.items():
            if name == "__metadata__":
                continue
            tensors[name] = {"shape": list(description["shape"]), "dtype": description["dtype"]}

    payload = {
        "note": (
            "Real tensor inventory of a sharded checkpoint, read from the safetensors headers "
            "only (no weights downloaded) by tools/make_qwen36_fixture.py. Shapes and dtypes "
            "are what an importer must account for."
        ),
        "source": {
            "repo": repo,
            "revision": revision or "see the fetch",
            "shards": len(set(weight_map.values())),
            "index_sha256": hashlib.sha256(json.dumps(index, sort_keys=True).encode()).hexdigest(),
        },
        "config": config,
        "tensors": tensors,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(f"wrote {out}: {len(tensors)} tensors from {len(set(weight_map.values()))} shards, {out.stat().st_size:,} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
