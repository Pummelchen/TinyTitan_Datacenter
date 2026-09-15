#!/usr/bin/env python3
"""One reader for a checkpoint, whether it is a single file or twenty-six shards.

The 35 B model ships as 26 shards plus `model.safetensors.index.json`, and the index is the only
thing that knows which shard holds which tensor. A reader that takes `sorted(glob(...))[0]` — which
is what both of this project's Python readers did — gets a checkpoint that **looks complete and is
missing five sixth of its layers**, with every tensor it does find the right shape. The engine side
was fixed first (`sources/DatacenterEngine/ShardedSafetensors.swift`); this is the same rule for the
contract and the install builder, so the Python side cannot quietly disagree with the Swift side
about what the model is.

Handles are opened once and kept. `safetensors` memory-maps, so holding 26 of them holds 26
mappings rather than 67 GB, and `get_slice` means a row read really reads rows.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

__all__ = ["SafetensorsSource", "shard_map"]


def shard_map(snapshot: Path) -> dict[str, str] | None:
    """`tensor name -> shard filename`, or None when the checkpoint is a single file."""
    index = snapshot / "model.safetensors.index.json"
    if not index.exists():
        return None
    return json.loads(index.read_text())["weight_map"]


class SafetensorsSource:
    """`tensor` for a whole tensor and `rows` for a row range, from one shard or many."""

    def __init__(self, snapshot: Path):
        from safetensors import safe_open

        snapshot = Path(snapshot)
        owner = shard_map(snapshot)
        if owner is None:
            single = snapshot / "model.safetensors"
            if not single.exists():
                shards = sorted(snapshot.glob("model.safetensors-*"))
                if shards:
                    raise SystemExit(
                        f"{snapshot} has {len(shards)} shards but no model.safetensors.index.json; "
                        "refusing to read part of a model"
                    )
                raise SystemExit(f"no safetensors file in {snapshot}")
            self._owner = {name: "model.safetensors" for name in safe_open(str(single), framework="pt").keys()}
            self._handles = {"model.safetensors": safe_open(str(single), framework="pt")}
        else:
            self._owner = owner
            self._handles = {
                shard: safe_open(str(snapshot / shard), framework="pt") for shard in sorted(set(owner.values()))
            }
        self.names = set(self._owner)
        self.shard_count = len(self._handles)

    def _handle(self, name: str):
        shard = self._owner.get(name)
        if shard is None:
            raise KeyError(f"no tensor named '{name}' in the checkpoint")
        return self._handles[shard]

    def tensor(self, name: str) -> np.ndarray:
        return self._handle(name).get_tensor(name).float().numpy().astype(np.float32)

    def rows(self, name: str, start: int, end: int) -> np.ndarray:
        return self._handle(name).get_slice(name)[start:end].float().numpy().astype(np.float32)
