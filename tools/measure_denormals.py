#!/usr/bin/env python3
"""How many quantized scales are denormal, zero, or not finite.

The measurement that decides the Metal question. `DC-087` found that Metal flushes denormal
operands to zero while the CPU does not, so a GPU kernel matches the fp32 contract everywhere
except where an operand is denormal — and a group's scale is exactly the quantity that can land
there, because it is the group's span over fifteen. The decision is between two renegotiations:
flush denormals on the CPU as well, or keep the GPU away from ops whose operands can be denormal.
Both are expensive, and which one is worth paying for depends on how often the case occurs.

Reads only the **scale section** of each quantized tensor, uncached, in bounded windows — for the
35 B install that is about 640 MB of a 20 GB artifact, and it is a read, not a run of the model.
Needs numpy, so it lives in the venv like the other measuring tools rather than gating CI.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from quantize import open_uncached  # noqa: E402

#: The smallest positive normal fp32 value.
MIN_NORMAL = float(np.finfo(np.float32).tiny)


def scan(install: Path, window: int = 4 << 20) -> dict:
    manifest = json.loads((install / "install.json").read_text())
    descriptor = open_uncached(install / "data.bin")
    totals = {"scales": 0, "denormal": 0, "zero": 0, "nonfinite": 0, "tensors": 0, "worst": 0.0}
    per_tensor: list[str] = []
    try:
        for entry in manifest["tensors"]:
            if entry["dtype"] != "int4":
                continue
            totals["tensors"] += 1
            rows = 1
            for dimension in entry["shape"][:-1]:
                rows *= dimension
            groups = rows * (entry["padded_columns"] // entry["group"])
            code_bytes = rows * (entry["padded_columns"] // 2)
            start = entry["offset"] + code_bytes
            remaining = groups * 4
            entry_denormal = 0
            while remaining > 0:
                count = min(window, remaining)
                raw = os.pread(descriptor, count, start)
                if not raw:
                    raise SystemExit(f"{entry['name']}: short read at {start}")
                values = np.frombuffer(raw, dtype=np.float32)
                magnitude = np.abs(values)
                denormal = np.count_nonzero((magnitude > 0) & (magnitude < MIN_NORMAL))
                entry_denormal += int(denormal)
                totals["denormal"] += int(denormal)
                totals["zero"] += int(np.count_nonzero(magnitude == 0))
                totals["nonfinite"] += int(np.count_nonzero(~np.isfinite(values)))
                totals["scales"] += int(values.size)
                start += count
                remaining -= count
            if entry_denormal:
                per_tensor.append(f"{entry['name']}: {entry_denormal} of {groups}")
    finally:
        os.close(descriptor)
    totals["per_tensor"] = per_tensor
    return totals


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("install", type=Path)
    parser.add_argument("--show", type=int, default=10, help="how many tensors to name")
    args = parser.parse_args(argv)
    totals = scan(args.install)
    scales = max(totals["scales"], 1)
    print(f"install: {args.install}")
    print(f"  quantized tensors: {totals['tensors']}")
    print(f"  scales:            {totals['scales']}")
    print(f"  denormal:          {totals['denormal']} ({100 * totals['denormal'] / scales:.6f} %)")
    print(f"  zero:              {totals['zero']} ({100 * totals['zero'] / scales:.6f} %)")
    print(f"  non-finite:        {totals['nonfinite']}")
    for line in totals["per_tensor"][: args.show]:
        print(f"    {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
