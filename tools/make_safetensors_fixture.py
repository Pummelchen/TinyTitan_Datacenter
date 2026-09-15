#!/usr/bin/env python3
"""Emit a tiny safetensors file covering the header cases the reader must handle.

The real checkpoints are 1–5 GB and are never committed, so the reader's parsing — the
`__metadata__` block that is not a tensor, upper-case dtype names, mixed dtypes — is
covered by a fixture small enough to keep in the repository.

    .venv/bin/python tools/make_safetensors_fixture.py
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
import torch

FIXTURE = Path(__file__).resolve().parent.parent / "tests" / "DatacenterEngineTests" / "Fixtures" / "tiny.safetensors"


def main() -> int:
    from safetensors.torch import save_file

    tensors = {
        # bf16 and f32 and f16 together, because the reader switches on the dtype.
        "bf16.tensor": torch.tensor([[1.0, -2.5, 0.5], [0.0, 3.25, -0.125]], dtype=torch.bfloat16),
        "f32.tensor": torch.tensor([1.5, 2.5, -3.5, 4.5], dtype=torch.float32),
        "f16.tensor": torch.tensor([[0.25, -0.75]], dtype=torch.float16),
    }
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    # `metadata` is written into the header's `__metadata__` key, which the format
    # reserves and which a reader that treats every key as a tensor fails on.
    save_file(tensors, str(FIXTURE), metadata={"format": "pt", "purpose": "reader-test"})
    print(f"wrote {FIXTURE} ({FIXTURE.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
