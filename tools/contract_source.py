#!/usr/bin/env python3
"""The one place that decides which reader a contract gets, and the flag that goes with it.

Two rounds fixed the same defect in two different places. `D69` added `--stream-experts` to the M1 gate, which
had never passed it; `D73` added `--uncached` to the same gate, which had never passed that either, so its
checkpoint runs mapped 67 GB while `docs/m1-gate.md` recorded both flags as the pair that makes such a run
survivable. Each time the question — *which reader, and with which flag* — had been answered independently
wherever it was asked, and one of the answers was wrong. So it is answered once, here.

`--stream-experts` is deliberately **not** here. Whether it exists is a property of the model family, not of the
reader: `qwen3_5` is dense and its forward has no `stream_experts` parameter, while `qwen3_5_moe` has routed
experts whose stacks must be fetched by index. A flag declared where it does nothing is worse than a flag
absent where it means nothing, so the MoE CLI declares it and the dense one does not. What both share is
`--uncached`, because both can be pointed at a checkpoint that does not fit in this machine's memory.

Standard library only, like everything else under `tools/` that gates the repository.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

UNCACHED_HELP = (
    "read the checkpoint through pread instead of `safe_open`'s mmap. The default is unchanged, because the "
    "contract is the authority and its reader should change only deliberately; this exists because a 67 GB "
    "mapping on an 8 GB node is the mechanism behind two panics, and it is byte-identical to the mapped reader "
    "(tools/test_uncached_safetensors.py checks that on a real shard of this checkpoint)."
)


def add_uncached_argument(parser: argparse.ArgumentParser) -> None:
    """Declare the safe-reading flag on a contract CLI.

    Every contract CLI gets this, and `tools/test_contract_source.py` asserts that, so a new one cannot be
    written without it and a gate cannot invoke one without being able to pass it.
    """
    parser.add_argument("--uncached", action="store_true", help=UNCACHED_HELP)


def open_source(snapshot: Path, *, uncached: bool = False) -> Any:
    """The reader for `snapshot`: an install if that is what it is, else a checkpoint, mapped or pread.

    An install directory is read through the install's own dequantiser, which is what the Swift reader mirrors.
    That is how the gate's real question gets asked: same weights on both sides, so a difference is a difference
    in arithmetic rather than in what was quantised (`D55`).
    """
    if (snapshot / "install.json").exists():
        from install_source import InstallSource

        return InstallSource(snapshot)
    if uncached:
        from uncached_safetensors import UncachedSafetensorsSource

        return UncachedSafetensorsSource(snapshot)
    from safetensors_source import SafetensorsSource

    return SafetensorsSource(snapshot)
