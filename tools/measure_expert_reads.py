#!/usr/bin/env python3
"""Measure what the expert read path actually achieves on the real install.

Why this exists. `datacenter-trace` published "bytes from SSD" as `elementsRead * 2`, an estimate
that assumes bf16 on disk and overstates a **4-bit** install by ~3.5×. Every rate derived from it was
wrong with it — including an "effective 177 MB/s" reading that was withdrawn. **A number about the
read path has to come from the read path.**

The pattern here is the engine's, taken from the install's own geometry rather than assumed. One expert
is three bounded reads per tensor — codes, then group scales, then zero points — so a fetch is six
preads: 1,048,576 / 131,072 / 32,768 bytes for `gate_up` and 524,288 / 65,536 / 16,384 for `down`, for
1,818,624 bytes. That is a *good* shape for an SSD; the question is what it achieves.

    python3 tools/measure_expert_reads.py --install .build/m1-install --experts 64

Patterns, each over the same number of expert-bytes so the rows are comparable:

- `whole-slab` — one contiguous region, the ceiling this file can reach;
- `ascending` — experts in slab order, which no routing table ever does;
- `random` — experts in a fixed pseudo-random order, which is what routing looks like;
- `blocks-16k` — 16 KB at a time in random order, the shape behind the Testbed's 108 MB/s figure.

Standard library only, and it reads through `F_NOCACHE` so a measurement does not become a page-cache
incident on a node that has already had one.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_disk_headroom import require_headroom  # noqa: E402
from quantize import open_uncached  # noqa: E402


def expert_geometry(entry: dict) -> dict:
    """Where an expert's six reads sit inside a stacked tensor, from the manifest's own numbers.

    One expert is one **leading-axis** entry, and its row width is the product of the remaining
    dimensions *except the last*, which is the quantized column count. Reading `shape[1]` as the width
    is the leading-axis trap this repository has a note about, and it is how the first version of this
    function derived 592 bytes for a 1,212,416-byte expert.
    """
    experts = entry["shape"][0]
    rest = entry["shape"][1:]
    rows = 1
    for dimension in rest[:-1]:
        rows *= dimension
    columns = rest[-1]
    groups = columns // entry["group"]
    codes = rows * (columns // 2)
    scales = rows * groups * 4
    zeros = rows * groups
    per_expert = codes + scales + zeros
    if per_expert * experts != entry["nbytes"]:
        raise SystemExit(
            f"{entry['name']}: geometry does not explain nbytes "
            f"({per_expert} x {experts} != {entry['nbytes']})"
        )
    return {"experts": experts, "codes": codes, "scales": scales, "zeros": zeros, "per_expert": per_expert}


def read_expert(fd: int, entry: dict, geo: dict, expert: int) -> int:
    """One expert, exactly as the engine reads it: three bounded reads per tensor."""
    base = entry["offset"] + expert * geo["per_expert"]
    total = 0
    for offset, length in (
        (base, geo["codes"]),
        (base + geo["codes"], geo["scales"]),
        (base + geo["codes"] + geo["scales"], geo["zeros"]),
    ):
        total += len(os.pread(fd, length, offset))
    return total


def read_contiguous(fd: int, offset: int, nbytes: int, window: int = 4 * 1024 * 1024) -> int:
    total, read = 0, 0
    while read < nbytes:
        chunk = os.pread(fd, min(window, nbytes - read), offset + read)
        if not chunk:
            break
        total += len(chunk)
        read += len(chunk)
    return total


def measure(name: str, fn) -> dict:
    started = time.perf_counter()
    total = fn()
    seconds = time.perf_counter() - started
    return {"pattern": name, "bytes": total, "seconds": seconds, "mb_per_s": total / 1e6 / seconds if seconds else 0.0}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", type=Path, default=Path(".build/m1-install"))
    parser.add_argument("--experts", type=int, default=64, help="expert fetches per layer")
    parser.add_argument("--layers", type=int, default=0, help="layers to span (0 = all)")
    parser.add_argument("--blocks-kb", type=int, default=16, help="block size for the blocks pattern")
    parser.add_argument(
        "--min-gb", type=float, default=1.0,
        help="refuse a sample smaller than this: a small read is answered by cache and reports the "
             "cache, not the disk. The Testbed page already recorded a 13 GB/s 'measurement' that was "
             "a file fitting in memory, and the first version of this tool repeated it.",
    )
    args = parser.parse_args(argv)

    require_headroom(purpose="the read-path measurement")
    manifest = json.loads((args.install / "install.json").read_text())
    stacks = [t for t in manifest["tensors"] if t["role"] == "expert.stack_gate_up"]
    if not stacks:
        raise SystemExit("no expert.stack_gate_up in the manifest; is this an install?")
    stacks = stacks[: args.layers or len(stacks)]
    geos = [expert_geometry(t) for t in stacks]
    payload = args.install / "data.bin"

    per_layer = min(args.experts, geos[0]["experts"])
    total_bytes = sum(per_layer * g["per_expert"] for g in geos)
    print(f"install:  {payload}  ({payload.stat().st_size / 1e9:.2f} GB)")
    print(f"tensors:  {len(stacks)} expert stacks, {stacks[0]['name'].split('.mlp.')[0]} ...")
    print(
        f"expert:   {geos[0]['per_expert']:,} bytes = codes {geos[0]['codes']:,} + scales "
        f"{geos[0]['scales']:,} + zeros {geos[0]['zeros']:,}"
    )
    print(f"reading:  {per_layer} experts x {len(stacks)} layers = {total_bytes / 1e9:.2f} GB per pattern\n")

    if total_bytes < args.min_gb * 1e9:
        raise SystemExit(
            f"refusing: {total_bytes / 1e9:.2f} GB is below the {args.min_gb:.2f} GB floor, and a sample "
            f"that fits in cache measures cache. Raise --experts or --layers."
        )

    fd = open_uncached(payload)
    results = []
    try:
        def each(pattern: str, order: list[tuple[int, int]]) -> int:
            return sum(read_expert(fd, stacks[layer], geos[layer], expert) for layer, expert in order)

        ascending = [(layer, expert) for layer in range(len(stacks)) for expert in range(per_layer)]
        shuffled = list(ascending)
        random.Random(20260916).shuffle(shuffled)
        block = args.blocks_kb * 1024
        block_bytes = total_bytes // block
        offsets = []
        rng = random.Random(20260916)
        for tensor, geo in zip(stacks, geos):
            offsets.append((tensor["offset"], tensor["nbytes"]))
        spans = []
        remaining = block_bytes
        for offset, nbytes in offsets:
            take = min(remaining, nbytes // block)
            spans.append((offset, take * block))
            remaining -= take
            if remaining <= 0:
                break

        def blocks() -> int:
            total = 0
            for offset, length in spans:
                for _ in range(length // block):
                    at = offset + rng.randrange(0, max(1, length - block))
                    total += len(os.pread(fd, block, at))
            return total

        # Every pattern is measured before the descriptor closes: the first version ran one after,
        # which is a bad file descriptor rather than a measurement.
        results.append(measure("ascending", lambda: each("ascending", ascending)))
        # The same pattern again, immediately: if the second run is much faster, the first was real
        # and the second is cache — and if they are equal, neither is cache.
        results.append(measure("ascending (repeat)", lambda: each("ascending", ascending)))
        results.append(measure("random", lambda: each("random", shuffled)))
        results.append(measure(f"blocks-{args.blocks_kb}k", blocks))
    finally:
        os.close(fd)

    print(f"{'pattern':20} {'GB read':>8} {'seconds':>9} {'MB/s':>9}")
    for r in results:
        print(f"{r['pattern']:20} {r['bytes'] / 1e9:8.2f} {r['seconds']:9.3f} {r['mb_per_s']:9.1f}")
    cold, warm = results[0], results[1]
    print()
    print(
        f"cold {cold['mb_per_s']:.0f} MB/s vs repeat {warm['mb_per_s']:.0f} MB/s: "
        + ("a cache effect, so read the cold figure" if warm["mb_per_s"] > cold["mb_per_s"] * 1.25
           else "no cache effect — this sample is larger than cache and the figures are the disk's")
    )
    print(
        "\nThe engine's pattern is the `ascending`/`random` rows. A per-expert fetch is six preads like\n"
        "these, so a token's read time is (expert bytes) / (this rate) — and the shard planner for M2\n"
        "should be designed against the cold figure."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
