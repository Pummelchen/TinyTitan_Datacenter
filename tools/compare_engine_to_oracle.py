#!/usr/bin/env python3
"""Compare an engine trace against a reference trace, as numbers and as decisions.

`trace_diff.py` is the gate: it reports every differing byte between two traces that are
supposed to be *identical*. This is the other comparison — the engine against the
**semantic oracle**, which is a different implementation with a different summation order
and therefore can never be byte-identical (D3, R13). It reports two things separately:

- the numeric spread, per shared tensor, relative to that tensor's own scale;
- the **discrete decisions** — the argmax of the logits — asserted as an index set.

Keeping them apart is the point of I3: a 1-ULP logit difference can flip a marginal argmax,
and from there the continuations are unrelated while every per-tensor check still looks
green. So "the numbers are close" and "the decisions match" are two claims, and this prints
them as two.

    .venv/bin/python tools/compare_engine_to_oracle.py .build/engine-2b .build/torch-2b
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import trace_format as tf  # noqa: E402


def values(trace, name: str) -> np.ndarray:
    return np.frombuffer(trace.tensor(name).payload, dtype=np.float32)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", type=Path)
    parser.add_argument("oracle", type=Path)
    parser.add_argument("--quiet", action="store_true", help="only the summary and the verdict")
    args = parser.parse_args(argv)

    engine = tf.read_trace(args.engine)
    oracle = tf.read_trace(args.oracle)
    shared = [name for name in oracle.tensor_names if name in engine.tensor_names]
    if not shared:
        print("no shared tensor names: these traces are not comparable", file=sys.stderr)
        return 2

    print(f"engine {len(engine.tensor_names)} tensors, oracle {len(oracle.tensor_names)}, shared {len(shared)}")
    worst_relative = 0.0
    worst_name = ""
    if not args.quiet:
        print(f"  {'tensor':22s} {'max|Δ|':>11s} {'mean|Δ|':>11s} {'scale':>11s} {'relative':>10s}")
    for name in shared:
        a, b = values(engine, name), values(oracle, name)
        if a.shape != b.shape:
            print(f"  {name}: SHAPE MISMATCH {a.shape} vs {b.shape}", file=sys.stderr)
            return 2
        difference = np.abs(a - b)
        scale = float(np.abs(b).max())
        relative = float(difference.max()) / max(scale, 1e-30)
        if relative > worst_relative:
            worst_relative, worst_name = relative, name
        if not args.quiet:
            print(f"  {name:22s} {difference.max():11.3e} {difference.mean():11.3e} {scale:11.3e} {relative:10.2e}")

    # The discrete decision, as an index set (I3), from the logits alone.
    verdict = "not compared"
    if "logits" in shared:
        width = engine.entry("logits")["shape"][-1]
        engine_logits = values(engine, "logits").reshape(-1, width)
        oracle_logits = values(oracle, "logits").reshape(-1, width)
        engine_tokens = engine_logits.argmax(-1)
        oracle_tokens = oracle_logits.argmax(-1)
        agree = bool(np.array_equal(engine_tokens, oracle_tokens))
        margins = np.sort(oracle_logits, axis=-1)[:, -2:]
        smallest = float((margins[:, 1] - margins[:, 0]).min())
        print(f"\n  argmax engine: {list(engine_tokens)}")
        print(f"  argmax oracle: {list(oracle_tokens)}")
        print(f"  smallest top-1 margin: {smallest:.4f}")
        verdict = "MATCH" if agree else "DIFFER"
        print(f"  discrete decisions: {verdict}")
        if not agree:
            return 1

    print(
        f"\nworst relative difference: {worst_relative:.2e} at {worst_name}\n"
        f"these traces are two implementations, so the numbers are expected to differ by "
        f"rounding order; the decisions are not (D3, R13)"
    )
    return 0 if verdict in ("MATCH", "not compared") else 1


if __name__ == "__main__":
    raise SystemExit(main())
