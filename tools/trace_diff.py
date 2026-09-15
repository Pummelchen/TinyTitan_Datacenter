#!/usr/bin/env python3
"""Golden-trace differ: bit-exact comparison with first-divergence localisation,
plus a separate, exact assertion for discrete decisions.

Two jobs, deliberately not merged:

- **Numerics** are compared byte for byte. The first differing tensor is reported
  with the element index, both values and the fp32 ULP distance, and the scan stops
  there — a divergence at layer 3 makes everything after it noise, and a report
  that buries the cause under a thousand consequences is a report nobody reads.
- **Discrete decisions** (router top-k index sets, sparse-attention block
  selections) are compared for exact equality and reported even when every float
  tensor matches. This is I3's teeth: a flipped top-k index diverges the output
  while every per-tensor check still looks healthy, so it can never be a tolerance
  question.

Exit status: 0 identical, 1 different, 2 the traces could not be read.

Usage::

    python3 tools/trace_diff.py <reference> <candidate> [--json] [--quiet]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from trace_format import Trace, TraceError, read_trace, ulp_distance_f32

FLOAT_DTYPES = ("f32", "f16", "bf16")


@dataclass
class Finding:
    kind: str
    where: str
    detail: str

    def line(self) -> str:
        return f"{self.kind:16s} {self.where:32s} {self.detail}"


@dataclass
class Report:
    reference: str
    candidate: str
    checked_tensors: int = 0
    checked_elements: int = 0
    discrete_checked: int = 0
    digest_shortcut: bool = False
    findings: list[Finding] = field(default_factory=list)

    @property
    def identical(self) -> bool:
        return not self.findings

    def add(self, kind: str, where: str, detail: str) -> None:
        self.findings.append(Finding(kind, where, detail))

    def summary(self) -> str:
        if self.identical:
            how = "matching digests" if self.digest_shortcut else "element comparison"
            return (
                f"IDENTICAL — {self.checked_tensors} tensor(s), "
                f"{self.checked_elements} element(s), {self.discrete_checked} discrete "
                f"decision(s) checked ({how})"
            )
        kinds = {}
        for finding in self.findings:
            kinds[finding.kind] = kinds.get(finding.kind, 0) + 1
        parts = ", ".join(f"{count} {kind}" for kind, count in sorted(kinds.items()))
        return f"DIFFERENT — {parts}"


def _first_differing_element(a: bytes, b: bytes, width: int) -> int:
    for index in range(0, min(len(a), len(b)), width):
        if a[index : index + width] != b[index : index + width]:
            return index // width
    return min(len(a), len(b)) // width


def _decode_element(trace: Trace, name: str, index: int):
    tensor = trace.tensor(name)
    width = len(tensor.payload) // max(1, _element_count(tensor.shape))
    payload = tensor.payload[index * width : (index + 1) * width]
    if tensor.dtype == "f32":
        import struct

        return struct.unpack("<f", payload)[0]
    if tensor.dtype == "f16":
        import struct

        return struct.unpack("<e", payload)[0]
    return tensor.values()[index]


def _element_count(shape) -> int:
    count = 1
    for dim in shape:
        count *= dim
    return count


def _reference_signature(trace: Trace) -> tuple | None:
    """What produced this trace, when that is a reference stack we can compare.

    A trace captured by transformers 5.16.1 and one captured by 5.17.0 are not two
    measurements of the same thing, and a gate that mixes them reports a difference
    that has nothing to do with the engine. Engine traces carry no such stack, so
    they are exempt.
    """
    reference = trace.manifest.get("reference") or {}
    if reference.get("tool") != "trace_capture":
        return None
    return (
        reference.get("transformers"),
        reference.get("torch"),
        reference.get("compute_dtype"),
        reference.get("weight_dtype"),
        reference.get("attn_implementation"),
    )


def compare(reference: Trace, candidate: Trace, stop_at_first_float: bool = True) -> Report:
    report = Report(reference=str(reference.root), candidate=str(candidate.root))

    # Comparability first, and before the digest shortcut: two traces from different
    # reference builds can have identical bytes and still not be two measurements of
    # the same thing.
    left, right = _reference_signature(reference), _reference_signature(candidate)
    if left is not None and right is not None and left != right:
        report.add(
            "provenance",
            "reference stack",
            f"reference {left}, candidate {right} — these traces are not comparable",
        )
        return report

    if reference.digest == candidate.digest:
        report.digest_shortcut = True
        report.checked_tensors = len(reference.tensor_names)
        report.discrete_checked = len(reference.discrete_names)
        return report

    ref_names = reference.tensor_names
    cand_names = set(candidate.tensor_names)
    first_float_reported = False

    for name in ref_names:
        if name not in cand_names:
            report.add("missing_tensor", name, "present in the reference, absent in the candidate")
            continue
        ref_entry, cand_entry = reference.entry(name), candidate.entry(name)
        if ref_entry["dtype"] != cand_entry["dtype"]:
            report.add(
                "dtype",
                name,
                f"reference {ref_entry['dtype']}, candidate {cand_entry['dtype']}",
            )
            continue
        if ref_entry["shape"] != cand_entry["shape"]:
            report.add(
                "shape",
                name,
                f"reference {ref_entry['shape']}, candidate {cand_entry['shape']}",
            )
            continue

        report.checked_tensors += 1
        ref_payload = reference.tensor(name).payload
        cand_payload = candidate.tensor(name).payload
        report.checked_elements += _element_count(ref_entry["shape"])

        if ref_payload == cand_payload:
            continue
        if first_float_reported and stop_at_first_float:
            continue

        width = max(1, len(ref_payload) // max(1, _element_count(ref_entry["shape"])))
        element = _first_differing_element(ref_payload, cand_payload, width)
        ref_value = _decode_element(reference, name, element)
        cand_value = _decode_element(candidate, name, element)
        detail = (
            f"element {element}: reference {ref_value!r}, candidate {cand_value!r}"
        )
        if ref_entry["dtype"] in FLOAT_DTYPES:
            detail += f", {ulp_distance_f32(float(ref_value), float(cand_value))} ULP apart"
        report.add("float", name, detail)
        first_float_reported = True

    for name in sorted(cand_names - set(ref_names)):
        report.add("extra_tensor", name, "present in the candidate, absent in the reference")

    # Discrete decisions: exact, separate, and reported whatever the floats did.
    ref_discrete = {d["name"]: d for d in reference.manifest.get("discrete", [])}
    cand_discrete = {d["name"]: d for d in candidate.manifest.get("discrete", [])}
    for name, entry in ref_discrete.items():
        if name not in cand_discrete:
            report.add("missing_discrete", name, "present in the reference, absent in the candidate")
            continue
        report.discrete_checked += 1
        other = cand_discrete[name]
        if entry["shape"] != other["shape"]:
            report.add(
                "discrete_shape",
                name,
                f"reference {entry['shape']}, candidate {other['shape']}",
            )
            continue
        if entry["values"] != other["values"]:
            missing = [v for v in entry["values"] if v not in other["values"]]
            extra = [v for v in other["values"] if v not in entry["values"]]
            order_only = not missing and not extra
            detail = f"reference {entry['values']}, candidate {other['values']}"
            if order_only:
                detail = "same values, different order — " + detail
            else:
                detail = f"missing {missing}, unexpected {extra} — " + detail
            report.add("discrete", name, detail)
    for name in sorted(set(cand_discrete) - set(ref_discrete)):
        report.add("extra_discrete", name, "present in the candidate, absent in the reference")

    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--json", action="store_true", help="machine-readable report")
    parser.add_argument("--quiet", action="store_true", help="summary only")
    args = parser.parse_args(argv)

    try:
        reference = read_trace(args.reference)
        candidate = read_trace(args.candidate)
    except TraceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    report = compare(reference, candidate)

    if args.json:
        print(
            json.dumps(
                {
                    "identical": report.identical,
                    "summary": report.summary(),
                    "checked_tensors": report.checked_tensors,
                    "checked_elements": report.checked_elements,
                    "discrete_checked": report.discrete_checked,
                    "reference_digest": reference.digest,
                    "candidate_digest": candidate.digest,
                    "findings": [
                        {"kind": f.kind, "where": f.where, "detail": f.detail}
                        for f in report.findings
                    ],
                },
                indent=2,
            )
        )
    else:
        if not args.quiet:
            for finding in report.findings:
                print(finding.line())
        print(report.summary())

    return 0 if report.identical else 1


if __name__ == "__main__":
    sys.exit(main())
