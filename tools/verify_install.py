#!/usr/bin/env python3
"""Verify a model install from Python, independently of the engine (`DC-108`).

The engine reads the install through its own Swift reader, and M1's gate compares its output to
`tools/ordered_reference.py`. Neither of those ever asks whether the **container** is what the packer
claimed: that every tensor's declared layout matches its payload, that the tensors tile the payload
exactly, that each role's quantisation is the one `tools/quant_policy.json` requires, and that the bytes
hash to the manifest's digest. This does, with `tools/install_reader.py`, in the standard library.

    python3 tools/verify_install.py .build/m1-install                 # structure and policy
    python3 tools/verify_install.py .build/m1-install --digests        # ... and every payload hash

`--digests` reads the whole payload, so it asks the disk guard first and streams: on the 21.7 GB M1
install it is a single sequential pass with bounded memory, not a job that fills the page cache.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import check_disk_headroom  # noqa: E402
from install_reader import Install, InstallError  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def policy_for(root: Path, script_dir: Path | None = None) -> dict[str, str]:
    """`role -> quantisation`, from the same policy file the packer was given.

    The manifest records that policy's **absolute path on the machine that built the install**, which does
    not exist anywhere else. The first version fell back to a path relative to the *repository*, so a copy
    of this tool on another node — where the install is all that travelled — died with a `FileNotFoundError`
    looking for `/Users/<builder>/tools/quant_policy.json`. The fallback is now the file **beside this
    script**, which is where a copied tool finds its own data, and the repository path is the last resort.
    """
    manifest = json.loads((root / "install.json").read_text())
    here = Path(__file__).resolve().parent if script_dir is None else script_dir
    candidates = [Path(name) for name in manifest.get("policy_files") or []]
    candidates.append(here / "quant_policy.json")
    candidates.append(ROOT / "tools" / "quant_policy.json")
    policy: dict[str, str] = {}
    chosen = None
    for path in candidates:
        if path.exists():
            chosen = path
            break
    if chosen is None:
        raise SystemExit(
            "no quant policy found: looked at the manifest's path, "
            + f"{here / 'quant_policy.json'}, and {ROOT / 'tools' / 'quant_policy.json'}. Copy "
            + "`tools/quant_policy.json` beside this script — a tool that travels has to travel with the "
            + "data it checks against, and this is that data."
        )
    policy.update(json.loads(chosen.read_text()).get("quant", {}))
    return policy


def coverage(install: Install) -> tuple[list[str], int]:
    """Every byte of the payload claimed exactly once, and nothing beyond it.

    A reader can satisfy every per-tensor check and still be wrong about where the tensors *are*, and
    the symptom would be one tensor reading another's bytes. Tiling is the check that catches it.
    """
    problems: list[str] = []
    spans = sorted((tensor.offset, tensor.offset + tensor.nbytes, tensor.name)
                   for tensor in install.tensors.values())
    expected = 0
    for start, end, name in spans:
        if start != expected:
            problems.append(
                f"{name}: starts at {start} where the payload is at {expected}"
                if start > expected else
                f"{name}: overlaps the tensor before it ({start} < {expected})"
            )
        expected = max(expected, end)
    size = install.data_path.stat().st_size
    if expected != size:
        problems.append(f"the tensors end at {expected} and the payload is {size} bytes")
    return problems, len(spans)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("install", type=Path)
    parser.add_argument("--digests", action="store_true", help="hash every payload (reads it all)")
    parser.add_argument("--sample", type=int, default=0, help="hash this many payloads instead")
    parser.add_argument("--json", type=Path, default=None, help="write the report here")
    args = parser.parse_args(argv)

    problems: list[str] = []
    with Install(args.install) as install:
        print(
            f"{args.install}: schema 1, family {install.family}, revision "
            f"{(install.revision or '')[:12]}…, {len(install.tensors)} tensor(s)"
        )
        policy = policy_for(args.install)
        by_quant: dict[str, int] = {}
        for tensor in install.tensors.values():
            by_quant[tensor.quant] = by_quant.get(tensor.quant, 0) + 1
            try:
                tensor.geometry()
            except InstallError as error:
                problems.append(str(error))
                continue
            required = policy.get(tensor.role)
            if required is None:
                problems.append(f"{tensor.name}: role {tensor.role!r} is not in the policy")
            elif required != tensor.quant:
                problems.append(
                    f"{tensor.name}: role {tensor.role!r} is {tensor.quant}, the policy requires "
                    f"{required}"
                )
        print(f"  quantisation: " + ", ".join(f"{k} {v}" for k, v in sorted(by_quant.items())))
        print(f"  policy roles: {len(policy)}")

        tiling, count = coverage(install)
        problems.extend(tiling)
        print(f"  payload: {count} tensor(s) tiling {install.data_path.stat().st_size:,} byte(s)")

        to_hash = list(install.tensors.values())
        if args.sample:
            # The largest first: a sample that only reads the small tensors proves little.
            to_hash = sorted(to_hash, key=lambda t: -t.nbytes)[: args.sample]
        elif not args.digests:
            to_hash = []
        if to_hash:
            check_disk_headroom.require_headroom(purpose="the install verification")
            total = sum(tensor.nbytes for tensor in to_hash)
            print(f"  hashing {len(to_hash)} payload(s), {total / 1e9:.2f} GB, streamed")
            for position, tensor in enumerate(to_hash, start=1):
                if not install.verified(tensor):
                    problems.append(f"{tensor.name}: the payload digest does not match the manifest")
                elif position % 100 == 0:
                    print(f"    {position}/{len(to_hash)}")
            print(f"    {len(to_hash)} payload(s) hashed")

    report = {
        "install": str(args.install),
        "family": install.family,
        "revision": install.revision,
        "tensors": len(install.tensors),
        "payload_bytes": install.data_path.stat().st_size,
        "hashed": len(to_hash),
        "problem_count": len(problems),
        "problems": problems[:50],
    }
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"  report: {args.json}")

    if problems:
        print(f"INSTALL VERIFICATION FAILED: {len(problems)} problem(s)", file=sys.stderr)
        for problem in problems[:20]:
            print(f"  {problem}", file=sys.stderr)
        return 1
    print(f"INSTALL VERIFIED: {len(install.tensors)} tensor(s), structure and policy"
          + (f", {len(to_hash)} payload digest(s)" if to_hash else " (digests not checked)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
