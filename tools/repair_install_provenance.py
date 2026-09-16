"""Repair an install's provenance block from the source files it was built from.

`I6` says the converted artifact must say which weights it came from. `tools/quantize.py` now digests every
source weight file and records the repo and revision as inputs, but the M1 install **predates that fix**: its
`source.files` is empty, its `source.repo` holds a commit hash, and its `source.revision` says `"local"`. The
audit therefore stands at `partly`, and it is the only `partly` left.

The values are recoverable, and this repairs them **without touching a byte of the payload**:

* the **revision** is the snapshot directory's own name, and the **repo** the cache directory's name —
  `…/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96e…/` names both, so neither has to be supplied;
* `files` is `quantize.digest_snapshot`, the same function the packer calls, so the digests are computed the
  way the artifact would have computed them.

Three rules keep it honest rather than convenient:

1. **The payload is provably untouched.** The manifest is rebuilt with only `source` and `passes` changed, and
   every other key is compared in canonical JSON before anything is written. A repair that could alter a
   tensor digest would be a rewrite, not a repair.
2. **A repair is recorded as a pass.** `provenance-repair` is appended to `passes`, so the artifact describes
   its own history instead of pretending it was built that way.
3. **A disagreement is refused, not overwritten.** An install that already carries file digests is compared
   against the recomputed ones: equal is a no-op, different is an error for a human, because it means the
   artifact and the source on disk are not the same pair.

    python3 tools/repair_install_provenance.py --install .build/m1-install --dry-run
    python3 tools/repair_install_provenance.py --install .build/m1-install
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from quantize import digest_snapshot, provenance_warnings  # noqa: E402

PLACEHOLDERS = {"local", "unknown", "none", "test", "placeholder", "todo"}
PASS_NAME = "provenance-repair"


class RepairError(Exception):
    """The repair cannot be made honestly."""


def repo_from_cache(path: Path) -> str:
    """`…/models--Qwen--Qwen3.6-35B-A3B/snapshots/<sha>/` -> `Qwen/Qwen3.6-35B-A3B`."""
    for parent in path.parents:
        if parent.name.startswith("models--"):
            parts = parent.name[len("models--") :].split("--")
            if len(parts) >= 2:
                return "/".join(parts)
    raise RepairError(f"no `models--…` cache directory above {path}; pass --repo explicitly")


def discover_snapshot(root: Path, given: Path | None) -> Path:
    if given is not None:
        if not (given / "config.json").exists():
            raise RepairError(f"{given} has no config.json, so it is not a snapshot")
        return given
    candidates = sorted(root.glob("models--*/snapshots/*"))
    if len(candidates) != 1:
        raise RepairError(
            f"expected exactly one snapshot under {root}, found {len(candidates)}; pass --snapshot"
        )
    return candidates[0]


def repair(install_root: Path, snapshot: Path, repo: str | None, dry_run: bool) -> dict:
    manifest_path = install_root / "install.json"
    if not manifest_path.exists():
        raise RepairError(f"{manifest_path} does not exist")
    manifest = json.loads(manifest_path.read_text())
    source = dict(manifest.get("source") or {})

    revision = snapshot.name
    repo = repo or repo_from_cache(snapshot)
    files = digest_snapshot(snapshot)
    if not files:
        raise RepairError(f"{snapshot} holds no *.safetensors, so there is nothing to trace")

    existing = source.get("files") or {}
    if existing:
        if existing == files:
            print("provenance already correct; nothing to do")
            return {"changed": False, "reason": "already correct"}
        differing = sorted(set(existing) ^ set(files))
        raise RepairError(
            "this install already records source file digests and they differ from the source on disk "
            f"({len(differing)} name(s) differ, e.g. {differing[:3]}); the artifact and the weights are not "
            "the same pair, which is a question for a human and not something to overwrite"
        )

    repaired = json.loads(json.dumps(manifest))
    repaired["source"] = dict(source, repo=repo, revision=revision, files=files)
    passes = list(repaired.get("passes") or [])
    if PASS_NAME not in passes:
        passes.append(PASS_NAME)
    repaired["passes"] = passes

    # Rule 1: nothing outside `source` and `passes` may move. Compare canonically, key by key.
    for key in sorted(set(manifest) | set(repaired)):
        if key in ("source", "passes"):
            continue
        if json.dumps(manifest.get(key), sort_keys=True) != json.dumps(repaired.get(key), sort_keys=True):
            raise RepairError(f"the repair would change {key!r}, which it must not; refusing")

    print(f"repo      {source.get('repo')!r} -> {repo!r}")
    print(f"revision  {source.get('revision')!r} -> {revision!r}")
    print(f"files     {len(existing)} -> {len(files)} digest(s)")
    for warning in provenance_warnings(repo, revision, files):
        print(f"  warning: {warning}")
    if dry_run:
        print("dry run: nothing written")
        return {"changed": False, "reason": "dry run"}

    backup = install_root.parent / "reverify" / f"{install_root.name}.install.json.before-{PASS_NAME}"
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest_path, backup)
    manifest_path.write_text(json.dumps(repaired, indent=2) + "\n")
    print(f"repaired {manifest_path}; the previous manifest is at {backup}")
    return {"changed": True, "backup": str(backup), "files": len(files)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, default=None)
    parser.add_argument("--repo", default=None)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / ".build" / "hf-cache",
        help="where to look for the snapshot when --snapshot is not given",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        snapshot = discover_snapshot(args.cache_root, args.snapshot)
        print(f"source snapshot: {snapshot}")
        repair(args.install, snapshot, args.repo, args.dry_run)
    except RepairError as error:
        print(f"refusing: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
