"""Re-check the milestone claims that can be re-checked cheaply, and name the ones that cannot.

`run_all_gates.py` checks the repository: its tests, its links, its claims about numbers. This checks the
**milestones**, and it exists because of a specific failure of that habit.

The M1 gate was recorded as passing, with a digest. The engine's digest has since moved — visible in this
repository's own later records — and the contract was never re-run against it, so the stored pass quietly
became history. Nobody noticed for rounds, because nothing re-checked it: the finding came from a manual
comparison, thirty seconds of work that no tool was doing.

Two claims are easy to conflate and this keeps them apart:

* **the engine is stable** — it still produces the digest its own record says it does. A cheap single-node
  trace on the install answers this, and any *new* divergence fails.
* **the engine matches the reference** — the milestone's actual claim. That needs the contract, which on
  the 35 B model means reading the checkpoint, which this machine cannot safely do. So it is reported as
  `stale` or `not checked` **with the reason**, never assumed from the first.

`tools/milestones.json` holds the claims and their expected digests. A divergence that is already known and
tracked is declared there with its task id, so the tool stays useful rather than permanently red; a
divergence that is **not** declared fails, which is how a new one arrives loudly instead of silently.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_DATA = Path(__file__).resolve().parent / "milestones.json"
TRACE_BIN = ROOT / ".build" / "release" / "datacenter-trace"


def load_milestones(path: Path | None = None) -> dict:
    document = json.loads((path or DEFAULT_DATA).read_text())
    milestones = document.get("milestones") or []
    if not milestones:
        raise ValueError("milestones.json declares no milestones")
    for entry in milestones:
        for field in ("id", "claim", "agreement", "source"):
            if field not in entry:
                raise ValueError(f"milestone {entry.get('id')!r} is missing {field!r}")
        if entry["agreement"] == "stale" and not entry.get("known_mismatch"):
            raise ValueError(
                f"milestone {entry['id']} is stale and declares no known_mismatch: a stale agreement has to "
                f"be tracked somewhere, or it is a fact nobody owns"
            )
    return document


def read_digest(trace_dir: Path) -> str | None:
    manifest = trace_dir / "manifest.json"
    if not manifest.exists():
        return None
    return json.loads(manifest.read_text()).get("digest")


def run_engine_trace(install: Path, tokens: list[int], out: Path) -> str | None:
    """One single-node trace on the install, under the heavy-job lock."""
    from heavy_job import require_heavy_headroom

    require_heavy_headroom(0.35, purpose="the milestone re-check trace")
    if not TRACE_BIN.exists():
        raise SystemExit(f"{TRACE_BIN} is missing: run `swift build -c release` first")
    if out.exists():
        shutil.rmtree(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [str(TRACE_BIN), str(install), str(out), ",".join(str(token) for token in tokens)],
        cwd=ROOT, capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise SystemExit(f"the engine trace failed: {result.stdout}{result.stderr}")
    return read_digest(out)


def install_report(install: Path, sample: int = 0) -> dict | None:
    """The install verifier's report, or `None` when there is no install to verify.

    `I4` says the policy is data and `I6` says the artifact carries its own provenance, and both are checked by
    `tools/verify_install.py` — which until now only ever ran **by hand**. The milestone check is where this
    repository re-checks its claims, so it is where the artifact those claims rest on gets checked too: schema,
    roles, policy coverage, tiling, the provenance header, and **every payload digest**.

    The digests are included rather than left to a separate run because their cost was measured before this
    decision: on the 21.7 GB install the whole pass takes **26.7 s** and peaks at **29.4 MB**, and free disk and
    swap do not move, because it is one sequential `pread` pass and never a page-cached mapping. What that buys
    is `I6` in its strongest form — the manifest *records* a sha256 per payload and, until this round, nothing
    compared them to it. `--sample` bounds the work for a quick run.

    The verifier's own output is inherited rather than captured, because it is the evidence: it prints the
    payload coverage and, on failure, the problems themselves. A verifier that cannot run at all is a problem
    rather than a pass, so a missing report is reported — the one outcome this must never produce is silence.
    """
    if not (install / "install.json").exists():
        return None
    report_path = ROOT / ".build" / "milestone-check" / "install-verify.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if report_path.exists():
        report_path.unlink()
    command = [
        sys.executable, str(ROOT / "tools" / "verify_install.py"), str(install), "--json", str(report_path),
    ]
    command += ["--sample", str(sample)] if sample else ["--digests"]
    done = subprocess.run(command, cwd=ROOT, text=True)
    try:
        return json.loads(report_path.read_text())
    except (OSError, ValueError):
        return {"problems": [f"the install verifier wrote no readable report (exit {done.returncode})"]}


def install_problems(install: Path, sample: int = 0) -> list[str] | None:
    """The install verifier's problems, or `None` when there is no install to verify."""
    report = install_report(install, sample=sample)
    return None if report is None else list(report.get("problems", []))


def evaluate(document: dict, digest: str | None) -> tuple[list[str], list[tuple[str, str, str, str]]]:
    """`(problems, rows)` — a problem is an undeclared divergence, not a declared one."""
    problems: list[str] = []
    rows: list[tuple[str, str, str, str]] = []
    for entry in document["milestones"]:
        expected = entry.get("engine_digest")
        if expected is None:
            rows.append((entry["id"], "not checkable here", entry["agreement"], entry.get("recheck", "")))
            continue
        reproduces = digest is not None and digest.startswith(expected.rstrip("…"))
        if not reproduces:
            declared = entry.get("known_mismatch")
            detail = f"the engine produced {digest}, the record says {expected}"
            if declared:
                rows.append((entry["id"], f"differs ({declared})", entry["agreement"], detail))
            else:
                rows.append((entry["id"], "DIFFERS, UNDECLARED", entry["agreement"], detail))
                problems.append(f"{entry['id']}: {detail} — and no known_mismatch declares it")
            continue
        rows.append((entry["id"], "reproduces", entry["agreement"], entry.get("agreement_note", "")))
    return problems, rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--install", type=Path, default=ROOT / ".build" / "m1-install")
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument(
        "--trace-dir", type=Path, default=None,
        help="use an existing trace instead of running one (the tests do this)",
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--sample", type=int, default=0,
        help="verify this many payload digests rather than all of them (the default reads the whole payload)",
    )
    args = parser.parse_args(argv)

    document = load_milestones(args.data)
    tokens = document["prompt"]["tokens"]

    if args.trace_dir is not None:
        digest = read_digest(args.trace_dir)
        source = str(args.trace_dir)
    else:
        # Verify the artifact before trusting a digest computed from it. A trace that reproduces its recorded
        # digest on an install whose provenance or policy coverage is broken is a green light over a broken
        # artifact, which is the shape of failure this repository keeps finding.
        report = install_report(args.install, sample=args.sample)
        if report is not None and report.get("problems"):
            # The verifier has already printed the problems themselves, above.
            print(
                f"MILESTONE CHECK FAILED: the install does not verify "
                f"({len(report['problems'])} problem(s))",
                file=sys.stderr,
            )
            return 1
        out = ROOT / ".build" / "milestone-check" / "single-node"
        digest = run_engine_trace(args.install, tokens, out)
        source = str(out)

    problems, rows = evaluate(document, digest)
    if args.json:
        print(json.dumps({"digest": digest, "rows": rows, "problems": problems}, indent=2))
    else:
        print(f"engine digest on {source}: {digest}")
        print(f"prompt: {document['prompt']['id']} ({document['prompt']['length']} tokens)")
        width = max(len(row[0]) for row in rows)
        for milestone, engine, agreement, detail in rows:
            print(f"  {milestone.ljust(width)}  engine {engine:22s}  contract {agreement}")
            if detail:
                print(f"  {' ' * width}  {detail}")
    for problem in problems:
        print(f"MILESTONE: {problem}", file=sys.stderr)
    if problems:
        print(f"MILESTONE CHECK FAILED: {len(problems)} undeclared divergence(s)", file=sys.stderr)
        return 1
    stale = [row[0] for row in rows if row[2] == "stale"]
    print(
        f"MILESTONE CHECK OK: no undeclared divergence"
        + (f"; {len(stale)} agreement(s) declared stale ({', '.join(stale)})" if stale else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
