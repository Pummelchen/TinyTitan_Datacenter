"""Run every gate, in one command, with the test counts read from the runs themselves.

Two things this exists for.

**One place, not a list in a shell.** The gates were run by chaining commands by hand, and a chain that is
remembered rather than executed eventually skips a link — this repository has already shipped a commit whose
message described documents a failed doc-patch never wrote, and a gate that only runs when someone remembers
it is a gate that stops being true quietly.

**The counts come from the runs.** `check_status_claims.py` checks the numbers the documentation states
against the suites' actual output, and for three rounds it was fed those numbers by hand — twice after they
had gone stale. Here the Python and Swift suites run first, their counts are parsed from their own output,
and those counts are what the claims gate is given. The class of mistake stops being possible rather than
being watched for.

`--skip-swift` runs the standard-library gates alone, which is what a machine without the project toolchain
can still do — and what the CI job that has no Xcode 27 runs. A Python test file that cannot even be
imported, because it needs a pinned package, is reported **NOT CHECKED** with the reason: a check that
cannot run must never look like one that passed.

Standard library only, like every gate here.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"

PYTHON_RAN = re.compile(r"^Ran (\d+) tests?", re.MULTILINE)
SWIFT_TESTS = re.compile(r"Executed (\d+) tests?,")
SWIFT_FAILURES = re.compile(r"with (?:\d+ tests? skipped and )?(\d+) failures?")
SWIFT_SKIPPED = re.compile(r"with (\d+) tests? skipped")


def run(command: list[str], cwd: Path = ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(command, cwd=cwd, capture_output=True, text=True)


def python_test_files(tools: Path = TOOLS) -> list[Path]:
    """Every `test_*.py`, sorted, so the order of the summary does not depend on the filesystem."""
    return sorted(tools.glob("test_*.py"))


def run_python_tests(tools: Path = TOOLS, python: str | None = None) -> tuple[int, list[str], list[str], list[str]]:
    """`(tests, problems, not_checked, detail)` — one file at a time, so a missing package is local.

    `unittest discover` stops the whole run when one file cannot be imported, which would make a machine
    without `numpy` look like a machine where the tools are broken. Each file is therefore run on its own and
    a file that cannot be imported is NOT CHECKED, by name.
    """
    interpreter = python or sys.executable
    total = 0
    problems: list[str] = []
    not_checked: list[str] = []
    detail: list[str] = []
    for path in python_test_files(tools):
        # `discover -s tools -p <file>` rather than `-m unittest tools.<stem>`: the test files import their
        # siblings by bare name (`from check_markdown_links import …`), which `discover` makes importable by
        # putting `tools/` on the path and `-m unittest tools.x` does not. The first version of this used the
        # dotted form and reported nine first-party modules as missing packages.
        result = run([interpreter, "-m", "unittest", "discover", "-s", str(tools), "-p", path.name])
        output = result.stdout + result.stderr
        if result.returncode != 0 and "ModuleNotFoundError" in output:
            missing = re.search(r"ModuleNotFoundError: No module named '([^']+)'", output)
            not_checked.append(f"{path.name} (needs {missing.group(1) if missing else 'a package'})")
            continue
        match = PYTHON_RAN.search(output)
        if match is None:
            problems.append(f"{path.name} did not report a test count")
            continue
        total += int(match.group(1))
        if result.returncode != 0:
            problems.append(f"{path.name}: {match.group(1)} test(s), FAILED")
            detail.extend(line for line in output.splitlines() if line.startswith(("FAIL:", "ERROR:")))
        else:
            detail.append(f"{path.name}: {match.group(1)} test(s), OK")
    return total, problems, not_checked, detail


SWIFT_FAILURE = re.compile(r"error: -\[([^\]]+)\]\s*:?\s*(.*)")


def swift_failure_names(output: str) -> list[str]:
    """Which tests failed, because a count is not a diagnosis.

    This runner reported "1 failure(s)" with no name when the suite failed once during a battery run — and
    that failure has not recurred in three consecutive runs since, so it can never be explained. A gate that
    cannot say *what* failed turns a flake into a mystery.
    """
    names: list[str] = []
    for line in output.splitlines():
        match = SWIFT_FAILURE.search(line)
        if match is not None:
            detail = match.group(2).strip() or "failed"
            names.append(f"{match.group(1)}: {detail}")
    return names


def parse_swift_summary(output: str) -> tuple[int, int, int] | None:
    """`(tests, skipped, failures)` from the **overall** summary line, or None.

    XCTest prints one `Executed …` line per suite and then the total, and SwiftPM repeats the total — so the
    overall figure is the **last** such line. The first version took the first match, which was a suite's six
    tests, and reported 13 of 186 to a gate whose whole purpose is checking that number.
    """
    lines = [line for line in output.splitlines() if "Executed " in line and " tests" in line]
    if not lines:
        return None
    total = lines[-1]
    tests = SWIFT_TESTS.search(total)
    if tests is None:
        return None
    failures = SWIFT_FAILURES.search(total)
    skipped = SWIFT_SKIPPED.search(total)
    return int(tests.group(1)), int(skipped.group(1)) if skipped else 0, int(failures.group(1)) if failures else 0


def run_swift_tests(swift: str = "swift") -> tuple[int | None, int | None, list[str], str]:
    """`(tests, skipped, problems, detail)` from `swift test --no-parallel`."""
    result = run([swift, "test", "--no-parallel"])
    output = result.stdout + result.stderr
    parsed = parse_swift_summary(output)
    if parsed is None:
        tail = "\n".join(output.strip().splitlines()[-6:])
        return None, None, [f"swift test did not report a count: {tail}"], "no count"
    tests, skipped, failures = parsed
    if result.returncode == 0 and failures == 0:
        problems: list[str] = []
    else:
        names = swift_failure_names(output)
        problems = [
            f"swift test: {failures} failure(s)"
            + (": " + "; ".join(names) if names else " (no failure line named — the output has been lost)")
        ]
    return tests, skipped, problems, f"{tests} test(s), {skipped} skipped, {failures} failure(s)"


def counts_completeness(python_tests: int, python_missing: list[str], swift_tests: int | None) -> str | None:
    """Why the documentation's counts cannot be checked from this run, or None when they can.

    An **incomplete** Python run cannot check them: if a file was skipped for want of a package, the total is
    smaller than the documentation states for a complete run, and comparing them would report the
    documentation as wrong when the machine is what is short.
    """
    if python_missing:
        names = ", ".join(item.split(" (")[0] for item in python_missing)
        return f"the Python suite was incomplete here: {names}"
    if not python_tests:
        return "no Python tests ran"
    if swift_tests is None:
        return "no Swift count to check them against"
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skip-swift", action="store_true", help="the standard-library gates only")
    parser.add_argument(
        "--baselines", action="append", default=[], help="a metrics.json for the baseline gate (repeatable)"
    )
    args = parser.parse_args(argv)

    problems: list[str] = []
    not_checked: list[str] = []
    summary: list[tuple[str, str]] = []

    def simple(name: str, command: list[str]) -> None:
        result = run(command)
        if result.returncode == 0:
            summary.append((name, "OK"))
        else:
            summary.append((name, "FAILED"))
            problems.append(f"{name}: {result.stdout.strip() or result.stderr.strip()}")

    simple("toolchain", [sys.executable, str(TOOLS / "check_toolchain.py")])
    simple("markdown tables", [sys.executable, str(TOOLS / "check_markdown_tables.py")])
    simple("markdown links", [sys.executable, str(TOOLS / "check_markdown_links.py")])
    simple("provenance", [sys.executable, str(TOOLS / "check_provenance.py")])

    python_tests, python_problems, python_missing, python_detail = run_python_tests()
    summary.append(("python tests", f"{python_tests} test(s)" + ("" if not python_problems else ", FAILED")))
    problems.extend(python_problems)
    not_checked.extend(python_missing)

    swift_tests: int | None = None
    swift_skipped: int | None = None
    if args.skip_swift:
        not_checked.append("swift build and test (--skip-swift: this machine has no project toolchain)")
    else:
        swift_tests, swift_skipped, swift_problems, swift_detail = run_swift_tests()
        summary.append(("swift tests", swift_detail))
        problems.extend(swift_problems)

    # The counts are the runs' own, which is the point of the ordering above. An **incomplete** Python run
    # cannot check the documentation's counts: if a file was skipped for want of a package, the total is
    # smaller than what the documentation states for a complete run, and comparing them would report the
    # documentation as wrong when the machine is what is short. It is NOT CHECKED, with the reason.
    reason = counts_completeness(python_tests, python_missing, swift_tests)
    if reason is not None:
        not_checked.append(f"status claims ({reason})")
    else:
        result = run([
            sys.executable, str(TOOLS / "check_status_claims.py"),
            "--swift-tests", str(swift_tests),
            "--swift-skipped", str(swift_skipped or 0),
            "--python-tests", str(python_tests),
        ])
        summary.append(("status claims", "OK" if result.returncode == 0 else "FAILED"))
        if result.returncode != 0:
            problems.append(f"status claims: {result.stdout.strip()}{result.stderr.strip()}")

    if args.baselines:
        result = run([
            sys.executable, str(TOOLS / "check_baselines.py"),
            *[item for path in args.baselines for item in ("--metrics", path)],
        ])
        summary.append(("baselines", "OK" if result.returncode == 0 else "FAILED"))
        if result.returncode != 0:
            problems.append(f"baselines: {result.stdout.strip()}{result.stderr.strip()}")
    else:
        not_checked.append("baselines (no --metrics given; they need a run to compare against)")

    width = max(len(name) for name, _ in summary)
    print("\nGATES")
    for name, detail in summary:
        print(f"  {name.ljust(width)}  {detail}")
    for line in not_checked:
        print(f"  NOT CHECKED: {line}")
    for line in problems:
        print(f"  PROBLEM: {line}", file=sys.stderr)
    if problems:
        print(f"\nGATES FAILED: {len(problems)} problem(s)", file=sys.stderr)
        return 1
    print(f"\nGATES OK: {len(summary)} gate(s), {len(not_checked)} not checked here")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
