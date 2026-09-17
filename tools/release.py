#!/usr/bin/env python3
"""Build, verify and (only on an explicit flag) publish a release.

`RELEASE.md` is the standard this follows, and the parts that shape the code are worth naming here:

- **Dry run by default** (§1.2.6). Nothing is pushed, tagged or uploaded without `--publish`.
- **Gates run in order and each is able to fail** (§1.5): the repository's own gates, the full test suite
  serially, the milestone parity check, and a **clean scratch build** whose log is scanned for warnings —
  a fresh scratch path, because a warning scan over an incremental build compiles nothing and passes
  vacuously. The *plan* is checked too (`swift package describe`), not just the artifacts.
- **Assert the architecture, do not assume it** (§1.2.2): `lipo -archs` must say exactly `arm64`, checked on
  the binaries **inside the packaged archive**, not on the ones the build left behind.
- **One checksummed artifact** (§1.2.5), and the notes quote the digest of the archive that was actually
  built. A dry run's size is never copied forward (§1.8): publishing rebuilds and re-digests.
- **The notes must carry the placeholder or quote the real digest** (§1.8); `--publish` refuses otherwise,
  and that refusal is tested in `tools/test_release.py`.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"
REPOSITORY = "Pummelchen/TinyTitan_Datacenter"
PROJECT = "TinyTitan_Datacenter"
VERSION_TOOL = TOOLS / "version.py"
CHANGELOG = ROOT / "CHANGELOG.md"
EXECUTABLES = ["datacenter-generate", "datacenter-trace", "datacenter-node"]
SHA256_PLACEHOLDER = "SHA256_PENDING"
BYTES_PLACEHOLDER = "ARCHIVE_BYTES_PENDING"

README_BINARIES = """\
TinyTitan Datacenter {version} — prebuilt binaries
==================================================

Platform floor
  macOS 26 or newer, on Apple silicon (M1-M6). The binaries are built natively for arm64;
  there is no x86_64 build and no universal binary. Verify with:  lipo -archs <binary>

What is here
  bin/datacenter-generate   generation and the throughput measurements
  bin/datacenter-trace      one forward, written as a trace
  bin/datacenter-node       one node of a sharded run, as its own process
  VERSION, CHANGELOG.md, LICENSE, THIRD_PARTY_NOTICES.md

  Every tool answers `--version`.

These binaries are NOT code-signed and NOT notarized
  They were built and packaged by the repository's own `tools/release.py`, and macOS will therefore
  quarantine them when they are downloaded. If you have verified the SHA-256 alongside the archive,
  you can clear the quarantine yourself:

      xattr -dr com.apple.quarantine <extracted-directory>

  This is not a Gatekeeper-approved build and nothing here should be read as implying one.

Running them needs a model install
  Generation needs an install built by `tools/quantize.py` from this repository, plus a plan file for a
  sharded run. The install is not part of this archive: weights are never distributed with the engine.

Checksums
  The archive is published with a `.sha256` beside it; the release notes quote the same digest.
  `shasum -a 256 -c <archive>.sha256` verifies it.
"""


def run(command: list[str], cwd: Path = ROOT, capture: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(command, cwd=cwd, capture_output=capture, text=True)


def say(message: str) -> None:
    print(message, flush=True)


class Refused(Exception):
    """A precondition, gate or notes check refused to continue. Never swallowed."""


def version() -> str:
    result = run([sys.executable, str(VERSION_TOOL), "--print"])
    if result.returncode != 0:
        raise Refused(f"VERSION is unusable: {result.stderr.strip()}")
    return result.stdout.strip()


def preconditions() -> list[str]:
    """§1.4 — confirm and record, and stop on anything that would make the measurement a lie."""
    recorded: list[str] = []
    for command in (["sw_vers"], ["swift", "--version"], ["sysctl", "-n", "machdep.cpu.brand_string"]):
        result = run(command)
        recorded.append(f"{' '.join(command)}: " + " / ".join(result.stdout.strip().splitlines()[:2]))

    free_gb = shutil.disk_usage(ROOT).free / 1e9
    recorded.append(f"disk free: {free_gb:.1f} GB")
    if free_gb < 8:
        raise Refused(f"{free_gb:.1f} GB free; a clean scratch build plus the archive needs about 8 GB")

    pressure = run(["memory_pressure", "-Q"])
    recorded.append("memory_pressure -Q: " + pressure.stdout.strip().splitlines()[0])

    # §1.4: no competing build or model process. Never terminate one — name it and stop.
    competing = run(["pgrep", "-fl", "swift-build|swift-frontend|datacenter-generate|datacenter-node"])
    if competing.stdout.strip():
        raise Refused("a competing build or model process is running, and this script never terminates one:\n"
                      + competing.stdout.strip())

    auth = run(["gh", "auth", "status"])
    if auth.returncode != 0 or "Pummelchen" not in auth.stdout + auth.stderr:
        raise Refused("`gh auth status` is not the repository owner's account")
    recorded.append("gh auth: Pummelchen")

    dirty = run(["git", "status", "--porcelain"]).stdout.strip()
    if dirty:
        raise Refused(f"the tree is not clean:\n{dirty}")
    recorded.append("tree: clean")
    return recorded


def gates() -> list[str]:
    """§1.5 — in order, and each of them able to fail."""
    recorded: list[str] = []

    version_check = run([sys.executable, str(VERSION_TOOL), "--check"])
    if version_check.returncode != 0:
        raise Refused(version_check.stdout + version_check.stderr)
    recorded.append("version: " + version_check.stdout.strip())

    # The repo's own gates, the full suite serially, the milestone parity check and — when a recorded run
    # is there to compare against — the baselines. Naming a gate that could not run is `RELEASE.md` §1.8's
    # rule; giving it its input when the input exists is better than naming it.
    command = [sys.executable, str(TOOLS / "run_all_gates.py"), "--milestones"]
    recorded_run = ROOT / ".build/baseline-check/trace/metrics.json"
    if recorded_run.exists():
        command += ["--baselines", str(recorded_run)]
    all_gates = run(command)
    if all_gates.returncode != 0:
        raise Refused("the gate set failed:\n" + all_gates.stdout[-4000:] + all_gates.stderr[-2000:])
    for line in all_gates.stdout.splitlines():
        if line.strip().startswith(("swift tests", "python tests", "milestones", "GATES OK")):
            recorded.append(line.strip())

    # §1.5: guard the plan, not the byproduct.
    plan = run(["swift", "package", "describe", "--type", "json"])
    if plan.returncode != 0:
        raise Refused("`swift package describe` failed:\n" + plan.stderr[-2000:])
    missing = [name for name in EXECUTABLES if f'"name":"{name}"' not in plan.stdout.replace(" ", "")]
    if missing:
        raise Refused(f"the package no longer declares {missing}; the archive would be missing them")
    recorded.append("package plan declares: " + ", ".join(EXECUTABLES))
    return recorded


def scratch_build(scratch: Path) -> list[str]:
    """§1.5.4 — a clean scratch build, with the log scanned for warnings."""
    if scratch.exists():
        shutil.rmtree(scratch)
    say(f"  clean scratch build in {scratch} …")
    result = run(["swift", "build", "-c", "release", "--scratch-path", str(scratch), "--build-tests"])
    log = result.stdout + result.stderr
    warnings = [line for line in log.splitlines() if re.search(r"\bwarning:", line)]
    if result.returncode != 0:
        raise Refused("the clean scratch build failed:\n" + log[-4000:])
    if warnings:
        raise Refused("the clean build emitted warnings, and a release does not ship untriaged output:\n"
                      + "\n".join(warnings[:40]))
    return [f"clean scratch build: ok, 0 warnings ({scratch})"]


def archive_name(version_string: str) -> str:
    """§1.6 — `<project>[-<library>]-<version>-macos-arm64.tar.gz`. A single-library project omits the segment."""
    return f"{PROJECT}-{version_string}-macos-arm64.tar.gz"


def package(scratch: Path, version_string: str, stage: Path) -> tuple[Path, Path]:
    """§1.6 — the archive, with the licence, the notices and the binaries' own README."""
    release_dir = scratch / "release"
    if stage.exists():
        shutil.rmtree(stage)
    root = stage / f"{PROJECT}-{version_string}-macos-arm64"
    (root / "bin").mkdir(parents=True)

    for name in EXECUTABLES:
        source = release_dir / name
        if not source.exists():
            raise Refused(f"{source} is not there; the clean build did not produce {name}")
        shutil.copy2(source, root / "bin" / name)
    # §1.6: a Swift binary without its resource bundle fails at runtime, not at build time — so whatever
    # the build produced is carried, and the count is reported rather than assumed to be zero.
    bundles = sorted(release_dir.glob("*.bundle"))
    for bundle in bundles:
        shutil.copytree(bundle, root / "bin" / bundle.name)

    for name in ["LICENSE", "THIRD_PARTY_NOTICES.md", "CHANGELOG.md", "VERSION"]:
        shutil.copy2(ROOT / name, root / name)
    (root / "README-binaries.txt").write_text(
        README_BINARIES.format(version=version_string), encoding="utf-8"
    )

    archive = stage / archive_name(version_string)
    tar = run(["tar", "-czf", str(archive), "-C", str(stage), root.name])
    if tar.returncode != 0:
        raise Refused("tar failed:\n" + tar.stderr)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (stage / f"{archive.name}.sha256").write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    say(f"  archive: {archive.name}  {archive.stat().st_size} bytes  sha256 {digest[:16]}…")
    say(f"  resource bundles carried: {len(bundles)}")
    return archive, stage / f"{archive.name}.sha256"


def verify_archive(archive: Path, version_string: str) -> list[str]:
    """§1.2.2 — assert the architecture of what is *in the archive*, and smoke-test the shipped tools."""
    recorded: list[str] = []
    with tempfile.TemporaryDirectory() as scratch:
        extract = run(["tar", "-xzf", str(archive), "-C", scratch])
        if extract.returncode != 0:
            raise Refused("the archive could not be extracted:\n" + extract.stderr)
        extracted = Path(scratch) / f"{PROJECT}-{version_string}-macos-arm64"
        for name in EXECUTABLES:
            binary = extracted / "bin" / name
            if not binary.exists():
                raise Refused(f"{name} is not in the archive")
            archs = run(["lipo", "-archs", str(binary)]).stdout.strip()
            if archs != "arm64":
                raise Refused(f"{name} reports architectures {archs!r}; this release is arm64 only")
            answer = run([str(binary), "--version"]).stdout.strip()
            if answer != version_string:
                raise Refused(f"{name} --version says {answer!r}, and VERSION says {version_string!r}")
            recorded.append(f"{name}: {archs}, --version {answer}")
        for name in ["LICENSE", "THIRD_PARTY_NOTICES.md", "README-binaries.txt", "VERSION"]:
            if not (extracted / name).exists():
                raise Refused(f"{name} is missing from the archive (§1.6)")
    return recorded


def notes(archive: Path, version_string: str, destination: Path) -> str:
    """§1.8 — the changelog's section for this version, with the digest of *this* archive substituted."""
    text = CHANGELOG.read_text(encoding="utf-8")
    marker = f"## [{version_string}]"
    if marker not in text:
        raise Refused(f"{CHANGELOG} has no {marker} section")
    section = text[text.index(marker):]
    following = section.find("\n## [", 1)
    if following != -1:
        section = section[:following]

    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    size = archive.stat().st_size
    # §1.8: refuse unless the notes carry the placeholder or quote the real value. Asked exactly that way —
    # "does this text carry the value I just computed" — rather than by pattern-matching a digest-shaped
    # string, because a notes file may quote it in prose as well as in the checksum block, and a wrong
    # digest is refused by the same comparison.
    has_placeholder = SHA256_PLACEHOLDER in section and BYTES_PLACEHOLDER in section
    if not has_placeholder and digest not in section:
        raise Refused(
            "the notes quote no digest this script just computed, and carry no placeholder — refusing to "
            "publish notes that name a checksum nobody can verify"
        )
    substituted = section.replace(SHA256_PLACEHOLDER, digest).replace(BYTES_PLACEHOLDER, str(size))
    header = f"# TinyTitan Datacenter {version_string}\n\nApple silicon (arm64) binaries for macOS 26 or newer.\n"
    destination.write_text(header + substituted, encoding="utf-8")
    return digest


def publish(tag: str, archive: Path, checksum: Path, notes_file: Path) -> None:
    """§1.7 — pinned repository, and §1.4's 'HEAD is the tag' checked rather than hoped."""
    head = run(["git", "rev-parse", "HEAD"]).stdout.strip()
    existing = run(["git", "rev-parse", "-q", "--verify", f"{tag}^{{commit}}"])
    if existing.returncode != 0:
        created = run(["git", "tag", "-a", tag, "-m", f"TinyTitan Datacenter {tag.lstrip('v')}"])
        if created.returncode != 0:
            raise Refused("could not create the tag:\n" + created.stderr)
        say(f"  created tag {tag} at HEAD")
    elif existing.stdout.strip() != head:
        raise Refused(f"{tag} is {existing.stdout.strip()[:12]} and HEAD is {head[:12]}; HEAD must be the tag")
    pushed = run(["git", "push", "origin", tag])
    if pushed.returncode != 0:
        raise Refused("could not push the tag:\n" + pushed.stderr)

    command = [
        "gh", "release", "create", tag, str(archive), str(checksum),
        "--repo", REPOSITORY, "--title", f"TinyTitan Datacenter {tag.lstrip('v')}",
        "--notes-file", str(notes_file), "--latest",
    ]
    result = run(command)
    if result.returncode != 0:
        raise Refused("gh release create failed:\n" + result.stdout + result.stderr)
    say(f"  published {tag}: {result.stdout.strip()}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build, verify and publish a release (dry run by default).")
    parser.add_argument("--publish", action="store_true", help="actually tag, push and upload (§1.2.6)")
    parser.add_argument("--scratch", default=str(ROOT / ".build" / "release-scratch"))
    parser.add_argument("--stage", default=str(ROOT / ".build" / "release-stage"))
    arguments = parser.parse_args()

    version_string = version()
    tag = f"v{version_string}"
    say(f"TinyTitan Datacenter {version_string} — {'PUBLISH' if arguments.publish else 'DRY RUN'}")

    say("\nPRECONDITIONS (§1.4)")
    for line in preconditions():
        say("  " + line)

    say("\nGATES (§1.5)")
    for line in gates():
        say("  " + line)
    for line in scratch_build(Path(arguments.scratch)):
        say("  " + line)

    say("\nPACKAGE (§1.6)")
    archive, checksum = package(Path(arguments.scratch), version_string, Path(arguments.stage))

    say("\nVERIFY (§1.2.2)")
    for line in verify_archive(archive, version_string):
        say("  " + line)

    say("\nNOTES (§1.8)")
    notes_file = Path(arguments.stage) / f"release-notes-{tag}.md"
    digest = notes(archive, version_string, notes_file)
    say(f"  {notes_file.name} quotes {digest[:16]}… and {archive.stat().st_size} bytes")

    if not arguments.publish:
        say("\nDRY RUN: nothing tagged, pushed or uploaded. Re-run with --publish.")
        return 0

    say("\nPUBLISH (§1.7)")
    publish(tag, archive, checksum, notes_file)
    say("\nVERIFY THE RELEASE (§1.9): the notes quote the digest in the .sha256 beside the archive.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Refused as refusal:
        print(f"\nREFUSED: {refusal}", file=sys.stderr)
        sys.exit(1)
