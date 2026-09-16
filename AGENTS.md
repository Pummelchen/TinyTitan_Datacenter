# TinyTitan Datacenter

<!-- agent-harnesses:begin -->
> **One instruction file.** This is it. Codex, DeepSeek Harness, OpenCode,
> Qwen Code, Qoder and Zed read `AGENTS.md` directly, and Claude Code reads it
> through the committed `CLAUDE.md`, which contains nothing but `@AGENTS.md`.
> **Edit only this file** — do not add a second set of instructions anywhere.
>
> Do **not** add `.rules`, `.cursorrules`, `.windsurfrules`, `.clinerules`,
> `.github/copilot-instructions.md` or `AGENT.md`. Zed takes the *first match*
> from that list, **ahead of `AGENTS.md`**, so any one of them silently
> replaces this file for every Zed user.
<!-- agent-harnesses:end -->

A distributed inference engine for large MoE language models on a cluster of Mac
minis and Mac Studios, over LAN/SFP/QSFP and Thunderbolt. The Swift engine under
`sources/` **builds and passes its tests on Swift 6.4 / Xcode 27**, the toolchain
`swift-tools-version:6.4` requires: `swift build` clean, `swift test --no-parallel`
at **165 tests, 2 skipped, 0 failures** (the skips are the Metal kernel tests, which
need a GPU). **M0 and M1 are done and their gates have passed** — M0 on `Qwen/Qwen3.5-2B`
(three frozen prompts, **40,683,520 bytes identical** to the contract, every discrete
decision matching: `docs/m0-gate.md`), M1 on the real 35 B model, whose trace is
**byte-identical to the contract** (83 tensors, 40 discrete decisions, digest
`b8c976c5e7ba8816…`) with generation at **0.108 tok/s** cached and **348.6 MB** peak
memory (`docs/m1-gate.md`, re-established 2026-09-16). It is nevertheless **incomplete**:
no Python contract reads an install (`DC-108`), `D12` is an open design question, and
**M2 has started**: the reduction contract (`D17`), the wire protocol (`D18`), the failure semantics
(`D19`), the shard plan as data (`D20`), bring-up (`D21`) and a transport that binds and connects
(`D22`) are implemented and demonstrated — a two-node exchange driven by a **loaded plan file** is
bit-identical to the single-node forward, over a socket pair **and over TCP**. The engine **runs
sharded end to end across two machines**: `tools/run_m2_gate.py --remote node1@node1` stages a node on a
peer and `trace_diff` reports IDENTICAL on both against the single-node trace — M2's gate as a functional
test on the fixture. The farm's nodes are shared with other work, so cluster runs are functional rather
than benchmarked until the timing phase, and the 35 B model across two nodes needs its 20 GB install
staged on a peer. **M3–M5 have not started**.
There are **no releases and no tags**. The design, the plan and the status live in the
wiki; the measurements live in `docs/`.

## Scope of this checkout

- [README](README.md) — what the project is, for a first-time reader.
- [Wiki](https://github.com/Pummelchen/TinyTitan_Datacenter/wiki) — the working
  documents: [Roadmap](https://github.com/Pummelchen/TinyTitan_Datacenter/wiki/Roadmap)
  (phases and gates),
  [Project Tracker](https://github.com/Pummelchen/TinyTitan_Datacenter/wiki/Project-Tracker)
  (`DC-nnn` tasks, risks, open questions),
  [Architecture](https://github.com/Pummelchen/TinyTitan_Datacenter/wiki/Architecture),
  [Testbed](https://github.com/Pummelchen/TinyTitan_Datacenter/wiki/Testbed),
  [Glossary](https://github.com/Pummelchen/TinyTitan_Datacenter/wiki/Glossary).
- The sister project [TinyTitan](https://github.com/Pummelchen/TinyTitan) holds the
  single-node streaming runtime, the install format and the repacker. Treat it as
  read-only unless a change there is explicitly requested. It is **Apache-2.0**; this
  repository is **MIT**, so check the obligations before reusing any of its code here
  (tracked as `DC-013`).
- `docs/` holds the decision records (`m0-decisions.md`, `m1-decisions.md`,
  `m2-decisions.md`), the gate docs (`m0-gate.md`, `m1-gate.md`,
  `m0c-quantization.md`), the contracts
  (`ir-schema.md`, `trace-format.md`, `reference-*.md`) and `repository-layout.md`.

## This node's hard limit — read before running anything

**Never run GB-scale jobs here.** The machine is an 8 GB Mac mini (about 4.5 GB usable after
macOS), and on 2026-09-16 it **panicked twice** while doing exactly that: the 35 B engine at
~4.5 GB resident plus a 20 GB install build, concurrently. macOS grew swap to *13 swapfiles and
LOW swap space*, the system stopped responding for 90 s, and the hardware watchdog panicked it
(`watchdog timeout: no checkins from watchdogd in 90 seconds`). No bug in the engine caused
either panic; the memory budget did.

The rules that follow from it:

- **Tiny-fixture work only.** `tests/DatacenterEngineTests/Fixtures/tiny-qwen36/` runs the same
  code paths at megabytes instead of gigabytes — importer, mixture, cache, quantisation, gate.
- **One heavy job at a time, never concurrent**, and never a heavy job alongside an engine run.
- **Real-model runs need explicit human approval**, with `sysctl vm.swapusage` checked first.
- **A large page-cached read is a hazard here, not a neutral operation.** Verifying the 20 GB
  install — a read and a hash — took free disk from 17 GB to 2.96 GB in half a minute, because the
  page cache filled memory, memory pressure grew swap, and swap is disk. The disk watchdog stopped
  it. Model payloads now read through `UncachedFile` / `open_uncached` (`F_NOCACHE` + `pread`), and
  a full 20 GB verification peaks at 34 MB and leaves free disk steady — but a read that does *not*
  go through those (an `mmap`, a `read_bytes`, a `Data(contentsOf:)`) still is the old hazard, so
  treat any new multi-gigabyte read as a heavy job until you have checked which one it is.
- The 93 GB under `.build/` is excluded from Spotlight with a `.metadata_never_index` marker, so
  a reboot does not start re-indexing it — that re-indexing is itself sustained I/O on a machine
  that has just panicked.

**A 5 GB disk floor is enforced, not promised.** Two tools, both standard-library only:

```bash
# The monitor. Polls free space; three readings in a row below the floor and it writes
# .build/DISK_STOP and terminates the running heavy jobs (the engine, the contract, the install
# builder) — SIGTERM, then SIGKILL. It never terminates itself or the agent harness.
#
# The three-reading rule is not decoration. The first version acted on one reading of 4.67 GB
# that was 17.55 GB six seconds later, because APFS "purgeable" space appears and disappears as
# the system reclaims caches, and it killed a read-only verification that had done nothing wrong.
python3 tools/disk_watchdog.py --threshold-gb 5 --interval 5 --consecutive 3

# The preflight guard. quantize.py, the contract CLIs and run_m1_gate.py call require_headroom()
# before they touch a model, so nothing heavy starts below the floor either — and a stop marker
# from the watchdog refuses a new run until an operator clears it, because a run that tripped the
# limit must not resume by itself.
python3 tools/check_disk_headroom.py
```

Clearing `.build/DISK_STOP` is a deliberate act: read it, free space, then `rm` it.

## Layout

- `sources/DatacenterEngine/` — the runtime: `Qwen3Forward`, `Qwen3_5Forward`,
  `MixtureOfExperts`, `GatedDeltaNet`, `ModelCache`, `Safetensors` and
  `ShardedSafetensors`, `UncachedFile`, `Install`, `Ops`, `TraceWriter`.
- `sources/DatacenterIR/` — the importer: one importer per model family,
  `TensorRole`, `IRSpec`, `Validation`.
- `sources/DatacenterGenerate/`, `sources/DatacenterTrace/` — the two CLIs.
- `tools/` — the Python reference implementation and every gate: `ordered_*.py`
  (the numeric contracts), `run_m0_gate.py`, `run_m1_gate.py`, `trace_capture.py`,
  `trace_diff.py`, `quantize.py`, `disk_watchdog.py`, `check_disk_headroom.py`,
  the fixture builders and their tests.
- `tests/` — mirrors `sources/` path for path.

## Build and run

```bash
python3 tools/check_toolchain.py   # refuses anything but Xcode 27 with Swift 6.4

swift build                  # release: swift build -c release
swift test --no-parallel

# The CI link gate: local links and #anchors, offline
python3 tools/check_markdown_links.py --verbose

# Tests for the gate itself
python3 -m unittest discover -s tools
```

**The Python side needs a venv, and the interpreter is pinned.** Everything under
`tools/` that *gates* the repository is standard-library-only, so it runs on any
`python3` with nothing installed; anything needing a package (Core ML conversion,
torch, numpy) lives in `.venv` at CPython 3.14 with versions pinned in
`tools/requirements-*.txt`. Never install into the system interpreter.

```bash
uv venv --python 3.14 .venv
uv pip install --python .venv/bin/python -r tools/requirements-coreml.txt
uv pip install --python .venv/bin/python -r tools/requirements-reference.txt
```

The milestone gates are documented rather than restated here — `docs/m0-gate.md`,
`docs/m1-gate.md` and `docs/m0c-quantization.md` carry the commands and what each
asserts. Each gate reads a frozen prompt set (`tools/m0_prompts.json`,
`tools/m1_prompts.json`), writes its report under `.build/`, and exits non-zero on
any failure.

## Working rules

1. **Follow the loop: code, test, audit, document, update the tracker.** A change is
   not finished when it works. It is finished when it is tested, documented, and the
   [Project Tracker](https://github.com/Pummelchen/TinyTitan_Datacenter/wiki/Project-Tracker)
   says so with the evidence that closed it.
2. **Documentation is synced in the same push.** A commit that changes behaviour, the
   plan or a decision updates the README, the affected wiki page and the tracker
   together.
3. **Gates are never quietly weakened.** A gate that cannot be met is renegotiated in
   the tracker, with the measurement that forced it. A silently relaxed gate
   invalidates every claim built on it.
4. **No secrets, ever.** The repository and its wiki are public: no host names,
   addresses, passwords, keys, tokens or model-access credentials — in files, commit
   messages, issues, fixtures or test data.
5. **No large artifacts.** Model installs, weights and similar payloads are never
   committed. `.gitignore` covers the usual paths.
6. **Quality over speed.** Prefer the technically clean solution over the quick fix,
   and say when a task is not worth doing rather than doing it badly.
7. **Do not invent numbers.** Every figure in a document is either measured and
   reproducible, or explicitly marked as an estimate or a design intent. Unmeasured
   claims say so.

## Conventions

- **Swift 6.4 on Xcode 27 — the only supported toolchain, with no exceptions.**
  `swift-tools-version:6.4`, the Swift 6 language mode, **no version conditionals and no
  lowered manifest floor**, so there is nothing to fall back to on an older toolchain;
  `tools/check_toolchain.py` asserts the pairing and CI fails without it. And no
  architectural changes to imported models. `tests/` mirrors `sources/` path for path.
  The language-feature register is `DC-015`.
- The `sources/` and `tests/` directories are lower-case and declared explicitly in
  `Package.swift`: SwiftPM's conventional capitals resolve silently on a
  case-insensitive disk and fail on a case-sensitive one.
- **Models**: faithful ports only. A new family costs an importer (a pure
  name-to-role map) plus whatever kernel work its attention genuinely needs.
- **Commits**: imperative subject; the body explains *why*, not *what*. Work lands on
  `main`; a change that needs a gate is not merged before the gate passes.
- **Markdown**: wrapped to a readable width, tables where they carry structure, and
  every link checked by the link gate.
- **Python 3.14**, the project's standard and what the farm runs. Anything that needs
  a package lives in the pinned `.venv`; never install into the system interpreter.

## Traps

- **Bit-exactness is asserted two ways**: byte-identical trace bytes and exact
  discrete decisions (router top-k index sets). Numeric tolerance is explicitly
  **not** a substitute for the discrete check (I3).
- **A trace is only comparable within one pinned reference build** (`torch`,
  `transformers` in `tools/requirements-reference.txt`), and the model revision must
  be pinned too. `tools/compare_reference_modules.py` exists to catch a `transformers`
  upgrade changing the arithmetic.
- **Run `tools/make_contract_vectors.py` whenever an op in
  `tools/ordered_reference.py` changes**, then `swift test` — the Swift contract tests
  assert those regenerated golden bit patterns.
- **A role missing from `tools/quant_policy.json` stops the install** rather than
  defaulting, by design.
- **Test fixtures are `.copy` resources loaded via `Bundle.module`**; reading them
  from a source-relative path fails in a built test bundle.
- `coremltools==9.1.dev1` is pinned to a pre-release because there is no stable cp314
  wheel.
- `.gitignore` excludes `models/`, `*.gturbo`, `.venv/`, `.wiki/`, `.inspect/`: model
  weights are never committed.
- **The Swift CI job has no skip branch, and that is deliberate.** `macos-26` runner
  images carry Xcode 26.x, below the manifest's 6.4 floor, so the job **fails** there —
  which is the correct signal: the runner cannot meet the project's toolchain. Do not
  "fix" it with a warning-and-skip, a `#if swift(>=…)`, or a lowered manifest floor. The
  step this replaced read `swift --version | tail -1` — the **target** line — so its
  pattern could never match and it took its skip branch on every runner it ever saw;
  `tools/check_toolchain.py` parses the output properly and its tests pin both halves of
  that mistake. Run the gates locally, on Xcode 27.
- **A patch script that aborts leaves the code committed without the documentation.** Twice in one
  session a documentation patch stopped on a failed anchor, the shell carried on past it, and a commit
  went out with code whose message described documents that were never written. The pattern is not the
  mistake — an assert that refuses to guess is right — the mistake is a commit that is not **gated on the
  documentation step**. Write the docs with asserts, check the exit status, and only then commit; and if
  it does happen, land the documentation in its own commit that says so rather than rewriting history.
- **No architecture assertion exists anywhere in the repository**, and there is no
  release artifact to assert against — do not invent a `lipo` step.
- **The public status of this repository has swung three times, and only the latest is
  evidenced.** The first revision said there was no source code; the second said the
  engine runs; the third said it is "untested and does not run". The first two were
  wrong, and so is the third as written — on the toolchain the manifest requires, the
  build is clean and **165 Swift tests pass**, while on the `macos-26` CI image (Xcode
  26.x, below the 6.4 floor) the manifest does not even parse. **Any status claim must
  name the toolchain**, because that is the whole difference between "does not build"
  and "builds and passes". Point at a command and its output, never at an adjective.

## Releasing

**Read [`RELEASE.md`](RELEASE.md) before cutting a release.** It is this repository's
own release standard — edited here, not deployed from anywhere — and it carries both
the general rules and this repository's own section. Do not improvise a release.

The non-negotiables:

- **Apple Silicon only** — build native `arm64` (M1–M6). Never `--arch x86_64`,
  never `ARCHS=arm64 x86_64`, and never `lipo -create`, which is how a universal
  binary gets made.
- **Assert it** — `lipo -archs <binary>` must report exactly `arm64`. A build that
  silently produced a fat binary is a release defect, not a build option.
- **Every release carries the artifacts.** A tag alone is not a release.
- **Identity is single-sourced and enforced** — never bump one declaration of the
  version or build number on its own; the build or CI must fail on a mismatch.
- **Dry run first**; publish only on an explicit flag.
- **Never fetch a model, dataset or dependency to make a gate pass.** A check that
  cannot run is reported *not checked*, and the release notes must name it.
