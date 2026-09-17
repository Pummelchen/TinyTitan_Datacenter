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
at **200 tests, 0 skipped, 0 failures** — the Metal kernel tests run on the node's GPU
since `D34`. **M0, M1 and M2 are done and their gates have passed** — M0 on `Qwen/Qwen3.5-2B`
(three frozen prompts, **40,683,520 bytes identical** to the contract, every discrete
decision matching: `docs/m0-gate.md`), and **M1's gate passes in both of its forms** — the **checkpoint**
pair and the **install** pair, which is the restatement — on **all five frozen prompts**, every comparison
`IDENTICAL — 83 tensors, 0 differing elements, 40 discrete decisions`, with digests `b8c976c5e7ba8816…` for
the checkpoint and `b0d382dbabf36df0…` for the install, peaks 3.60-3.79 GB and 0.98-1.41 GB
(`docs/m1-gate.md`, re-verified 2026-09-17). Generation is measured at **0.108 tok/s** cached, **348.6 MB**
peak. The history matters because it is *why* the claims are shaped as they are: a re-check on 2026-09-17
first reported the engine differing by **40 discrete decisions and 1 float**, and **it was never the
arithmetic**. The install holds the Gated DeltaNet's three projections (`linear.in_qkv`, `linear.in_z`,
`linear.out`) and `attn.q/k/v/o` as **int4-affine** by `tools/quant_policy.json`, while that contract had been
generated from the **checkpoint's bf16**: the two sides were reading different weights. Point the contract at
the **install** — `tools/install_source.py`, which reads through the same dequantiser the Swift reader
mirrors — and the pair is byte-identical (`D56`). Against a checkpoint contract the difference is the
**declared, measured** cost of int4 — `D55`: 0.0156 max / 24.5% median relative at layer 0 — tracked in
`DC-112`. M1's claim is restated in `tools/milestones.json` to the form that can be falsified, and
`docs/m1-gate.md` has the evidence. It is nevertheless **incomplete**:
`D12` was an open design question and is now decided from a measurement (`D31`: the expert slot bank
is sized from a budget, one slot, because the measured hit rate is 0 at every size), and
**M2 shards the real model across two machines**: the reduction contract (`D17`), the wire protocol (`D18`), the failure semantics
(`D19`), the shard plan as data (`D20`), bring-up (`D21`) and a transport that binds and connects
(`D22`) are implemented and demonstrated — a two-node exchange driven by a **loaded plan file** is
bit-identical to the single-node forward, over a socket pair **and over TCP**. The engine **runs
across two machines, on the real 35 B model**: a 256-expert plan over two contiguous halves, one node on
a peer and one here, and `trace_diff` reports **83 tensors, 0 differing elements, 40 discrete decisions,
matching digests `b0d382dbabf36df0…`** — the same digest as the single-node M1 baseline. The farm's nodes
are shared with other work, so cluster runs are functional rather than benchmarked until the timing phase;
generation shards as well (`datacenter-generate --plan/--config/--node`), which is the instrument the
tok/s gates will use. **M3 is under way**: all four
machines run one forward in a full mesh and produce the single-node trace exactly (`--mesh`), and
`datacenter-generate` shards, so a two-machine cached generation produced the reference's tokens and
digest. Its ≥3× throughput gate is a deliberate measurement for a quiet farm. **M4–M5 have not
started**.
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
- The sister project [TinyTitan](https://github.com/Pummelchen/TinyTitan) holds a
  single-node streaming runtime and the `GTurbo*V1` container family. Its int4 is **unsigned with a bias**
  while this repository's install container is its own — signed codes, fp32 scales, int8 zero points
  (`D39`) — so a file from one is not readable by the other; the earlier claim here that it "holds the
  install format" was imprecise. Treat it as
  read-only unless a change there is explicitly requested. It is **Apache-2.0**; this
  repository is **MIT**, and `DC-013`'s review found **no code copied from it**, so no `NOTICE` transfers —
  the measurement and the three conditions that would change it are in `THIRD_PARTY_NOTICES.md` and
  `D36`, and `tools/check_provenance.py` guards the position offline.
- `docs/` holds the decision records (`m0-decisions.md`, `m1-decisions.md`,
  `m2-decisions.md`, `repository-decisions.md` for decisions about the repository itself), the gate docs (`m0-gate.md`, `m1-gate.md`,
  `m0c-quantization.md`), the contracts
  (`ir-schema.md`, `trace-format.md`, `wire-protocol.md`, `reference-*.md`), `repository-layout.md`, and
  `invariants-audit.md` — where each of I1–I6 stands, with its evidence and what would falsify it.

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

**Memory has a floor too, and heavy jobs take a lock (`D44`).** `tools/heavy_job.py` refuses a job whose
declared peak exceeds what the machine can use, warns when the machine is already under pressure, and holds
one heavy-job slot so two cannot run at once — the pairing that panicked this node. `quantize.py`,
`run_m1_gate.py` and `run_m3_gate.py` call it before they load anything:

```bash
python3 tools/heavy_job.py --needs-gb 4.2 --purpose "the M1 checkpoint path"   # what would this cost?
python3 tools/heavy_job.py --release                                          # after reading the holder

# And the ceiling on real memory is enforced too (D54), because a declared need is a promise and resident
# size is a measurement: `rss`, three readings in a row, then SIGTERM/SIGKILL and a MEMORY_STOP marker.
python3 tools/memory_watchdog.py --limit-gb 4 --interval 5
python3 tools/memory_watchdog.py --once --limit-gb 4        # a single check, safe to run beside a job
```

**Start the disk watchdog with the job and let only its own marker stop it.** A guard removed by hand during
tidying is the guard that was needed: on 2026-09-17 the watchdogs had been killed while cleaning up, a GPU
run grew swap to 1.8 GB, free disk touched **0 GB** and nothing was watching. It recovered by itself, and
the three-reading rule is what keeps one bad sample from being mistaken for a trend — but the watchdog runs
*with* the heavy job, always (`D58`).

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
  `heavy_job.py`, `run_all_gates.py`, `check_milestones.py`, the fixture builders and their tests.
- `tests/` — mirrors `sources/` path for path.

## Build and run

```bash
python3 tools/check_toolchain.py   # refuses anything but Xcode 27 with Swift 6.4

swift build                  # release: swift build -c release
swift test --no-parallel

# Every gate, in one command, with the test counts read from the runs themselves rather than
# typed in beside them. `--skip-swift` is the standard-library half, which is all a machine
# without Xcode 27 can do (DC-036, D41).
python3 tools/run_all_gates.py

# The CI link gate: local links and #anchors, offline
python3 tools/check_markdown_links.py --verbose

# The documentation's own numbers, against the suites' actual output
python3 tools/check_status_claims.py --swift-tests 200 --swift-skipped 0 --python-tests 391

# The provenance position: no copied code, and no NOTICE to carry
python3 tools/check_provenance.py

# The recorded baselines, against a run's metrics.json. Counts are asserted exactly; seconds and
# memory are reported and only asserted with --assert-observed, which belongs on a quiet farm (D40).
python3 tools/check_baselines.py --metrics .build/baseline-check/trace/metrics.json

# M3's throughput gate. It refuses to run unless the farm is quiet, which is a measurement of the
# nodes rather than an honour system — see D38.
python3 tools/run_m3_gate.py --install .build/m1-install --mesh node4@<addr>,node1@<addr>,node2@<addr>,node3@<addr>

# The functional half of M3, which is a different question from the throughput gate and can be asked at any
# time: the cluster runs, every node is checked against the single-node result, and NO speedup is computed,
# printed or recorded (D68).
python3 tools/run_m3_gate.py --install .build/m1-install --mesh node4@<addr>,node1@<addr>,node2@<addr>,node3@<addr> --functional-only

# Re-check the milestone claims, not just the repository: which digest the engine produces, and
# whether it was ever checked against the contract. A divergence that is not declared fails (D46).
python3 tools/run_all_gates.py --milestones

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
- **A `+ 0.0` normalisation is not a normalisation if the optimiser may fold it.** `D34` needed a zero to
  carry no sign on the CPU, the GPU and the Python reference. The additive idiom (`value + 0.0f`) is correct
  IEEE and the Metal compiler **folded it away**, so nine values of 195 still came back as `-0.0` while the
  test looked like it was checking the rule. Write the rule as a comparison with a bit-pattern result, and
  remember that a "two implementations agree today" test is what finds the second implementation you forgot:
  a fix in `dequantizeInt4` was silently absent from `dequantizeInt4Scalar` for two shapes out of sixty.
- **An instrument narrower than the thing it diagnoses reports success.** `DC-087`'s divergence diagnostic
  covered fewer shapes than the test that failed, so it printed "identical" while the gate stayed red. When
  a diagnostic and a gate disagree, widen the diagnostic to at least the gate's range before believing it.
- **A gate whose configuration is a list will go stale.** `check_status_claims.py` named the three
  decision records explicitly, so the `D36` it was written to check was reported as *cited but undefined*
  the moment a fourth record existed — the gate caught its own stale configuration, which is better than
  not catching it, but the fix was to **discover** `docs/*-decisions.md` rather than list it. Prefer a rule
  that finds its inputs to a list that has to be maintained beside them.
- **Check the form a reader copies, not only the prose.** The claims gate matched test counts in
  sentences and missed the `--swift-tests` flag in the command example beside them — the number most
  likely to be copy-pasted, and stale for a round. (This paragraph first quoted that stale command, and the
  widened gate flagged *it*: a note about a violation is still a violation if it reproduces the literal,
  which is exactly what the provenance fixture learned in the same round.) It now checks the flags too, and found it immediately: a gate that
  reads every *sentence* but no *command* is checking the least actionable form of the claim.
- **A test fixture that simulates a violation contains the violation.** The provenance check flags
  third-party copyright lines, and its own test file held one because the fixture wrote it literally — so
  the fixture now assembles the line at runtime rather than the checker gaining an exemption for the file
  that tests it. A rule with an exemption for its own test is a rule that stops being true quietly.
- **A streaming iterator wrapped in `list()` is not streaming.** Two of my own comparison scripts grew to
  gigabytes on this 8 GB node in one round: one wrapped `install.rows()` — an iterator whose entire point is
  to yield row blocks — in `list()`, and the other asked `dequantize()` for a large tensor, which returns a
  Python `list[float]` at twenty-four bytes per value. The OS killed the first; the second tripped the disk
  watchdog at **3.42 GB free**, the same shape as the panic. `tools/memory_watchdog.py` now enforces a
  **4 GB real-memory** ceiling (three readings, then SIGTERM/SIGKILL, and a `MEMORY_STOP` marker), but an
  ad-hoc `python - <<EOF` has no command line for any watchdog to match, so a script that loads model data
  asserts its own peak (`ru_maxrss`) and aborts above its budget (`D54`). Never `dequantize` a tensor you are
  not going to use whole; `rows()` is the streaming form.
- **No architecture assertion exists anywhere in the repository**, and there is no
  release artifact to assert against — do not invent a `lipo` step.
- **The public status of this repository has swung three times, and only the latest is
  evidenced.** The first revision said there was no source code; the second said the
  engine runs; the third said it is "untested and does not run". The first two were
  wrong, and so is the third as written — on the toolchain the manifest requires, the
  build is clean and **200 Swift tests pass**, while on the `macos-26` CI image (Xcode
  26.x, below the 6.4 floor) the manifest does not even parse. **Any status claim must
  name the toolchain**, because that is the whole difference between "does not build"
  and "builds and passes". Point at a command and its output, never at an adjective.

- **A pipe hides the exit status of the thing you are testing, and it has now happened three times in one
  session.** `swift test | tail` on a failing suite; `datacenter-generate … | tail -3` on a run whose *token
  line* was the thing being verified; and `run_m3_gate.py … | tail` on a gate that had **crashed** — each
  printed a reassuring tail, and each reported the exit status of `tail`, which is 0. The first is recorded
  as a toolchain trap, the second cost a re-run, and the third was caught only because a report file was
  missing. The lesson is not "be careful": it is that a pipe must never stand between a command and the
  question of whether it succeeded. **Write the output to a file, then read `$?`** — or use `PIPESTATUS` —
  and check the exit status of the command itself.

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
