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

A distributed inference engine for large MoE language models across a cluster of
Mac minis and Mac Studios over LAN/SFP/QSFP and Thunderbolt: it streams expert
weights from SSD so a model larger than the cluster's total RAM still runs,
optimized for a single interactive user rather than for serving throughput.

**Status: the engine runs and M1's correctness claim holds on the real 35B
checkpoint; throughput is the open finding. Nothing is released** — there are no
releases and no installers. The README still says *"Status: design phase. Nothing
here runs yet."*; that line is stale, so do not repeat it or treat it as current.

## Layout

- `sources/DatacenterEngine/` — the runtime: `Qwen3Forward`, `Qwen3_5Forward`,
  `MixtureOfExperts`, `GatedDeltaNet`, `ModelCache`, `Safetensors` and
  `ShardedSafetensors`, `Install`, `Ops`, `TraceWriter`.
- `sources/DatacenterIR/` — the importer: one importer per model family,
  `TensorRole`, `IRSpec`, `Validation`.
- `sources/DatacenterGenerate/`, `sources/DatacenterTrace/` — the two CLIs.
- `tools/` — the Python reference implementation and every gate: `ordered_*.py`
  (the numeric contracts), `run_m0_gate.py`, `run_m1_gate.py`, `trace_capture.py`,
  `trace_diff.py`, `quantize.py`, the fixture builders and their tests.
- `docs/` — `repository-layout.md`, `trace-format.md`, `ir-schema.md`, the
  `m0-*` / `m1-*` gate and decision records, and `reference-*.md`, which record
  each family's exact dtype boundaries and op order by file and line. Kernel
  comments cite those contracts rather than restating them.
- `tests/` — mirrors `sources/` path for path.

## Build and run

```bash
swift build -c release
swift test

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
`docs/m1-gate.md`, `docs/m0c-quantization.md` carry the commands and what each
asserts. Each gate reads a frozen prompt set (`tools/m0_prompts.json`,
`tools/m1_prompts.json`), writes its report under `.build/`, and exits non-zero on
any failure.

## Working rules

1. **Follow the loop: code, test, audit, document, update the tracker.** A change
   is not finished when it works, but when it is tested, documented, and the
   [Project Tracker](https://github.com/Pummelchen/TinyTitan_Datacenter/wiki/Project-Tracker)
   says so with the evidence that closed it.
2. **Documentation is synced in the same push.** A commit that changes behaviour,
   the plan or a decision updates the README, the affected wiki page and the
   tracker together.
3. **Gates are never quietly weakened.** A gate that cannot be met is renegotiated
   in the tracker, with the measurement that forced it. A silently relaxed gate
   invalidates every claim built on it.
4. **Do not invent numbers.** Every figure in a document is either measured and
   reproducible, or explicitly marked as an estimate or a design intent.
5. **No secrets, ever.** The repository and its wiki are public: no host names,
   addresses, passwords, keys, tokens or model-access credentials — in files,
   commit messages, issues, fixtures or test data.
6. **No large artifacts.** Model installs, weights and similar payloads are never
   committed; `.gitignore` covers the usual paths.
7. **Quality over speed.** Prefer the technically clean solution over the quick
   fix, and say when a task is not worth doing rather than doing it badly.

## Conventions

- **Swift 6.4 on Xcode 27**: `swift-tools-version:6.4`, the Swift 6 language mode,
  and no architectural changes to imported models. The language-feature register —
  which upcoming features are enabled, which deliberately are not, and what each
  costs in diagnostics — is `DC-015`, modelled on the sister project's
  `docs/swift-language-standard.md`.
- **Models**: faithful ports only. A new family costs an importer (a pure
  name-to-role map) plus whatever kernel work its attention genuinely needs.
- **Commits**: imperative subject; the body explains *why*, not *what*. Work lands
  on `main`; a change that needs a gate is not merged before the gate passes.
- **Markdown**: wrapped to a readable width, tables where they carry structure, and
  every link checked by the gate above.

## Traps

- **A trace is only comparable to another captured by the same reference build.**
  The reference implementation is pinned exactly in
  `tools/requirements-reference.txt`; a `transformers` upgrade can change the
  arithmetic under you, which `tools/compare_reference_modules.py` exists to catch.
- **Run `tools/make_contract_vectors.py` whenever an op in
  `tools/ordered_reference.py` changes**, then `swift test` — the Swift contract
  tests assert those regenerated golden bit patterns.
- **A role missing from `tools/quant_policy.json` stops the install** rather than
  defaulting, by design.
- **Do not invent token ids.** Fetch just the tokenizer files of a checkpoint whose
  weights are still downloading rather than guessing them.
- The sister project [TinyTitan](https://github.com/Pummelchen/TinyTitan) holds the
  single-node streaming runtime, the install format and the repacker. Treat it as
  read-only unless a change there is explicitly requested. It is **Apache-2.0**
  while this repository is **MIT**, so check the obligations before reusing any of
  its code here (tracked as `DC-013`).

<!-- release-rules:begin -->
## Releasing

**Read [`RELEASE.md`](RELEASE.md) before cutting a release.** It carries the
generic rules every Pummelchen repository follows, plus this repository's own
section. Do not improvise a release.

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
<!-- release-rules:end -->
