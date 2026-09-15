# TinyTitan Datacenter

A distributed inference engine for large MoE language models on a cluster of Mac
minis and Mac Studios, over LAN/SFP/QSFP and Thunderbolt. This checkout is in
**design phase**: there is no source code yet. The repository holds the README, the
licence, the repository scaffolding, the tooling and the reference contracts under
`docs/`; the design, the plan and the status live in the wiki.

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

- **Swift 6.4 on Xcode 27**, once code lands: `swift-tools-version:6.4`, the Swift 6
  language mode, and no architectural changes to imported models. `tests/` mirrors
  `sources/` path for path. The language-feature register — which upcoming features
  are enabled, which are deliberately not, and what each costs in diagnostics — is
  `DC-015`, modelled on the sister project's `docs/swift-language-standard.md`.
- **Models**: faithful ports only. A new family costs an importer (a pure
  name-to-role map) plus whatever kernel work its attention genuinely needs.
- **Commits**: imperative subject; the body explains *why*, not *what*. Work lands on
  `main`; a change that needs a gate is not merged before the gate passes.
- **Markdown**: wrapped to a readable width, tables where they carry structure, and
  every link checked by the gate below.
- **Python 3.14**, the project's standard and what the farm runs. Everything under
  `tools/` that gates the repository is standard-library-only, so it runs on any
  `python3` with nothing installed. Anything that needs a package — Core ML conversion,
  for instance — lives in the project venv (`.venv`, CPython 3.14) with its versions
  pinned in `tools/requirements-*.txt`. Never install into the system interpreter.

## Commands

```bash
# The CI link gate: local links and #anchors, offline
python3 tools/check_markdown_links.py --verbose

# Tests for the gate itself
python3 -m unittest discover -s tools

# Core ML / Neural Engine tooling. 3.14 has no stable coremltools wheel yet, so the
# pin is an exact pre-release (9.1.dev1); see tools/requirements-coreml.txt.
uv venv --python 3.14 .venv
uv pip install --python .venv/bin/python -r tools/requirements-coreml.txt

# The golden-trace harness (M0): a model-free fixture, then a comparison.
# A diff is only meaningful against a trace captured by the same reference build.
python3 tools/make_synthetic_trace.py .build/ref-trace
python3 tools/trace_diff.py .build/ref-trace .build/ref-trace

# The reference-side capture needs torch, which lives in the venv: run it there.
.venv/bin/python tools/trace_capture.py .build/tiny-trace --tiny

# The controlled-order numeric contract (the bit-exactness target) needs numpy.
.venv/bin/python -c "import sys; sys.path.insert(0,'tools'); import ordered_reference"

# The Gated DeltaNet's chunked delta rule, transcribed from the reference and checked
# against the reference's own function (needs the venv).
.venv/bin/python -m unittest discover -s tools -p 'test_ordered_gdn.py'
.venv/bin/python -m unittest discover -s tools -p 'test_ordered_qwen35.py'
.venv/bin/python -m unittest discover -s tools -p 'test_ordered_qwen35_model.py'
.venv/bin/python -m unittest discover -s tools -p 'test_ordered_moe.py'

# Regenerate the golden bit patterns the Swift contract tests assert
# (tests/DatacenterEngineTests/Fixtures/contract-vectors.json). Run this whenever an op
# in tools/ordered_reference.py changes, then run `swift test`.
.venv/bin/python tools/make_contract_vectors.py

# M0's central claim as one command: run the contract in Python, run the same
# forward in the engine, and compare the two traces byte for byte. It reads the
# checkpoint's model_type and picks the contract, so it covers both families; the
# qwen3_5 path streams both sides, which is how the 2 B model is checked on a node
# that cannot hold it.
.venv/bin/python tools/check_engine_contract.py \
    --snapshot .build/hf-cache/models--Qwen--Qwen3-0.6B/snapshots/<revision> \
    --tokens 1,2,3,4,5,6,7,8 --model Qwen/Qwen3-0.6B --revision <sha>

# The engine's generation, checked token by token against the contract, with the
# generated text decoded so "coherent" is something you can read.
.venv/bin/python tools/check_engine_generation.py \
    --snapshot .build/hf-cache/models--Qwen--Qwen3-0.6B/snapshots/<revision> \
    --prompt "The capital of France is" --max-new-tokens 8 \
    --model Qwen/Qwen3-0.6B --revision <sha>

# Regenerate the Qwen3.5-2B inventory fixture the qwen3_5 importer is tested against
# (names and shapes only; no weights).
python3 tools/make_qwen35_fixture.py \
    .build/hf-cache/models--Qwen--Qwen3.5-2B/snapshots/<revision> \
    tests/DatacenterIRTests/Fixtures/qwen35-2b-tensors.json

# Rebuild the tiny qwen3_5 checkpoint and the contract's golden output that the engine's
# whole-tower test asserts bit for bit (157 KB; needs the venv).
.venv/bin/python tools/make_tiny_qwen35_checkpoint.py

# Run the pinned 2 B model on one node and compare it to the reference implementation.
# The engine streams the weights, so this runs on a node that cannot hold the model.
.build/release/datacenter-trace \
    .build/hf-cache/models--Qwen--Qwen3.5-2B/snapshots/<revision> \
    .build/engine-2b 1,2,3,4,5,6,7,8 --model Qwen/Qwen3.5-2B --revision <sha>
.venv/bin/python tools/trace_capture.py .build/torch-2b \
    --snapshot .build/hf-cache/models--Qwen--Qwen3.5-2B/snapshots/<revision> --dtype f32
.venv/bin/python tools/compare_engine_to_oracle.py .build/engine-2b .build/torch-2b

# THE M0 GATE (DC-026): the frozen prompt set, both halves of the claim, one report.
# Reads tools/m0_prompts.json, writes .build/m0-gate/report.json, exits non-zero on any
# failure. Slower than the other commands because it runs the oracle too.
.venv/bin/python tools/run_m0_gate.py \
    --snapshot .build/hf-cache/models--Qwen--Qwen3.5-2B/snapshots/<revision> \
    --model Qwen/Qwen3.5-2B --revision <sha>

# M0c: build an int4 install from a checkpoint and measure what it costs against the
# frozen prompt set. The policy is data (tools/quant_policy.json); a role missing from
# it stops the install rather than defaulting.
.venv/bin/python tools/quantize.py build \
    .build/hf-cache/models--Qwen--Qwen3.5-2B/snapshots/<revision> .build/m0c/install \
    --spec .build/spec-2b.json
.venv/bin/python tools/measure_quantization.py \
    --snapshot .build/hf-cache/models--Qwen--Qwen3.5-2B/snapshots/<revision> --spec .build/spec-2b.json

# M1's text tower, checked against the reference's own Qwen3_5MoeTextModel: the numbers
# layer by layer, and the router's decisions at every layer as a separate assertion (I3).
.venv/bin/python -m unittest discover -s tools -p 'test_ordered_qwen36.py'

# Does one reference family do the same arithmetic as another, or merely have the same
# names and shapes? Parses both modules, renames one onto the other, and compares every
# top-level entity. Exits non-zero on a difference not listed in --allow-differ, so a
# transformers upgrade cannot quietly change the arithmetic under us.
.venv/bin/python tools/compare_reference_modules.py \
    .venv/lib/python3.14/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py \
    .venv/lib/python3.14/site-packages/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py \
    --rename Qwen3_5Moe=Qwen3_5 --allow-differ Qwen3_5DecoderLayer,Qwen3_5ForCausalLM,Qwen3_5ForConditionalGeneration,Qwen3_5PreTrainedModel,Qwen3_5RMSNorm,Qwen3_5ModelOutputWithPast,Qwen3_5CausalLMOutputWithPast

# M1's validation model: read the inventory from the shard headers and map it.
# No weights are downloaded — the index lists the names and each shard's header the shapes.
.venv/bin/python tools/make_qwen36_fixture.py Qwen/Qwen3.6-35B-A3B

# The reference implementation that produces golden traces. Pinned exactly: a
# trace is only comparable to another captured from the same transformers build.
uv pip install --python .venv/bin/python -r tools/requirements-reference.txt
```

Reference material for the current milestone lives in `docs/`: `m0-decisions.md` records
the resolved `D1`–`D7` decisions and the reasoning behind them, `trace-format.md` is the
gate's data contract, and the two `reference-*.md` contracts record each model family's
exact dtype boundaries and op order, by file and line. Kernel comments cite those
contracts rather than restating them.

Once code exists, the release gate is the sister project's model: a warning-free
release build, the lint gates, the full test suite, and byte-identical golden
baselines — never a subset of those reported as if it were all of them.
