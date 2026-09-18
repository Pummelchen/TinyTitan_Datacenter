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
at **259 tests, 0 skipped, 0 failures** — the Metal kernel tests run on the node's GPU
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
digest. Its ≥3× throughput gate is a deliberate measurement for a quiet farm. **A first throughput
observation was taken on a busy farm on 2026-09-17 with the operator's authorisation** — the gate recorded it
as `observation_only` with the loads, so it is **not** a gate result and the threshold was not asserted — and
it measured **0.93×** with bit-identity intact on all four nodes: baseline 5.662 s/step against the slowest
node's 6.086 s/step. The per-node metrics say why, and this is `DC-051`'s budget: a node reads **2.65× less
payload per step** (0.592 GB against 1.570 GB — the dense 0.261 GB is **replicated** and the experts really
are 3.95× fewer) and is nevertheless **slower**, while the exchange is **15.0%** of the step moving 4.49 MB at
**5.0 MB/s effective**, i.e. **1.63 ms per term**, which is latency rather than bandwidth. **The step is not
expert-read-bound, so ≥3× is not what an expert plan delivers on this design** (`D84`, `DC-107`). **The
per-phase breakdown then made that arithmetic** (`D87`): over 40 layers and 17.363 s of a profiled release
forward, `mix.read` is **37.6%**, `attn.core` 23.6%, `head` 13.5% (the LM head's 1.02 GB of weights),
`load` 13.5% — **reads together 64.5%** — and the profiler came out with the same digest as ever. So
**perfect, instantaneous, cost-free sharding of the expert reads is worth `6.521 × 3/4 = 4.891 s`,
17.363 → 12.472 s, or 1.39×`**, before any exchange. The speed target lives in the **replicated** reads
(dense 13.5% plus head 13.5%), the attention core at 23.6%, and the read path itself — not in the expert plan.
**The cached step now has its own breakdown too** (`D88`): profiled on the same install and the same command the
M3 baseline runs, **21.795 s over 4 steps = 5.449 s/step** — `mix.read` **33.0%**, `load` **30.5%**, `head` 19.2%,
`attn.core` 9.6% — and the three big phases are not the same kind of cost. The dense payload is read **once** for
the whole run (1.0437 GB held, 4888 cache hits), so `load` is **not I/O**: it is the same constants dequantised
and released on every step, 160 times per generation, while the GPU sits idle. `head` is cached weights and a
matmul. Only `mix.read` is device-bound — about 0.33 GB/step in 1.80 s, **184 MB/s** — and it is the only one a
plan divides, so perfect, free sharding is worth **1.09×** on the measured step. **The decoded-weight budget was
built and measured (`D89`), and on this node it loses.** The cache is bit-identical and does what it was aimed at
— about **0.035 s per layer per step**, which across all 40 layers is **1.4 s of a 5.42 s step** — but with the
budgets alternated on one binary, 1 GB and 2 GB made the step *worse* (5.33 → 5.46 and 5.60 s/step), because the
resident fp32 arrays cost more elsewhere than they saved (`head` +0.20 s/step, `attn.core` +0.27) on a machine
already swapping. So its **default budget is 0** and `SHARD_LAYER_CACHE_MB` is the knob for a node with headroom.
**The cluster then gave the same answer about itself** (`D90`): with the gate now forwarding `SHARD_PROFILE` to
every node, a four-node run shows **the plan working** — reads fall from 6.281 GB to 2.31-2.40 GB per node,
`mix.read` 1.80 → ~0.49 s/step, `mix.gateup` 0.25 → 0.06 — and **the exchange eating it**: 4.250, 4.136, 0.672 and
3.739 s/step, that is **71.5%, 69.5%, 11.2% and 63.1%** of the step. Four nodes doing identical work, one waiting
0.67 s and three waiting ~4 s: the cause is in `allReduce`, which sends to every peer and then receives
**sequentially, in peer order, blocking** — so a slow peer delays everyone queued behind it, forty times per
token. **That, not the expert plan, is the 1.5×**: receive concurrently and skip peers that own none of the
chosen experts, and the exchange returns to the shape of node 2's 0.67 s. The head shard (`head` 19.2%,
identical on every node) is the margin on top. **The exchange was then split into its parts and the answer was
unambiguous** (`D92`): on a four-node run the segments are **encode 0.002 s, send 0.007 s, receive 1.11-2.31 s,
merge 0.002 s** — receive is **99.6%** of the exchange and the segments cover **100.0%** of the total. The
cluster is not waiting on the *protocol*, it is waiting for its peers to **have** something to send: 90 reduces
per step at 13-26 ms each is compute imbalance, not overhead. With the farm quieter (loads 1.8-4.5 rather than
2.4-9.3) the same run measured **1.13×** — the engine's own cluster step is 4.790 s/step against a 5.389 s/step
baseline — so the earlier 0.93× was largely the farm. The target needs 4.79 → 3.59 s/step, and the two levers
are the exchange (1.1-2.3 s/step of waiting) and the head shard (−0.79 s/step). **The waiting was then tested
with a plan change and turned out to be the farm** (`D92`): four interleaved runs of `contiguous` against
`round-robin` (which interleaves expert ownership) gave **1.06, 0.96, 0.86 and 0.88×** — indistinguishable —
with a **24%** run-to-run spread, and **the run with the slowest cluster step is the one whose peer sat at load
18.7**. On this farm the cluster step measures the farm: the per-id skew hypothesis is falsified, the receive
wait is a busy peer, and no distribution fixes it. So a 1.5× figure cannot be *certified* without a quiet
window — which is what the gate's `--quiet-load 1.0` rule exists for (`D38`). What is in our hands is to make
the **replicated** work smaller, since the ratio is `(replicated + experts) / (replicated + experts/n +
exchange)` and the head's 1.05 s/step is identical on every node, and to **overlap the wait** with work that is
not on the critical path. **The head is done and it worked** (`D93`): it is now **vocabulary-parallel** — each
node computes its own rows with the block decomposition unchanged and the slices are gathered, so every node
ends with the identical full logits array and the trace digest is untouched — and the cluster went from **1.13×
to 1.36×** with bit-identity intact on all four nodes: step **4.790 → 4.021 s**, `head` **1.045 → 0.26-0.32 s**,
and all four nodes now identical to the millisecond. The gap to 1.5× is **0.36 s/step**. One instrument question
is recorded rather than used: the ledger's `exchange_seconds` (1.4-1.9 s/step) is larger than the `ff` phase
that contains it (0.575 s) even though the marks are right and the phases sum to the step, and `D90`/`D92` both
reasoned from that counter. **And the target is met** (`D94`): `load` was ~1 G parameters of int4 constants
dequantised with SIMD4 **on one core of eight**, the same constants on every token and every node, so the row
loop of `InstallFile.dequantizeInt4` — whose rows are independent — is now spread across the cores
(`SHARD_DECODE_THREADS=1` restores the single-threaded path, so the two are compared on one binary). Bit-exactness
is guarded by `Int4UnpackTests`, which compares the vector path against `dequantizeInt4Scalar` over more than
fifty shapes. Single node, alternated: `load` **1.33 → 0.56 s/step** (2.4x) and the step **5.14 → 4.36 s**. Cluster,
four alternated runs, every one bit-identical on all four nodes, loads 1.6-5.5: **1.74x and 1.70x** with eight
threads, and **1.66x with one** — so **the 35 B runs at 1.70-1.74x over a single node**, and at 1.66x even with the
thread work disabled. The farm was **busy** throughout and load hurts the ratio (four nodes are exposed to a spike
and the baseline is one), so these are lower bounds. **The gate's own roadmap target of ≥3x on a quiet farm
(`DC-053`) is not asserted by any of this**: every figure is an `observation_only` run with its loads beside it,
and the gate remains the certification instrument. `DC-113`'s GPU GEMV is retired from the path to this number —
the CPU path took the same phase down without a device, a protocol, or `D91`'s silent-CPU-fallback risk — and
remains a candidate for absolute speed. **M4–M5 have not
started**.
**The operator re-scoped the work on 2026-09-17, and it overrides the order above: single-node throughput
first, and *no network or cluster measurement until this engine reaches **7 tok/s** decode on one Mac mini M2*
(`D97`, `DC-117`).** The engine is at **0.230 tok/s** on that machine, so the gap is ~30x, and the reference is
the sister project's own measurement of a 35 B-A3B at 4-bit on a comparable 8 GB M-series node: **5.164 tok/s
at a 1 GB expert cache, 6.019 at 2 GB, 7.075 at 3 GB, and 2.756 at 4 GB** — the last one collapsing because a
4 GB wired cache, the dense weights, the KV and the prompt cache no longer fit in 8 GiB, so the machine swaps
(host wait per token 55 ms → 219 ms, GPU occupancy 42.8% → 17.4%, and the spread 3.167 → 2.756 → 2.429
degrading run over run). Its verdict is that the workload is **bandwidth-bound**: the lever is expert I/O, not
prefill, and its launcher's "at most 30% of physical RAM" warning is calibrated by that data. **v1.0.0 is released** (`RELEASE.md` §1.3 gave it an identity: `VERSION` is the authority,
`sources/DatacenterEngine/Version.swift` is generated from it, every tool answers `--version`, and both
`tools/version.py --check` and `Package.swift` refuse a disagreement). `tools/release.py` builds the
archive — **dry run by default**, gates first, a clean scratch build scanned for warnings, `lipo -archs`
asserted on the binaries *inside* the archive, one checksum beside it — and publishes only with
`--publish`. The design, the plan and the status live in the wiki; the measurements live in `docs/`.
**The GPU side of the int4 path was then built and measured (`D108`): the fused kernel is real, and the unpack
was never the bottleneck.** `MetalInt4Matmul` dequantises inside the matmul, so the fp32 slab the engine
materialises — about **6.5 GB of `Float` per token** — never exists, and it is **bit-identical** to
`InstallFile.dequantizeInt4` followed by `Ops.orderedMatmul` over the `D107` grid. In release on the real shapes
it is **1.23x / 1.45x / 2.36x** faster than unpack-then-matmul, which across a token's 320 slices of each
projection is **0.114 s of a 1.74 s step — about 6.6%**, the same order as the 7% `D107` bounded from the CPU.
So the arithmetic is not the route to 7 tok/s: `mix.read` is the **disk read**, and the route is **residency**.
The grid also found a boundary that had been assumed away — **an Apple GPU flushes a denormal product to zero
where the CPU keeps it**, for this kernel and for the `MetalMatmul` already in the tree, and no math mode changes
it — now pinned by a named test rather than left implicit.

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
  read-only unless a change there is explicitly requested. **Reading it for approach is authorised, and so is *taking code from it***
  (the operator said both, on 2026-09-17 and 2026-09-18; `docs/reference-tinytitan-decode.md` is the study that
  came out of the first). Taking code is **not free**: it is Apache-2.0 and this repository is MIT, so anything
  taken must carry its attribution — a root `NOTICE` with their notice and the `turbo-fieldfare` dependency it
  names, the licence text, and a mark on every file that was modified. `D100` records the change and
  `THIRD_PARTY_NOTICES.md` says what is required; `tools/check_provenance.py` is where the requirement is
  enforced, and it is updated in the same commit as the first code taken rather than after it. It is **Apache-2.0**; this
  repository is **MIT**. `DC-013`'s review found **no code copied from it** *at the time it was made*, and
  `D36` wrote down the three conditions that would change that; the operator's permission of 2026-09-18 is the
  first of them, so the `NOTICE` obligation is now **live** rather than hypothetical whenever a file is taken.
  `THIRD_PARTY_NOTICES.md` records what is required and `tools/check_provenance.py` enforces it — it still
  guards the position offline, and it changes from forbidding attribution to requiring it in the same commit
  as the first code taken (`D100`).
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
python3 tools/check_status_claims.py --swift-tests 259 --swift-skipped 0 --python-tests 427

# The provenance position: no copied code, and no NOTICE to carry
python3 tools/check_provenance.py

# Every command the documentation gives a reader, against the flags its tool actually accepts. A tool that
# cannot be asked is reported NOT CHECKED rather than passed.
python3 tools/check_documented_commands.py

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
  that mistake. Run the gates locally, on Xcode 27. The job now runs the **whole gate set** rather than
  `swift build` and `swift test` by hand, because it is the only job that can see both halves at once and the
  documented test counts are a claim about them: the other workflow skips the Swift half and reports it *not
  checked*, so before that change the claimed count was compared to a real run nowhere in CI, and a deleted test
  would have left every document correct about a suite that no longer existed (`D82`).
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
- **The architecture is asserted on the artifact, not in CI.** There is still no CI `lipo` step and none
  should be invented from nothing; what exists is `tools/release.py`, which checks `lipo -archs` on every
  binary **extracted from the packaged archive** and refuses anything but exactly `arm64` (`D95`). A check
  that runs against the build directory instead would assert the wrong thing — the archive is what ships.
- **The public status of this repository has swung three times, and only the latest is
  evidenced.** The first revision said there was no source code; the second said the
  engine runs; the third said it is "untested and does not run". The first two were
  wrong, and so is the third as written — on the toolchain the manifest requires, the
  build is clean and **259 Swift tests pass**, while on the `macos-26` CI image (Xcode
  26.x, below the 6.4 floor) the manifest does not even parse. **Any status claim must
  name the toolchain**, because that is the whole difference between "does not build"
  and "builds and passes". Point at a command and its output, never at an adjective.

- **A path spelled in two places will drift, and the copy that breaks is the one a reader uses.** Both cluster
  gates staged the install into the peer's directory **under its own name** (`m1-install`) and then launched the
  peer with the literal `./install`. `--remote-install` worked and every earlier cluster run had passed it, so
  the broken default was never exercised — until the documented command in the gate's own docstring was run on
  the real farm, and every peer died on a missing `install/install.json` after 21.7 GB had been copied to each.
  The fix is one function that decides the name, used by the stager and the launcher alike; the test that
  matters is the one that reads **both** files and refuses the literal anywhere, because the defect was the two
  halves disagreeing and a test of either half alone passes (`D83`).
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
