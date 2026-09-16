<img width="1254" height="1254" alt="TinyTitanDatacenter" src="TinyTitanDatacenter.png" />



# TinyTitan Datacenter

[![Stars](https://img.shields.io/github/stars/Pummelchen/TinyTitan_Datacenter?style=flat-square&logo=github&label=Stars&color=e3b341)](https://github.com/Pummelchen/TinyTitan_Datacenter/stargazers)
[![Views (14d)](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Pummelchen/TinyTitan_Datacenter/main/.github/traffic.json)](https://github.com/Pummelchen/TinyTitan_Datacenter)
[![Last Commit](https://img.shields.io/github/last-commit/Pummelchen/TinyTitan_Datacenter?style=flat-square&logo=git&label=Last%20Commit&color=2ea44f)](https://github.com/Pummelchen/TinyTitan_Datacenter/commits/main)
[![Contact](https://img.shields.io/badge/Contact-0xa0b1%40gmail.com-blue?style=flat-square&logo=gmail&logoColor=white)](mailto:0xa0b1@gmail.com)


A distributed inference engine for large MoE language models on clusters of Mac minis and Mac Studios over LAN/SFP/QSFP and Thunderbolt.

**Status: M0 and M1 are complete and their gates have passed.** On the required
toolchain — **Xcode 27 with Swift 6.4, and nothing else** — `swift build` is clean and
`swift test --no-parallel` runs **114 tests, 2 skipped, 0 failures** (the skips are the
Metal kernels, which need a GPU), with **174** standard-library Python tests run in CI.
M0 matched the reference on `Qwen/Qwen3.5-2B`: 40,683,520 bytes of trace data identical
to the contract, every discrete decision matching. M1 runs the real 35 B model on one node
and its trace is **byte-identical to the contract** — 83 tensors and 40 discrete
decisions, digest `b8c976c5e7ba8816…` — generating at **0.108 tok/s** cached with
**348.6 MB** peak memory. Those are current as of 2026-09-16; what M1 leaves open is
`D12` and an install-path contract (`DC-108`). There are **no releases and no tags**.

News, measurements and the live work list are in the **[wiki](https://github.com/Pummelchen/TinyTitan_Datacenter/wiki)** —
start with [News](https://github.com/Pummelchen/TinyTitan_Datacenter/wiki/News) for what
has closed and what it cost, and the
[Project Tracker](https://github.com/Pummelchen/TinyTitan_Datacenter/wiki/Project-Tracker)
for what is still open.

## What it does

Runs models far larger than your total RAM across a cluster of Macs, streaming expert
weights from SSD, optimized for a single user rather than for serving throughput.

Development target is 4x Mac mini M2 (8 GB each) over Thunderbolt. The engine is
scaled to allow an unlimited count of Mac nodes (same model type).

## Why

Single-node SSD streaming already works in the sister project
[TinyTitan](https://github.com/Pummelchen/TinyTitan) and gets ~4-6 tok/s on a
180B-class MoE. The obvious next step — splitting layers across machines — doesn't
help: with one sequence in flight only one node is ever busy, so bytes-read-per-token
is unchanged.

TinyTitan Datacenter uses **expert parallelism** instead. Every node holds the dense backbone
replicated and a disjoint 1/N slice of the routed experts. All nodes work on the same
token simultaneously and all-reduce the MoE output.

| | Pipeline parallel | Expert parallel |
|---|---|---|
| Nodes busy per token | 1 of N | N of N |
| Aggregate SSD bandwidth | 1x | N x |
| Aggregate expert cache | 1x | N x |
| Sync per token | N-1 hops | 1 all-reduce per MoE layer (~4 KB) |

That's the thesis: speedup from topology and I/O layout, not from more compute. It is a
**read fraction**, not a constant — the dense backbone, the router and the all-reduce are
replicated work on every node, so the gain approaches N only while expert reads dominate.

## Approach

- **Faithful ports only.** No architectural changes — no fewer layers, no weight sharing,
  no substituted attention. All speed comes from sharding, expert repacking and
  quantization. A reference implementation to diff against is what makes this debuggable.
- **One IR, thin importers.** A declarative model IR dispatches on tensor *role*, not
  tensor name; importers are pure name-to-role mapping (144–235 lines per family here).
  Transform passes — fusing, repacking, reordering, quantizing, sharding — are written once,
  architecture-agnostic. Quantization and shard policy are data files shared across models.
- **Transcode, don't requantize.** DeepSeek ships FP4 experts / FP8 dense natively and Qwen
  ships an FP8 variant; direct transcoding avoids stacking our error on theirs. (Not yet
  applicable: M1's checkpoint is bf16, so no vendor error is being added on top of today.)
- **Bit-reproducibility is a hard invariant.** N-node output must be bit-identical to 1-node
  output — fp32 accumulation, fixed reduction order. Without that you can't tell a conversion
  bug from a scheduling artifact, and on these models you will need to.
- **Discrete decisions are checked separately from numerics.** Router top-k index sets must
  match the reference *exactly*. Small numeric drift flips them, after which output diverges
  completely while every per-tensor MSE check still looks green.

## Target models

In order: **Qwen3.6-35B-A3B** (the validation model), **DeepSeek-V4-Flash** and
**Qwen3.8-Flash-Next**. M0 used the dense Qwen3.5-2B. The verified configuration of each —
with the checkpoint revision every figure came from — is on the wiki's
[Target models](https://github.com/Pummelchen/TinyTitan_Datacenter/wiki/Target-Models) page.

## Milestones

- **M0** — single node, small dense model, bf16. Gate: bit-matches reference golden traces.
  **Done**; ships the trace-capture and diff harness.
- **M1** — Qwen3.6-35B-A3B, single node, 4-bit, SSD-streamed. Gate: correct output and a
  recorded tok/s baseline. **Gate open** — the figures on record are historical.
- **M2** — 2 nodes, expert-parallel. **Gate: bit-identical to M1.** The project's real gate.
- **M3** — 4 nodes. Gate: ≥3x the M1 tok/s.
- **M4** — DeepSeek-V4.1-Flash. Gate: matches the reference at 128K context.
- **M5** — Qwen3.8-Flash-Next. Gate: same.

## Non-goals

Training or fine-tuning. Multi-user serving and continuous batching. Architectural
modification of imported models. Being a universal model translator — each new
attention family costs real kernel work, and that's expected.

## Stack

Swift + Metal. Raw sockets over the Thunderbolt bridge for the all-reduce. macOS only.
`swift-tools-version:6.4`, Swift 6 language mode, Python 3.14.

## Documentation

| Page | What it holds |
|---|---|
| [News](https://github.com/Pummelchen/TinyTitan_Datacenter/wiki/News) | Everything that has closed, dated, with the measurement it closed on |
| [Roadmap](https://github.com/Pummelchen/TinyTitan_Datacenter/wiki/Roadmap) | The phases M0–M5 with the gate each one has to pass |
| [Project Tracker](https://github.com/Pummelchen/TinyTitan_Datacenter/wiki/Project-Tracker) | Only what is still open: tasks, risks, open questions |
| [Architecture](https://github.com/Pummelchen/TinyTitan_Datacenter/wiki/Architecture) | Expert parallelism, the invariants, the IR and the transport |
| [Target models](https://github.com/Pummelchen/TinyTitan_Datacenter/wiki/Target-Models) | The verified configuration of the three models |
| [Testbed](https://github.com/Pummelchen/TinyTitan_Datacenter/wiki/Testbed) | The hardware and toolchain the work assumes |
| [Glossary](https://github.com/Pummelchen/TinyTitan_Datacenter/wiki/Glossary) | The terms used on those pages |

`docs/` in this repository holds the contracts and the decision records; every milestone
gate is documented there with the command that reproduces it.

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 André Borchert.

## Contact

Questions, bug reports and suggestions are always welcome. You can contact André Borchert by email at [0xa0b1@gmail.com](mailto:0xa0b1@gmail.com).
