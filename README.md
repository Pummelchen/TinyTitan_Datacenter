<img width="1254" height="1254" alt="TinyTitanDatacenter" src="TinyTitanDatacenter.png" />



# TinyTitan Datacenter

[![Stars](https://img.shields.io/github/stars/Pummelchen/TinyTitan_Datacenter?style=flat-square&logo=github&label=Stars&color=e3b341)](https://github.com/Pummelchen/TinyTitan_Datacenter/stargazers)
[![Views (14d)](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Pummelchen/TinyTitan_Datacenter/main/.github/traffic.json)](https://github.com/Pummelchen/TinyTitan_Datacenter)
[![Last Commit](https://img.shields.io/github/last-commit/Pummelchen/TinyTitan_Datacenter?style=flat-square&logo=git&label=Last%20Commit&color=2ea44f)](https://github.com/Pummelchen/TinyTitan_Datacenter/commits/main)
[![Contact](https://img.shields.io/badge/Contact-0xa0b1%40gmail.com-blue?style=flat-square&logo=gmail&logoColor=white)](mailto:0xa0b1@gmail.com)


A distributed inference engine for large MoE language models on clusters of Macs Minis/Studio's over LAN/SFP/QSFP and Thunderbolt.

**Status: design phase. Nothing here runs yet.**

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

Shard uses **expert parallelism** instead. Every node holds the dense backbone
replicated and a disjoint 1/N slice of the routed experts. All nodes work on the same
token simultaneously and all-reduce the MoE output.

| | Pipeline parallel | Expert parallel |
|---|---|---|
| Nodes busy per token | 1 of N | N of N |
| Aggregate SSD bandwidth | 1x | N x |
| Aggregate expert cache | 1x | N x |
| Sync per token | N-1 hops | 1 all-reduce per MoE layer (~4 KB) |

That's the whole thesis: 4–10x single-user tok/s from topology and I/O layout, not
from more compute.

## Approach

**Faithful ports only.** No architectural changes — no fewer layers, no weight
sharing, no substituted attention. All speed comes from sharding, expert repacking,
and quantization. This keeps a reference implementation to diff against, which is
what makes the project debuggable.

**One IR, thin importers.** A declarative model IR dispatches on tensor *role*, not
tensor name. Importers are pure name-to-role mapping (~500–800 lines per model
family). Transform passes — fusing, repacking, expert reordering, quantization,
sharding — are written once and architecture-agnostic. Quantization and shard policy
are data files shared across models.

**Transcode, don't requantize.** DeepSeek ships FP4 experts / FP8 dense natively;
Qwen ships an FP8 variant. Direct transcoding avoids stacking our error on theirs.

**Bit-reproducibility as a hard invariant.** N-node output must be bit-identical to
1-node output. fp32 accumulation, fixed reduction order. Without this you can't tell
a conversion bug from a scheduling artifact — and on these models you will need to.

**Discrete decisions checked separately from numerics.** Router top-k index sets and
sparse-attention block selections must match the reference *exactly*. Small numeric
drift flips these, after which output diverges completely while every per-tensor MSE
check still looks green.

## Target models

DeepSeek-V4.1 Editions
Qwen 3.8/4.0 Editions

Both frontier families are converging on the same shape — fine-grained MoE, shared
expert, compressed or sparse hybrid attention, constant-size recurrent state, MTP
speculative decoding, and sparse n-gram lookup tables. That shape happens to suit
SSD streaming well, and an IR built around it should absorb the next generation with
importer changes only.

## Milestones

- **M0** — Single node, small dense model, bf16. Gate: bit-matches reference golden
  traces. Ships the trace-capture and diff harness.
- **M1** — Qwen3.6-35B-A3B, single node, 4-bit, SSD-streamed. Gate: correct output,
  recorded tok/s baseline.
- **M2** — 2 nodes, expert-parallel. **Gate: bit-identical to M1.** This is the real
  gate for the project.
- **M3** — 4 nodes. Gate: ≥3x the M1 tok/s.
- **M4** — DeepSeek-V4.1-Flash. Gate: matches reference at 128K context.
- **M5** — Qwen3.8-Flash-Next. Gate: same.

## Non-goals

Training or fine-tuning. Multi-user serving and continuous batching. Architectural
modification of imported models. Being a universal model translator — each new
attention family costs real kernel work, and that's expected.

## Stack

Swift + Metal. Raw sockets over Thunderbolt bridge for the all-reduce. macOS only.


## Documentation

The design, the plan and the live task list are kept in the
[wiki](https://github.com/Pummelchen/TinyTitan_Datacenter/wiki):

| Page | What it holds |
|---|---|
| [Roadmap](https://github.com/Pummelchen/TinyTitan_Datacenter/wiki/Roadmap) | The phases M0–M5 with the gate each one has to pass |
| [Project Tracker](https://github.com/Pummelchen/TinyTitan_Datacenter/wiki/Project-Tracker) | Every task (`DC-nnn`), its status, the risks and the open questions |
| [Architecture](https://github.com/Pummelchen/TinyTitan_Datacenter/wiki/Architecture) | Expert parallelism, the invariants, the IR and the transport |
| [Testbed](https://github.com/Pummelchen/TinyTitan_Datacenter/wiki/Testbed) | The hardware and toolchain the work assumes |
| [Glossary](https://github.com/Pummelchen/TinyTitan_Datacenter/wiki/Glossary) | The terms used on those pages |

A commit that changes the plan or a decision updates the README, the affected wiki page
and the tracker in the same push.

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 André Borchert.

## Contact

Questions, bug reports and suggestions are always welcome. You can contact André Borchert by email at [0xa0b1@gmail.com](mailto:0xa0b1@gmail.com).
