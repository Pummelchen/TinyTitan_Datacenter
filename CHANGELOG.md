# Changelog

Every release's notes are its section below, and the changelog **is** the announcement
(`RELEASE.md` §1.8) — the README does not carry a release callout, only the facts it states.

The sections are extracted verbatim by `tools/release.py`, which substitutes the checksum block at
packaging time and refuses to publish notes that quote a digest it has not just computed.

## [1.0.0] — 2026-09-17

**Tag:** [`v1.0.0`](https://github.com/Pummelchen/TinyTitan_Datacenter/releases/tag/v1.0.0)

First release. The engine runs **Qwen3.6-35B-A3B** across four Apple-silicon nodes and produces the
single-node result exactly, with a measured decode speed-up of **1.7x** over one node.

### Highlights

- **Distributed 35 B inference that is bit-identical to a single node.** All four machines run one
  forward in a full mesh and reproduce the single-node trace exactly: 83 tensors, 0 differing elements,
  40 discrete decisions, matching digests (`--mesh`; M2's gate).
- **Vocabulary-parallel output head.** The head was 1.05 s/step *identical on every node* — the largest
  piece of replicated work. Each node now computes only its own vocabulary rows, inside the existing block
  decomposition, and the slices are gathered so every node ends with the same full logits array: the
  argmax, the margin and the trace digest are untouched. `head` **1.045 → 0.26 s/step**; cluster **1.13 → 1.36x**
  (`D93`; `ShardedGenerateTests` asserts the gathered logits equal the single node's).
- **The int4 dequantiser runs across the machine's cores.** `load` was ~1 G parameters decoded with SIMD4
  on one core of eight, on every token and every node. Rows are independent, so the row loop is spread;
  `SHARD_DECODE_THREADS=1` restores the single-threaded path so the two are compared on one binary.
  `load` **1.33 → 0.56 s/step**; one node **5.14 → 4.36 s/step** (`D94`; `Int4UnpackTests` compares the
  vector path against `scalar` bit-for-bit over more than fifty shapes).
- **Cluster decode at 1.70–1.74x over a single node**, bit-identical on all four nodes, measured with the
  conditions alternated on one binary. The farm was busy (loads 1.6–5.5) and load hurts the ratio, so these
  are **lower bounds** (`D94`; the M3 gate, recorded as an observation — see *Checks that did not run*).
- **The exchange is measured in its parts.** Encode, send, receive and merge are reported separately and
  cover 100.0% of the exchange: receive is **99.6%** of it, and the wire format and protocol cost 9 ms per
  step. The imbalance hypothesis that followed was then **falsified** by four interleaved runs
  (`D92`; `ShardExchangeTests`).
- **Shard plans can interleave expert ownership.** `--distribution contiguous|round-robin`, with unknown
  values refused rather than defaulted (`D92`; `test_run_m2_gate.py`).
- **A version, single-sourced and enforced.** `VERSION` is the authority, `sources/DatacenterEngine/Version.swift`
  is generated from it, `--version` answers on all three tools, and both `tools/version.py --check` (a gate)
  and `Package.swift` (at configure time) refuse a disagreement (`RELEASE.md` §1.3; 7 tests).

### Requirements

- **Apple silicon only** (M1–M6). Built natively `arm64`; `lipo -archs` reports exactly `arm64` on every
  binary in the archive.
- **macOS 26 or newer** — the package's declared platform floor.
- A model **install** built by this repository's `tools/quantize.py`, plus the plan file for a sharded run.

### Binaries

- `datacenter-generate` — generation and the throughput measurements.
- `datacenter-trace` — one forward, written as a trace.
- `datacenter-node` — one node of a sharded run, as its own process.
- **Not code-signed and not notarized.** See `README-binaries.txt` in the archive for the quarantine
  command.

### Checks that did not run

- **M3's throughput gate was not asserted.** It refuses a busy farm by design (`D38`), and the farm was
  shared throughout; every speed-up figure above is an `observation_only` run with its loads recorded
  beside it. The gate's own roadmap target of **≥3x on a quiet farm** is therefore **not** claimed.
- **The warning scan covers the release products, not the test targets.** A clean release build of the
  products is what this release scans; the test targets are built and run by the gate set in the **debug**
  configuration, where a fresh scratch build passes all 226 tests. A release-configuration test build does
  not resolve `DatacenterIR` on this toolchain, which is recorded as observed and not diagnosed.
- **No GPU matmul path is enabled.** `MetalMatmul` remains opt-in (`SHARD_GPU_MATMUL=1`) because `D63`
  measured it slower than the CPU; the threadgroup-tiled kernel written since has **not** been re-measured,
  so this release claims nothing about it.

### Checksums

```
SHA256_PENDING  TinyTitan_Datacenter-1.0.0-macos-arm64.tar.gz
ARCHIVE_BYTES_PENDING  bytes
```
