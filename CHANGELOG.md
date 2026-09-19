# Changelog

Newest first. This is the record of what each release is; the wiki's
[News](https://github.com/Pummelchen/TinyTitan_Datacenter/wiki/News) page carries the longer
history of what was measured and what went wrong on the way.

## 1.1 — 2026-09-20

**The baseline.** The first release of TinyTitan Datacenter (`ttd`) under its own identity: a
dedicated repository, built from scratch, MIT, with the release line its own.

| | |
| --- | --- |
| Tag | `v1.1` |
| Commit | `e99d9d31` |
| Version | `1.1` — two components; there is no patch component in this project |
| Archive | `TinyTitan_Datacenter-1.1-macos-arm64.tar.gz` + `.sha256` |
| Platform | macOS, Apple Silicon, `arm64` only |

**Measured position.** A streaming MoE runtime, one node at **7.146–7.937 tok/s** across four Mac
minis (mean 7.73, 128 tokens, temperature 0, one binary), and **6.016 tok/s** through a four-stage
layer pipeline across all four — correct, balanced within 9%, and *slower than a single node*. The
step is 141.6 ms/token: 47% expert I/O and 50% GPU wait, with the expert I/O already fully
overlapped (`exposed_io` 0.0 ms).

**Not checked at this release.** Every golden baseline — no install was present under `models/`,
and none may be fetched to change that. The full list is named in
[`docs/release-notes-v1.1.md`](docs/release-notes-v1.1.md).

### Reverting to this baseline

```bash
git fetch --tags origin
git checkout -b work-at-1.1 v1.1      # the tree, exactly as released
```

The wiki is versioned separately: `git clone .../TinyTitan_Datacenter.wiki.git` and check out
`v1.1` there for the documentation as it stood.

## 1.0.0

The previous release, of the engine this project replaced. Preserved as the annotated tag
`retired-datacenter-engine`; restore it with
`git checkout -b datacenter-engine retired-datacenter-engine`.
