# Handover: TinyTitan Datacenter (`ttd`), after release 1.1

**Paste this into the next session:**

> Continue the TinyTitan Datacenter work in this checkout — `~/Downloads/TinyTitan Datacenter`,
> repository `Pummelchen/TinyTitan_Datacenter`. Read `AGENTS.md` first, then this file, then the
> wiki (a **separate git repository**, `.wiki/`, itself tagged `v1.1`). **The project is
> TinyTitan Datacenter, short name `ttd`** — a dedicated repository, built from scratch, MIT, with
> no upstream and no fork relationship. **`README.md` is human-owned: do not edit it, not even its
> badges** (`AGENTS.md` has the rule and the reason it exists). The next release is **1.2**;
> versions are `x.x`, never `x.x.x`. **1.1 is published** — tag `v1.1` → `315540df`,
> `TinyTitan_Datacenter-1.1-macos-arm64.tar.gz`. **Verification uses only what is installed under
> `models/`**; this checkout has no install there, so every golden baseline is *not checked* and
> that is the correct answer — never fetch, convert or repack a model to make a check pass. Report
> measurements, not assurances.

This is the only current brief. It supersedes the handover that preceded it, which described this
codebase under its old identity and name; its traps that still bite are folded in below.

## Where the work stands

| Piece | State |
| --- | --- |
| Repository | `Pummelchen/TinyTitan_Datacenter` — **not** a GitHub fork (`fork: false`, `parent: null`) |
| Checkout | `~/Downloads/TinyTitan Datacenter` |
| `main` | `3073c2f0`; this handover lands on top of it |
| Release | **1.1 published** — tag `v1.1` → `315540df`; `TinyTitan_Datacenter-1.1-macos-arm64.tar.gz` + `.sha256`, 25,830,004 bytes, sha256 `eb154f05…` |
| Wiki | separate repo `TinyTitan_Datacenter.wiki.git`, `.wiki/` in the checkout, **tagged `v1.1`** at `8f0dc2f`; entries after that are post-release |
| Licence | **MIT** (`LICENSE`, Copyright (c) 2026 André Borchert). Third-party notices live in `THIRD_PARTY_NOTICES.md` and are an obligation, not a link — keep them |
| Version identity | root **`VERSION`** = `1.1`, mirrored by `CFBundleShortVersionString` in `tools/install_tinytitan.sh`; `tools/release.sh` **refuses** when the tag, `VERSION` and the plist disagree |
| Models here | none under `models/`. Each node has `~/Downloads/qwen36-4bit.gturbo` (19 GB; 40 layers; expert stride 1,769,472 B) |
| Goldens | **all NOT CHECKED** — named as such in `docs/release-notes-v1.1.md` |
| Farm | four 8 GB M2 minis: node1 `192.168.18.27`, node2 `192.168.18.25`, node3 `192.168.18.29`, node4 `192.168.18.26` |

## The current numbers

Measured 2026-09-20, one binary (`md5 f0f7c5f2` on all four nodes), 128 tokens, temperature 0:

| Configuration | tok/s |
| --- | --- |
| node1 / node2 / node3 / node4 alone | 7.971 / 7.949 / 7.902 / 7.590 — **mean 7.853** |
| **four-stage layer chain** | **6.143** (runs: 6.206, 6.091, 6.133) |
| node1 alone, page-cache reader, 40 slots | 8.552 |
| server, no cache flag (marginal decode) | 9.27 |

**The chain is ~22% slower than one node, and that is structural.** The layer pipeline is serial,
so stage 1's period (**161 ms/token**) is the *sum* of the stage times, not the slowest of them.
Run it with `benchmark/run_four_stage_chain.sh`; stage 4's output is the model's, and stage 1
printing garbage is correct — a ten-layer embedder cannot produce tokens.

## What the 1.1 session landed

1. **The expert-cache budget is now a third of physical memory** (`0536724d`). It was a half, which
   on an 8 GB mini licensed 64 slots of a 70.8 MB slot and **paged**; the server went **2.26 →
   9.27 tok/s** and the CLI's default path improved with it. The justification is arithmetic rather
   than taste: 8 GiB — the old constant — is exactly a third of the 24 GiB machine those budgets
   were tuned on, so a third reproduces the tuned value where it was tuned and scales down where a
   constant could not. Tests: `tests/TinyTitan/Runtime/Configuration/ExpertCacheBudgetTests.swift`.
2. **The server can now load every model this project installs** (`3dfc51e4`, tests `23c7cf39`). It
   demanded a sidecar `chat_template.jinja` and threw `missingToolTemplate` without one, while the
   tokenizer's own `tokenizer_config.json` already carried a 7,764-character template — and it
   hashed that missing file into the prompt cache's identity, so the config form had no identity
   either.
3. **`benchmark/run_four_stage_chain.sh`** — the launch helper that never existed. Its absence is
   why the published cluster figure could not be reproduced; it now carries the exact commands,
   the addresses, and the measured results.
4. **Three published claims corrected**: "the expert I/O is already fully overlapped" (false — the
   phases are serial) withdrawn from `docs/release-notes-v1.1.md` **and the live v1.1 release
   body**; the page-cache inversion that would have cost 21% removed (`670be5a3`); the cluster
   figure's unverifiability fixed by the script above.

## What is open

1. **The largest unexplored term is the GPU phase: 71.3 ms of the 141.6 ms step.** The I/O half is
   now understood; the GPU half was never touched. The bounding arithmetic matters: GPU + encode is
   **75.1 ms = 13.3 tok/s**, so ~13 tok/s needs expert I/O to become *entirely* free — a 97%
   elimination that is not available on this hardware. Treat any target above ~13 tok/s on one node
   as requiring a different model or machine, not tuning.
2. **The page-cache reader is 8% faster and deliberately not enabled.** Serial `pread` gives
   **8.334** against the bounded default's **7.717**, re-verified after the budget fix with no
   overlap between the groups. `docs/v4-core-design.md:318` records that as *"20% for a footprint
   that is actually bounded"* — a design decision, so changing the default is a product call, not a
   fix. Opt in per-run with `TINYTITAN_BOUNDED_IO=0`.
3. **The ~22% cluster penalty** needs a different decomposition — tensor or expert parallelism. The
   wire is the constraint: **765 µs** round trip on 1 GbE, and the Thunderbolt/10GbE ports read
   `status: inactive`. A layer pipeline divides memory, not time.
4. **The wiki's plan pages still describe the retired plan** (`Roadmap.md`, the milestone pages,
   most of `Project-Tracker.md`, the `Architecture.md` sharding section). Flagged in `News.md`,
   never done.
5. **`main` is well past the `v1.1` tag.** The tag is intact at the release commit, so nothing
   needs doing — but a reader expecting `main` to *be* 1.1 will be surprised.
6. Carried forward unchanged from the previous brief: the `thread-sanitizer` intermittent report at
   `HTTPServerSupport.swift:106`; publishing `plugins/dsh-tinytitan` (submitted, awaiting the
   catalogue maintainer); `TINYTITAN_KEEP_WIRED` being unable to *un*wire the cache; and the
   hardware blockers in tracker section 3 (validation on M1/M2/M4/M5/M6, ANE across generations,
   long-context parity past the exactness window).

## Traps worth carrying forward

- **Cross-node connects must use an IP, never a hostname.** With a hostname `connect()` fails with
  `NSPOSIXErrorDomain Code=22 "Invalid argument"` and the stage retries 450 times before giving up.
  This cost three attempts at the four-stage chain.
- **node4 cannot originate outbound connections** (macOS Local Network Privacy), so it must only
  ever listen; `TINYTITAN_STAGE_BACK_SWAP=1` on the *dialling* neighbours is what makes that work.
- **node4 accumulates swap across consecutive engine runs.** One engine at a time per machine; one
  app, CLI or model-using test at a time anywhere (`AGENTS.md`).
- **Stage roles, as validated:** node4 listens (`STAGE_LISTEN=47721`, `BACK_LISTEN=47712`,
  `BACK_ROLE=sink`); middles are `BACK_ROLE=both` with `BACK_SWAP=1`; the embedder is `source`. The
  engine prints every connect it makes (`[back]`, `[wire]`) — **read the logs on the nodes before
  theorising**, they name the broken hop directly.
- **`totalExposedIoNanos` / `exposed_io` is misleading.** It read `0.0 ms` while 64 ms/token of
  await sat on the critical path. Trust deltas: removing 42.5 ms of await removed 42.8 ms of step
  (ratio 1.007).
- **The default cache size is derived, not fixed.** If anything touches
  `affordableExpertCacheBudget` or `defaultExpertCacheBudgetBytes`, re-measure the *default* path on
  an 8 GB machine specifically. Both front ends were silently slowed by the old value, and every
  tuned figure had been taken with the flag passed by hand — which is exactly why nobody noticed.
- **A release tag that is not yet published may be force-moved**, but a published one is a
  different matter: the sequence is commit prep → tag → dry run → fill in `### Verification` →
  commit → `git tag -f` → `git push --force origin vX.Y` → `--publish`. `--publish` re-runs every
  gate and rebuilds the archive, so the published digest and size are never the dry run's.
- **`release.sh` needs `HEAD` to be the tag** and a release build at `.build/release/TinyTitanCLI`
  to exist before it starts. Its golden phase refuses to run beside any model process, and it
  reports a *refused* start as a "mismatch".
- **The repository carries two release lineages.** Tags `v2.0`–`v5.7` belong to the shipped runtime;
  `ttd`'s own line is `v1.0.0`, `v1.1`. Do not treat the old numbers as this project's history.
- **The wiki is a second repository with its own history** — commit and push `.wiki/` separately.
- **Pushing needs `git -c credential.helper=store`**; `~/.git-credentials` is valid. `gh` is
  authenticated from `~/.config/ttd/git-access-token` (and `~/.config/gh/hosts.yml`), so
  `gh release create` / `edit` work without hand-passing a token.
- **A `file:` plugin install is a copy**: after editing `plugins/dsh-tinytitan/`, re-install it or
  the harness keeps running the copy.
- **Verification is what is installed.** A stored baseline is never deleted because its model is
  currently absent, and an uninstallable check is reported *not checked* — including in release
  notes.
- **Say what a guard actually reads, not what it intends**, and **test a claim rather than trusting
  it**. Every defect that reached a release in this project was a claim broader or narrower than the
  code — the two fixed in this session included.
