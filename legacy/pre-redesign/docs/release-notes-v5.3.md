## NVMAI 5.3 — KAT-Coder-V2.5-Dev, and the conversion bug that hid inside it

This release adds a model and fixes the four defects that adding it exposed. One
of those defects had been producing installs that loaded, passed every byte-level
check, and answered confidently from the wrong weights.

### KAT-Coder-V2.5-Dev 35B-A3B

Kwaipilot's agentic-coding fine-tune of Qwen 3.6 35B-A3B joins the family at
4-bit and 8-bit (`tools/install_models.sh katcoder`, `katcoder-8bit`, or
`katcoder both`). Same geometry as Qwen 3.6 and its own sampling — temperature
1.0, from the checkpoint's `generation_config.json`, not the series' 0.6.

Verified on the real install at both widths: the three continuations this
project uses behave (`The capital of France is` → ` Paris`, `Once upon a` →
` time`, `The quick brown fox jumps over the lazy` → ` dog`), each receipt
verifies, and each width has a golden baseline that re-checks byte-identical.
The baselines' 96-token answer is coherent technical prose about mutexes.

Measured on this base M3 with 24 GB through `benchmark/nvmai_maxthroughput.py`,
512-token greedy generations: **17.86 tok/s** at 4-bit and **6.91** at 8-bit. The
8-bit build streams 36.9 GB of routed experts from SSD, which makes it the most
expert-locality-sensitive install of the 35B family; its `count` prompt is the
worst case at both widths (11.34–17.86 tok/s at 4-bit, 1.00–6.91 at 8-bit).

### The expert axis was ordered by arrival, not by index

KAT's checkpoint is the first this project has converted whose routed experts
ship **one tensor per expert** rather than fused. The converter stacked them by
appending, which is only correct when they arrive in ascending index order. They
do not: the checkpoint index is lexicographically sorted, so layer 0's experts
arrive `0, 1, 10, 100, … 109, 11, 110, …` — 48 non-consecutive steps — and 12 of
the 40 layers also split their experts across two shards. The fused axis was
ordered by arrival, so the runtime routed to expert *k* and read expert *j*'s
weights.

**Nothing in the pipeline could see it.** Every expert's bytes matched the
checkpoint exactly; the shapes, the manifest and the `packed_experts` layout
were right; `validateRoleUniformity` passed; the receipt verified; and
`gturbo_diff_snapshot` reported all 613 resident tensors byte-identical. The
model answered fluently and *partly* correctly — "the capital of France"
appeared, then collapsed into repetition — at both widths, because the logits
came from the wrong experts.

It was found by converting Qwen 3.6 through the same converter as a control:
same geometry, but its source ships experts already fused, and it answered
correctly. That isolated the fault to the per-expert path, and comparing **expert
255** rather than 0 or 1 exposed the ordering — two earlier readings had called
the fusion correct because they spot-checked the first two experts, which happen
to arrive first.

The accumulator now preallocates the expert axis and files each expert at its own
index, rejecting a duplicate instead of overwriting. `tools/lint.sh converter`
feeds experts in shuffled order and asserts each lands at its index; reverting to
append reproduces `[3,0,7,1,5,2,6,4]` and fails it. It lives in the lint gate
rather than the Swift suite because no Swift test can observe a Python converter
bug.

### Three size caps that refused legitimate files

Found in the same run, each a literal chosen when an unbounded read was made
bounded, and each below what a real checkpoint produces:

- **The snapshot index** was capped at 4 MiB and KAT's is 9.7 MB, because the
  bound scales with tensors × key length and the converter's renames roughly
  double key length. The same literal was copied into the remote loader, so
  neither install path could build the model.
- **The resident index** was capped by the per-worker *staging* budget (1 MB)
  rather than the format's own ceiling, refusing an index of about 28 MB.
- **The runtime's manifest cap** was 4 MiB while `--verify-install` accepted the
  same 6.25 MB file against its own 64 MiB cap — so the install verified and
  then refused to load. A cross-module test now asserts the two ceilings agree.

### The runtime now streams KAT's experts from SSD

Before the fusion fix, KAT's routed experts were classified as *resident*
weights: the install declared `expertsPerLayer: 0`, carried no packed expert
files, and would have held all 256 experts per layer in RAM instead of
streaming them. The install built by this release declares **256** experts,
`expertStride 1769472`, **41 packed expert files**, and 1.8 GB of resident
weights with 17 GB streamed.

### Also in this release

- **`tools/install_models.sh <model> both`** installs 4-bit and 8-bit from one
  download for every Qwen3.5-MoE checkpoint (Ornith 1.5, Qwen 3.6,
  Qwen-AgentWorld, KAT). One width alone already converted both and kept the
  other snapshot; the install path now reuses it in either spelling, so a
  second width never re-fetches the checkpoint.
- **A downloader that survives a truncating link.** Shards are fetched as
  length-checked 64 MiB ranges, three at a time, with `--http1.1` (this host
  resets HTTP/2 streams continuously), a stall floor that aborts a dead
  connection, and no resume that could append to a truncated prefix. A
  truncation costs one chunk instead of 5 GB.
- **The tool scripts resolve their own Python** by capability — 3.10+ with
  `numpy`, `ml_dtypes` and `safetensors` — instead of a pinned `python3.13`, so
  they work wherever the analysis stack lives; `NVMAI_PYTHON` overrides.
- **The app recognizes both KAT widths**, with descriptors carrying each
  snapshot's own fingerprint, and the install table in
  `AppModelInstallTests` now covers ten builds.
- **`benchmark/nvmai_maxthroughput.py --engine cpu|gpu`** selects the engine and
  folds it into the result label, so a CPU row cannot be read as a GPU one.
- **The README benchmark table** carries KAT's measured rows and a
  GPU-versus-CPU table for the dense Qwen 3.5 models.

### Performance

KAT's rows are the only numbers newly measured for this release, on this base M3
with 24 GB, and they are in the README table above. The dense Qwen 3.5
GPU-versus-CPU table added to the README is **not** a fresh measurement: it quotes
the decode rates already recorded on the wiki's
[One Prompt, Every Model](https://github.com/Pummelchen/NVMAI/wiki/Capital-of-Paris-Smartness)
page, which was measured on this machine. Attempts to re-run the dense CPU
numbers for this release were abandoned as unreliable — the 9B 8-bit thrashes on
the CPU engine (0.4–1.2 tok/s, one 512-token generation taking 1225 s) because
that engine holds the model resident instead of streaming experts — and the
recorded numbers are short-generation rates, not 512-token peaks like the rows
above them.

### Verification

**Incomplete — the release has not been cut.** The dry run reached the golden
phase on the tagged commit: `tools/lint.sh` clean, `swift test --no-parallel`
**1470 tests in 228 suites passed** (122.6 s), and the first baseline checked
(Ornith 1.5 8-bit) was byte-identical. The phase then stopped on `ornith-4`:

```
  FAILED (exit 1)
  error: parallel expert read failed: Operation timed out
error: golden baseline mismatch (4)
```

**That is the storage provider, not the runtime.** Dropbox had left seven of the
ten installs *online-only*: `fileproviderctl evaluate <path>` reported
`isDownloaded = 0`, and reading one outside NVMAI reproduces the failure (`cat`
exits 1 with `Operation timed out`) — so the error the streaming pool reports is
a refused read, not a defect it found. Six of the seven installs were
materialized (24 GB, at 2–15 MB/s). The seventh needs **134 GB** materialized,
the gate loads every install in one run, and the volume had 123 GB free; it was
not made local by deleting other data, and the provider then paused downloads
outright (`FP -1004 "Sync paused" … domain: serverUnreachable`), with Dropbox's
own `speculative disk management` the mechanism that evicted the files in the
first place.

The skip is named and reason-carrying rather than deleted from the gate's list
(`NVMAI_RELEASE_SKIP_GOLDENS`, enforced by `--publish` to appear in these
notes), and it is *not* an approval to publish: the golden phase — 9 of 10
baselines, `qwen38-8` skipped — the clean scratch build, and the staged archive
still have to run before this section can state a result. What follows is what
it must say once they have: the counts, the archive size, and this paragraph
with the outcome filled in.

### Checksum

`nvmai-5.3-macos-arm64.tar.gz` sha256: `SHA256_PENDING`
