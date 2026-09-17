# The M1 gate

**Status update, 2026-09-17 (later): the gate FAILS on today's artifacts, and the failure is localised.**
The earlier update said "not current" because the only comparison available used a contract predating the
changes. That gap is now closed: the contract was **re-run** — see below for how, since it had been
impossible — and it still differs from the engine by **40 discrete decisions and 1 float**. So this is not a
stale artifact. The milestone does not hold today, and the divergence is small enough to hunt.

**The first divergence, in the differ's own order:**

```
float     layer.00.hidden_out  element 0: reference -0.006576654966920614,
                                          candidate -0.005132569000124931, 3101151 ULP apart
discrete  layer.00.router.topk missing [231, 71], unexpected [72, 19]  (and 39 more layers)
```

`layer.00.hidden_in` **matches**, so the divergence is not in the embedding: it is **inside layer 0**. From
there the router flips and the decisions cascade for all forty layers, which is exactly the failure mode
`I3` exists to catch.

**Resolved, with matched weights (`D56`).** The gate's claim was being tested with two different inputs: the
engine read the install and the contract read the checkpoint. `tools/install_source.py` points the contract at
the **install**, through the same dequantiser the Swift reader mirrors, and then the comparison is about
arithmetic and nothing else:

```
trace_diff  engine (install)  vs  contract (install)     IDENTICAL — 83 tensors, 0 elements, 40 discrete
                                                         decisions, matching digests b0d382dbabf36df0…
trace_diff  engine (install)  vs  contract (checkpoint)   DIFFERENT — 40 discrete, 1 float
```

**The engine is correct, end to end**, not merely at layer 0: byte-identical traces and identical digests over
all 83 tensors and all 40 router decisions. The second line is the **declared, measured** effect of the
install's quantisation — the same 40 and 1 that `D50`–`D55` chased — and it is tracked in `DC-112` rather
than hidden. `tools/milestones.json` now carries the claim in this falsifiable form, so the milestone check
re-verifies the right thing.

**Root cause, proved bit-exactly (`D55`).** The two sides are not running the same weights. The install
holds the Gated DeltaNet's three projections — `linear.in_qkv`, `linear.in_z`, `linear.out` — and
`attn.q/k/v/o` as **int4-affine**, by `tools/quant_policy.json`; the contract is generated from the
checkpoint, where they are bf16. Everything else in layer 0 is byte-identical between the two.

Running the reference's layer 0 on the identical `hidden_in`, once with the checkpoint's bf16 projections and
once with **the install's own int4 values** read back through the install reader:

```
reference bf16-proj vs engine (install)    max abs 0.0155785   median rel 0.245   identical=False
reference int4-proj vs engine              max abs 0           median rel 0       identical=True
reference bf16 vs reference int4-proj      max abs 0.0155785   median rel 0.245   identical=False
```

**The engine is correct**, and this is stronger evidence than the gate's original claim ever was: given the
same weights it reproduces the reference byte for byte through the convolution, the gates, the l2 norms, the
chunked delta rule, the triangular solves, the gated norm and the projection. M1's failure is the
**quantisation policy**, and the milestone's claim that an int4 install reproduces a bf16 contract
byte-for-byte **cannot hold as configured**. That is a decision for `DC-112`, not a bug to fix in the engine —
and it also means the pass recorded on 2026-09-16 cannot be reconciled with these artifacts.

**Narrowed further, with capture points inside the layer.** The trace captures layer boundaries only, so
both sides gained an **opt-in** internals capture — `SHARD_TRACE_INTERNALS=1` for the engine,
`--capture-internals` for the reference — recording `attn_out` (the mixture's input, after the attention
residual) and `ff_out` (the feed-forward's output) per layer. Off by default and it has to be: the digest
covers the tensor list, so a trace with extra tensors is a different artifact. The check that it is genuinely
opt-in is that the default trace is still **83 tensors with digest `b0d382dbabf36df0…`**, byte for byte.

With 163 tensors captured on each side, the first divergence is:

```
layer.00.hidden_in   identical
layer.00.attn_out    differs — all 10240 values
layer.00.ff_out      differs
```

So the divergence is in the **attention half of layer 0** — the input norm and the Gated DeltaNet — and the
router, the mixture and every layer after it are **downstream consequences**. That eliminates the whole
mixture path as a cause, and the router's marginal flip is now explained rather than suspected.

**Correction, same day, to how that first line was read.** "1 float" in the differ's output means one float
**tensor**, not one element: **all 10240 values** of `layer.00.hidden_out` differ, and the single value quoted
is merely the first of them. The earlier reading — "one element, 0.0014 absolute, a boundary difference" —
was wrong, and what it suggests changes with it: this is a **systematic** difference in the layer, not a
rounding artefact at one position. `layer.00.hidden_in` matching still stands, and the layer's convolution has
since been eliminated by reading both implementations: the reference and the engine index
`position + tap - (kernel - 1)`, accumulate in the same order, zero-pad identically, and activate with silu,
which is why neither is the cause. The open candidates are the delta rule's first step, the gates'
exponentiation, and the gated RMSNorm's ordering — the engine's own comment records that its bf16 oracle
rounds the normalised value *before* the weight multiply while an all-fp32 contract does not.

**And the re-run is no longer a hazard, which is what made this possible.** Two changes: the reference can
read the checkpoint through `pread` instead of `safe_open`'s mmap (`D47`), and it can fetch the routed
experts **by index** instead of materialising a layer's 3.2 GB stack (`D48`). With both, the contract run
finishes in about two minutes with swap **flat at ~1.2 GB** and disk **steady at 11 GB**, where the attempt
without them drove swap to 5.1 GB and disk down to 8.0 GB in a minute. The streaming change is proven
arithmetically invisible on the **real model**: the contract produced with the streamed reference is
**IDENTICAL** to the one produced with the stacked reference — 83 tensors, 0 elements, 40 decisions.

The recorded pass below is real and was real; it belongs to the artifacts of 16 September 01:5x. Nothing
here has been rewritten: the pass is kept, and this is what re-checking it found.

The evidence, measured today:

| comparison | result |
| --- | --- |
| the stored contract vs the stored engine trace (both 2026-09-16 01:5x) | `IDENTICAL — 83 tensors, 0 elements, 40 discrete` |
| a **fresh engine trace** vs the **stored engine trace** | `DIFFERENT — 40 discrete, 1 float` |
| a fresh engine trace vs the **stored contract** | `DIFFERENT — 40 discrete, 1 float` |

The first row is why the other two mean something: the two stored traces have **byte-identical**
`data.bin` (`a7c77b63…`), so they agreed when they were written. Today's engine trace has
`data.bin f52ca4c3…` and prints digest `b0d382dbabf36df0…`, where the stored pair prints `b8c976c5e7ba8816…`.

**The contract is not the stale side.** The reference's only change since the stored contract is **two
lines** adding a headroom guard (`d26b419`), and its arithmetic is untouched; it reads the **checkpoint**,
which has not changed. So the drift is on the engine's **install** path, and the candidates are exactly the
changes that landed after 01:5x: the install was **rebuilt at 04:50**, and the dequantiser's zero and NaN
canonicalisation landed with `D34` at 23:36.

**Why this is not settled here, corrected by measurement.** The first answer was "the reference's read is a
page-cache hazard", and that half is now **fixed rather than assumed**: `tools/uncached_safetensors.py` reads
the shards through `pread` and never maps them, and it is **byte-identical to `safe_open` on a real 4 GB shard
of this checkpoint** — a row slice of the 1074 MB expert tensor and a one-dimensional norm, with every dtype
the checkpoint uses. The reference takes it behind an explicit `--uncached` flag, because the contract is the
authority and its reader should change only deliberately.

**And the run still cannot finish here, for a different reason — measured, not assumed.** With the uncached
reader active, on this node, one attempt produced:

```
t+20s   swap total = 4096.00M  used = 2400.44M   disk free 9.0 GB
t+40s   swap total = 4096.00M  used = 3412.38M   disk free 9.0 GB
t+60s   swap total = 5120.00M  used = 3825.75M   disk free 8.0 GB
```

Swap grew from 2048 MB to **5120 MB** and free disk fell from 11.8 GB to **8.0 GB** in sixty seconds, with
the page cache out of the picture — so the binding constraint is the **reference's own working set**: one
`gate_up_proj` is 1074 MB of bf16 that the reference materialises as **2 GB of fp32**, before the layer's
other tensors and its activations. It was stopped deliberately at 8 GB free rather than letting the disk
floor stop it, and the machine recovered to 10 GB free. The remaining step needs a machine with more memory
than this one has, which is what `DC-111` now says: run the **engine on the checkpoint** and the **reference
on the checkpoint** where both fit, and the two are then comparable without guessing.

**Corrected 2026-09-17: that step has been taken, on this node.** Both sides now read the checkpoint through
`pread`, and the whole gate passes on all five frozen prompts — the status section below has the numbers. What
remains specific to a small machine is the **install** path, and the reason is speed rather than size: its
contract reads cost hours where the engine's cost minutes (`DC-114`, `D70`).

**And this document's own record was inconsistent, which is how it stayed hidden.** Further down, a status
table lists the trace digest as `b0d382dbabf36df0…` — the value the engine produces today, the value
`D34`'s record confirms, and the value the M2 and M3 gates report — while the status section above it still
calls `b8c976c5e7ba8816…` the current digest. Both were true at different times, and nothing re-checked which
was true *now*. `tools/check_milestones.py` does exactly that, and it reports this agreement as **stale**
with its task rather than letting a superseded digest stand as a pass (`D46`).

**Status: passing**, re-established 2026-09-16 after `D15` and `D16`. See the status section below
for the evidence, and for three claims in this document that the code has since outgrown.

M1's gate is three things: **correct output**, a **recorded tok/s baseline**, and a **measured
cache hit rate**. One command runs all three on one checkpoint and writes a report:

```bash
.venv/bin/python tools/run_m1_gate.py \
    --snapshot .build/hf-cache/models--Qwen--Qwen3.6-35B-A3B/snapshots/<rev> \
    --model Qwen/Qwen3.6-35B-A3B --revision <sha>
```

| | |
| --- | --- |
| Correctness | every prompt in `tools/m1_prompts.json`, the engine's trace against the contract's, byte for byte, through `tools/trace_diff.py` — which also compares the **discrete** router decisions exactly, because I3 makes them a different kind of claim from "the numbers are close" |
| Throughput | greedy generation with `datacenter-generate`, its own timing parsed into the report |
| Cache | the engine's expert-traffic counters, written beside each trace in `metrics.json` |
| Memory | each engine run's **peak resident set size**, from the platform's `/usr/bin/time -l`, because `DC-032`'s gate is a budget and a budget needs a number. It counts clean file-backed pages, so it is an upper bound on the process rather than a claim about private dirty memory |

## Status: passing, re-verified 2026-09-17 on all five frozen prompts

**The whole gate, on the real checkpoint, on this node.** Not one prompt and not a subset: all five of
`tools/m1_prompts.json`, engine against contract, both reading the same checkpoint through `pread`.

| prompt | tokens | engine | peak RSS | |
| --- | --- | --- | --- | --- |
| `capital` | 5 | 34.94 s | 3.79 GB | IDENTICAL |
| `arithmetic` | 33 | 89.65 s | 3.76 GB | IDENTICAL |
| `code` | 40 | 115.92 s | 3.63 GB | IDENTICAL |
| `repeat` | 60 | 128.13 s | 3.60 GB | IDENTICAL |
| `long` | 67 | 146.26 s | 3.67 GB | IDENTICAL |

`GATE PASSED`, with `IDENTICAL — 83 tensor(s), 0 element(s), 40 discrete decision(s) checked` on every one, and
the first prompt's digest `b8c976c5e7ba8816…` on **both** sides. Expert traffic runs 2,240 to 8,040 requests
and 3.5 to 12.6 GB of elements read, with a cache hit rate of **0.0000** on every prompt — `D31`'s finding,
arriving again from another direction.

**That digest answers the question this document left open.** The narrative above worried that "the drift is on
the engine's install path", because a fresh engine trace printed `b0d382dbabf36df0…` where the stored pair
printed `b8c976c5e7ba8816…`. Point the engine at the **same input the stored pair used** — the checkpoint — and
it prints `b8c976c5e7ba8816…` again, byte for byte, after the GPU unpack became the default and every contract
matmul went through a chooser. There is no engine drift to find: the two digests are two **inputs**, and `D55`
measures the distance between them.

**And the gate was not passing the pair of flags this document records as survivable.** `run_m1_gate.py` passed
**neither** until `D69` added `--stream-experts` and `D73` added `--uncached`, so its checkpoint runs went
through `safe_open` and **mapped 67 GB** — the hazard this file names two sections above as the thing that drove
swap to 5.1 GB and disk to 8.0 GB in a minute. Measured after the fix, the contract alone is **66.20 s and
0.397 GB** peak on the `capital` prompt, so what a checkpoint run spends is the **engine's** trace over bf16
weights rather than the contract's.

**The declarations follow that measurement, and they differ on purpose.** A checkpoint run declares **4.2 GB**,
because the engine's trace peaks at 3.60-3.79 GB; an install run declares **1.5 GB**, against a contract
measured at 1.21 GB (`D69`). The first draft of the fix set the checkpoint figure to 1.5 GB from the contract's
0.397 GB and the gate's own peak disproved it the same day (`D73`); an under-declared guard admits a job the
machine cannot take, which is worse than a conservative one.

**And the same gate passes against the INSTALL, which is M1's restated claim.** Round 52 made the gate able
to read an install at all and round 59 made it able to finish one: the install dequantiser was visiting every
value in Python, 0.253 s for one real expert and about eleven minutes of expert reads for a forward. Vectorised,
and proven **bit-identical** to the scalar definition on the real install and over both fixtures, the whole gate
now completes:

| prompt | tokens | engine | peak RSS | |
| --- | --- | --- | --- | --- |
| `capital` | 5 | 17.40 s | 0.98 GB | IDENTICAL |
| `arithmetic` | 33 | 54.51 s | 1.30 GB | IDENTICAL |
| `code` | 40 | 67.63 s | 1.34 GB | IDENTICAL |
| `repeat` | 60 | 86.90 s | 1.41 GB | IDENTICAL |
| `long` | 67 | 98.45 s | 1.31 GB | IDENTICAL |

`GATE PASSED`, digests `b0d382dbabf36df0…` on both sides, about an hour for the set. The install path declares
**2.0 GB**, because the measured peak is 1.41 GB and 1.41 against the previous 1.5 was a 6% margin (`D75`).

## Status: passing, re-established 2026-09-16 after `D15` and `D16`

**M1's gate passes.** The three parts, with evidence measured today rather than quoted from the
original run, on the frozen `capital` prompt against the pinned checkpoint:

| Part | Result | Evidence |
| --- | --- | --- |
| Correct output | **IDENTICAL** | the contract (`ordered_qwen36_trace.py`) was **re-run today** and `trace_diff` reports `IDENTICAL — 83 tensor(s), 0 element(s), 40 discrete decision(s) checked (matching digests)`. Engine and contract digest are both `b8c976c5e7ba8816…` — the *same* digest as the original gate run, which is the evidence that `D15` and `D16` changed no arithmetic, and that `D11`'s flush is install-only and does not touch the checkpoint path |
| Throughput | **0.108 tok/s** cached (9.25 s/token), 0.0374 uncached | `datacenter-generate` on the install, 4 steps after the 5-token prompt; cold run 37.0 s, warm 22.1 s, and **both paths generate the identical tokens** (`11751,11,264,3177`) |
| Cache hit rate | **0.0000** over 2,240 requests | structural: the expert banks are rebuilt per forward. `D12` is the open design question about whether one may survive a token; the gate asks for a *measured* rate, and this is it |
| Memory | **348.6 MB peak RSS** | `/usr/bin/time -l` on the install path, which reads `F_NOCACHE` + `pread` and maps nothing |

**Three claims in this document that the code has outgrown**, each recorded rather than edited away:

1. **"M1 has no KV cache" is no longer true.** `datacenter-generate --cached` decodes one position per
   token against a per-layer state, and three tests hold it to the same numbers as the sequence path:
   `testCachedGenerationProducesTheSameTokensAsUncached`,
   `testCachedAttentionIsBitIdenticalToTheSequenceAttention`, and `testTheCachedPathChoosesTheSameExperts`.
   The paragraph below said the cache was "the next piece of correctness-preserving work"; it was already
   there, and this run's identical token list is the model-level confirmation.
2. **The throughput baseline is 5.6x the historical figure** — 0.0191 tok/s then, **0.108** now — because
   `D15` and `D16` removed 24 s of hashing and duplicate reads from every forward. The uncached path
   (0.0374 tok/s) is the honest comparison with the original 0.0191, which was uncached by definition.
3. **The memory figure is 12x smaller on the install path** — 4.16 GB peak RSS then, **348.6 MB** now —
   because the checkpoint run mapped safetensors shards and counted their clean file-backed pages as
   resident, while the install reads uncached and maps nothing. Both numbers are correct for what they
   measured; they are not the same measurement.

**What the install path's verification is, since `DC-108` closed it.** There is still **no Python
contract for an install *trace*** — `ordered_qwen36_trace.py` accepts only a checkpoint — so the
byte-identical claim above remains a claim about the **checkpoint** path. What the install path has now,
and did not have when this paragraph first said "no Python contract reads an install", is a reader in the
other language: `tools/install_reader.py` reads the container and `tools/verify_install.py` checks what a
per-tensor reader cannot — that the 693 tensors **tile** the 21,700,655,616-byte payload exactly, that
every role's quantisation is the one `tools/quant_policy.json` requires, and that all 693 payloads hash
to the manifest's digests (26.3 s, uncached, free disk flat at 15 GB, swap unchanged). The op-level
golden tests and the bit-identical re-runs across bank sizes are unchanged evidence on top of that.

**Still not checked, and it says so rather than being implied.** The quantisation against the **source
checkpoint** — the one check that would close the loop between the 67 GB source and the 21.7 GB artifact —
needs that source, which is not in this checkout. `DC-108` asked for a reader and a check; it did not
promise the source.

**And a resource note for whoever runs this next.** The contract step re-reads the 67 GB checkpoint
through a cached path, and on the 8 GB development node it drove swap to **402 MB free** before the run
was stopped. The engine's install path is the safe one here (349 MB peak RSS, uncached reads); the
checkpoint contract run should be treated as a heavy job that runs alone.

## THE GATE, on the real model: one prompt, all three parts

`.venv/bin/python tools/run_m1_gate.py --snapshot <35B> --only capital --max-new-tokens 4`

```
capital      IDENTICAL   41.87 s  peak    4.16 GB  hit rate 0
0.0191 tok/s mean over 4 steps
GATE PASSED
```

| Part | Result |
| --- | --- |
| Correct output | **IDENTICAL** — 83 tensors and 40 discrete decisions against the contract, digests equal |
| Peak memory | **4.16 GB** against ~4.5 GB of usable memory per node |
| Cache hit rate | **0.0000** over 2240 requests |
| Throughput | **0.0191 tok/s** — **52.2 s per token** |

The throughput is the number to sit with. It is not a tuning result; it is what a faithful fp32
CPU port with **no KV cache** costs: every generation step re-runs the whole sequence, so a
step's cost grows with the sequence and the mean over four steps after a five-token prompt is
52 s. At a hundred tokens of context that arithmetic is minutes per token. The brief's M0
deferred the cache deliberately — "a cache is a second numeric path through attention and M0's
job is to establish one correct path before there are two" — and that deferral has now been
paid for honestly. **For M1 the KV cache is not an optimisation, it is the next piece of
correctness-preserving work**, and it comes before Metal: a kernel is a constant factor, and
this is a growth term.

The expert traffic behind the token: 2240 requests, **3.52 G elements**, which is **7.05 GB read
from the SSD** in bf16 and 14.1 GB held in fp32.

## First result on the real model — one prompt, M1's correctness claim holds

`capital` (5 tokens) against the real 27 GB checkpoint, engine and contract:

| | engine | contract |
| --- | --- | --- |
| Wall time | **45.6 s** | 461.8 s |
| Peak resident memory | **4.31 GB** | not measured |
| Tensors, discrete | 83, 40 | 83, 40 |
| Digest | `b8c976c5e7ba8816…` | `b8c976c5e7ba8816…` |

`trace_diff` says **IDENTICAL — 83 tensors, 40 discrete decisions checked**, so the engine
reproduces the contract on the 35 B model **byte for byte, with the router's decisions asserted
separately**. That is M1's correctness claim, measured rather than argued, and it is the first
time the whole stack — sharded reader, importer, streaming provider, mixture, head — has run on
the real weights.

Three numbers worth reading carefully:

- **4.31 GB peak resident memory against the brief's ~4.5 GB of usable memory per node.** That is
  at the budget edge, and on macOS the figure includes clean file-backed pages, so most of it is
  the streamed weights the page cache is holding. It is a measured argument for the brief's own
  I/O rule — expert slabs read with `F_NOCACHE`/`O_DIRECT`, which **was** implemented after this
  measurement was taken and is now the rule in both languages: the same 20 GB install verifies
  uncached in about twenty seconds at a **33.7 MB** peak with free disk steady, where reading it
  through the page cache once took free disk from 17 GB to 2.96 GB.
- **Cache hit rate 0.0000**: 2240 expert requests across 40 layers and no hits. A layer's slot
  bank is built as the layer loads and dropped with it, so nothing survives a token, and within
  one token two of five positions rarely agree on all eight experts.
- **Engine 10.1× faster than the contract** on the same prompt (45.6 s against 461.8 s), which is
  the first honest apples-to-apples figure for the fp32 path on this hardware.

Traffic per prompt: 40 layers × **28 distinct experts** × 2 projections = 2240 slices, which is
consistent with 5 positions × top-8 = 40 pairs and no repeats across layers. In fp32 that is
3.52 G elements ≈ 14.1 GB of decoded weights, or ≈ 7.05 GB as the bf16 the shards actually store.

## Status of the instrument

**The instrument is verified, and one prompt has now been measured end to end.** Run against the 236 KB
`tiny-qwen36` fixture, the gate reports `identical everywhere: True` on every prompt and exits
zero, and `tools/test_run_m1_gate.py` drives exactly that — so the gate is not untested code
waiting for the one input that matters. The numbers below are therefore the *shape* of the
report, not M1's result:

| | |
| --- | --- |
| Engine vs contract, tiny fixture | identical, digests equal |
| Cache hit rate, tiny fixture | **0.0000** |
| Throughput, tiny fixture | below the tool's tenth-of-a-second resolution |
| Peak resident memory, tiny fixture | 0.01 GB |

The hit rate of zero is worth reading rather than dismissing: the slot bank belongs to one layer
and is built as that layer loads, so a hit needs two positions **in the same layer of the same
forward pass** to choose the same expert. On a nine-token prompt that happens rarely, and nothing
is reused across tokens. Whether the real model's eight-of-256 routing repeats enough to make the
bank earn its memory is exactly the question M1's gate exists to answer — and if it does not, the
answer is a cache that survives a token, which is a design change and will be recorded as one.

## What the gate does *not* yet claim

- **No KV cache.** Every generation step re-runs the whole sequence, so the mean step time grows
  with it and the recorded figure is a baseline to improve on, not a steady-state rate. The
  report says so in its own `note`.
- **One node.** M2's bit-identity across nodes is a separate gate and the one that matters most.
- **The prompt set is short.** `long` is 67 tokens: beyond one delta-rule chunk, but nothing like
  the 128K the later milestones must hold at. Long-context conversion failures are real and this
  gate does not look for them.

## Status, 2026-09-16: correctness holds on one prompt of five, throughput is superseded

Written because the tables above are the only surviving record of the first real-model run — the
report JSON under `.build/` is gone — and **their throughput figures are no longer current**. Nobody
should quote `0.0191 tok/s` as M1's baseline without reading this section first.

### Proven

| | |
| --- | --- |
| Correct output | **IDENTICAL** on one prompt of five (`capital`, 5 tokens): 83 tensors and 40 discrete decisions, digests equal, through `trace_diff.py` |
| Discrete decisions | exact, asserted separately from the numbers, as `I3` requires |
| The cached path | now agrees with the uncached one on **tokens and margins** (`11751,11,264,3177`; `1.6400, 0.0972, 1.2532, 2.6414`) after the weight-layout fix in the attention cache |
| The 4-bit install | builds (20 GB, peak footprint 4.44 GB), and a full verification reads it uncached in ~20 s at a 33.7 MB peak with free disk steady |

### Superseded, and by how much

The measured throughput above was taken **before** three changes, each of which is bit-identical and
each of which was measured locally:

| change | measured effect |
| --- | --- |
| the `int4` row read decodes **one expert** instead of the whole stack | one expert of eight now costs 1184 of 9472 bytes where it cost all of them |
| the ordered matmul vectorised across outputs | **1.66×** (4.1 → 6.8 GFLOP/s) |
| the int4 unpack vectorised by group and widened to eight codes per load | **1.83×** (648 → 1185 M values/s) |

The first of those is the one that should matter most, and the arithmetic is why: the active experts
are ~2.0 B parameters per token, so unpacking them at the old rate is ~9.3 s, and decoding whole
stacks instead of one expert multiplied that by thirty-two — **~55 s**, which is the measured
52.2 s/step reached independently by a different route.

**So the prediction is that throughput improves by most of an order of magnitude. It is not a
result.** No real-model run has happened since those fixes, and this section exists so that the
stale number and the prediction cannot be confused for one another.

## Covered already — checked before proposing work

An audit aimed at the two largest engine surfaces nobody in this session had personally probed returned
"already covered, and specifically", which is worth recording so the next round does not propose the
same probing again.

**The Gated DeltaNet recurrence**, where `D8` makes the chunked rule authoritative and the cache a
second numeric path — the exact shape that diverges silently:

| test | what it pins |
| --- | --- |
| `testSingleChunkMatchesTheContract` | one chunk, against the contract |
| `testMultipleChunksMatchTheContract` | the chunked rule itself |
| `testTheDecodeStepMatchesTheSequencePathOnTheLongCase` | the **cache** against the sequence path |
| `testTheDecodeStepMatchesTheSequencePathWithAsymmetricHeads` | the same, where head widths differ |
| `testTheConvIsCausalAndLeftPadded` | the convolution's causality and padding |
| `testSoftplusUsesTheThreshold` | the gating nonlinearity's threshold |
| `testTriangularSolveMatchesForwardSubstitution` | the solve, against an independent formulation |

**The cache path**, including the property this audit was going to add before finding it present:

| test | what it pins |
| --- | --- |
| `testCachedGenerationProducesTheSameTokensAsUncached` | prefill plus decode against one long sequence |
| `testCachedAttentionIsBitIdenticalToTheSequenceAttention` | bit identity, not tolerance |
| `testTheCachedPathChoosesTheSameExperts` | the discrete decision through the cache (I3) |
| `testTheReplayLandsOnThePromptsLastPosition` | the position the cache resumes at |
| `testASteppedPositionIsCheaperThanTheWholeSequence` | that a step reads less than the sequence |

**Configuration provenance**, which independently confirms that `DC-093`'s claim was false:
`testTheConfigurationComesFromTheCheckpointNotFromDefaults`, `testTheRopeTablesArePartial` and
`testTheFixtureExercisesBothLayerKinds` all exist and pass.

### Not measured

### The spec carries the checkpoint's geometry, field by field

Verified against the artifact rather than asserted: every geometry field in
[`Qwen/Qwen3.6-35B-A3B`'s `config.json`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/raw/main/config.json)
appears in the real install's `spec.config` with the same value, and **no field is silently
defaulted** — which is the failure that matters, because a default looks exactly like agreement.

| checkpoint | install `spec.config` | value |
| --- | --- | --- |
| `hidden_size` | `hiddenSize` | 2048 |
| `head_dim` | `headDim` | 256 |
| `num_attention_heads` / `num_key_value_heads` | `numAttentionHeads` / `numKeyValueHeads` | 16 / 2 |
| `num_hidden_layers` | `numLayers` | 40 |
| `num_experts` / `num_experts_per_tok` | `numExperts` / `numExpertsPerToken` | 256 / 8 |
| `moe_intermediate_size` | `moeIntermediateSize` | 512 |
| `vocab_size` | `vocabSize` | 248320 |
| `rms_norm_eps` | `rmsNormEps` | 1e-06 |
| `rope_parameters.partial_rotary_factor` | `partialRotaryFactor` | 0.25 |
| `rope_parameters.rope_theta` | `ropeTheta` | 1e7 |
| `full_attention_interval` | `fullAttentionInterval` | 4 |
| `linear_num_key_heads` / `linear_num_value_heads` | `linearKeyHeads` / `linearValueHeads` | 16 / 32 |

`tools/test_fixture_spec_matches_config.py` holds that correspondence as a check with tests, and reads
the config at **top level, under `text_config`, and under `rope_parameters`** — because all three
nestings are in play and two of them have already fooled a hand-written lookup this session.

> **A withdrawn claim, kept visible.** An earlier revision of this page listed "the MoE fixture does not
> exercise partial RoPE" here. That was **wrong**: `Fixtures/tiny-qwen36/config.json` has carried
> `"partial_rotary_factor": 0.5` and `"rope_theta": 10000000.0` all along, and `spec.json` carries
> `"partialRotaryFactor": 0.5`. The check that produced the claim read the fixture's config under
> `text_config`, where the *real* checkpoint nests its geometry, while the fixture keeps those keys at
> the top level — so the check looked in the wrong place, printed `(absent)`, and I reported my own
> tooling's blind spot as a fact about the repository. `DC-093` is withdrawn. The fixture covers the
> partial-RoPE path.


## Run on all five prompts, 2026-09-16: it works, and I1 holds on the real model

The operator approved one real-model run with the instruction to test **that it works rather than
for benchmark statistics**, because every node in the farm is doing coding work. So this section
reports correctness, and the timings below are explicitly **not** a baseline.

| prompt | tokens | result | seconds, under load |
| --- | --- | --- | --- |
| `capital` | 5 | trace written, exit 0 | 33 |
| `arithmetic` | 33 | trace written, exit 0 | 85 |
| `code` | 40 | trace written, exit 0 | 109 |
| `repeat` | 60 | trace written, exit 0 | 119 |
| `long` | 67 | trace written, exit 0 | 138 |

Those seconds are wall-clock on a machine running other work. They are recorded because they exist,
**not** as a throughput figure, and the difference matters: `capital` took 45.6 s in the recorded run
and 33 s here, which is *not* a speedup claim — the earlier figure was taken on an idle machine and
these were not.

### I1 verified on the real model

`capital` and `long` were each run **twice**, and both pairs of traces are **byte-identical**,
`data.bin` and `manifest.json` alike. That is `I1` — identical input, identical output bytes — tested
directly on the real 35B checkpoint rather than argued from the design.

### And the strongest result of the session

`capital`'s canonical digest is **`b8c976c5e7ba8816e8322a41137b568790b7ca08d081483fc392c8ae1401b8f3`**,
which is the value recorded in the first gate run — the trace that was **bit-identical to the
contract**. So the engine still produces the contract's exact bytes after all of it: the `int4` row
read decoding one expert instead of a whole stack, the matmul vectorised across outputs, the unpack
vectorised and widened to eight codes per load, and five upcoming language features. Each of those
was proven bit-identical in isolation; this is the composition of them all, on the real model,
against the digest on record.

### Still unmeasured, and why

The **contract** half for the four new prompts. Its cost is ~10× the engine's, and 461.8 s for a
five-token prompt extrapolates to roughly **five hours** for the five-prompt set — which is not a
reasonable thing to do to a node that other work depends on. So M1's byte-identity claim remains
proven on **one prompt of five**, and the other four are now known to *run* rather than to *match*.

## I4 and L2 audited: covered, and the shared policy is not rot

Auditing "quantization and sharding policy are data, not code" and the importer layer against the
brief found both satisfied, with the evidence rather than the intent:

- **A role absent from `quant_policy.json` stops the install**, which `AGENTS.md` states as a trap and
  which is tested **four times** (`assertRaises(quantize.PolicyError)` in `test_quantize.py`).
- **The policy carries its own rationale** — a `why` block justifying each role against an invariant:
  routers stay bf16 for I3, the routed experts are "92.9 % of this model's parameters", the shared
  expert's error is not amortised because it is active on every token, and the Gated DeltaNet's decay
  is exponentiated so a 4-bit error there is not a small one. That is I4 done as data *with a reason*.
- **Both directions of coverage now have a test.** The real install's 27 roles all have policy entries;
  the three entries no MoE tensor uses (`mlp.down`, `mlp.gate`, `mlp.up`) appear in the **dense**
  fixture, which is the shared policy format working as I4 intends rather than dead weight. The test
  asserts the union across fixtures, so rot in either direction fails it.
- **The importers are 144–235 lines**, comfortably under L2's 500–800 ceiling: `Qwen3Importer` 144,
  `Qwen3_5MoEImporter` 223, `Qwen3_5Importer` 235.

Sharding policy is **not** audited here because it does not exist yet: it is M2's work, and the brief
puts it there.

## Measuring the cache hit rate: the sweep instrument

M1's gate asks for "a measured cache hit rate", and `D12` is open because the brief asks for per-layer
LRU slot banks without saying how large. The literal was `16`, which `SlotBudgetTests` reports as an
expected failure: 201 MB per layer, **8.05 GB across 40 layers** against roughly 4.5 GB usable.

Rather than guess a size, the run should **measure** it — the honest answer needs a hit rate at more
than one size, and the gap between "how often does routing repeat" and "how much RAM is there" needs a
number between the two. `SHARD_EXPERT_SLOTS=<n>` sets the bank for one process:

```bash
SHARD_EXPERT_SLOTS=2  datacenter-trace ...   # 40 x 2 x 12.58 MB = 1.0 GB, inside the budget
SHARD_EXPERT_SLOTS=16 datacenter-trace ...   # the historical literal, 8.05 GB, does not fit
```

Read **once per process** as a `static let`, because Swift 6 forbids a mutable global — and because two
bank sizes in one process would share banks, so one process per setting is what a sweep wants anyway.
The parse is a separate function so it can be tested without a process: `"0"`, `"-3"`, `"many"` and `""`
all fall back to 16, since a bank of zero would silently disable the cache the measurement exists to
observe.

## The slot bank's contract, measured

M1's gate asks for a **measured** cache hit rate, and `D12` is open because the bank's size was never
derived from a budget. Before a real-model sweep can be read, the mechanism underneath it has to be
pinned — an instrument that has never been run is not an instrument. `ExpertSlotCacheTests` does that
against a counting stub, so every number is deterministic and none of it needs a checkpoint:

- **The brief asks for LRU**, so eviction order is a promise and is tested: touch 0, 1, 0, then 2, and
  0 survives while 1 is evicted.
- **The bank is bounded**: ten distinct experts through a bank of two leaves
  `peakResidentExperts <= 2`, which is the figure the budget arithmetic multiplies by a per-expert
  byte count.
- **The hit rate is non-decreasing in the bank size**, or a swept curve could not be read.

The curve for one skewed routing sequence (24 steps, 8 experts, top-k shaped) is:

| capacity | hits |
| --- | --- |
| 1 | 2 / 48 |
| 2 | 18 / 48 |
| 4 | 26 / 48 |
| 8 | 32 / 48 |

**32 and not 48**, because eight experts times two projections is sixteen *compulsory* first misses —
which is the number my first version of the test got wrong, not the engine. A fixture this small says
nothing about the real model's routing, and it is not meant to: it says the mechanism honours its own
contract, so that when the real sweep prints a curve, the curve means something.

## The sweep on the real model, 2026-09-16: the cache is transparent, and the hit rate is zero at every size

Run under the operator's standing approval, with `check_disk_headroom` clean (17.01 GB free, floor
5 GB), swap at 935 MB of 2048 MB used, and `tools/disk_watchdog.py` alongside. `capital`'s five frozen
tokens — the same ids as the earlier run — through the engine at bank sizes **2, 8 and 16**:

| bank size | trace digest | wall clock | bytes from SSD *(estimate — see the correction below)* | requests | hits | decoded in memory |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | `b0d382db…` | 39.5 s | 6,977,224,704 | 2,218 | **0** | 13,954,449,408 |
| 8 | `b0d382db…` | 39.3 s | 6,977,224,704 | 2,218 | **0** | 13,954,449,408 |
| 16 | `b0d382db…` | 39.9 s | 6,977,224,704 | 2,218 | **0** | 13,954,449,408 |

**The traces are byte-identical, checked independently rather than inferred from the digests:**
`trace_diff` reports *83 tensors, 0 differing elements, 40 discrete decisions* and matching digests. So
the bank's size cannot change what the engine outputs, which is the least a cache has to be — and it is
the same property M2 will need from sharding, on the same harness.

**What the numbers say per token:** 1,395 MB read from SSD per prefill token *by the estimate that has since been withdrawn — the measured figure is 2.6×–5.2× smaller, see the correction below*; 13.95 GB decoded into
memory, exactly twice the SSD bytes because the int4 payload becomes fp32; prefill at **7.90 s per
token, 0.127 prefill tok/s**. That last figure is a **prefill** rate and the gate's baseline is a
**generation** rate — they measure different things and must not be compared.

**The hit rate is 0.0000 at every size**, which is the structural result `DC-091` found by audit: the
banks are rebuilt per forward, so no token can hit what an earlier one read, and the size is therefore
irrelevant. `D12` is not "how big should the bank be" but "may a bank survive a token, and what total
budget may it hold".

**Two things this run also establishes by accident, both worth keeping:** the run pushed swap from
**935 MB to 1,504 MB** used, and the sweep's own guard now **refuses a second run** at that level — the
check is doing its job, not decorating the script. And the digest differs from the pre-`D11`
`b8c976c5…`, exactly as `D11` predicted, so **the contract comparison under the flushed definition is
still the outstanding half** of M1's correctness claim.

**One number is not yet explained and is recorded rather than guessed:** 2,218 requests is 221.8 expert
*fetches* per token (2,218 = 2 projections × 1,109 fetches), against 8 experts × 40 layers = 320
selections per token. A mixture that de-duplicates selections within a layer would land below 320, and
5.5 distinct experts per layer per position is plausible — but that is an inference, not a measurement.
Settling it means counting distinct experts per layer in the mixture and comparing.

### Correction, 2026-09-16: the "bytes from SSD" column was an estimate, and a wrong one

It came from `expertElementsRead * 2`, which assumes two bytes per element on disk. This install stores
experts at **0.578 bytes per weight** — 4-bit codes with group-64 scales and zeros, 4.625 bits — so the
estimate is **≈3.5× too large**. The install had been counting the bytes it actually read all along
(`InstallFile.bytesRead`, exposed as `WeightSource.bytesReadFromSource` and now reported by
`datacenter-trace` as `install_bytes_read_this_forward`), so the fix was to report the measurement
instead of arithmetic about it. Three tests hold it: that a forward reports the bytes it read, that the
counter only grows, and that **a forward does not read the whole install** — which is the streaming
claim, and the same class of bug as `DC-088`'s guard that touched a whole entry to answer a question
about one expert.

**What the real traffic was, bounded by the geometry rather than guessed.** One expert's payload is
**1,818,624 bytes** (gate+up 1,212,416 + down 606,208, read from `install.json`), so the run's traffic
is 1,818,624 × the number of expert fetches. The counters do not pin the fetch count down — that is the
open question `Q9` — so:

| if a request is | fetches for 5 tokens | real traffic | per prefill token |
| --- | --- | --- | --- |
| an (expert, projection) read | 1,109 | 2.02 GB | 404 MB |
| one expert fetch | 2,218 | 4.03 GB | 807 MB |

**And the derived "effective 177 MB/s" is withdrawn with it**: it was computed from the estimate, so it
was wrong by the same factor. A number about the read path has to come from the read path, which is the
next measurement rather than another division.

## The read path, measured 2026-09-16: ~1 GB/s, so the reads are not the bottleneck

The sweep's numbers left a question — 39.5 s for five tokens, and various wrong readings of where
that went — so the read path was measured directly with `tools/measure_expert_reads.py`, which
performs the engine's own fetch: **six preads per expert** (codes, scales, zeros for `gate_up`, then
the same three for `down`), taken from the install's geometry rather than assumed.

**3.10 GB per pattern — larger than cache, which matters.** The first version of this measurement read
77.6 MB and reported **2957 MB/s**, above this SSD's measured ceiling: it was measuring cache, exactly
the trap the Testbed page records ("a file that fit in cache" gave 13 GB/s). The tool now refuses a
sample below 1 GB by default and measures the same pattern twice to show whether a cache effect exists.

| pattern | GB | seconds | MB/s |
| --- | --- | --- | --- |
| `ascending` — the engine's pattern, slab order | 3.10 | 3.113 | **997** |
| `ascending`, repeated immediately | 3.10 | 3.139 | 989 |
| `random` — the engine's pattern, shuffled order | 3.10 | 3.169 | **979** |
| `blocks-16k` — 16 KB random | 3.10 | 22.940 | 135 |

Cold and warm agree (997 against 989), so there is no cache effect and these are the disk's figures.
**The engine's access pattern achieves ~86% of the SSD's sequential ceiling, and randomising the expert
order costs it 2%.** 16 KB blocks, by contrast, are 7.4× slower — which is why the layout is contiguous
and why `D13` records what would change if the read path ever moved to `O_DIRECT`.

**Two earlier claims of mine are withdrawn with this measurement.** An "effective 177 MB/s" was
computed from the `elementsRead * 2` estimate and was wrong by that estimate's factor. And the
conclusion drawn from it — that the single node was "running at a fraction of its own SSD's
capability", with local read-path work as the first priority — does not survive contact with the
measurement: **the reads are already near the ceiling.** Both are withdrawn in the [News](https://github.com/Pummelchen/TinyTitan_Datacenter/wiki/News) record
rather than edited out of it.

**So where does 39.5 s go?** Bounded by measurement, the components are:

| component | basis | time |
| --- | --- | --- |
| expert reads | 2.02–4.03 GB (fetch count is `Q9`) at ~1 GB/s | 2–4 s |
| dense weights re-read per forward | 3.08 GB of the 21.7 GB install is not expert stacks; `loadLayer` runs per layer **per forward** and every read is uncached | ~3 s |
| expert dequantisation | 3.49 G elements at the measured 1185 M values/s | ~2.9 s |
| matmuls | ~10 GFLOP per five tokens at the measured 6.8 GFLOP/s | ~1.5 s |
| **accounted for** | | **~9–11 s** |
| **measured** | the sweep's wall clock | **39.5 s** |

**A factor of about 3.5 is unaccounted for, and it is not the disk.** The next measurement is a profile
of the forward itself rather than more arithmetic about it — the pattern that has been wrong three times
now when it replaced a measurement.

## The forward, profiled: what the 39.5 s was, and one 2x speedup

`DC-105` asked where a five-token forward's time goes. `SHARD_PROFILE=1` now times the real code path
(`Profiler` in `ForwardPass.swift`, marks in the forward and inside the mixture), and the trace writes
the phases into `metrics.json` beside the trace. The phases sum to the run: **39.82 s of a 39.9 s
wall clock**, so this is a profile and not arithmetic about one.

| phase | seconds | share |
| --- | --- | --- |
| `mix.read` — the expert fetch | **25.53** | **64%** |
| `attn.core` | 3.91 | 10% |
| `head` | 3.76 | 9% |
| `load` — a layer's dense weights | 3.26 | 8% |
| `embed` | 1.59 | 4% |
| `mix.gateup` (matmul) | 1.03 | 3% |
| `mix.down` (matmul) | 0.49 | 1% |
| everything else | 0.25 | 1% |

**`DC-106`'s hypothesis was wrong, and the profile says so.** The dense weights re-read per forward
were 8%, not the missing 3.5x; it is a real inefficiency but a small one. The expert fetch was the
cost, and splitting it open inside the reader (`SourceTiming`) named the reason: **SHA-256 slab
verification on every read, 21.04 s — 53% of the whole forward** — against 0.82 s of actual reads and
4.06 s of unpacking. `D15` records the decision: verification is now a parameter, off by default,
established out of band by `tools/quantize.py verify`.

**Verified, same prompt, same install:**

| | before | after |
| --- | --- | --- |
| wall clock | 39.9 s | **19.9 s** |
| `mix.read` | 25.53 s | 6.28 s |
| digest seconds | 21.04 s | **0** |
| read seconds | 0.82 s (cache-assisted) | **3.04 s** |
| unpack seconds | 4.06 s | 3.63 s |
| trace digest | `b0d382dbabf36df0…` | **`b0d382dbabf36df0…`** (unchanged) |

**Two independent confirmations came out of it.** The reads now measure **3.04 s for 3.06 GB =
1.0 GB/s**, which reproduces the standalone cold read-path measurement of 997 MB/s from a different
direction — so the engine's read path is doing what the disk can do. And the 3.06 GB is the measured
install traffic of one five-token forward: 2.02 GB of expert payloads (1,109 fetches at 1,818,624
bytes, bounded by the total) plus ~1.04 GB of dense weights, the head included.

**What is left, and where the next work is.** After the fix the largest components are `attn.core`
(3.91 s), the expert unpack (3.63 s) and the head (3.20 s) — none of them I/O. The prefill rate is now
**0.251 tok/s** (19.9 s for five tokens) against 0.127 before; the gate's *generation* baseline is
still a separate measurement, because generation re-runs the sequence without a KV cache and is not
this number.

## The instrument lied again: verification reads were not counted, and cost ~3 GB a forward

`D15` removed per-read slab hashing and took the forward from 39.9 s to 19.8 s. Reading the same code
path for `DC-107` found the same shape in the whole-tensor path:

- **`digestMatches` verified whatever the `verify` flag said.** Reading **five rows of the 1.02 GB
  embedding read all of it** to hash it. That is the `embed` phase: **1.64 s for kilobytes of data**.
- **Those reads were never counted** — `digestMatches` called `blob.readData` directly — so the
  "measured 3.06 GB" from the round-2 section above was **use-traffic reported as a total**. The third
  instrument artefact in this project, after an `elementsRead * 2` estimate and a cache-assisted
  benchmark.
- **`payload` read each entry twice**: once to hash it, once to return it.

`D16` fixes all three: verification and use share one buffer, `verifyOnFirstUse` is explicit and off by
default, and `SourceTiming.verifiedBytes` reports the verification share so a total can be read
correctly.

| phase | round 2 | now | change |
| --- | --- | --- | --- |
| `embed` | 1.64 s | **0.00 s** | −100% |
| `head` | 3.20 s | 1.64 s | −49% |
| `load` | 2.95 s | 1.74 s | −41% |
| `mix.read` | 6.28 s | 5.99 s | −5% |
| **total (profiled)** | **19.80 s** | **14.97 s** | **−24%** |
| wall clock | 19.9 s | **15.5 s** | |
| trace digest | `b0d382dbabf36df0…` | **identical** | |

**Two rounds together: 39.9 s → 15.0 s, a 2.67× speedup, with the same digest** — 0.127 tok/s → **0.321
tok/s** on this prefill. Both changes are performance changes with no numerical effect, which is what
I1 demands of one.

**And 3.06 GB is now the true install traffic of the forward**, rather than use-traffic dressed as a
total: 2.02 GB of expert payloads (1,109 fetches at 1,818,624 bytes) plus ~1.04 GB of dense weights,
the head included. Nothing runs uncounted, so the earlier `Q9` arithmetic stands on a number that now
means what it says.

**What is left in the 14.97 s**, none of it I/O: the expert fetch is 5.99 s — a genuine 2.99 s of
reads at ~1 GB/s and 3.45 s of unpacking at ~1,012 M values/s, both at their measured limits — and
`attn.core` is **3.86 s**, the largest single compute phase, followed by `load` 1.74 s, `head` 1.64 s
and the mixture matmuls 1.50 s. The attention core is the next target (`DC-107`).

## Running the sweep

The sweep is a script, not a sequence typed from memory, because it is a heavy run on a node that has
panicked twice while doing heavy things:

```bash
python3 tools/run_m1_sweep.py \
    --snapshot .build/m1-install --tokens 1,2,3,4,5 \
    --binary .build/release/datacenter-trace \
    --slots 2,8,16 --out .build/m1-sweep

python3 tools/run_m1_sweep.py ... --dry-run     # the plan and the preconditions, nothing executed
```

**It refuses before it starts**, on the two conditions that preceded both panics:

- `tools/check_disk_headroom.py`'s 5 GB floor, consulted through `require_headroom`, which also refuses
  when the watchdog has left a stop marker — a run that tripped the limit must not resume by itself;
- **swap already in use** above `--swap-used-limit-gb` (default 1.0), because a machine well into swap
  is not the machine to start a 35 B run on. Reported always, so the number is in the log either way.

Each bank size is a separate process (`SHARD_EXPERT_SLOTS=<n>`), which is what a sweep needs, and the
per-run `metrics.json` is collected into `.build/m1-sweep/sweep.json` with a printed table. The metrics
file's schema is **read, not assumed**: the expert figures are extracted wherever they appear, in
document order — a LIFO walk reversed "the last layer's figure wins" and its own test caught that. Seven
tests cover the refusals, the disk-floor consultation, the command shape against the CLI's own usage
line, the dry run executing nothing, and the metrics reading.
