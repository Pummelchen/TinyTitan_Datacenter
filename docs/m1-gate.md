# The M1 gate

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

| bank size | trace digest | wall clock | bytes from SSD | requests | hits | decoded in memory |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | `b0d382db…` | 39.5 s | 6,977,224,704 | 2,218 | **0** | 13,954,449,408 |
| 8 | `b0d382db…` | 39.3 s | 6,977,224,704 | 2,218 | **0** | 13,954,449,408 |
| 16 | `b0d382db…` | 39.9 s | 6,977,224,704 | 2,218 | **0** | 13,954,449,408 |

**The traces are byte-identical, checked independently rather than inferred from the digests:**
`trace_diff` reports *83 tensors, 0 differing elements, 40 discrete decisions* and matching digests. So
the bank's size cannot change what the engine outputs, which is the least a cache has to be — and it is
the same property M2 will need from sharding, on the same harness.

**What the numbers say per token:** 1,395 MB read from SSD per prefill token; 13.95 GB decoded into
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
