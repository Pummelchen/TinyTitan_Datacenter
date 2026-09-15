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
