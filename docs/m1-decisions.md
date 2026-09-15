# M1 decisions

The decisions M1 rests on, with the evidence that forced each. `m0-decisions.md` holds M0's
`D1`–`D7`; the numbering continues here.

## D9 — The fast matmul must be bit-identical, so the contract's order is the specification

**Decision.** `orderedMatmul` uses a formulation that vectorises across the **output** dimension and
keeps each lane's accumulation over `k` ascending, one rounding per multiply and one per add. It is
adopted because it is **bit-identical** to the scalar contract, not because it is close. The scalar
body is retained by name (`orderedMatmulScalar`) as the definition, and a test compares the two on
more than a hundred shapes. **Fused multiply-add and therefore BLAS are excluded**, on measurement.

**Why this needed deciding before any kernel work.** `I1` and `I2` are defined by the *rounding
sequence*, not by the algebra: two formulations that agree to within any tolerance can still produce
different output bytes, and a sharded run has to match a single-node run exactly. So the question for
every kernel is not "is it faster" but "does it round in the same order".

**The measurements** (this node, `Swift 6.4`, release, 256×256×256 and a synthetic int4 payload):

| what | number |
| --- | --- |
| vector vs scalar matmul | **bit-identical** on every shape tried, including `out` not a multiple of four and odd `k` |
| the whole engine after the swap | 86 tests green, golden and contract comparisons included — end-to-end bit-identity |
| scalar matmul | 4.1 GFLOP/s |
| vector matmul | **6.8 GFLOP/s — a 1.66× speedup**, free because the bits are unchanged |
| `Float.addingProduct` against the ordered sum | **177 of 256 dot products differ in the last bit** |
| `dequantizeInt4` | 370.9 M values/s, so one 35 B token's ~3.45 B values cost **~9.3 s of unpacking** |

The FMA row is the decisive one: an FMA is one rounding instead of two, it differs from the contract
on **69 % of inputs**, and every BLAS uses it. `cblas_sgemm` is therefore not available for any op the
gate compares — not "would need care", *cannot*. The scalar rate is also the answer to "is this
matmul-bound": 6.9 GFLOP of matmul per token is about 1.7 s at 4.1 GFLOP/s, against a measured 52.2 s
per step, so the matmul is not where the time is going.

**A prediction, labelled as one.** The 52.2 s/step was measured *before* the row-read fix in
`DC-033`. Forty layers of whole-stack decoding is 40 × 537 M values ≈ 21.5 G values, which at the
measured 370.9 M values/s is ≈ 58 s — the same number by a different route. With eight of 256 experts
fetched per layer it becomes ≈ 0.67 G values ≈ 1.8 s. So that fix, made for memory, should also be
worth most of an order of magnitude in throughput. **This is not verified and will not be claimed as
a result until a real-model run measures it**, which needs the operator's approval.

**Consequence for the remaining `DC-033` work.** The next kernels are the int4 unpack and the expert
fetch path, not the matmul — and the unpack has to preserve the same rounding sequence, which is a
tighter constraint than a GEMM kernel faces.

### The unpack, which was the easier kernel and the bigger number

`D9`'s measurement put the unpack at 370.9 M values/s against the matmul's 4.1 GFLOP/s, so the
unpack is where a token's time goes. It turned out to be a **much easier** kernel than the matmul,
for a structural reason: its only floating-point operation is one multiply, `Float(code - zero) *
scale`, with **no summation anywhere**. There is no accumulation order to preserve, so a vector
formulation is bit-identical by construction rather than by luck — `Float(code - zero)` is exact
because the codes and the zero point are small integers, and `SIMD4<Float> * scale` rounds once per
lane exactly as the scalar multiply does. Contrast with `cblas_sgemm`, which cannot be used at all
because it fuses.

The win comes from elsewhere: the scale and zero point are per **group** (sixty-four values), and
the scalar loop reloaded both for every element.

| | scalar | vector |
| --- | --- | --- |
| unpack rate | 648.3 M values/s | **1038.3 M values/s — 1.60×** |
| one 35 B token (3.45 G values) | 5.3 s | **3.3 s** |

Both are bit-identical on a grid of sixty shapes — `columns` not a multiple of four, the padded
tail, a group of one, a single row — and on the end-to-end golden tests.

**Two notes for whoever comes next.** The remaining factor is nibble extraction: four values per
iteration is one 128-bit register, so widening to eight nibbles at a time with integer lanes is the
obvious next step and is **not measured here**. And the grid above cost an hour to a Swift footgun
worth writing down: **Swift's `%` keeps the sign of the dividend**, so `(-5) % 4` is `-1` and
`columns + that` can be *smaller* than `columns` — the test compared the two implementations on an
impossible layout and very nearly sent me hunting a bug in the wrong function.

**The combined prediction, still a prediction.** The row-read fix divides the values unpacked per
token by thirty-two (eight of 256 experts), and this divides the rate by 1.60, so the unpack should
fall from ~9.3 s to ~0.6 s per token. That, plus the 21.5 G-value → 0.67 G-value arithmetic behind
it, is the basis of the "most of an order of magnitude" prediction. **Nothing about the real model
has been re-measured, and it needs the operator's approval to be.**

## D8 — The chunked Gated DeltaNet rule is authoritative, and a cache is a second numeric path

**Decided:** the cache's decode arithmetic is built so that it agrees with the **chunked** rule,
which is what prefill uses and what M0 verified against the reference.

**Why it needed deciding.** `Qwen3_5MoeGatedDeltaNet.forward:625` switches functions:

```python
if use_precomputed_states and seq_len == 1:
    torch_recurrent_gated_delta_rule(...)   # decode
else:
    torch_chunk_gated_delta_rule(...)       # prefill
```

Test-first work on the decode path measured the two functions against each other on identical
inputs in the pinned reference (v5.17.0):

| comparison | max abs / scale |
| --- | --- |
| our recurrent rule vs our chunked rule | **3.2e-07** |
| the reference's recurrent vs **the reference's own chunked** | **3.4** |

Our chunked rule is the one M0 validated against the reference's chunked function, so the
disagreement is between the reference's two functions rather than between ours. A 3.4 relative
difference is a complete divergence, not rounding, and the working agreement is explicit that
this is a thing to ask about rather than guess past.

**The answer taken:** the chunked path is authoritative.

**What follows.**

- The cache is a **second numeric path**: algebraically equivalent to the uncached path and, on
  the measurement above, numerically equivalent to about 1e-7 relative — not bit-identical,
  because the chunk grouping differs.
- **M1's bit-identity claim stays where it was measured**: the uncached path against the
  contract, 83 tensors and 40 discrete decisions with matching digests.
- **The cached path is validated differently**: against the uncached engine at the agreed
  protocol, with the **discrete decisions asserted exactly** — I3 does not relax because the path
  changed.
- **I1 is not weakened.** The same prompt in the same mode is still deterministic. What I1 never
  promised, and what the reference itself does not deliver, is that two different numeric paths
  agree.

**Recorded because:** a cache built by assumption would have taken the reference's decode path as
the definition of correctness, and that path contradicts the reference's own prefill by 3.4
relative. The discrepancy is unreconciled in the reference and is noted as such in
`reference-qwen36-35b-a3b.md`; `tools/test_ordered_gdn_recurrent.py` carries the two measurements
as a passing test and an expected failure so the marker removes itself if the discrepancy turns
out to be ours.

## The decode path, implemented under `D8`

`GatedDeltaNet.decodeStep` carries a `State` — the convolution's window (the last `kernel - 1`
raw projections per channel, which `causal_conv1d_update:252` concatenates onto the new input)
and the recurrent state `[heads, keyHeadDim, valueHeadDim]`. It is checked against the layer's
**sequence** path on the golden vectors, including the asymmetric two-key-heads-to-four case,
at a tolerance rather than at the bit — which is `D8`'s consequence stated as a test:

| | |
| --- | --- |
| decode step vs sequence path, long case | passes at 1e-5 relative |
| decode step vs sequence path, asymmetric heads | passes at 1e-5 relative |
| the sequence path vs the contract | **bit-identical** (unchanged, and still asserted) |

What is not yet wired: a prefill that leaves the states behind, and the attention layers' KV
cache. The unit is verified before either, because the wiring is where a state can be threaded
into the wrong layer or the wrong batch element and still produce plausible text.

## The cache: the bug, and what it was

The cached path diverged from the uncached one on the real model from the second token, while the
reference's own two paths agreed on the fixture. The margins said it was a defect rather than a
`D8` difference: **both paths were confident in different tokens**, which means their logits
differed by more than the top-2 gap — of the order of one, not of rounding.

Isolation, in the order the evidence arrived:

| check | result |
| --- | --- |
| `GatedDeltaNet.decodeStep` vs the sequence path, fixture | **1.6e-06 relative** — rounding, not the cause |
| cached attention vs sequence attention, fixture | **6.1e-03** — the defect, and the test that found it |
| after the fix | **0.000e+00**, bit-identical |
| real model, cached vs uncached, 4 steps | tokens **and** margins identical (`11751,11,264,3177`; `1.6400, 0.0972, 1.2532, 2.6414`) |

**The bug:** `Ops.orderedMatmul` takes its weight as `[out, k]` — the layout of a
`Linear.weight` — and `attentionStep` built both of its weight matrices as `[k, out]`. Every
shape was right, every number was plausible, and the pairs being multiplied were the wrong ones.
The sequence path was passing the key head *as stored* and the value head *transposed*, which is
what gave the layout away.

**And one measurement corrected a claim:** for a prompt inside one 64-position chunk the cached
replay is **bit-identical** to the chunked prefill, because the chunked rule *is* the recurrence
there. `D8`'s divergence needs a second chunk — 3.2e-07 relative over seventy positions. So the
honest statement of `D8` is narrower than it first read: **within a chunk the paths coincide
exactly; across chunks they agree to rounding.**

## The cache: implemented and measured

`sources/DatacenterEngine/ModelCache.swift` decodes one position per token against a per-layer
state — the Gated DeltaNet's window and recurrence, and the full-attention layers' keys and
values — with `GatedDeltaNet.decodeStep` doing the recurrence and `attentionStep` doing the cached
attention. `datacenter-generate --cached` measures it.

| | |
| --- | --- |
| Cached decode, real 35 B model, 4 steps | **66.1 s (16.5 s/step)** |
| Uncached, same prompt and steps | **208.9 s (52.2 s/step)** |
| Speedup at a 5-token prompt | **3.2×**, and it grows with context because the uncached path re-runs the sequence |
| Tiny fixture: cached vs uncached tokens | **identical** (5 steps) |
| Tiny fixture: router decisions, cached vs uncached | **2 of 2 layers exactly** |
| Tiny fixture: replay vs chunked prefill logits | **2.2e-03 relative** |
| Reference, cached vs uncached greedy, same fixture | **identical** (8 steps) |

The bug above was found and fixed, and after it the real model agrees on **tokens and margins**.
The cache is therefore trustworthy as an optimisation. It remains a second numeric path by `D8`,
so M1's gate continues to rest on the uncached path — which is bit-identical to the contract — and
the cached path is checked against it, with the router's decisions compared exactly.

## Reading the install is itself a hazard on a 4.5 GB node

Verifying the 20 GB install — a sequential read plus a sha256 over every entry — drove free disk
from **17 GB to 2.96 GB in about thirty seconds**. The mechanism is a loop, and every step of it
is ordinary: the read fills the page cache, the page cache fills memory, memory pressure makes
macOS grow swap (one gigabyte per swapfile in `/System/Volumes/VM/`, transiently about fourteen
gigabytes), and swap is disk. The debounced disk watchdog stopped the job on the third
consecutive below-floor reading; macOS shrank the swap back to three gigabytes once the pressure
went away, and free space returned to fourteen.

This is the brief's runtime I/O rule, measured rather than taken on faith:

> Expert slabs: `F_NOCACHE` / `O_DIRECT`, async worker pool, per-layer LRU slot banks …

On a node this small that rule is **not a throughput optimisation, it is a stability
requirement**. A page-cached read of a model is what turns "reading the weights" into "exhausting
swap", and swap exhaustion is precisely what panicked this machine twice — `watchdog timeout: no
checkins from watchdogd in 90 seconds`, with thirteen swapfiles and LOW swap space.

Two consequences:

- **The install's byte-level verification is outstanding, not passed.** It needs an uncached
  reader or a machine with headroom, and it is now `DC-086` rather than a claim.
- The engine's reads have the right *shape* — one tensor at a time, never the model — and that is
  not sufficient. `sources/DatacenterEngine/` contains no `F_NOCACHE` and no `fcntl` at all, so
  every one of those reads is page-cached today. The fix belongs in the file handle the provider
  opens, which is `DC-033`'s neighbourhood.

## The fix: uncached reads for the install, and verification moved to the read

`InstallFile` did two things that were each, on this node, a hazard. It **memory-mapped the whole
payload** (`Data(contentsOf:, options: [.mappedIfSafe])`), so every read populated the page cache;
and it **hashed all twenty gigabytes on every open**, so simply starting the engine was a
full-payload read. Neither is visible in the arithmetic, and together they are the loop that
took free disk from 17 GB to 2.96 GB in half a minute.

Both are gone. `UncachedFile` opens the payload with `O_RDONLY` and asks for `F_NOCACHE`, reads
byte ranges with `pread` — not a seek plus a read, so two readers cannot move each other's offset
— and loops on short reads. `InstallFile` keeps that one descriptor instead of a mapping.

**Verification moved from the open to the read, and `I6` is not weakened by it.** Each payload's
digest is checked the first time that payload is read, and remembered; a tampered tensor
therefore still cannot produce plausible numbers, which is the property `I6` asks for. What
changed is *when* the check costs something: a tensor nobody reads costs nothing, and opening a
20 GB install is no longer a 20 GB read. `InstallFile(url:verify: true)` and `verifyAll()`
restore the eager whole-payload check for a gate that wants it stated explicitly.

Six tests cover the reader: uncached and mapped reads return the same bytes, a windowed read
matches the same window of the whole file, reading past the end is an error rather than zeros,
the streaming digest matches the one-shot digest, the cached mode is available for files whose
pages are worth keeping, and a missing file reports an open failure. Four more cover the moved
verification: a tampered tensor opens fine and throws **when read**, an untouched one still
reads, `verify: true` catches it at open, and an intact install passes `verifyAll()`.

**What is not done, stated plainly:** the *checkpoint* reader (`SafetensorsFile`) still
memory-maps its shard, so the streaming path over a safetensors snapshot is page-cached exactly
as the install was. That is the remaining work on `DC-086`, and the byte-level verification of a
20 GB install on this node is still outstanding — it needs a machine with headroom, or the
checkpoint reader fixed first.

## `DC-086` closed: the measurement

The checkpoint reader now has the same treatment as the install. `SafetensorsFile.rowsStreaming`
reads a row range through `UncachedFile` while `float32(_:rows:)` keeps the mapping, and the split
is by **consumer** rather than by tensor: the routed expert slabs stream (they are read once per
token and would evict everything useful) and the embedding and head stay cached (they are read
every token and are exactly what the cache is for). All three read paths now share one decoder, so
they cannot drift apart.

The Python tooling got the same fix, because the verification that caused the incident was a
Python read: `open_uncached`, `pread_exact` and `digest_of` replace `read_bytes()`, which on a
20 GB install is twenty gigabytes **resident**, not merely cached.

The measurement the task asked for:

| | before | after |
| --- | --- | --- |
| free disk during a full 20 GB verification | 17 GB → **2.96 GB** | steady at **16 GB** |
| peak memory for the same verification | ~20 GB resident (`read_bytes`) | **33.7 MB** |
| how long the check takes | a full read on every open | once per payload, on first read |

So: a 20 GB install verifies on an 8 GB node without free disk crossing the floor, which is what
`DC-086` said would close it. It is closed.

## The row rule, and the fourth time it bit

`InstallFile.rows(named:range:)` took a range in **leading-axis entries** and, for an `int4`
tensor, passed it straight to the payload arithmetic as though it were a row index. For a rank-3
stack those are different numbers: one expert spans `shape[1]` payload rows, so `range 1..<2` read
the first row of expert 1 rather than expert 1. The fixture reported *"expert 0 of
gate_up_proj has 32 values, expected 1024"* — every shape plausible, every byte in range.

The rule, which this project has now learned four times and should stop learning:

> **A row is one leading-axis entry.** For a `[experts, rows, columns]` stack, one row is one
> expert. Any arithmetic that touches the payload's flattened row axis must translate first, by
> the rows-per-entry factor.

It has bitten both contract readers (each decoded a stacked expert tensor as flat rows), the
install builder's chunking (which asked the reader for payload rows where the reader offers
leading-axis entries), and now the row read itself.

The change that came with the fix is the one `DC-033` needed: the payload is section-major, so a
row range is **three bounded reads and one decode** rather than a whole-stack decode. Measured on
the fixture, one expert of an eight-expert stack costs **1184 of 9472 bytes** — exactly its share,
against eight times that before.

One cost is deliberate and worth stating: the first read of an entry also hashes its whole payload,
which is `I6`'s price and is why the measurement warms the entry first. For the real model that is
537 MB once per expert tensor per process. A per-slab digest in the format would remove it, and
that is a format change, so it is not in M1.
