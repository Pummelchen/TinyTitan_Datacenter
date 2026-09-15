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

| | scalar | four-wide | eight-wide |
| --- | --- | --- | --- |
| unpack rate | 648.3 M values/s | 1038.3 M values/s (1.60×) | **1185.5 M values/s — 1.83×** |
| one 35 B token (3.45 G values) | 5.3 s | 3.3 s | **2.9 s** |

Eight codes per load is worth a further 14 %, and it is guarded: a wider block must not straddle
two groups, because the scale and the zero point change at the boundary, so the wide path runs only
when the group is a multiple of eight and the four-wide loop remains the general case.

Both are bit-identical on a grid of sixty shapes — `columns` not a multiple of four, the padded
tail, a group of one, a single row — and on the end-to-end golden tests.

**Two notes for whoever comes next.** The eight-wide path above is the nibble extraction, measured
and adopted. What is left in Swift is thinner: the widest sensible FP lane here is 128 bits, and
Swift offers no cheap widening from integer lanes, so the next real factor is **Metal** rather than
more of this. And the grid above cost an hour to a Swift footgun
worth writing down: **Swift's `%` keeps the sign of the dividend**, so `(-5) % 4` is `-1` and
`columns + that` can be *smaller* than `columns` — the test compared the two implementations on an
impossible layout and very nearly sent me hunting a bug in the wrong function.

**The combined prediction, still a prediction.** The row-read fix divides the values unpacked per
token by thirty-two (eight of 256 experts), and this divides the rate by 1.60, so the unpack should
fall from ~9.3 s to ~0.6 s per token. That, plus the 21.5 G-value → 0.67 G-value arithmetic behind
it, is the basis of the "most of an order of magnitude" prediction. **Nothing about the real model
has been re-measured, and it needs the operator's approval to be.**

## D10 — What a Metal kernel may do to the bits, and the fast-math trap

**The decision, before any kernel is written.** `D9` established that the contract is a rounding
sequence, so a kernel is admissible only if it reproduces that sequence. Generalised for the GPU:

> **One accumulator per output, `k` ascending.** A kernel may parallelise across outputs, across
> rows, across layers — but not across `k`, and it may not reassociate.

That constraint is narrower than it sounds and it does **not** exclude the GPU. A one-thread-per-output
kernel satisfies it trivially, and so does a shared-memory **tiled** kernel: a tile contributes a
contiguous run of `k` values, so as long as each output's accumulator takes the tiles in order and
the values inside a tile in order, the sequence is the scalar one. What is excluded is split-K and
any reassociation — which is what a fast GEMM does *because* it is fast.

**The trap, and it is a real one.** Metal compiles shaders with **fast math enabled by default**,
and fast math is precisely the licence to reassociate and to contract `a * b + c` into an FMA. A
kernel that looks identical to the Swift one, with identical source arithmetic, can therefore
produce different bits by default. Every kernel must be compiled with `fastMathEnabled = false`, and
every kernel must be **checked against the scalar op bit for bit**, not merely against a tolerance —
which is the same discipline `D9` used to rule BLAS out.

**Measured on the development node (Apple M2):**

| | |
| --- | --- |
| `MTLCreateSystemDefaultDevice()` | returns an **Apple M2**, `hasUnifiedMemory = true` |
| runtime-compiled MSL from a source string | **compiles** (`library.functionNames == ["unpack4"]`) |

**Consequence for CI, and it is a gate rule rather than a detail.** The GitHub `macos-26` runner has
no GPU, so `MTLCreateSystemDefaultDevice()` returns nil there. Every Metal test must therefore
**skip** when there is no device — the same shape as the Swift-6.4 skip in `DC-036`, which has
already been bitten twice by a toolchain difference between CI and the farm. This is that lesson
applied before it costs anything.

**The first kernel is the unpack**, and for the reason `D9` gives: it is element-wise, its only
floating-point operation is one multiply, and it has no summation at all, so the accumulation rule
does not even apply to it. It is the Metal kernel least able to argue with the contract, which makes
it the right one to prove the toolchain, the test harness and the fast-math discipline on.

## D11 — The denormal rate, measured, and the two options it leaves

`DC-087` established that Metal flushes denormal operands while the CPU does not. The question was
how often that matters — so `tools/measure_denormals.py` reads the **scale sections** of an install,
uncached, and counts.

**On the real 35 B install** (`.build/m1-install`, 210 quantized tensors, 523,304,960 scales read in
2.13 s with free disk steady at 17 GB, which is the uncached path doing its job):

| | |
| --- | --- |
| scales | 523,304,960 |
| **denormal** | **3,781,952 — 0.722705 %** |
| zero | 0 |
| non-finite | 0 |

And they are **not spread evenly**: layer 0's `gate_up_proj` has 3,472,448 of its 8,388,608 scales
denormal — 41 % of that one tensor — while layer 10 has 64 and most layers are near zero. The first
layers of a quantized model are where groups of near-zero weights live.

**0.72 % is not a rounding detail when the gate compares every value.** It is roughly four million
differing weights per pass, and it is enough to decide the Metal question rather than defer it.
There are two options and no third:

1. **Flush denormals in the contract.** Round denormal scales to zero in `tools/quantize.py` and in
   the engine's unpack, so both sides agree because the flush is *defined* rather than discovered.
   The values lost are ~1e-38 against weights of order 1e-1, invisible at any tolerance; `I3` is
   untouched because routers and gating are bf16 and never quantized. The cost is that the install
   and every comparison against the oracle must be re-measured, and the M1 bit-identity claim is
   against a *redefined* contract.
2. **Keep Metal off anything downstream of a quantized weight.** A flushed scale makes the unpacked
   weight zero where the CPU makes it denormal, and that propagates into every matmul, so this is
   not "the unpack stays on the CPU" — it is the **entire expert path** staying on the CPU, which is
   the hot path. Metal's role would be reduced to the dense, unquantized ops.

**The operator chose (1)**, so the flush is now part of the contract rather than a recommendation, and the implementation notes live in `DC-089`. It was recommended because the flush is a *definition* rather than an error, it is one
line in each of the two implementations, and it is testable exactly the way everything else in this
project is — assert the flushed behaviour on both sides and re-run the oracle comparison with the
new error budget recorded. But it changes what "bit-identical" means, so it is a renegotiation
rather than a fix, and it is recorded here for a human decision rather than taken unilaterally.

## The GPU flushes denormals, which is a decision and not a bug

`DC-087` began as "the Metal unpack disagrees with the CPU" and ends as a constraint on every kernel
this project will write.

The hunt, in order, because two of my three guesses were wrong:

| guess | verdict |
| --- | --- |
| a partly filled final group | **wrong** — a diagnostic grid of twenty-four shapes showed full groups failing too |
| the group-index arithmetic | **wrong** — it agrees with the CPU on 127 of 128 values in the shape that fails |
| denormal operands | **right** |

In the one failing shape, `columns = 64, group = 1, rows = 2`, exactly **one value of 128** differs:
index 117, where the GPU returns `0.0` and the CPU returns `2.6e-37`. That value's *scale* is a
**denormal** — the CPU computes `(5 - 41) * scale` with `scale ≈ -7e-39`, below the fp32 normal
floor — and **Metal flushes denormal operands to zero**. Every other value in the shape, and every
value in the other twenty-three shapes, is normal and bit-identical.

The consequence is not about the unpack. **Metal cannot reproduce the fp32 contract bit-for-bit
wherever an operand is denormal**, and quantized scales are exactly the kind of quantity that lands
there. `I1` and `I2` are defined on output bytes, so this is a decision to make rather than a defect
to fix:

- **Either** the contract flushes denormals on the CPU as well, which makes the two agree by
  defining the flush into the contract — a renegotiation, needing the oracle re-validated against
  it and its own measurement;
- **or** the GPU is used only for ops where denormals cannot arise, which for a faithful port of
  fp32 arithmetic is close to nowhere, and Metal's role becomes the element-wise ops whose
  *operands* are known normal.

Until that is decided the kernel is not called by anything. The task's Done-when has been rewritten
to the achievable claim rather than quietly dropped, and the diagnostic that found this stays in the
suite — "which shapes disagree" is the first question about the next kernel too.

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

## The leading-axis trap, five times, and where it hid the last time

This project has now been bitten five times by the same confusion, and the fifth is the instructive
one because it was not an index at all:

| # | where | what it was |
| --- | --- | --- |
| 1 | both contract readers | a stacked expert tensor read as flat rows |
| 2 | the install builder's chunking | asked the reader for payload rows where it offers leading-axis entries |
| 3 | the `int4` row read | used the entry range as a payload-row index, so "expert 1" became "the first row of expert 1" |
| 4 | Metal's unpack | the same, caught by a bit-identity test |
| 5 | **the slab-digest guard** | `range.count % inner == 0`, which for one expert of a stack is `1 % 32` — so the whole branch was skipped, silently and correctly, at full cost |

The fifth had no wrong index, no wrong value and no failing assertion: the code took the other
branch and the result was *right*. It was only visible as a **cost**, which is why it survived a
review of the arithmetic and two rounds of reading. The lesson is narrower than "mind your indices":

> A guard that decides between a fast path and a correct fallback needs a test that would **fail when
> the guard stops matching** — not merely one that passes when the answer is right.

The `bytesRead` counter on `InstallFile` is that test, and it is the reason this was found at all.

### And the sixth time came with a second face

Finishing `DC-088` produced one more form of the same confusion, worth separating from the others:

> **A global index and a range-local offset are different numbers, and both are called "the row".**

The slab loop needed the slab's **global** index to choose which digest to compare (one expert of a
stack is slab `range.lowerBound + index`) and the **range-local** payload rows to slice the buffers
it had just read (`index * inner`, because `codes` holds only the requested range). Using the global
number for both sliced a 1024-byte buffer at `1024..<2048` and trapped — and because the trap
happened before the diagnostic print, four rounds of instruments reported nothing at all.

The same file also carried **two** whole-entry digest checks on that path, one before the reads and
one after. Every attempt changed exactly one, so a whole-entry check always survived and rejected
any tamper anywhere in the entry — which made a tamper test report the *untouched* expert as bad and
sent me chasing offsets that were correct all along.
