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

**What `D11` changes about `D10`.** `D10` said Metal flushes denormal operands and the CPU does not,
so a GPU kernel could not be bit-identical for free. With the flush now **defined in the contract**,
that objection is gone: re-checking `DC-087` against the flushed CPU shows the GPU and CPU agreeing on
**every value** — 0 differences out of 130 — on the partly-filled groups that exposed the divergence.
So a GPU kernel is not ruled out by denormals, and the remaining question about Metal is the measured
**1.30×** against a CPU path that is already bit-identical, not correctness. A separate divergence
remains on some `group = 1` shapes, undiagnosed and recorded as such.

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

**The operator chose (1), and it is implemented** — four sites, and the placement is the whole
difficulty:

| where | what |
| --- | --- |
| `tools/quantize.py`, `flush_denormals` | the definition, one function |
| `quantize_rows` | flushes the scale that gets **stored** |
| `dequantize_tensor` (Python reader) | so an install that already carries denormals reads the same |
| `Install.swift`, `flushed` | one helper, used by the vector **and** the scalar unpacker |

`quantize_rows` is the interesting one: packing divides by the scale and the zero point is derived
from it, so flushing early divides by zero — the Python suite failed on exactly that on the first
attempt. The flush belongs on the value that finally gets stored, after the codes are built.

Proven by tests the fixture could never provide, because it contains no denormals at all: a crafted
denormal scale decodes to **zero** on both Swift paths and in the Python reader, the two agree
bit-for-bit, normal scales keep their value and their sign, and the quantizer stores zero.

**One consequence to state rather than discover.** The real install's stored scales still contain
denormals — they are not rewritten — but every reader now flushes them, so the engine and the
contract still agree with each other while both differ from the traces recorded before this change.
**`capital`'s digest `b8c976c5e7ba8816…` is therefore historical**, and re-establishing the
contract comparison on the real model needs another run of the kind the operator authorised. It was recommended because the flush is a *definition* rather than an error, it is one
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

## D12 — the slot bank's size, which the brief leaves open

Found by auditing `DC-091`'s own fix one round after making it, and it is a good argument for the audit
step existing.

`Qwen3_5Forward.expertSlotsPerLayer = 16`, and the brief asks for "per-layer LRU slot banks" without
saying how large. Making those banks persist across tokens — which is what the hit rate needs — turns a
**per-layer** budget into a **total** one, and the total does not fit:

| quantity | value |
| --- | --- |
| one expert, gate+up | 8.39 MB |
| one expert, down | 4.19 MB |
| **one expert** | **12.58 MB** of fp32 |
| 16 slots, one layer | **201 MB** |
| 16 slots × 40 layers | **8.05 GB** |
| usable RAM per node | **~4.5 GB** |

The geometry is the checkpoint's, not a recollection: `hidden_size` 2048, `moe_intermediate_size`
512, `num_hidden_layers` 40 with **every** layer a mixture, `num_experts` 256, `num_experts_per_tok`
8. Source: [`Qwen/Qwen3.6-35B-A3B` `config.json`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/raw/main/config.json)
(`transformers_version` 4.57.1) — and the same figures were already in this repository's
`docs/reference-qwen36-35b-a3b.md`. The first version of this table used numbers from memory
(`768`, `48`, `128`) that are **all wrong**, which is why the rule about not inventing numbers exists
and why the arithmetic here is now cited rather than remembered. The conclusion is unchanged and in
fact sharper: 8.05 GB still does not fit in 4.5 GB.

So the persistent-bank change was reverted rather than shipped: on this node it would have swapped
during the very real-model run it was meant to improve, and the fixture cannot see it because the
fixture's experts are kilobytes. `97` Swift tests are green again, at the pre-change count.

**The correct shape of the fix is a total budget, not a per-layer count.** A 1.5 GB cache across 40
layers allows **2.98 slots per layer** — a bank of one or two, not sixteen — and the split between
"more layers" and "more slots per layer" is a decision with a measurable trade: slots buy cross-token
hits, layers buy nothing at all if the bank is dropped. With top-k of 8 against 256 experts, a
one-slot-per-layer bank will still hit rarely, so the honest options are:

1. **size the bank from a total budget** (1–2 slots per layer), accept a low hit rate, and rely on the
   measured uncached read bandwidth — which is the bottleneck the brief names anyway;
2. **weight the budget towards the layers where routing concentrates**, since the router is not
   uniform and the first layers were where the denormal scales concentrated too;
3. **keep banks only for a window of layers** (the per-layer LRU evicts whole layers), which bounds
   memory but only hits when consecutive tokens reuse a layer's expert, which they do not.

**The constraint is now a test, not a paragraph.** `tests/DatacenterEngineTests/SlotBudgetTests.swift`
does this arithmetic on the checkpoint's geometry and asserts the total against the brief's own
hardware limit, `~4.5 GB` usable per node. It fails today — `8053063680` against `4500000000`, which
XCTest reports in full — and it is marked `XCTExpectFailure` with `D12`'s reasoning, so that the number
is visible to anybody who runs the suite without reading this page, the suite stays honestly green
(a known-unmet constraint recorded as met would be a lie), and **the day the bank is sized from a
budget that test reports an unexpected pass**, which forces the marker to be removed deliberately
instead of quietly left behind.

This is open, and it is the operator's call because it trades RAM against hit rate on a node whose RAM
limit has already caused two panics.

## D13 — 16 KB alignment is deferred with `O_DIRECT`, and the measurement says why

The brief's L3 requires experts repacked "contiguous as [gate|up|down], **16 KB aligned**", and the
brief's runtime rule names the read path as `F_NOCACHE` / **`O_DIRECT`**. Those two sentences are one
requirement: **`O_DIRECT` is what needs 16 KB alignment**, because it requires the buffer, the length
and the file offset all to be aligned to the device's block size. `F_NOCACHE` + `pread` does not: it
bypasses the page cache and reads at any offset.

This engine uses `F_NOCACHE` + `pread` (`UncachedFile.swift`), and `O_DIRECT` appears **exactly once**
in the repository — in a comment quoting the brief. So the alignment requirement is currently
unnecessary, and it is deferred *with the read path* rather than dropped.

**Measured on the real 20 GB install, 693 tensors, `install.json`:**

| alignment | offsets that fail it |
| --- | --- |
| 2 bytes | 0 |
| **64 bytes** | **0** |
| 512 bytes | 375 |
| 4096 bytes | 667 |
| **16384 bytes** | **688** |

So the layout is **64-byte aligned**, which is what `SIMD4<Float>` loads and a `pread` want, and the
gaps between consecutive tensors run 128, 512, 1024, 4096, 8192 and 65536 bytes. That is by design, not
by accident: the writer packs tensors to a small alignment and the reader loads unaligned words
explicitly (`loadUnaligned`), which is why the bit-exactness work never had to care.

**The revisit condition is a single measurement**: if the read path moves to `O_DIRECT` — for instance
if `F_NOCACHE` turns out not to bypass the cache on a future macOS, which is exactly what the 20 GB
page-cache incident would look like — then the install must be rewritten with 16 KB alignment and every
trace regenerated, because the offsets change. Writing that down is what keeps the requirement from
being quietly forgotten.

## D14 — I6 is half-implemented, and the missing half is cheap to close

Auditing the brief's **I6** against the artifact, rather than against the intent, found two of its four
requirements unmet — in the real install **and** in both committed fixtures, so it is the writer and not
one build:

| I6 requires | what the artifact has |
| --- | --- |
| source repo **+ commit** | `repo` holds either a commit hash (real install) or a *name* (`"tiny-qwen36"`), and `revision` is always **`"local"`** — so the source commit is not recorded, and one field is carrying two meanings |
| sha256 of **each source weight file** | `source.files` is **`{}`** — empty, always. The 693 per-tensor digests are of the *converted payloads*, which is a different claim |
| the ordered list of transform passes | `passes: ["quantize-group-affine-int4"]` — present |
| the policy files used | `policy_files: ["tools/quant_policy.json"]` — present, and relative in the fixtures |

**What is not the problem.** No absolute path is committed: `git grep "/Users/"` over tracked files
returns nothing, the fixtures carry `["tools/quant_policy.json"]`, and the `/Users/...` path seen in
`.build/m1-install/install.json` is in a gitignored build artifact only. And the per-tensor digests, the
spec, the family and the pass list are all present, which is most of what makes an artifact
self-describing.

**Why this is worth closing rather than noting.** I6 exists so that a converted model can be traced to
the weights it came from. An artifact that says `revision: "local"` and lists no source files cannot
answer "which checkpoint is this", which is the one question a provenance header is for — and the
failure is silent, because every other field looks complete.

**The cost is measured, not estimated.** The remedy is to hash the source safetensors and record the
commit. The source checkpoints are far larger than RAM, but this project already has the tool for that:
`UncachedFile` / `open_uncached` with `pread`, which verified the 20 GB install in **~20 s at a 34 MB
peak** with free disk steady. Hashing source files is the same shape of work. **`data.bin` does not
change**, so no install must be rebuilt and no trace regenerated: only the manifest gains fields, which
makes this a safe change to batch with the next real-model run rather than a reason for one.

## D15 — Per-read slab verification is opt-in, because it cost 21 s of a 40 s forward

`DC-088` added per-slab SHA-256 verification to the row-range read so a tampered install is caught
before it can become plausible numbers. It ran on **every** read. A phase profile of a real forward
(`SHARD_PROFILE=1`) then measured what that cost:

| phase | seconds | share of the forward |
| --- | --- | --- |
| the expert fetch (`mix.read`) | 25.53 | 64% |
| — of which **digest verification** | **21.04** | **53%** |
| — of which unpacking | 4.06 | 10% |
| — of which the actual reads | 0.82 | 2% |

**The integrity check was over half the model's runtime, and the disk was 2% of it.** The decision:
`InstallFile(url:verifySlabs:)` is a parameter, **false by default**, and a caller who wants
verification asks for it. Integrity is established out of band — the manifest carries a digest per
tensor *and* per slab, and `tools/quantize.py verify` checks the whole install once, uncached, at a
34 MB peak (measured). Re-hashing every payload the model reads is not a stronger guarantee than
checking the install once; it is the same guarantee at 53% of the runtime.

The trap this avoids in the other direction: with verification off, falling through to
`digestMatches(entry)` would read and hash the **whole** entry — 310 MB — to answer a question about
one expert, which is `DC-088`'s bug in reverse. The gated block falls through to nothing, not to the
whole-entry check.

**The cost is visible and tested rather than asserted.** `InstallFile.sourceTiming` reports bytes,
read seconds, digest seconds and unpack seconds; a test pins that the default reader records
**exactly zero** digest seconds and that a caller who asks records more than zero. Writing that test
found a flaw in the instrument itself: the clock had started outside the branch, so the default
"cost nothing" reading was 4.2e-08 s of the branch test rather than zero.

Verified on the real 35 B model: the same prompt, **39.9 s → 19.9 s**, `mix.read` 25.53 s → 6.28 s,
digest seconds 0, and the trace digest **unchanged** (`b0d382dbabf36df0…`) — the option is a
performance change with no numerical effect, which is what I1 demands of one.
