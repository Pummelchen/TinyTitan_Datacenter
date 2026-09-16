# M2 decision records

The single-node decisions are in `m0-decisions.md`, `m1-decisions.md` and `m0c-quantization.md`. These
are the ones the cluster forces, and they are written **before** the code that depends on them, because
each one is a contract two nodes have to agree on rather than a choice one node can make.

## D17 — The deterministic reduction contract: reduce contributions, never per-node partials

**The problem.** M2's gate is "2 nodes bit-identical to the 1-node baseline". Floating-point addition is
not associative, so the moment the sum is split across nodes the low bits are at risk. Concretely, in
fp32 at magnitude 2e7 the spacing is 2:

```
a = 2e7, b = 3, c = 3
(a + b) + c = 20000008      the left-to-right sequence
a + (b + c) = 20000006      a per-node partial summed at the end
```

Both are "the sum of the same three numbers". Only one of them is the sequence the single-node engine
performs. That example is a test (`OrderedReductionTests`), not an illustration: if those two agreed,
this contract would be unnecessary.

**The decision.** The all-reduce carries **per-`(token, expert)` contributions** — the expert's `down`
projection and the routing weight — and the reduction sums them in a canonical order:

1. **Order key: token ascending, then expert id ascending.** This is exactly the sequence
   `MixtureOfExperts.experts` performs today, so an N-node run replays the 1-node arithmetic instead of
   approximating it.
2. **The routing weight is applied before the addition** (`output += values[i] * scale`), the same
   expression shape as the single-node site. A compiler that contracted one site into an FMA and not
   the other would break bit-identity invisibly; both sites must round identically.
3. **fp32 accumulation**, as the architecture requires.
4. **No per-node pre-aggregation, ever.** A node that sums its own experts and ships the total has
   already lost the property, and no downstream care can recover it.
5. **No reduction for the dense backbone.** It is replicated and computed identically on every node, so
   there is nothing to combine — which is why there is exactly **one all-reduce per MoE layer** and not
   one per layer.

**This is stronger than "a fixed ring order".** The brief asks for a fixed ring order rather than
arrival order, and `docs/brief-response.md` notes the consequence: the ring membership would have to be
pinned too, or two runs at the same N could disagree. Ordering by `(token, expert)` removes that
requirement. Re-numbering the ring, replacing a node, or re-partitioning the experts cannot move a bit,
because neither the ring nor the arrival order appears in the order key. What *must* be pinned is the
**ownership map** — which node holds which experts — and that is a data artifact (`DC-011`), not a
property of the network.

**What it costs, stated honestly.** A pre-summed partial for one token is `hidden × 4 B` = 8 KB, which
is where the architecture's "roughly 4 KB per layer" comes from. Per-contribution transport is up to
`top-k × hidden × 4 B` = **64 KB per token per layer** at `hidden = 2048`, `top-k = 8`, and only the
*other* node's terms need to cross, so ~half of that on a balanced pair. That is a real 8× on the
synchronisation payload, paid deliberately: it is the price of I2, and it is still small enough that
**latency per layer, not bandwidth, remains the design constraint** (`DC-051` measures it). If that
measurement ever says otherwise, the escape hatch is an order-independent accumulator (exact
fixed-point or a superaccumulator), which is more work and a different decision — not a return to
per-node partials.

**What it forbids, and why each matters.**

- Pre-summing per node, for the reason above.
- Summing in arrival order or ring order, which makes the result depend on the network's timing.
- Reducing a **subset**: a run that summed seven of a token's eight experts would be quietly wrong and
  numerically plausible. The reduction cannot distinguish "no contribution" from "a contribution of
  zero", so absent terms must be an error at the transport layer (`DC-009`, `DC-043`), including the
  node-failure path — a missing node is a failed run, never a smaller sum.
- Trusting a wire value's shape: the width check in `OrderedReduction.accumulate` is a `precondition`
  for a programming error, and untrusted input is the transport's job to validate before it gets there.

**Where it lives and how it is held.** `sources/DatacenterEngine/OrderedReduction.swift` implements the
contract as a pure function over `ExpertContribution`, so it can be tested without a cluster. Four
tests hold it:

| Test | What it pins |
| --- | --- |
| `testTheReductionReproducesTheSingleNodeSequenceBitForBit` | the reduction **is** the single-node sequence, compared bit pattern by bit pattern against an independently written reference |
| `testPartitioningAndArrivalOrderDoNotMoveTheBits` | 1, 2, 4 and 8 partitions, each with its terms reversed and the partitions delivered in reverse order, produce identical bits |
| `testPreSummedPartialsWouldNotBeBitIdenticalAndTheContractIs` | the trap is real (20000008 against 20000006) and the contract avoids it |
| `testTheOrderKeyIsTokenThenExpert` | the order key itself, so a refactor cannot quietly change it |

### Status: implemented, and demonstrated on the fixture's real weights

`MixtureOfExperts.experts` is now a thin wrapper over `expertContributions` +
`OrderedReduction.accumulate`, so the one-node and N-node paths **are the same code** — a second loop
beside the contract would be a second chance to accumulate in a different order, which is the thing this
decision exists to prevent. `ShardedMixtureTests` takes the fixture's **real quantized weights**, the
**real** router's selection, and two and four simulated nodes, and compares bit pattern by bit pattern
with the single-node forward; it also checks that every selected expert is owned exactly once, and that a
node asked for an expert it does not own is refused rather than handed zeros.

Three things the fixture run forced, two of them because the first version was wrong:

- **A node must be able to decline an expert.** The router selects across all experts while a node owns a
  subset, so `ExpertWeightProvider` gained `serves(_:)` — defaulting to `true`, so nothing single-node
  changed — and the expert path **skips** what a provider does not serve. The first attempt made the
  wrapper throw instead, and the run failed at once with "node 1 was asked for expert 6, which it does
  not own": loud absence, before there was any guard to make skipping safe.
- **Completeness is a run-level guard, not a hope.** Skipping is only safe if something checks that the
  reduction saw exactly the selected terms, so `OrderedReduction.selectedKeys`/`isComplete` are
  production API, tested, and required of every sharded run — including the node-failure path, where the
  answer is to fail the run (`DC-043`).
- **The cost is real and accepted.** Routing the single-node forward through the contract added ~1.4 s to
  a 16.3 s forward on the real 35 B model (40 contributions per layer × 40 layers, materialised and
  sorted). Keeping a fast single-node loop beside the contract would buy that second back and reintroduce
  the divergence risk, so it is not done. The real-model digest is **unchanged**
  (`b0d382dbabf36df0…`), which is the evidence that the refactor is numerically neutral.

**What remains.** The transport and the wire protocol (`DC-008`, `DC-009`). Nothing above has crossed a
network, so the cluster's real failure modes — a slow node, a retry, a node that dies mid-run — are still
untested, and `Q9`-style questions about the ring are answered on paper rather than over a wire.

## D18 — The wire protocol: bit-exact terms, length-framed, hostile input refused before allocation

`D17` decided *what* crosses; this decides *how*, and the encoding is not an implementation detail —
a codec that rounded, normalised or re-ordered would break I2 while looking like arithmetic.

**The layout**, little-endian and fixed, so two nodes cannot disagree about a byte order:

```
magic "TTDC" · version u16 · tokens u32 · hiddenSize u32 · count u32
count × ( token u32 · expert u32 · width u32 · scale f32 · width × f32 )
```

**Floats travel as their IEEE-754 bit patterns.** Not as decimal, not through a canonicalising step:
`-0.0` stays `-0.0`, a subnormal stays a subnormal, and a NaN keeps its payload. A test feeds the wire
`0.0`, `-0.0`, `±leastNonzeroMagnitude`, `±infinity`, a NaN and a NaN *with a payload*, and compares bit
patterns on the way back — because every other test in the file would pass a codec that cleaned them up.

**Framing is a 32-bit length prefix, and it is a claim rather than a promise.** Both the length prefix
and every declared dimension and term count are bounded (`2^26` bytes per frame, `2^20` for dimensions
and terms), and the bound is checked **on the number, before anything is allocated** from it. A decoder
that trusts a length field is a denial-of-service with extra steps; two tests hand it a `2^31` length and
a `2^30` width and require the refusal, not an attempt.

**No checksum, deliberately.** `D15` removed per-read hashing from the hot path because it cost 53% of a
forward to duplicate a guarantee the manifest already provided. A per-frame hash here would be the same
mistake at a smaller scale: TCP checksums, and a frame that arrived corrupt but *plausible* is caught by
the completeness guard (`OrderedReduction.isComplete`) and by the trace digest. A third check that
duplicates both is not worth a millisecond per layer.

**What is demonstrated, and on what.** `ContributionWireTests` runs the fixture's real weights through a
genuine two-node exchange: each node computes only its own experts' terms, encodes them, sends them over
a `socketpair`, decodes what arrives, checks `isComplete` against the router's selection, reduces, and
compares with the single-node forward **bit for bit** — on both nodes. It also checks that two frames
written before either is read come back one at a time, because a stream does not preserve message
boundaries and a reader that assumed it did would return one and a half frames.

**What this is not.** It is one host and one process, so the network is a socket pair and the peer cannot
die mid-frame. A real connection needs the transport measured (`DC-008`), and the failure semantics —
timeout, retry, a node that stops answering, and the rule that a missing term **fails the run** rather
than shrinking the sum — are `DC-043`. `SO_NOSIGPIPE` is set on every descriptor, because the alternative
is that a peer which has gone away kills the process with `SIGPIPE`, and that presents as "the node
vanished" rather than "the write failed".
