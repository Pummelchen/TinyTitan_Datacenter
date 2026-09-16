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

## D19 — Failure semantics: a retry cannot move the bits, and a missing term fails the run

Three ways a cluster run ends badly, and they must not look alike. `ShardExchange` implements the
rules; `ShardExchangeTests` holds them, using a counting transport so "was this retried?" is an exact
assertion rather than a stopwatch.

| Failure | Rule | Why |
| --- | --- | --- |
| A peer says nothing | **retry**, up to `ExchangePolicy.attempts` | the reduction order is canonical (`D17`), so the same terms sent twice produce the same bits — retrying is safe *because* of that, not in spite of it |
| A peer stops mid-frame | **fail, no retry** | the stream is desynchronised: the next read would return the previous frame's tail, and the failure would surface much later, somewhere unrelated to its cause |
| A peer is gone | **fail** | `receive` reports a closed connection rather than waiting out a deadline |
| A term never arrives | **fail the run** | `merge` checks completeness before anything is summed; a reduction over seven of a token's eight experts is a number that looks like an answer and is not one |

**A duplicate is allowed only if it is bit-identical.** That is what a resend looks like, so `merge`
collapses it — and a duplicate that *differs* means two nodes disagree about a term they both claim to
have computed, which nothing downstream can repair. It is refused at the point where the diagnosis is
still possible. The comparison is on bit patterns, not `==`: `Float` says `-0.0 == 0.0` and `NaN != NaN`,
and either would hide exactly the divergence it exists to catch.

**Two defects, both the same class, both caught by asserting a duration.** The first version had an
`ExchangePolicy` timeout that never reached the transport, so a test configured 60 ms, waited 30 s per
attempt, and **passed** — three attempts, 90 seconds, green, because it asserted behaviour and not time.
Wiring the policy through produced a second one: the test's own counting decorator implemented `send` and
`receive` but not `applyTimeout`, took the protocol's no-op default, and reintroduced the identical bug a
layer up. Corrected, the same test runs in **0.191 s** and asserts `deadlines == [60, 60, 60]`, so the
deadline is now known to arrive rather than assumed to. The lesson is in the code: a decorator that wraps
a transport **must forward the deadline**, and a parameter that is not wired up is a lie that only a
clock catches.

**What is still not exercised.** A real network and a real second process: everything here runs over a
socket pair on one host, where a peer can be silenced but not made slow, and where no packet is ever
lost. The transport measurement (`DC-008`) and M2's own gate (`DC-045`) need the cluster. What is decided
here is what the run *does* when a peer misbehaves; that the failure is *detected* over a real link is the
next question, and the rules above are the contract it will be tested against.

## D20 — The shard plan is data, and the format makes the dangerous mistakes unrepresentable

`D17` reduced what the planner has to pin to exactly one thing: the **ownership map**. The ring order no
longer matters, arrival order never did, and the reduction order is canonical — so the artifact is a map
from expert id to node, and nothing else.

**The format is chosen so the property the reduction needs is structural rather than validated.**

```json
{"schema":1,"family":"qwen3_5_moe","experts":256,"nodes":4,
 "distribution":"contiguous","owners":[0,0,...,1,1,...]}
```

`owners` is a flat array **indexed by expert id**. An expert cannot be owned twice, because there is one
slot per expert; it cannot be quietly unowned, because the length is checked against the model's expert
count. The obvious alternative — a list of expert ids per node — makes both mistakes representable and
leaves a validator to catch them after the fact, which is the kind of check that is forgotten in the one
path that matters.

**Compatibility is checked at load, not discovered at the first token.** `schema` is the format version;
`family` and `experts` are checked against the model a node actually opened
(`validate(forFamily:experts:)`), so a plan made for another model is refused before a run rather than
manifesting as a missing term. A file whose `owners` length disagrees with its declared expert count is
refused on the count — a test writes exactly that file, because a truncated plan is what an editor
produces.

**Identity: canonical JSON, then a digest.** Keys are sorted and whitespace removed, so two nodes that
agree produce the same bytes; `canonicalDigest()` is compared during bring-up (`DC-042`) and a mismatch
stops the run. A test asserts two independent generations agree and that moving one expert changes the
digest — which is the whole point of comparing before starting.

**Contiguous blocks rather than round-robin, and it changed no result.** Rounds 6 to 8 assigned experts
round-robin; this assigns contiguous *blocks* so a node's experts are adjacent in the stacked tensor and
its reads walk forward through the payload instead of striding across it. The bit-identity tests were not
adjusted and still pass: **who computes a term never moves a bit**, which is `D17`'s claim demonstrated
by a refactor rather than by argument. Both distributions are available, and a plan whose owners do not
match its own label is refused.

**The dense backbone is not in the plan, deliberately.** It is replicated on every node and therefore has
nothing to assign, which is why there is one all-reduce per MoE layer and none per layer elsewhere.

**Dead code removed rather than kept.** A "node owns no experts" check was written, could not be reached
by any test — a missing node always fails the contiguous-shape check first — and a model with fewer
experts than nodes legitimately leaves nodes idle. It was deleted, and the test that could not reach it
was repointed at the shape check that does the work.

**What this leaves.** A command-line writer for the plan file is a convenience, not a gap: any node
generates the same plan deterministically from `(family, experts, nodes, distribution)` and the digest
proves it. What remains for M2 is the cluster: two real processes on a real link (`DC-045`).

## D21 — Bring-up: the six things nodes must agree about, checked before a token

A cluster run claims that N machines are computing one model together, and every way that claim can be
false produces numbers with a plausible shape. So the agreement is checked once, up front, on exactly the
fields whose disagreement would invalidate the run:

| Field | Why it is compared |
| --- | --- |
| `schema` | the bring-up protocol itself |
| `family`, `revision` | the same model, pinned the same way (I6) |
| `experts`, `hiddenSize`, `topK` | the same geometry, because a plan for another shape is another model |
| `planDigest` | the same ownership map (`D20`) — the one artifact `D17` says must be pinned |

**What is deliberately not compared: an install digest.** In a sharded run each node holds the experts it
owns, so the installs differ *by construction* and comparing them would refuse every correct run. This is
the kind of check that is obvious to add and wrong to add, which is why it is written down.

**A refusal names the field.** `ClusterIdentity.differences(from:)` compares field by field, so a run
stops with "node 1 disagrees about planDigest" rather than "nodes disagree". A test changes each field in
turn and requires it to be reported alone.

**The declaration frame carries its own magic (`TTDH`).** A declaration and a contribution frame travel
over the same connection, so a codec that inferred the type from the payload would eventually read a
declaration as terms — and a contribution decoded from JSON is exactly the kind of plausible nonsense
this project keeps having to eliminate. Two tests hold the boundary from both sides: a contribution frame
is refused as a declaration, and a declaration frame is refused as contributions.

**The peer set is checked, not assumed.** A peer that claims a node twice, a peer claiming a node outside
this cluster, and a node that never answered are three distinct failures with three messages. The first
version collapsed all three into one branch and could index an empty array while doing it — caught by
reading it back, not by a test, which is why the fix arrived with the tests.

**Discovery is data for now, and that is a statement about the network rather than a shortcut.**
`ClusterConfig` lists endpoints and is validated (this node present, no duplicates, no empty host or
impossible port). Finding nodes — mDNS, address negotiation, or binding and connecting at all — needs a
real link, and belongs with the transport measurement (`DC-008`).

**Demonstrated.** Twelve tests over a socket pair: a matching cluster completes bring-up in both
directions (this node learns its peer, and the peer receives this node's declaration unchanged), and every
disagreement above is refused with its own error before anything is computed. What is left for M2 is the
socket that a real machine binds — `DC-008` for the implementation and measurement, `DC-045` for the gate.

## D22 — The transport binds, listens and connects: a socket rather than a socket pair

The socket pair carried the framing, the deadline and the exchange correctly, and it could not fail the way
a socket fails: nothing bound, no port was ever busy, and no connection was ever refused. `TCPListener`
and `TCPTransport.connect` put those paths in reach — IPv4 loopback, with the link itself left to
`DC-008`'s measurement.

**A blocking `connect` is not an option, and this is the kind of thing that only shows up on a real
socket.** Connecting to a host that is not answering blocks for the kernel's own timeout, which is
minutes: the run would look hung rather than refused, and the timeout policy would never apply. So the
socket goes non-blocking for the connect, `poll` waits for writability under the policy's deadline, and
`SO_ERROR` is asked for the answer — because `POLLOUT` also becomes ready when a connection *failed*.
Then the socket goes back to blocking, so the read path behaves exactly as the pair's did.

**Two diagnostics were wrong and are fixed.** A failed `connect` was reported as `cannotBind`, which
names the wrong operation and would send someone looking at the listener. The `accept` and `connect`
implementations also silently shadowed the global functions of the same name; Swift refused to compile
rather than picking one, and they are now qualified.

**Demonstrated.** Six tests: a listener asked for port `0` reports the port it got and carries a frame
both ways; bring-up completes over TCP and the peer receives this node's declaration unchanged; a
**contribution exchange over TCP is bit-identical to the single-node forward** on the fixture's real
weights; an accept nobody answers times out on its deadline; connecting where nothing listens is refused;
and a second listener on a busy port is refused.

**What is still not exercised, and it is now a short list.** IPv6, the choice and measurement of the real
link (`DC-008`'s other half), and two machines running a forward together (`DC-045`). The engine also does
not yet *use* any of this: the exchange is exercised by tests, and the next piece is a forward that
actually shards across it.

## D23 — The engine runs sharded, and the trace is byte-identical

Everything up to here was a contract, a codec, a transport or a test harness. This is the engine: a
`Qwen3_5Forward` with a `ShardExecution` runs one node of a cluster, and **the trace it produces is
byte-identical to the single-node one, discrete router decisions included.**

**It is a branch, not a second forward.** The mixture case computes `routed` and `indices` either way and
the rest of the layer loop is untouched; the sharded branch calls `expertContributions` with an
`OwnedExpertProvider`, then `ShardExchange.allReduce`, then the **same** `MixtureOfExperts.combine` the
single-node path uses. `block` was split into `sharedPart` + `combine` so there is exactly one
implementation of each — a second one would be a second chance to differ, which is what `D17` exists to
prevent. The split was verified numerically neutral the way that matters: the suite was unchanged and
green before and after it.

**The router runs on every node, and its decision does not travel.** The dense backbone is replicated, so
every node can decide for itself; shipping the decision would create a second source of truth for it — and
I3 makes that decision a claim in its own right, not a number to compare approximately.

**One all-reduce per mixture layer**, inside the layer loop, which is the architecture's line and now also
the code's.

**The lockstep is real and worth naming.** Each node blocks on its peers inside every layer, so the two
nodes must make progress *concurrently* — the test runs them on two threads for that reason. This is not a
test artefact: it is the shape of the run, and it is why the synchronisation budget (`DC-051`) is a real
question for M3 rather than a detail. The fixture's frames are about a kilobyte and would fit in a socket
buffer either way; the real model's are tens of kilobytes per layer, where the concurrency is load-bearing
— and two machines provide it for free.

**Demonstrated, over real sockets.** Two nodes, TCP loopback, two threads: the trace is byte-identical to
the single-node run on **both** nodes, tensors compared by bit pattern and the router's top-k compared
exactly (I3). A silent peer makes the *forward* throw rather than producing a trace from whatever experts
this node happened to own, and the policy's deadline reaches the socket through the forward — the `D19`
guarantee, at the level a user would meet it. And with no shard context the forward is the one M1's gate
measured: two single-node runs are identical to each other, which pins that the branch did not disturb the
original path.

**What it is not.** Two *processes*: the two nodes here are two threads in one address space, each with
its own `Qwen3_5Forward` and its own `InstallFile`, so nothing is shared but the kernel. And two
*machines*: that is M2's gate (`DC-045`), and the real model at two nodes does not fit on this 8 GB
development host, which is why it needs the cluster.

## D24 — The gate runs as two processes, and the judge is the project's own differ

Two threads in one address space share a heap and a failure domain; M2's claim is about machines. So
`datacenter-node` runs **one node as its own process** — its own `InstallFile`, its own address space, its
own socket — and `tools/run_m2_gate.py` starts two of them, one listening and one connecting, waits for
both, and compares every trace with `tools/trace_diff.py`.

The trace is the evidence, and the judge is the harness M0 built rather than a new one:

```
[1/4] plan: 8 experts over 2 nodes, contiguous
[2/4] reference: 7 tensors, 2 discrete, digest 58518422914cfe2b…
[3/4] node 0: BRINGUP ok: node 0 of 2, plan 25d34fe36d62d793…
      node 1: BRINGUP ok: node 1 of 2, plan 25d34fe36d62d793…
[4/4] node 0: IDENTICAL — 7 tensor(s), 0 element(s), 2 discrete decision(s) checked (matching digests)
      node 1: IDENTICAL — 7 tensor(s), 0 element(s), 2 discrete decision(s) checked (matching digests)
M2 GATE (fixture scale) PASSED: 2 processes, one plan, one trace
```

**The agreement is checked by the run, not by the harness.** The node loads the plan, validates it against
the model it actually opened, handshakes, and only then computes. The first version of the harness guessed
the family (`tiny-qwen36`) while the fixture's family is `qwen3_5_moe`: the node refused the plan, which is
`D20` working exactly as intended — but the refusal came out as a Swift trap, because an uncaught
top-level `try` in Swift is a `SIGTRAP` and a stack line. The fix was two-sided: the harness reads the
family from the install's manifest, and the node reports a refused plan as a message.

**The node's metrics are measured, not derived.** `datacenter-node` counts the bytes the install actually
handed out, and the verification bytes separately. `elementsRead * 2` was removed from the single-node
trace in the first round of this work for overstating a 4-bit install by ~3.5x, and it does not come back
in the sharded one wearing a different name.

**The honest limits.** The CLI is two-node: the engine's `ShardExecution` takes any number of peers and the
plan covers any N, but the CLI takes exactly one `--listen` or `--connect`, and the harness says so rather
than appearing to support four nodes and hanging. The run is the **fixture**, because the real model at two
nodes does not fit on this 8 GB host — `--install` points the same harness at the real install on the
cluster, which is `DC-045`.

## D25 — M2's gate, on two machines

`DC-045` is M2's real gate and two threads on one host cannot satisfy it. With `--remote node1@node1` the
harness stages a node binary and the fixture on a **second machine**, runs node 1 there and node 0 here,
and fetches the remote trace back so the project's own differ judges both against the single-node
reference:

```
[3/4] two machines: node 1 listening on node1@node1, node 0 connecting from here
      node 1 bound port 55252 on its LAN address (the farm's names take the VPN)
      node 1: BRINGUP ok: node 1 of 2, plan 25d34fe36d62d793…
      node 1: wrote /Users/node1/m2-gate/node-1: 7 tensors, 2 discrete, digest 58518422914cfe2b…
      node 0: BRINGUP ok: node 0 of 2, plan 25d34fe36d62d793…
      node 0: wrote …/node-0: 7 tensors, 2 discrete, digest 58518422914cfe2b…
      fetched node 1's trace from the other machine (3 files)
[4/4] node 0: IDENTICAL — 7 tensor(s), 0 element(s), 2 discrete decision(s) checked (matching digests)
      node 1: IDENTICAL — 7 tensor(s), 0 element(s), 2 discrete decision(s) checked (matching digests)
M2 GATE (fixture scale) PASSED: node 0 here and node 1 on node1@node1, one plan, one trace
```

**What this establishes.** Two machines, a real link, a plan both read from the same bytes, a handshake
that refuses disagreement, one all-reduce per mixture layer, and a trace byte-identical to the
single-node run — on both nodes, router decisions included. The binary is built here and copied: ad-hoc
signing survives `scp`, so a peer needs no checkout and no build, which is what makes one `scp` of an
arm64 binary enough to turn a machine into a cluster node.

**Why it is the fixture and not the 35 B model, stated plainly.** The real model would need its **20 GB
install staged on the peer**, and the farm's other nodes are in active use by other work: the standing
instruction for this phase is *functional tests, not benchmarks*, and moving 20 GB onto a shared machine
to re-prove a property the fixture already proves is neither. When the timing phase opens, `--install`
points the same harness at a staged install and the same command becomes the real gate — and that run
will want the Ethernet path rather than the VPN one the names resolve to (a trap the Testbed records).

**Three harness bugs, all from assuming the local case.** The first version left node 0 listening on this
host *and* started a listener on the remote, so two nodes waited for a peer that was never coming; the
roles invert when the peer is remote, and only one node may listen. The second copied the remote trace
without `-r`, which `scp` reported as "not a regular file" — a trace is a directory. The third labelled
output by process index rather than node id, so a passing run read as though node 0 had written node 1's
file.

## D26 — M2's gate, on the real 35 B model, across two machines

The fixture proved the machinery; this is the gate. One node on `node3`, one on the development host, a
256-expert plan over two contiguous halves, the real install on both, and the M1 baseline as the
reference:

```
[1/4] plan: 256 experts over 2 nodes, contiguous
[2/4] reference (one node): 83 tensors, 40 discrete, digest b0d382dbabf36df0…
[3/4] node 1 (node3): BRINGUP ok, plan b561404b0cbbad71…, digest b0d382dbabf36df0…
      node 0 (here):  BRINGUP ok, plan b561404b0cbbad71…, digest b0d382dbabf36df0…
[4/4] node 0: IDENTICAL — 83 tensor(s), 0 element(s), 40 discrete decision(s) checked (matching digests)
      node 1: IDENTICAL — 83 tensor(s), 0 element(s), 40 discrete decision(s) checked (matching digests)
M2 GATE PASSED
```

**The digest is the M1 baseline's.** `b0d382dbabf36df0…` is what the single-node engine produced before
any of this existed, and both machines produced it from half the experts each. I2 is not an argument any
more; it is a measurement on the real model over a real link.

**Why a trace and not a token list.** The trace carries the `logits`, so byte-identical logits make
greedy generation *identical by construction* — a strictly stronger claim than comparing generated text,
which the M1 gate also rests on. What is not yet built is a **sharded `datacenter-generate`**: the
generation CLI opens a model without a shard context, so the tok/s gates M3 needs will require it
(`DC-109`).

**No timings are claimed here.** The run reported wall clocks (16.5 s for the single-node reference,
13.2 s and 12.7 s for the two nodes), and they are recorded as *how long the functional test took*, not
as throughput: the farm's nodes are shared, nothing was isolated, and the standing rule for this phase
is functional tests rather than benchmarks. The ~111 MB/s the install transfer sustained over the LAN is
likewise an observation about a copy, not a transport measurement — that is `DC-008`, on a quiet farm.

**The install was staged once.** 21.7 GB to `node3`'s Downloads folder, which is the only data movement
this needed: the development host already had the install, so one peer was enough for a two-node gate.
Both nodes read only the tensors they touch, which is why a full install per node is still more than the
design needs — a shard-repacked install is a later refinement, not a prerequisite.

**One thing to fix and it is mine.** The first version of `D25` printed this node's peer address in its
evidence block, and this repository is public: host names and addresses do not belong in it (R9). The
line is scrubbed here. It remains in the history of the commit that introduced it, which is a force-push
to remove and therefore the operator's decision rather than mine.

## D27 — The whole farm runs one forward: four nodes, bit-identical

`--mesh` runs every node as its own process on its own machine. A **mesh rather than a star**, and the
reason is the contract rather than taste: the all-reduce is pairwise (`D17`), so a leaf in a star would
hold its own terms and the coordinator's and nothing else. `isComplete` would refuse that run rather than
let it sum less — the right failure, but not a run. Joining is deterministic and needs no ordering
agreement: node *i* connects to every lower id and accepts from every higher one, so no pair connects
twice and every node joins in a single phase. Ports come from the cluster config (`D21`), which is also
what each node validates its own membership against.

```
[3/4] mesh of 4 nodes: node4(…26), node1(…27), node2(…25), node3(…29)
      node 0: BRINGUP ok: node 0 of 4, plan d614a352184da9cc…
      node 1: BRINGUP ok: node 1 of 4, plan d614a352184da9cc…
      node 2: BRINGUP ok: node 2 of 4, plan d614a352184da9cc…
      node 3: BRINGUP ok: node 3 of 4, plan d614a352184da9cc…
      every node: 7 tensors, 2 discrete, digest 58518422914cfe2b…
[4/4] node 0..3: IDENTICAL — 7 tensor(s), 0 element(s), 2 discrete decision(s) each
M2 GATE PASSED: a mesh of 4 machines, install the fixture, one plan, one trace
```

**That is M3's functional half**: four machines, expert-parallel, producing exactly what one machine
produces. M3's *gate* is a throughput claim — ≥3× the M1 rate — and it is **not** made here: the farm is
shared with other work, nothing was isolated, and the timing phase comes later by instruction. The 35 B
model at four nodes would also need its 20 GB install on three peers; the two-node gate already made the
real-model bit-identity claim (`D26`), and this one makes the N-node claim on the fixture, where the
payload and the runtime are small enough not to disturb anyone.

**The addressing is part of the run now.** These runs use the **internal LAN addresses**, not the node
names: the names resolve over the mesh VPN at roughly 2.5× the round trip, which is the trap the Testbed
records, and it is the difference that will matter when the synchronisation budget is measured. The
inventory on that page carries the addresses so the next run does not rediscover them.
