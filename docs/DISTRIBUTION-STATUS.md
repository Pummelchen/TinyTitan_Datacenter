# Distribution status

> **RESUME HERE (written at round 92, session context exhausted).**
>
> **Done and verified:** the exchange on both sides, bit-identical to the single-node answer over a real socket;
> the remote-buffer kernel, inert when unsharded; **the requesting half is wired** - the call site in
> `encodeDecodeRoutedMoE` (`52e24d9`), the CLI provider from `--shard-plan`/`--shard-node`/`--shard-peers`
> (`9b072ce`), both `nil` by default, behaviourally verified (unreachable peers exit 1; no peers runs single-node at
> 7.151 tok/s) and the single-node path confirmed unregressed by measurement (7.610-7.735 tok/s).
>
> **The one thing left: the serving half's real `Compute`.** `ShardExchangeServer` takes it as an injected closure
> and it is unwritten. Everything about it is specified below - its shape (`fetchRoutedExperts` ->
> `makeRoutedArgumentBuffer` -> phase 1 -> a down projection -> readback), its cache-sharing (free, because
> `fetchRoutedExperts(layer:experts:)` is the entry point the request path already uses), and **its silent-failure
> mode, recorded before the code** (the existing down projection applies the routing weight, so a peer built on it
> would send `w * value` and the requester would multiply by `w` again - a wrong number the bit-exactness contract
> cannot catch).
>
> **Then the four-node run**, with loads, a median over repeats, and the generation length recorded together. The
> prediction is **~8-9 tok/s (about 1.12x)** and it would **falsify** the 21 tok/s target rather than meet it - which
> is what the goal asked for.
>
> **Do not re-derive these** (each cost records to establish): the expert read is *not* on the critical path
> (`D239`, measured directly, two sweeps agreeing); the cache slot count is the lever and the candidate set is not
> (`D243`); the exchange costs 3.2 ms/step on raw frames, excluding the serving node's own work (`D208`);
> `totalExposedIoNanos` never increments in this configuration so the server's `exposedIo` reads zero for a reason
> unrelated to IO (`D237`, `D238`).

A handover note. This fork carries work that exists on **one disk with no remote it may be pushed to**
(`DC-135`), so the state is written down here rather than left to be rediscovered from `git log`.

## The one-line summary

**The exchange is built and exact; the decode path does not use it yet.** Everything below the call site is
finished and tested; the call site is not written.

## What is proven

| property | how it is established |
| --- | --- |
| **The reduce is exact under any partition** | `ShardReduce`. k is fixed at 8, partials are fp32, a node contributes **zero-padded** across all k, and IEEE addition of zero is exact — so any split of the eight slots gives the **identical bit pattern**. The tests assert `bitPattern`, not a tolerance, because a tolerance would not falsify it. |
| **Slot order is load-bearing** | A test asserts that reordering the same eight values **changes** the fp32 sum (`1e8 + 1.0 - 1e8 = 0.0` against `1e8 - 1e8 + 1.0 = 1.0`), and that the vector still demonstrates it. Slots travel with values and are never renumbered. |
| **Ownership narrows the routed set before planning** | `PreadExpertStreamer.ownedExpertFilter`, applied at the top of `makeExpertCachePlan`, so the slot-count check, the eviction choice and the read all see the owned set. **On the real forward path**: `encodeDecodeRoutedMoE` → `planRoutedExperts` → `planExpertsCached` → `makeExpertCachePlan`. `nil` is an **absent** filter, not an identity, so single-node is unchanged. |
| **Replicated experts never reach the wire** | The participant asks only when `!plan.isLocal(expert:to:)`. `owner(of:)` alone would ask for a replicated expert — exactly the bytes replication exists to remove. A test with `replicated: [1, 3]` pins it. |
| **The shadow kernel is inert when unsharded** | `moe_phase2_down_reduce_k8_remote` is a **separate** kernel; a single-node run dispatches the original, byte-for-byte. Verified on the real install: identical tokens, 7.230–7.299 tok/s. |
| **The exchange crosses a real socket** | `ShardExchangeIntegrationTests` — split by plan, frames over TCP, reassembled in slot order, landing on slot 3 and zeros elsewhere. |

## What is not

**`encodeDecodeRoutedMoE` (`RealForwardRunner+Decode.swift:1773`) does not call the exchange.** It plans only
locally-owned experts and passes no `remotePartials`. The three steps:

1. Read `moeActs` back for the layer (a GPU→CPU sync, which is the cost).
2. `ShardExchangeParticipant.contributions(...)` — which needs a persistent connection set to the peers, not a
   connection per layer.
3. Build the `[d][8]` fp32 buffer (this node's slots zero, peers' rows placed) and pass it as
   `remotePartials` to `encodeRoutedPersistentPhase2Reduce`.

`docs/sharded-phase2-reduce.md` has the arithmetic that constrains this: **the host cannot form the sum**,
because node-level results associate differently, so the peer's partials must reach the kernel.

## Why it is not worth much on this engine

`D182`/`D183` measured the step at 139.6 ms with only **13.2%** of it being expert work that sharding can divide.
A perfect, free four-way division gives **1.11×**; with the measured 17.3 ms exchange, **0.98×**. Reaching 3×
would need more than 88% of the step to divide. And `D179` removed the replication lever: R experts costs
`R × 40 × 1,769,472 B`, so R = 64 is **4.53 GB**, not the 116 MB an earlier plan assumed.

So this work is worth finishing **as the mechanism**, and it is not worth finishing **for the throughput** — the
vocabulary-parallel head (221.1 ms, 6.6% of the step) is worth more than the entire exchange and is not a
sharding change at all.

## The test discipline this work needed

Three defects were found by tests rather than by reasoning, and each is a comment that had been believed:

* a participant asking by `owner(of:)` instead of `isLocal` — would have put replicated experts on the wire;
* `if (remote != nullptr)` — **Metal leaves an unbound buffer argument undefined, not null**, so the guard was
  itself the bug and the single-node path silently ran the sharded kernel on garbage;
* a test that hung, and being **filtered around** is what hid the second one for a whole session. `swift test`
  now runs whole: 684 tests, 0 failure markers.

## A named gap: the two halves compose, but nothing tests that they do

The requesting half, the serving half and the reduce are each tested, and the whole suite is green. What is **not**
verified is the **composition** of the requesting and serving halves over one socket — a real server, a real
participant, the peer's contribution merged with the node's own and compared bit-for-bit against the single-node
answer. That test was written and **crashes**:

    Swift/SliceBuffer.swift:317: Fatal error: Index out of bounds

with no stack frame naming any of our code. It is removed rather than shipped — a test that kills the runner is
worse than a named gap — and this note exists so the gap is not mistaken for coverage.

**Ruled out by reading, so a retry does not repeat it:**

* the frame shapes. `dimensions` is `activation.count`, **one activation row shared by every slot**, and the
  engine's `moeActs` is `[topK * FmoE]` which is one row per slot — so a participant asking about two slots still
  sends one row, and it does. The reply is `slots.count * dimensions` and both the encoder and the decoder check
  it.
* the slot arithmetic. Experts `[0,2,4,6]` at slots `[0,1,2,3]` over a round-robin 4-node plan puts 2 and 6 on
  node 2, so exactly one request crosses, and the test's own placement is in range.
* the kernel layout. `remote[d * 8 + slot]` with `d < 4`, `slot < 8` addresses 32 floats, which is what
  `remotePartials` returns.

**Not ruled out:** `ShardPeerChannel.receive`'s frame slicing and `dataToFloats`, both of which slice and neither
of which the crash names. **The next attempt should call `contributions(...)` against a stubbed peer before adding
the socket**, so the crash bisects to one side of the wire.

## The two call sites, located, so the next attempt starts from a line number

Everything on both sides of the exchange is built and tested. What is not written is where the engine calls it.
This is what is known about that seam, taken from reading the code and confirmed against the line numbers as of
`3238872`.

### The requesting side

`RealForwardRunner+Decode.swift`, `encodeDecodeRoutedMoE`:

* **the routed expert set is planned at ~1387** (`model.planRoutedExperts`), and the decoded plan's `misses` are what
  the local streamer will read;
* **`readyBuffers` at ~1384** is already non-blocking and the ring lands about one layer;
* **the phase-2 reduce is encoded at ~1773** via `encodeRoutedPersistentPhase2Reduce`, which now takes
  `remotePartials: MTLBuffer?` and selects `phase2ReduceK8RemotePSO` when it is non-nil;
* **the seam that decides which experts this node reads is `ModelExpertIO.setOwnedExpertFilter`**, applied at the top
  of `makeExpertCachePlan` and therefore on the real forward path. It is unset by default, which is why every
  measurement in this repository remains valid.

So the call site is: after the plan is known (so the expert list is final) and before the phase-2 encode, read
`moeActs` back — `topK * FmoE` fp16, 8 KB at the real shapes — call
`ShardExchangeParticipant.remotePartials(layer:experts:slots:activation:dims:)`, upload the `[d][8]` result, and
pass it. **This node's own slots in that buffer must be zero** or they double-count (`D168`).

### The serving side

`ShardExchangeServer` exists and is tested, with the expert computation **injected**:

```swift
public typealias Compute = (_ layer: Int, _ experts: [Int], _ activation: [Float]) throws -> [Float]
```

The real `Compute` has to run the named experts over the supplied activation — the MoE's phase 1 for a subset of
slots — **sharing the node's expert cache with the generation loop** rather than owning a second copy of it. It must
reply in the requested order, and the server refuses a reply of the wrong width rather than padding.

### The lifecycle

`answer(_:)` loops until its peer **closes**, not for a fixed request count — a decode loop sends forty requests a
token and closes when the token is done. A test that awaits the server before closing **deadlocks**, and the first
version of `ShardExchangeServerTests` did, for the full timeout.

### What to measure, and against what

The four-node run must record the node loads and a median over repeats, and — per `D230` — **the generation length**,
because the step is not stationary: the layer body grows 8.5% and the attention kernel 47% from position 16 to 160.
The projection to compare against is **14.33 tok/s without attention sharding and 16.35 with**, on `D228`'s
calibrated model, with the caveat that it was calibrated against a single node and four columns of it are model
output rather than measurement.

**One engine caveat worth carrying:** `totalExposedIoNanos` never increments in this configuration — its clock
returns `nil` unless an observed completion count equals a *predicted* one (`D237`, `D238`) — so the server's
published `exposedIo` reads zero here for a reason unrelated to how much IO is exposed. Do not build on it.

## The cheapest measured step towards a four-node number, which needs no exchange at all

The four-node projection turns on one question: **does dividing the expert read divide the step?** The cache sweep
(`D217`, `D218`, `D236`) answers it indirectly - 156 MiB fewer per token buys 46 ms across four points - but it
cannot reach the four-node read volume, because 40 slots already reads 89 MiB/token and a node owning a quarter of
the experts would read about 22. **The sweep's lowest point is above the volume a sharded node would see.**

There is a shorter route, and it uses a seam that already exists and is already on the forward path:

* `ModelExpertIO.setOwnedExpertFilter(_:)` applies a filter at the top of `makeExpertCachePlan`, so a node reads only
  the experts it owns. **It is unset by default**, which is why every existing measurement is valid;
* `--shard-plan`, `--shard-node` and `--shard-peers` already parse and build a `ShardConfiguration`, but **nothing
  consumes it** (`D206`).

**Wiring only the first half** - build the plan from the flags and call `setOwnedExpertFilter` with
`ShardPlan.isLocal(expert:to:)` for this node, **without any exchange** - produces a run that **has the right read
volume and the wrong answer**. Its timing is therefore **not a tok/s claim** and must never be reported as one.

What it *is*: a **measured** step time for a node reading a quarter of the experts. That is the quantity `D228`'s
projection rests on, and it can be had on one machine, in one run, without the exchange, the peers, or a second node.
If the step falls by roughly the read's share, the read divides and the projection is sound; if it barely moves, the
read is not on the critical path and `D228` is wrong in the direction `D235` guessed.

**Two things to be careful about.** The run's *output is garbage* and its log must say so, or a later reader will
mistake a token stream for a result; and the ownership filter changes which experts are resident, so the cache
behaviour differs from a real sharded run in a way that has to be stated rather than assumed away.

The full call site is still what the objective asks for. This is the measurement that can be taken before it, and it
is the last one available on a single node.

## The read is NOT the lever, and this has been measured — do not re-derive it

`D239` wired `--shard-plan`/`--shard-node` to `setOwnedExpertFilter` and measured a node reading **64 of 256
experts** against one reading all 256, same binary and flags, three alternating pairs on node3:

    128.2 -> 128.3 ms     133.7 -> 134.5 ms     128.4 -> 128.5 ms       1.00x, 0.99x, 1.00x

**Reading a quarter of the experts changes the step by nothing.** The expert read is not on the critical path, and
any projection that recovers read time for a sharded node - including `D219`'s fitted intercept and `D228`'s overlap
column - is recovering a term worth zero.

**And the cache sweep is not evidence to the contrary** (`D240`). The filter cuts read volume *and* miss count by
about four times each with no effect on the step, so `--expert-cache-slots` is a proxy for something that is neither
- `D220`'s 0.521 ms is per **cache slot removed**, not per miss. The measurements (40 slots fastest, 64 collapses
into swap, the 9.02 tok/s floor) all stand; only the causal story attached to them was wrong.

**What the device does divide, and what a sharded node therefore saves** (`D221`, `D241`): the routed MoE kernels,
**19.4 ms of a 137.0 ms step**. The filter cannot show this because it leaves the phase-1 kernels running over all
eight routed slots with zeros in the unowned ones - a **real** sharded node does not run its peers' kernels at all.
Attention (338.1 ms), the shared expert (119.9) and the router (106.8) are replicated and stay that way unless
sharded separately, which nothing here does.

**So the ceiling is ~1.12x for expert sharding and ~1.21x with attention and the shared expert sharded as well,
against the ~2.9x that 21 tok/s needs.** The route to anything better is dividing the *dense* work, not the experts.

### The call site is smaller than the handover implied — checked at the lines

Reading the phase-2 site directly removes two uncertainties the earlier note carried:

* **`moeActs` is in scope there.** It is a property of the runner
  (`RealForwardRunner.swift:270`, `// [topK * FmoE] FP16`, allocated at `:763`), and it is already passed as
  `acts:` into the phase-1 encodes at `RealForwardRunner+Decode.swift:1528`, `:1552` and `:1787`. So the activation
  the exchange needs is **already reachable at the call site** — no plumbing to thread it there.
* **What is actually missing is one property.** There is **no** `shardParticipant` on the runner: grepping
  `ShardExchangeParticipant` and `remotePartials` across `RealForwardRunner.swift` and
  `RealForwardRunner+Decode.swift` returns nothing. So the work is: hold an optional participant, set it from the
  options the CLI already parses, and call it at `:1787`.

**One conversion to be careful about.** `moeActs` is **fp16** and the exchange carries **fp32** — `ShardExchange`
frames floats and `remotePartials` returns `[Float]`. The readback therefore widens `topK * FmoE` fp16 values to
fp32 before sending, and the reply's `[d][8]` buffer is already fp32 and goes straight to the kernel. **At the real
shapes that is 8 KB of fp16 read back per layer**, which is small but is a GPU sync and must be counted in the
exchange cost rather than assumed free — `D208`'s 3.2 ms/step was measured on raw frames and **excludes** it.

**And the participant must be optional in the same way the ownership filter is** (`D164`): absent, not an identity,
so the single-node path is untouched when no plan is given.

### The participant property forces a module decision, and there are two ways to avoid the obvious one

`01f54eb` narrowed the remaining work to "one property and one call". The property is the awkward part, because of
where the two types live:

* `ShardExchangeParticipant` is in **`TinyTitanDecodeProtocol`**, which `Package.swift:110` declares with **no
  `dependencies:` at all** — it is standalone;
* `RealForwardRunner` is in **`TinyTitan`**, and **no file under `sources/TinyTitan/` imports the protocol module**.
  The engine has never depended on the distribution.

So holding a `ShardExchangeParticipant` on the runner means adding **`TinyTitan` → `TinyTitanDecodeProtocol`**. There
is no cycle to worry about (the protocol module depends on nothing), so it compiles — but it is a **structural**
change: the single-node engine becomes coupled to the sharding code, in a repository whose whole measured record
rests on the single-node path being untouched when no plan is given (`D164`).

**Two ways to avoid it, and the second is the one this repository's own pattern suggests:**

1. **Put the exchange behind a protocol declared in the engine** — `func remotePartials(...) -> [Float]?` — and have
   the CLI supply the conforming participant. The engine then depends on nothing new, and the single-node path holds
   an optional existential that is `nil` unless a plan was given.
2. **Take a closure**, which is what `ShardExchangeServer` already does for its compute and what
   `PreadExpertStreamer.ownedExpertFilter` already does for ownership. **Both seams this session added to the engine
   are closures**, and both are `nil` by default — so a third one would be consistent rather than novel.

**This is a decision, not a measurement, and it should be made deliberately rather than by reaching for the first
thing that compiles.** The cost of getting it wrong is the one this document keeps warning about: a dependency that
makes the single-node path anything other than untouched invalidates the measurements that path produced.

### The call, in the shape it should take

Everything it needs is in scope at `RealForwardRunner+Decode.swift:~1787` and verified: `routedX` (`[D]` **fp16**, the
router's `hidden:` input and phase 1's `x:` input), `outIndices` (`topKExperts` `UInt32`, the router's ids),
`outWeights` (already passed as `routingWeights`), `D`, `topK`, and `L` for the layer. `remotePartialsProvider`
exists on the runner and is `nil` by default.

```swift
// before the encodeRoutedPersistentPhase2Reduce call
var remotePartialsBuffer: MTLBuffer? = nil
if let provider = remotePartialsProvider {
    let dims = Int(D), k = Int(topK)
    // routedX is [D] fp16 - ONE row, because a decode step's single hidden state feeds every routed expert.
    let src = routedX.contents().bindMemory(to: Float16.self, capacity: dims)
    var activation = [Float](repeating: 0, count: dims)
    for i in 0..<dims { activation[i] = Float(src[i]) }
    let ids = outIndices.contents().bindMemory(to: UInt32.self, capacity: k)
    let experts = (0..<k).map { Int(ids[$0]) }
    if let partials = provider(L, experts, Array(0..<k), activation, dims) {
        remotePartialsBuffer = try persistentRemotePartials(dims: dims)   // lazily created, see below
        remotePartialsBuffer!.contents().copyMemory(from: partials, byteCount: dims * 8 * 4)
        // THE NODE'S OWN SLOTS MUST BE ZERO (D168), or its own contribution is counted twice.
        let p = remotePartialsBuffer!.contents().bindMemory(to: Float.self, capacity: dims * 8)
        for slot in 0..<k where !isRemote(experts[slot]) { for d in 0..<dims { p[d * 8 + slot] = 0 } }
    }
}
// ... and pass `remotePartials: remotePartialsBuffer` to encodeRoutedPersistentPhase2Reduce
```

**Two things the shape makes explicit that prose kept losing.** The provider must decide ownership - the engine
does not know the plan - so `isRemote` is the provider's business and the zeroing has to agree with it, or a slot is
either double-counted or dropped. And `persistentRemotePartials` must be a **stored, lazily created** buffer, not a
per-layer allocation: 40 a token against a shared `MetalBufferCache` is `D114`'s failure exactly - 2.47 s/step
against 0.92, with the damage in a phase the change never named.

**Then the run.** Four nodes, `--shard-plan`/`--shard-node`, one `TinyTitanDecodeServer` per node with the real
`Compute` wired to `ShardExchangeServer`, and the loads, the repeat count with a median, and the generation length
recorded together. The prediction to test is **~8-9 tok/s** (about 1.12x), and it would falsify the 21 target rather
than meet it - which is what the goal asked for.

### The serving `Compute`: the last piece, and what it must produce

The requesting half is landed (`52e24d9`, `9b072ce`): the call site, the provider, the CLI wiring, all inert unless a
plan is given. **What is not built is the peer's `Compute`**, and its contract is fixed by what the requester does
with the reply:

    public typealias Compute = (_ layer: Int, _ experts: [Int], _ activation: [Float]) throws -> [Float]

and `ShardExchangeParticipant.remotePartials` takes the server's `experts.count` rows of `dims` and lays them out
`[d * 8 + slot]`. So `Compute` must return, **in the order asked**, one row of `dims` per expert: the **routed
expert's output** - gate/up, activation, down - for the activation it was given.

**`dims` is the hidden size**, not the expert intermediate. The requester sends `routedX`, a `[D]` row, and the
kernel consumes `[D][8]`, so the expert's down projection is part of what the peer computes. A `Compute` that
returned the post-gate_up activation would be `moeActs`-shaped and one step short - the same confusion that took
three records to resolve on the requesting side (`D247`, `D250`), and it would produce a wrong number rather than an
error.

**Three things the wiring has to respect.**

1. **The `Compute` is `throws` and the server refuses a wrong-width reply rather than padding it**, so a failure
   surfaces as a refused request rather than as a short row landing on the wrong slot.
2. **It must share the node's expert cache**, not open a second one. `ModelExpertIO` owns the streamers and the
   residency table; a server that built its own would double the resident footprint on an 8 GB machine, which is
   `D179`'s failure mode.
3. **It runs on the serving node's GPU**, so it is a second consumer of the device the generation loop is already
   using. The exchange's cost on the *requesting* side was measured at 3.2 ms/step on raw frames (`D208`) and that
   number includes none of this - **the serving node's own work is on the requesting node's critical path**, which
   `D207` flagged and nothing has yet measured.

**Then the run.** Four nodes, `--shard-plan`/`--shard-node`/`--shard-peers`, a server per node, and the loads, the
repeat count with a median, and the generation length recorded together. The prediction is **~8-9 tok/s (about
1.12x)** and it would falsify the 21 target rather than meet it - which is what the goal asked for.

### The serving `Compute` can be built from pieces that exist, with one thing to get right

Checked before assuming, because the last five rounds were a record of what happens when a nearby buffer is used
without checking that it is the one the arithmetic used:

* **the MoE kernels are expert-*list* bound, not slot-bank bound.** They take `routedBlobs: [MTLBuffer]` and
  `routedOffsets: MoEExpertOffsets` (`MoE.swift:330`, `:393`, and `MoEExpertOffsets` at `:9`), so a peer can
  evaluate **any** expert list it is asked for rather than only what occupies its own layer's slots;
* **`fetchRoutedExperts(layer:experts:)` exists** (`ModelExpertIO.swift:266`) and takes explicit expert ids - the
  same entry point the request path already uses, so a `Compute` shares the node's cache by construction rather than
  by discipline;
* **the argument buffer is built by `makeRoutedArgumentBuffer(routedBlobs:...)`**, which validates the blobs against
  `topK` before encoding.

**So the shape is:** `fetchRoutedExperts` for the requested ids -> `makeRoutedArgumentBuffer` -> phase 1 with the
supplied activation as `x:` and a scratch as `acts:` -> a down projection -> read back `[Float]`.

**And every input to that shape is a public model method**, checked rather than assumed:

    model.fetchRoutedExperts(layer:experts:)   -> [TensorView]        ModelExpertIO.swift:266
    model.routedExpertOffsets(layer:)          -> MoEExpertOffsets    called at RealForwardRunner+Decode.swift:1379
    moe.makeRoutedArgumentBuffer(routedBlobs:) -> MTLBuffer           MoE.swift:330
    moe.encodeRoutedPersistentPhase1U16Load(x:acts:...)               MoE.swift, called at :1528

There is no missing accessor and no plumbing to add: a serve-side `Compute` is a new function that calls four things
the request path already calls, plus a command buffer and a readback.

**And the one thing to get right is the down projection, because the existing one does three jobs at once.**
`encodeRoutedPersistentPhase2Reduce` applies the **routing weight**, **reduces** across slots and folds in the
residual - and a peer must send the **unweighted per-expert output**, because `D154`/`D168` put the weight on the
side that owns the router's decision, which is the requester. A `Compute` built on the reduce would send
`w * value` and the requester would multiply by `w` again: **a wrong number, silently, and one the bit-exactness
contract cannot catch** because each side is internally consistent.

**Two ways to get it.** Run the existing phase 2 with **unit routing weights** and `k = 1` per expert, which yields
the unweighted per-expert down output with no new kernel; or add a down-only encode. **The first reuses a kernel that
is already verified; the second is clearer.** Either way the trap is the same one `D168` records on the requesting
side, mirrored - and it is worth writing down before the code, because it is invisible in a passing test.

### The serve path has an async/sync seam, and it is the last thing to decide

Two facts that only meet when the code is written:

* **the method belongs on `RealForwardRunner`.** It holds `model: Model` (`:144`), `moe: MoE` (`:173`),
  `moeActs: MTLBuffer` (`:270`) and the device - everything the serve path needs, and nothing else in the tree has
  all four;
* **`model.fetchRoutedExperts(layer:experts:)` is `async throws`** (`ModelExpertIO.swift:266`) while
  **`ShardExchangeServer.Compute` is synchronous**: `(Int, [Int], [Float]) throws -> [Float]`.

**So the obvious implementation - a synchronous `Compute` that blocks on an async fetch - is the one to avoid.**
Bridging async to sync inside the server's accept loop means blocking a thread that may be a cooperative-pool thread,
and **`D197` is this repository's own record of exactly that failure**: `ShardExchangeIntegrationTests` hung for a
whole session because a blocking accept sat inside a `Task` and starved the very task that had to connect to it. The
fix there was to move the blocking call to `DispatchQueue.global()`; the same move here would work but hides the
cost, and the server would then hold a thread per request.

**Two clean ways, and they differ in what they cost:**

1. **Make `Compute` async** - `(Int, [Int], [Float]) async throws -> [Float]` - and let `answer(_:)` await it. The
   server already runs off the cooperative pool (`D197`), so this is the natural shape and needs no thread to be
   blocked. It changes `ShardExchangeServer`'s signature and its tests, which is the cost.
2. **Give the runner a synchronous fetch.** The MoE path already calls `routedExpertBuffers(for:)` and
   `routedExpertResidentIDs(layer:)` synchronously before falling back to the async fetch, so a synchronous
   "buffers for these ids, populating the bank if needed" is a shape the engine already contains.

**Neither is chosen here, and the choice should be made deliberately**, because the wrong one produces a server that
passes its tests and then deadlocks under a real four-node run - which is precisely the failure `D197` cost this
repository once already.

### Decided: the serve path is synchronous, and it needs no change to the server

`D245`-style, the decision is made by checking rather than by choosing. The MoE path already reaches the expert
buffers **synchronously**:

    model.planRoutedExperts(layer:experts:...)   ModelExpertIO.swift:107   synchronous
    model.routedExpertBuffers(for: plan)         ModelExpertIO.swift:165   synchronous
    model.routedExpertResidentIDs(layer:)        ModelExpertIO.swift:190   synchronous

while the async ones (`beginFetchRoutedExperts:230`, `fetchRoutedExperts:266`) are the *fallback* the path takes when
the synchronous route has nothing. **So a serve-side `Compute` can be written entirely on the synchronous entry
points**, which means `ShardExchangeServer.Compute` keeps its current signature, `answer(_:)` keeps its shape, and
none of its tests change. **Option 2 of the two recorded, and it is the one that costs nothing.**

**What that leaves is genuinely small.** A synchronous method on `RealForwardRunner` (which holds `model`, `moe` and
`moeActs`): plan for the requested ids, `routedExpertBuffers(for:)`, `makeRoutedArgumentBuffer`, one
`encodeRoutedPersistentPhase1U16Load` with the supplied activation widened from `[Float]` to the fp16 `x:` buffer,
a down projection, and a readback.

**One unknown is left and it is mechanical**: the runner holds **no command queue property** - a grep for
`commandQueue`, `queue` or `metalQueue` finds none, and the encode helpers all receive a `commandBuffer` from their
caller. So the serve method needs a queue, which means taking one from `context` or from `MoE`. **That is the last
thing to look up before writing the function**, and it is a one-line question rather than a design one.

**And the trap recorded earlier still stands unchanged**: the down projection must not apply the routing weight,
because the requester applies it. A `Compute` built on `encodeRoutedPersistentPhase2Reduce` as it stands would send
`w * value` and the requester would multiply by `w` again - silently, and invisibly to a bit-exactness contract that
each side satisfies on its own terms.

### The last unknown is closed: the queue is `ctx.queue`

The serve method needs a command queue and the runner holds no such property directly. The decode path shows where
it comes from - `ctx.queue.makeCommandBuffer()`, used at `RealForwardRunner+Decode.swift:249`, `:286` and `:391` -
so `ctx` is in scope on the runner and carries the queue.

**That completes the specification, and every part of it is now a checked fact rather than a plan:**

| piece | where | state |
| --- | --- | --- |
| the model and kernels | `RealForwardRunner.model` `:144`, `.moe` `:173`, `.moeActs` `:270` | in scope |
| a synchronous expert fetch | `planRoutedExperts` `ModelExpertIO.swift:107`, `routedExpertBuffers(for:)` `:165` | synchronous, no server change |
| the argument buffer | `moe.makeRoutedArgumentBuffer(routedBlobs:)` `MoE.swift:330` | exists |
| the phase-1 encode | `encodeRoutedPersistentPhase1U16Load(x:acts:…)`, called at `:1528` | exists |
| a command queue | `ctx.queue`, used at `:249`, `:286`, `:391` | in scope |
| the reply layout | `[d * 8 + slot]`, `ShardExchangeParticipant.remotePartials` | tested |
| the weight trap | the down projection must **not** apply the routing weight | recorded |

**What remains is the function and the four-node run, and nothing else.** The method is synchronous, it shares the
node's cache because `planRoutedExperts`/`routedExpertBuffers` are the same entry points the request path uses, it
needs no change to `ShardExchangeServer`, and its one silent-failure mode - sending `w * value` when the requester
also multiplies by `w` - is written down before the code.

### The last real piece: phase 2 reduces, so a peer needs either a new kernel or N encodes

The serve path's down projection was the one step written as "a down projection" without saying which. Checking it
finds that **the existing one cannot do the job**:

* `encodeRoutedPersistentPhase2Reduce` (`MoE.swift:555`) takes `routingWeights`, `residual` and writes a **single
  `y:`** - it **reduces** across slots. That is the right shape for the node that owns the router's decision and the
  wrong shape for a peer, which must return **one row per expert**;
* a grep for a down-only or per-expert output encode finds **nothing** - and one candidate that looks like it is
  not one. `moe_phase1_2_routed`, which appears in the kernel profile at 190 dispatches, is a **diagnostic role
  label** attached to the phase-1 + phase-2 chain at `RealForwardRunner+Decode.swift:1850` and named in
  `RealForwardRunner+Diagnostics.swift:75`, **not a separate kernel**. Following it costs a search and yields
  nothing, so it is recorded here rather than left for the next attempt to chase.

**So the last piece is real work, not wiring.** Two ways, and they differ in kind:

1. **A new down-only encode** - the same down projection with the weight and the reduce removed, writing
   `[expert][d]`. It is a small kernel, it is the clean shape, and it means a new `.metal` source, a new PSO and its
   pipeline state, which is why it is not written here.
2. **N single-slot phase-2 encodes** with **unit routing weights** and a zero residual, one per requested expert, so
   each call reduces a single slot and yields that expert's down output. No new kernel, but `experts.count` encodes
   per request and `experts.count` readbacks - and at eight experts a layer, forty layers, that is 320 encodes and
   320 readbacks per token **on the serving node**, which is very likely to dominate the exchange cost that `D208`
   measured at 3.2 ms on raw frames and explicitly excludes this.

**The choice is between a kernel and a lot of dispatches, and it should be made knowing the second's cost is
measured nowhere.** `D114` is the precedent for what dispatch counts do to this engine - 640 to 80 was **1.27x** -
and the second option adds 320.

**This is the honest end of the specification.** The serve path is not one function on existing pieces; it is one
function plus either a small kernel or a dispatch pattern whose cost is unknown. **Everything before it is built and
verified** - the exchange on both sides, the requesting half wired and behaviourally verified, the queue and the
synchronous fetch located - and this is what stands between that and a four-node number.

### The serve path's call sequence, with the real parameter names

"The down projection" is the phrase that hid the kernel. This is the same step written so it can be typed:

```
// once, on the runner: an fp16 activation buffer [D], an fp16 acts scratch [FmoE], a float y [D],
// a float weights[1] = 1.0, a float residual[D] = 0, all persistent - NOT per request, per D114.
let plan   = try model.planRoutedExperts(layer: layer, experts: experts)   // ModelExpertIO.swift:107, sync
let blobs  = try model.routedExpertBuffers(for: plan)                      // :165, sync -> [TensorView]
let offsets = try model.routedExpertOffsets(layer: layer)                  // MoEExpertOffsets

// write `activation` into the fp16 activation buffer, widened.

// ONE EXPERT AT A TIME, because phase 2 REDUCES and has no per-expert output:
for (i, expert) in experts.enumerated() {
    let cb = ctx.queue.makeCommandBuffer()
    try moe.encodeRoutedPersistentPhase1U16Load(
        commandBuffer: cb, routedArgBuffer: argBuf,
        routedBlobs: [blobs[i].buffer], routedOffsets: offsets,
        x: activationBuffer, acts: actsScratch, d: D, f: FmoE, topK: 1)
    try moe.encodeRoutedPersistentPhase2Reduce(
        commandBuffer: cb, routedArgBuffer: argBuf,
        routedBlobs: [blobs[i].buffer], routedOffsets: offsets,
        acts: actsScratch,
        routingWeights: unitWeights,        // [1.0] - the WEIGHT BELONGS TO THE REQUESTER (D168)
        residual: zeroResidual,             // zeros - the residual too
        y: yBuffer, d: D, f: FmoE, topK: 1) // remotePartials: nil
    cb.commit(); cb.waitUntilCompleted()
    // read yBuffer.contents() as [D] floats into out[i * D ..< (i + 1) * D]
}
```

**`topK: 1` in both calls is the whole trick**, and `validate(routedBlobs:topK:)` enforces it: one blob, one
slot, so the reduce has nothing to reduce and `y` is that expert's down output. **The weights are 1.0 and the
residual is zero for the reason `D168` gives** - the requester owns the router's decision and applies the weight, so
a peer that applied it too would be counted twice, silently.

**What this costs, stated so the run can be interpreted.** Per request: `experts.count` phase-1 encodes,
`experts.count` phase-2 encodes and `experts.count` blocking waits. On the serving node at eight experts a layer and
forty layers that is **640 encodes and 320 blocking waits per token**, which is the shape `D114` measured at
**2.47 s/step against 0.92** when buffer handling went wrong - not the same failure, but the same order of
dispatches. **The four-node number this produces will be dominated by the serving node's dispatch pattern, not by
the exchange**, and that must be said when it is reported.

**The clean alternative remains a down-only kernel** writing `[expert][d]` in one pass, which removes all of it.
