# The distribution: what is built, what is proven, and what is not

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
