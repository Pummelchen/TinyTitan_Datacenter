# Where the exchange joins the arithmetic

The one remaining edit, and the constraint that decides its shape.

## The seam

`MoE.encodeRoutedPersistentPhase2Reduce` (`sources/TinyTitan/Kernels/MoE/MoE.swift:539`) dispatches
`moe_phase2_down_reduce_k8`, which computes, per dimension `d`:

```
partial[sg] = float(routing_w[sg]) * value      // sg = 0 … 7
acc(float)  = float(residual[d])
acc        += partial[0]
acc        += partial[1]
…
acc        += partial[7]
y[d]        = half(acc)
```

The decode loop reaches it from `encodeDecodeRoutedMoE` (`RealForwardRunner+Decode.swift:1346`), which plans
the layer's experts at `:1387` through `ModelExpertIO.planRoutedExperts` → `PreadExpertStreamer.planExpertsCached`
→ `makeExpertCachePlan`, where `ownedExpertFilter` already narrows the routed set to what this node owns.

## The constraint: the host cannot form the sum

The obvious wiring — let a node reduce its own slots and add each peer's reduced value on the host — is
**not bit-exact**, and it is worth being precise about why, because it is the mistake that looks correct.

A node owning slots {0, 4} would compute

```
y_local = residual + partial[0] + partial[4]
```

and a peer owning the rest would send

```
y_peer  = partial[1] + partial[2] + partial[3] + partial[5] + partial[6] + partial[7]
```

Adding them gives `(residual + partial[0] + partial[4]) + (partial[1] + … )` — a **different association**
from the reference's `residual + partial[0] + partial[1] + … + partial[7]`. In fp32 that is a different
number, and `D154`/`D168` exist because it is. The `ShardReduce` tests demonstrate the sensitivity directly:
`1e8 + 1.0 - 1e8` is `0.0` while `1e8 - 1e8 + 1.0` is `1.0`.

So **the peer's partials must reach the ordered sum in their own slot positions**, which means the kernel has
to see them. Node-level results cannot be combined on the host.

## The edit

Give the phase-2 kernel the peers' partials as an input, and add them into `partial[sg]` **before** the
ordered sum:

```metal
partial[sg] = float(routing_w[sg]) * value;
if (remote != nullptr) { partial[sg] += remote[d * 8 + sg]; }   // fp32, zero where a peer owns nothing
acc = float(residual[d]);
acc += partial[0]; … acc += partial[7];
```

`remote` is a `[d][8]` fp32 buffer, **zero-filled in every single-node run**. Adding zero is exact, so the
single-node path keeps the bit pattern it has today, and a node's own slots simply have their remote entries
zero while a peer's have its contributions.

That is the whole of it, and it is one buffer and one line of kernel, because the arithmetic properties that
make it exact — fixed k = 8, fp32 partials, slot order, zero padding — are already the reference's.

## What the peer sends

`value` for each of its slots, **not** `routing_w[sg] * value`. Two reasons:

* The routing weights are the *same* on every node — the router ran identically — so sending the product would
  send numbers the receiver can already derive, and sending it again risks the two disagreeing if the gate's
  rounding ever differed.
* The product is the last operation before the sum, so computing it at the node that owns the expert keeps the
  order `routing_w × value` in exactly one place.

The frame is therefore `ShardExchange.Reply`'s existing shape: a slot list and fp32 rows, one row per slot.
`ShardExchangeParticipant.contributions(layer:experts:slots:activation:dims:)` already returns them placed in
their slots, and `ShardReduce.reduceRows` is the host-side check that the whole thing equals the single-node
sum — used in tests, not in the decode path, where the kernel does the sum.

## Sizing, which is not free

`D179`: replicating R experts costs `R × 40 layers × 1,769,472 B`, so R = 64 is 4.53 GB and does not fit
alongside `D178`'s measured 2.83 GB cache in 8 GB. The exchange must be sized against what actually fits, and
the per-step wire volume is the thing to measure before the four-node run, not after it.
