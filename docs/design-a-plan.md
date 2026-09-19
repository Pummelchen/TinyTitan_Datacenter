# Design A — layer pipeline: implementation plan

**Baseline preserved before any change.** Fork tag `pre-design-a-baseline` (HEAD `a15d25b`, the expert-sharding
engine with a correct three-node exchange measured at 0.85x a single node); main tag `pre-design-a-baseline`
(HEAD `8526edd`, the four 35B tests and the design document). Bundles in `~/tt-backup/` and on the node3.
**Nothing below changes the expert-sharding path, and the tag is what makes that recoverable.**

## What Design A is

Node *i* owns **ten consecutive layers** and **all 256 experts of those layers**, resident on its own SSD with a
per-layer cache. Only the hidden state crosses the wire. From `docs/distribution-design.md` section 4:

```
node0 L0-L9 ──4KB──> node1 L10-L19 ──4KB──> node2 L20-L29 ──4KB──> node3 L30-L39 → head → sample
   ^                                                                                        |
   └─────────────────────────────── one token index, 4 bytes ────────────────────────────────┘
```

**It is a ring, not a chain (`D311`).** A generation is a closed loop: each token's forward pass ends in a sample and
the sampled token is the next token's input, so the last stage's result must return to the first stage. The forward
leg carries 4 KB of hidden state per hop and the return leg carries **one token index** - closing the loop costs
nothing on a 117.8 MB/s wire. The chain above is right for a *single* forward pass and wrong for *generation*.

| | measured input | 4-stage projection |
| --- | --- | --- |
| compute per stage | 130 ms for 40 layers | **32.5 ms** |
| stream per stage | 566 MB/token active, 68.1% cached, 1.8 GB/s | **25.1 ms** |
| wire per stage | 4 KB hidden state | **0.1 ms** |
| **stage time** | | **max = 32.5 ms → 30.8 tok/s** |

## The seams, located

| seam | where | what changes |
| --- | --- | --- |
| **decode layer loop** | `RealForwardRunner+Decode.swift:192`, `for L in 0..<cfg.numLayers` | becomes the node's **layer range** |
| prefill layer loop | `RealForwardRunner+Prefill.swift:452`, `for L in layers` | **already takes a layer array** - precedent for the above |
| activation buffer | `routedX` in `encodeDecodeRoutedMoE` | becomes the payload between stages |
| head | `fusionHead`, last stage only | stages 0..n-2 return the hidden state, not logits |
| transport | `DecodeTCPSocket`, `ShardPeerChannel` | reused as-is for the hidden-state frames |

## The stages, in order

**A1. Layer range on the runner.** A `layerRange: Range<Int>` the decode loop honours, defaulting to all layers so
the single-node path is untouched. **This is the whole of the change that makes a node own layers**, and it is
verifiable on one node: a node with `0..<10` must produce a hidden state, and one with `0..<40` must still produce
the tokens it produces today. **Bit-identity on the unchanged configuration is the gate.**

**A2. The pipeline frame.** One frame carrying `[D]` fp16 plus a token index and a layer index, sent after the last
owned layer and awaited before the first. Reuses `ShardPeerChannel`'s framing.

**A3. Embed and head placement.** Stage 0 embeds; the last stage runs the head. A middle stage neither embeds nor
projects to vocabulary.

**A4. Chunked prefill.** Prefill by 4,096-token chunks per stage, because that is the chunk size the record
established for the ANE and because a chunk is what amortises a stage's weight reads.

**A5. Exactness at a layer boundary, then the ring.** `0:40` with a `hiddenOut` on layer 20 recorded, against
`0:20` with the same `hiddenOut`, compared **for the same token** - a forward pass truncated at the same point, which
tests that a stage's output is a function of its input and its own layers and nothing else. **A file handoff between
two separate runs cannot work** (`D311`): the first stage would embed the token its own truncated forward pass
sampled, so the states it wrote belong to the wrong tokens. **Then two-node measurement**, then four. The gate is the same discipline the whole record uses: node,
configuration, generation length, loads, and a median over repeats.

## What will falsify it

- **A stage time above 48 ms** - the target's budget - on the 35B, which would mean the projection's `max()` is
  wrong because streaming is not hiding behind compute.
- **A hidden-state frame that does not reproduce the single-node trace** when the range is made contiguous. Design A
  is exact by construction or it is not Design A.
- **A per-stage cache hit rate far below the measured 68.1%**, which would break the streaming term.

## What is NOT claimed

**Nothing here is built.** The ~31 tok/s is a projection from measured inputs - 130 ms of compute, a 68.1% hit rate,
566 MB of active bytes, 1.8 GB/s - and `docs/distribution-design.md` section 12 says so. **The 35B's ten layers is
4.53 GB against ~4.5 GB usable**, so a 35B run will look better than the 120-180B target ever will, and any result
must be reported with the cache deliberately capped.
