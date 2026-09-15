# M1 decisions

The decisions M1 rests on, with the evidence that forced each. `m0-decisions.md` holds M0's
`D1`–`D7`; the numbering continues here.

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

## The cache: implemented, measurable, and **not yet trusted**

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

**The last two rows are why the cache is not trusted yet.** On the tiny fixture my two paths
produce the same tokens, and so do the reference's — but on the **real** model my cached and
uncached paths agree on the first token and diverge from the second (`11751, 13, 561, 6511`
against `11751, 11, 264, 3177`). That asymmetry is what makes it a suspected bug rather than a
`D8` path difference: a numeric-path difference of the measured size usually leaves the argmax
alone, which is precisely what the reference shows on the same fixture.

So the state of this work is: **implemented, measured, and withheld from the gate** until the
divergence is isolated. The experiment that will settle it is per-layer: cache and no-cache over
the same prompt on the real model, comparing each layer's `hidden_out`, so the first layer that
disagrees names the bug. Until then the cached path is an optimisation under investigation, and
M1's gate continues to rest on the uncached path, which is bit-identical to the contract.
