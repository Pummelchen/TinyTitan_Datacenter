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
