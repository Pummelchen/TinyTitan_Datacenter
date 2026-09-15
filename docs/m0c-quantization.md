# M0c — what 4-bit costs (`DC-031`)

The M0 gate says the engine matches its reference in bf16. M0c asks the next question:
**what does 4-bit cost?** Measured on the pinned `Qwen/Qwen3.5-2B` at revision `15852e8c…`,
2026-09-15, using the gate's own instruments and its frozen prompt set.

## The install

| | |
| --- | --- |
| Checkpoint | 4,548,285,948 bytes (bf16, all 632 tensors) |
| **Install** | **1,813,618,928 bytes** — **2.51× smaller** |
| Quantized | 320 tensors, **7.71 bits/weight** including scales and zero points |
| Kept higher | 170 tensors (embedding, head, norms, the Gated DeltaNet's decay and convolution) |
| Build time | 18.7 s |

The 7.71 figure is not "4-bit" and should not be read as such: it is the average over the
quantized tensors *including* their per-group scales and zero points, and the model as a whole
also carries a 1 GB bf16 embedding. The quantized roles are the ten matmul families
(`attn.q/k/v/o`, `linear.in_qkv`, `linear.in_z`, `linear.out`, `mlp.gate/up/down`).

The layout is **ours, not a transcode**: group-wise affine int4, group 64 along a row, fp32
scales, int4 zero points, low nibble first. I5 says to transcode rather than requantize where
a vendor ships a native 4-bit format — DeepSeek does, and this family does not, so there is
nothing here to transcode and this is a genuine quantize. It is still recorded as a pass in
the install's provenance, with the policy file, per I6.

The policy is data (`tools/quant_policy.json`, I4): a role that is absent from it **stops the
install** rather than defaulting, because a default is how a policy stops describing the
model. Five roles are fp32 (the QK-norms, the gated norm, `A_log`, `dt_bias`) and seven are
bf16 (norms, embedding, head, the Gated DeltaNet's input gates and convolution) — I3 is
explicit that what decides a discrete outcome stays at bf16 or above.

## What it cost

The contract ran against the install on the frozen prompt set, with the dequantizer in place
of the checkpoint reader — the same contract code, so the difference measured is the
quantization's and nothing else.

| prompt | decisions preserved | worst relative divergence | smallest top-1 margin |
| --- | --- | --- | --- |
| `capital` | **5 / 5** | 1.64e-01 | 1.2448 |
| `fibonacci` | **17 / 18** | 2.68e-01 | 0.2460 |
| `boiling` | **6 / 6** | 2.49e-01 | 0.1243 |

**28 of 29 next-token decisions survive.** The one that does not is `fibonacci` position 1,
where bf16 predicts token 15 and 4-bit predicts 220 — a prompt whose margins are the
smallest of the three. From that position the two continuations are unrelated, which is
precisely the failure I3 describes: every numeric check would still look reasonable, and the
output is different.

For scale: the same comparison between the fp32 engine and the fp32 reference diverges by
about **2e-06 relative**. Quantization raises that to **1.6e-01 – 2.7e-01** — four to five
orders of magnitude — and costs one decision in twenty-nine. That ratio is the thing to
carry into M1, where the experts are quantized but the router, per I3, is not.

## The engine reads the install, bit for bit

Swift decodes the same packed codes, the same group scales and the same zero points, and the
two implementations agree **byte for byte** on all three frozen prompts — 40,683,520 bytes of
trace data, matching digests, the differ reporting `IDENTICAL`. There is no rounding in this
path to disagree about: the codes, the zero points and the scales are integers and exact
arithmetic, which is why this comparison can be byte equality rather than a tolerance, and why
it is a stronger statement than the bf16 path's.

The install carries its own **IR spec**, so the engine needs nothing beside it — and the
configuration the forward pass uses is reconstructed from that spec, which is the check that
the spec is sufficient to run the model rather than a description of one (L1).

## What this does not yet show

- **One prompt set, all short.** Quantization's effect on the Gated DeltaNet's recurrence
  over long sequences is exactly the kind of thing the brief warns is invisible at short
  context; M0c does not claim otherwise.
- **The flip is not characterised further.** Whether it is caused by the attention
  projections, the MLP, or the Gated DeltaNet's in/out projections is not yet attributed;
  the policy is per-role precisely so that question can be answered by changing one entry
  and re-running.
