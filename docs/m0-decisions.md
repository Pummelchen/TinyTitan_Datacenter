# M0 decision record (D1–D7)

Recorded 2026-09-15. Each decision names what it settles, the technical consequence, and
what it costs. Where the operator delegated the choice ("use the best technical
decision"), the reasoning and the numbers are here rather than in a commit message.

---

## D1 — M0 model: Qwen3.5-2B, 4-bit

**Decided by the operator. Accepted, with one conflict flagged and a staging proposal.**

Verified from the checkpoint, not from the model card (`Qwen/Qwen3.5-2B`, revision
`15852e8c16360a2fea060d615a32b45270f8a8fc`):

| | |
| --- | --- |
| Layers | **24: 18 `linear_attention` + 6 `full_attention`** (three linear, then one full, repeated) |
| Linear attention | Gated DeltaNet: `linear_conv_kernel_dim: 4`, key/value head dim 128, 16 key heads, 16 value heads |
| Full attention | 8 heads / 2 KV heads, `head_dim: 256`, `attn_output_gate: true` |
| Dense parts | hidden 2048, intermediate 6144, `rms_norm_eps: 1e-6`, tied embeddings, MTP head (1 layer) |
| Context / vocab | 262,144 / 248,320 |
| Checkpoint | `Qwen3_5ForConditionalGeneration` — **it carries a vision tower** (`Qwen3_5VisionModel`, patch embed/merger) and a `preprocessor_config.json`; 4,548,221,488 bytes in one shard |

**The conflict.** M0's stated purpose is to prove the harness on something trivially
conventional, so that a mismatch can only be the harness. Qwen3.5-2B is not that. Its
first layer is a **chunked Gated DeltaNet**, and the reference implementation
(`transformers` v5.17.0, `modeling_qwen3_5.py:301`) is a chunked delta rule with
cumulative decay, pairwise decay matrices, `l2norm` on q/k, a causal `conv1d` and a
gated norm — plus 4-bit dequantization if the 4-bit form is used from the start. That is
the kernel family the brief assigns to M4/M5, and it would be the first thing the
bit-exactness harness is asked to validate.

**Why it is still defensible.** D2 keeps Qwen3.6-35B-A3B as M1, and 30 of that model's 40
layers are the *same* linear-attention family. Building the Gated DeltaNet kernel in M0
therefore front-loads work M1 needs anyway instead of duplicating it. The cost is
schedule risk in the milestone that is supposed to be cheap, not wasted effort.

**Proposal — keep the model, stage the milestone (needs a yes/no):**

| Stage | Model | What it proves |
| --- | --- | --- |
| **M0a** | a synthetic deterministic model, plus `Qwen3-1.7B` (conventional, contract already written) | the capture and diff harness works **and can fail** — seeded 1-ULP perturbation, a transposed tensor and a flipped top-k each get located. No exotic kernel involved. |
| **M0b** | Qwen3.5-2B, **bf16** | the real architecture bit-exact: embedding → norms → full-attention layers → Gated DeltaNet layers → MLP → tied head, gated layer by layer, conventional path first. |
| **M0c** | Qwen3.5-2B, **4-bit** | the quantization/transcode path as a *separate* variable, with I5 provenance, gated against the 4-bit reference. |

Splitting bf16 from 4-bit is the point: if 4-bit lands in the same step as the first
bit-exactness proof and the traces disagree, two independent error sources are confounded
and the harness cannot tell them apart — which is exactly what the brief's "bf16, no
quantization" was protecting against. 4-bit is still delivered in M0; it is sequenced,
not dropped.

**Also required by this choice:** the importer maps the **text tower only**
(`Qwen3_5TextModel`); the vision tensors are 2.3 B-ish params of the 4.55 GB checkpoint
and must be filtered by name, not silently loaded.

---

## D2 — M1 model: Qwen3.6-35B-A3B stays

**Accepted.** Consequences recorded rather than re-argued: M1 includes 30 linear-attention
layers of 40 (the same Gated DeltaNet family as D1's model, so M0's kernel work carries
forward), plus an attention output gate and an MTP head. M1 is not the cheap validation
step the brief describes; it is the first *streaming* milestone on a hard architecture.

Two things that reduce it: the sister project already runs this model single-node, so
there is an implementation to compare against (licence boundary: `DC-013`), and M0b will
have already validated the linear-attention kernel that M1 depends on.

---

## D3 — M0 gate definition (delegated: best technical decision)

**Decision: both sides compute in IEEE fp32 — weights stored bf16 (or 4-bit), cast to
fp32 at use — with a fixed op decomposition and a fixed reduction order. The gate is a
byte comparison of traces. HF's bf16 model is used for *semantic* checks and exact
discrete decisions, never as the bit-exactness target.**

Why: bit-equality is only *provable* where both sides are IEEE-754 fp32 with a controlled
addition order. bf16 matmul accumulates in an implementation-defined order, so "bit-match
HF bf16" is not a goal that can be met by construction — it can only be hoped for, and a
hope is not a gate. fp32 accumulation also makes I3 cheap, since the discrete decisions
are then taken from full-precision logits.

Cost: fp32 activations roughly double activation memory and arithmetic against bf16. On an
8 GB node with a 2 B model that is affordable; it must be revisited before M4/M5, and if it
is ever relaxed, the gate is re-derived and recorded here — not silently loosened.

Reference harness configuration: `torch.float32` compute, `attn_implementation="eager"`,
`torch.use_deterministic_algorithms(True)`, thread count pinned, model revision pinned,
prompt set frozen, every one of those recorded in the trace header.

**Measured 2026-09-15**, on a tiny `qwen3` built from a config (4 layers, 64 wide, no
download — the plumbing and the reproducibility question settled before the checkpoint is
fetched): two captures of the same input with the same configuration are **bit-identical**,
and one thread versus four threads produced the **same bytes**. The thread count is still
pinned and recorded — the real checkpoint re-measures it, and a fact about a 164k-parameter
model is not a licence to assume it at 2 B.

The same experiment quantifies what a tolerance would have cost: bf16 and fp32 differ in
**every element**, with the absolute divergence growing through the stack (1.2e-4 at the
embedding, 2.8e-2 at the final norm on values of scale 2.7), while *relative* error reaches
13285% because those activations sit near zero. A relative bound is therefore both
unmeasurable and blind — it cannot see a top-k index flip at all.

**Measured on a real layer, 2026-09-15 — this is what pinned the gate down.** A matmul of
the shapes this model uses (8×1024 by 512×1024, fp32) computed with an explicitly ordered
accumulation — ascending `k`, one product added per step — does **not** reproduce
PyTorch's matmul: **3544 of 4096 outputs differed**, mean 17 ULP, max 22587 ULP. Both
results sat at exactly the same distance from an fp64 computation of the same product
(3.148e-07), so the difference is summation order, not accuracy — and PyTorch's order is
a property of its BLAS kernels, not of the model. No engine can be required to reproduce
it, and no Metal kernel ever will.

So M0 has **two references, with two different jobs**:

| Reference | Job |
| --- | --- |
| `tools/trace_capture.py` (torch) | the **semantic oracle**: is this the right model? Exact discrete decisions (I3), closeness per tensor |
| `tools/ordered_reference.py` | the **numeric contract**: a `qwen3` dense forward in which every sum has a stated order — ascending index, one product per step, no reassociation, no fused multiply-add, fp32 throughout. The engine's bit-exactness gate compares against this |

Validated on the real `Qwen3-0.6B` over 28 layers and 8 tokens: the ordered forward agrees
with torch semantically — **the logits' argmax agrees at every position**, mean |Δ| 8.6e-6
at the final norm — while never being bit-identical from the first matmul onward (mean |Δ|
grows 1.6e-7 → 2.8e-5, worst element 9.3e-3). That gap is the honest measure of what
"faithful port" means when the only difference is the order of the additions: same model,
same decisions, different last bits.

Consequences, recorded before any kernel is written:

- **The engine accumulates in the contract's order** — no split-K, no FMA, no
  reassociation — because that order is what makes I1 and I2 testable at all.
- **I2 compares our own implementation to our own implementation.** Comparing the engine
  to torch could never be bit-exact, and pretending otherwise would have produced a gate
  that fails for a reason that has nothing to do with the engine.

---

## D4 — Reduction canon (delegated: best technical decision)

**Decision: nodes exchange their *per-expert contributions*, not partial sums. Every node
then accumulates all top-k contributions in ascending global expert-id order. The
exchange itself uses recursive doubling.**

Why: this is the only scheme in which "N nodes ≡ 1 node" holds *by construction* at any N,
because the additions are performed in the same order in both cases. Pre-summing per node
was measured to break the invariant in 56% of random top-8 draws (R10).

Cost, on the measured transports, per MoE layer, `k=8`, hidden 2048:

| | payload per node (N=4) | transmit at 1 GbE | exchange latency | per layer |
| --- | --- | --- | --- | --- |
| Per-expert contributions, recursive doubling | ~2 experts × 8 KB = 16 KB | 0.13 ms | 2 rounds × 0.32 ms | ~0.77 ms |
| Flat all-to-all of all k | k × 8 KB = 64 KB | 0.52 ms | 1 round | ~0.85 ms |

The topology only moves the exchange, never the summation order, so determinism is
independent of N and of the transport. On the Thunderbolt bridge the transmit term
practically disappears (~13 µs), leaving latency — which is why the bridge matters for M2
and why M2 must not silently run over the VPN path.

Tie-break and ordering rule: experts are ordered by ascending global expert id,
independent of which node owns them; a node owning none of the selected experts sends
nothing and contributes nothing.

---

## D5 — Router precision (delegated: best technical decision)

**Decision: router computation — logits, normalization, comparison and top-k selection —
runs in fp32. Router *weights* may be stored bf16 or quantized, because the accumulation
is fp32 either way. Ties are broken by ascending expert id, and the reference applies the
identical rule so both sides agree by construction rather than by luck.**

Evidence: bf16 logits flip the top-k. Measured over 20,000 random 256-way routers with
round-to-nearest-even bf16, 922 (4.61%) changed the top-8 index set — for example fp32
logits 5.363673687 and 5.385571480 both become bf16 5.375, and the ordering that
separated them is gone (R11). One flipped index diverges the output while every
per-tensor check stays green. bf16 has 8 mantissa bits and is not "bf16 or higher" in any
useful sense; fp32 is.

---

## D6 — Reference machine: the four M2 nodes only

**Accepted — no other chips, no rented hardware.** The consequence needs stating, because
it interacts with D3:

A 2 B model in fp32 is ~9.2 GB of weights, so **an fp32 reference cannot be resident on an
8 GB node**. It does not have to be: the reference runs **layer by layer with
memory-mapped weights, casting each layer to fp32 as it is used**. The largest single
layer here is the MLP (3 × 2048 × 6144 ≈ 37.7 M params ≈ 151 MB in fp32), so a
layer-at-a-time fp32 reference fits comfortably; it is slow, and slow is fine for offline
trace generation (9.2 GB streamed at the measured 1161 MB/s ≈ 8 s per forward pass).

This also means the reference and the engine share a memory strategy — mmap, one layer
resident at a time, bf16 storage cast to fp32 at use — which makes them comparable
without either side needing the whole model in RAM.

**Validated on the real checkpoints (2026-09-15).** `tools/trace_capture.py` has two
paths: resident, and `--from-disk`, which builds the model on the `meta` device and loads
one decoder layer at a time. On `Qwen3-0.6B` the two paths produce **bit-identical** traces
(same digest, 87 tensors), so the streaming loader is the reference rather than an
approximation of it. On the M0 model `Qwen3.5-2B` — 18 Gated DeltaNet layers, ~9.2 GB of
fp32 weights — two independent layer-by-layer runs are **bit-identical** (75 tensors,
digest `72793ffc…`) at a peak RSS of **2.78–3.17 GiB**. The resident path could not have
produced that trace on this hardware at all, which is the whole argument for D6.

For M4/M5 the same arithmetic applies at a scale (284 B, 3.25 GB of expert reads per
token) where it does not fit; that stays parked as `B5`/`B7` rather than being solved now.

---

## D7 — Naming: TinyTitan Datacenter

**Decided by the operator.** "Shard" was the brief's working title; the project is
**TinyTitan Datacenter** and the repository keeps its name. Consequences: the README's
stray "Shard uses expert parallelism" line is corrected (this closes the last item of
`DC-002` except the two wording nits), the wiki already uses the right name, and no
document introduces "Shard" again.
