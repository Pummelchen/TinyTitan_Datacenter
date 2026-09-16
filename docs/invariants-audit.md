# The invariants, audited

`docs/brief-response.md` assessed I1–I6 **before** M1 and M2 existed, and several of its entries are now
stale: I2 says "not yet testable", I4 says its sharding half "does not exist yet", and I6 says its gap "is
now fixed". This is the audit of where each invariant actually stands, with the evidence, and with what
would falsify it. The stale assessments in that document are marked rather than rewritten, because it is a
record of what was known when it was written.

**Statuses**, and nothing else: `verified` — there is evidence and it is current; `partly` — the mechanism
exists and some evidence does not; `not applicable yet` — the model or phase it applies to is not here;
`not yet` — nothing claims it.

| Invariant | Status | The evidence that is current |
| --- | --- | --- |
| `I1` bit-reproducibility | `verified` | three milestone gates, below |
| `I2` sharding is semantically free | `verified` | `tools/run_m2_gate.py`, `D17` |
| `I3` discrete decisions match exactly | `verified` | the M0/M1 gates' index sets |
| `I4` policy is data | `verified` | `tools/quant_policy.json`, `D20` |
| `I5` transcode, do not requantize | `not applicable yet` | the checkpoint is bf16 |
| `I6` provenance in the artifact | `partly` | `tools/verify_install.py` names the gaps |

## I1 — bit-reproducibility — `verified`

**The claim**: the same input produces the same bytes, runs and components alike.

**Evidence.** M0 matched the reference implementation on `Qwen/Qwen3.5-2B` with **40,683,520 bytes
identical** over three frozen prompts (`docs/m0-gate.md`). M1's trace on the real 35 B model is
**byte-identical to the contract** — 83 tensors, 40 discrete decisions, digest `b8c976c5e7ba8816…`
(`docs/m1-gate.md`). M2 produced that same digest from **half the experts on each of two machines**
(`b0d382dbabf36df0…` for the shared prompt, `docs/m2-decisions.md`), and the four-node mesh reproduced the
single-node trace exactly. The GPU unpack is asserted **bit-identical** to the scalar one across a grid of
shapes (`MetalUnpackTests`), and the reduction is asserted to be bit-identical across arrival orders and
partitions (`OrderedReductionTests`).

**What actually threatened it, four times**, was machinery rather than arithmetic — a slot bank rebuilt per
forward, a denormal flush the GPU did not perform (`D11`, then `D34` finding the GPU had never implemented
it), a test fixture whose synthetic scales contained NaN payloads the real install never has, and a memory
budget that would have swapped the node. Each is now either defined in the contract or asserted by a gate.

**What would falsify it**: any gate's `trace_diff` reporting a differing element, or a digest in
`docs/` no longer matching the artifact it names.

## I2 — sharding is semantically free — `verified`

**The claim**: the same model over N nodes produces what one node produces.

**Evidence.** `tools/run_m2_gate.py --remote …` on the real 35 B install: reference, node 0 and node 1 all
digest `b0d382dbabf36df0…`, with `trace_diff` reporting **IDENTICAL — 83 tensors, 0 elements, 40 discrete
decisions** on both nodes (`docs/m2-decisions.md`, `D26`). The four-node mesh (`--mesh`) produced the same
trace from all four machines (`D27`), and sharded **generation** produced the single-node tokens and digest
(`D28`), with the fixture and the real model both checked.

**The concern the brief-response raised is answered rather than merely met.** It warned that "the ring
membership must be pinned too, or two runs at the same N could disagree". `D17` went further: the
all-reduce carries per-`(token, expert)` contributions summed in that order, so **the ring's composition
cannot move a bit** — renumbering, re-partitioning or replacing a node changes which pieces arrive, never
the order they are summed in. What must be pinned is the **ownership map**, and `D20` pins it as data with a
digest.

**What would falsify it**: a shard plan whose nodes disagree with the single-node trace at any N; or an
all-reduce that mixes per-node partials, which `D17`'s measured counterexample (`(a+b)+c = 20000008` against
`a+(b+c) = 20000006`) shows would be detectable.

## I3 — discrete decisions match exactly — `verified`

**The claim**: the router's argmax and top-k **index sets** match, not merely the numbers they come from.

**Evidence.** M0's three prompts each matched the reference implementation's argmax index sets while the
numeric distance stayed near `2e-06`; M1 checked **40 discrete decisions** on the real model; M2 and M3
checked the same decisions per node. The comparison is a **separate assertion** from the numeric one, so a
marginal flip cannot hide behind a tolerance (`tools/trace_diff.py` reports both). Routers are kept at bf16
with the reason recorded in `tools/quant_policy.json`, because a router divergence leaves "every per-tensor
check still green".

**What would falsify it**: a trace comparison reporting equal tensors and a differing `router.topk` set —
which the differ would report as `N discrete decision(s) differ`, never as a tolerance.

## I4 — policy is data, not code — `verified`

**The claim**: the quantisation policy and the shard plan are **data** in the artifact, not logic in the
binary.

**Evidence.** `tools/quant_policy.json` is a per-role table with a `why` block justifying each entry, a role
missing from it **stops the install** (asserted by tests and by the verifier, which refuses a role the policy
does not name), and one file serves two model families. The **sharding half** now exists: `ShardPlan` is read
from a file, validated against the model actually opened — family and expert count — loaded by the node CLI,
carried with a canonical digest, and the harness generates one from the install manifest rather than
hard-coding a distribution (`ShardPlanTests`, `D20`, `D25`). A plan for a different model is refused with a
named reason rather than run.

**What would falsify it**: a policy or plan decision that can only be expressed by editing Swift.

## I5 — transcode, do not requantize — `not applicable yet`

**The claim**: a vendor's quantised weights should be transcoded, not decoded and re-quantised, because that
adds this project's error to the vendor's.

**Why it does not apply**: the M1 checkpoint ships bf16 (`docs/reference-qwen36-35b-a3b.md` records the
checkpoint's dtypes from its own config), so dequantise-then-requantise is the **only**
option and no vendor error is being compounded. It becomes live at **M4**, where DeepSeek's FP4 experts exist
to transcode from — and M4 has no weights on this farm (`DC-060`–`DC-064`), so this invariant is not merely
unverified but *unreachable* here.

**What would falsify it**: at M4, an on-disk 4-bit layout chosen for convenience rather than for closeness to
the vendor's block scaling.

## I6 — provenance in the artifact — `partly`

**The claim**: the converted artifact says which weights it came from.

**What the code does.** `tools/quantize.py` records `source.repo` and `source.revision` as **inputs**
(`--repo`, `--revision`) and, when they are not given, as **null** — explicitly not a placeholder, because
`"local"` "reads like an answer and is not one". `digest_snapshot` hashes every source weight file, through
the uncached descriptor, so a 67 GiB checkpoint does not become page cache.

**What the artifact on disk actually says.** The M1 install predates both fixes:

```
  provenance: source.revision is 'local', which reads like an answer and is a placeholder
  provenance: source.files is empty, so the source weights cannot be traced to their digests
  provenance: source.repo holds a commit hash (995ad96eacd9…) while source.revision is a placeholder,
              so the two look swapped — which is what the M1 install records
```

Those three lines are `tools/verify_install.py`'s output on `--install .build/m1-install`, which is why the
status is `partly` rather than `verified` or `not yet`: the mechanism is right, the shipped artifact is
older than it, and **nothing checked the artifact until this audit**. The verifier now reports provenance on
every run and the packer warns when the two inputs look swapped, so the same mistake cannot be silent again.

**What would close it**: rebuilding the install with `--repo` and `--revision` (and the snapshot present for
`digest_snapshot`), then re-staging it to the farm — a 21.7 GB rebuild and three 21.7 GB copies, which is a
deliberate operation rather than a side effect of an audit. Until then the gap is named, in the verifier's
output and here.

**What would falsify it**: an artifact whose `source` block is absent, which the verifier now reports as a
**problem** rather than a note.
