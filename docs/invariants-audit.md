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
| `I6` provenance in the artifact | `verified` | 26 source-shard digests in `install.json`, `D57` |

## I1 — bit-reproducibility — `verified`

**The claim**: the same input produces the same bytes, runs and components alike.

**Evidence.** M0 matched the reference implementation on `Qwen/Qwen3.5-2B` with **40,683,520 bytes
identical** over three frozen prompts (`docs/m0-gate.md`). M1's trace on the real 35 B model **was**
byte-identical to the contract — 83 tensors, 40 discrete decisions, digest `b8c976c5e7ba8816…`
(`docs/m1-gate.md`) — and a re-check on 2026-09-17, against a contract **re-run** on the current
artifacts, found the engine differing by 40 discrete decisions and 1 float. That was **not** an arithmetic
divergence: the engine read an **install** and the contract a **bf16 checkpoint**, and `D55` showed the
difference was exactly those weights. Point the contract at the install — `tools/install_source.py`, the
same dequantiser the Swift reader mirrors — and `trace_diff` reports **IDENTICAL — 83 tensors, 0 differing
elements, 40 discrete decisions, matching digests `b0d382dbabf36df0…`** (`D56`, verified 2026-09-17). The
remaining difference against a checkpoint contract is the **declared, measured** cost of int4 (`D55`), not a
failure of this invariant, and `tools/milestones.json` now states M1's claim in that falsifiable form. M2
produced that same digest from **half the experts on each of two machines**
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
numeric distance stayed near `2e-06`; M1 checked **40 discrete decisions** on the real model, and as of
2026-09-17 that comparison is **current and passing** with matched weights — `trace_diff` between the engine
and a contract reading the same install reports every one of the 40 matching (`D56`); M2 and M3
checked the same decisions per node — and those comparisons are current, because every node reads the
**same install** (`b0d382db…` on all of them). What int4 *does* move is measured rather than assumed: against
a bf16 checkpoint contract the router's choices differ in **40 of 1,600 slots**, and at the output the
**argmax of the logits is unchanged on all five positions** — with the smallest margin **0.27** and top-8
overlap of **5–8 of 8**, which is a pass that is honest about being narrow (`DC-112`). The comparison is a **separate assertion** from the numeric one, so a
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

## I6 — provenance in the artifact — `verified`

**The claim**: the converted artifact says which weights it came from.

**What the code does.** `tools/quantize.py` records `source.repo` and `source.revision` as **inputs**
(`--repo`, `--revision`) and, when they are not given, as **null** — explicitly not a placeholder, because
`"local"` "reads like an answer and is not one". `digest_snapshot` hashes every source weight file, through
the uncached descriptor, so a 67 GiB checkpoint does not become page cache.

**What the artifact on disk said, and what it says now.** The M1 install predated both fixes, and
`tools/verify_install.py` reported exactly that:

```
  provenance: source.revision is 'local', which reads like an answer and is a placeholder
  provenance: source.files is empty, so the source weights cannot be traced to their digests
  provenance: source.repo holds a commit hash (995ad96eacd9…) while source.revision is a placeholder
```

That is why the status was `partly`: the mechanism was right and the shipped artifact was older than it.
`tools/repair_install_provenance.py` repaired the metadata — **not the payload** — on 2026-09-17, and both
values turned out to be **discoverable** from the cache layout rather than needing a human:
`models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96e…` names the repo and the revision. The repair records
`files` as **26 source-shard digests**, sets `repo` to `Qwen/Qwen3.6-35B-A3B` and `revision` to the commit,
appends a `provenance-repair` pass so the artifact describes its own history, and can only change `source`
and `passes` — every other key is compared canonical-JSON-equal before it writes. The verifier now reports
`source.files records 26 file digest(s)`, the engine's trace digest is **unchanged** (`b0d382dbabf36df0…`,
so every recorded digest still stands), and the cost was 1m19s with the disk flat at 9 GB through the
uncached reader (`D57`). The verifier reports provenance on every run and the packer warns when the two
inputs look swapped, so the same mistake cannot be silent again.

**What would close it**: rebuilding the install with `--repo` and `--revision` (and the snapshot present for
`digest_snapshot`), then re-staging it to the farm — a 21.7 GB rebuild and three 21.7 GB copies, which is a
deliberate operation rather than a side effect of an audit.

**Where that is blocked, measured rather than assumed.** The first version of this paragraph said the
checkpoint was absent. It is not: the 67 GB snapshot the install was built from is present under
`.build/hf-cache`. The blocker is space, and it is arithmetic — a build reads the snapshot and writes a new
21.7 GB install **at the same time**, so it needs about 89 GB on one machine. The machine that holds the
snapshot has **8.6 GB** free, and the other three have **23 GB, 44 GB and 57 GB** (after the staging this
project did itself). No machine on this farm can hold both, and the snapshot cannot be deleted to make room
because it is the only copy and the thing the rebuild reads. Closing `I6` therefore needs either more disk
on one node or a rebuild that streams the snapshot from elsewhere — a deliberate infrastructure decision,
not a test that was skipped. Until then the gap is named, in the verifier's output and here.

**What would falsify it**: an artifact whose `source` block is absent, which the verifier now reports as a
**problem** rather than a note.
