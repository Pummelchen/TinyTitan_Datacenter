# Repository decisions

Decisions about how this repository works rather than about a model milestone. The milestone records are
`m0-decisions.md`, `m1-decisions.md` and `m2-decisions.md`; this file exists because a decision that is
about the *process* has no honest home in any of them.

## D35 — A gate for the documentation's own claims

Every rule here ends with "the tracker says so with the evidence that closed it", and the numbers in the
README, `AGENTS.md` and the wiki are the visible half of that. They drift, and this class of defect has been
hand-fixed more than a dozen times in this project: a test count quoted after a test was added, `2 skipped`
after the skips were removed, `D12` described as open a round after it was decided. That is an argument for
a gate rather than for another careful read.

`tools/check_status_claims.py` checks three things, offline:

* the **test counts** each current-state document claims (`README.md`, `AGENTS.md`, the tracker, the
  Roadmap, the Testbed, the Architecture page);
* the **decision ids** cited anywhere in the repository's Markdown — each must have a heading in one of the
  decision records, so a citation cannot point at a decision nobody wrote;
* the **release state**: a document that says there are no tags must be right about that.

It reports how many claims it found and **refuses to pass on zero**, because a regex that quietly stopped
matching would turn the gate into a decoration — a check that is checking nothing must not look like a check
that passed. The CI job feeds it the tool suite's own count, parsed from the run that just happened, so the
number cannot be a literal that someone forgot to update.

**Its first run found `.wiki/News.md` claiming 105 tests with 2 skipped against 184 and 0.** That is not
drift: News is a dated log, and the number was true the day it was written. So the gate distinguishes
*claims* (documents that describe the repository now) from *records* (News, and the decision documents),
and checks citations in both. The false positive is why the distinction is written down rather than left to
judgement, and it is now a test.

**What it cannot check**: anything that needs a run — the digests, the throughput figures, the byte counts.
Those live in `docs/` next to the command that produced them, and the rule stays what it was: point at a
command and its output, never at an adjective.

**And four rows left the tracker with it.** `DC-004` (the CI gates, now including this one), `DC-034` (the
frozen prompt set and the throughput baseline), `DC-035` (M1's gate) and `DC-044` (the bit-identity harness)
were all *done* and still open: work that is finished but unrecorded is the same defect as a claim that is
recorded but wrong.

## D36 — Licence and provenance: the reusable units, their obligations, and why no NOTICE is required

`DC-013` asked what this repository owes the sister project, which is Apache-2.0 while this one is MIT, and
whether a `NOTICE` has to be carried. The answer is **no**, and the reason is a measurement rather than an
assurance: **nothing here is copied from it.**

The review compared every Swift and Python file here against a read-only clone of
[TinyTitan](https://github.com/Pummelchen/TinyTitan) — 19 MB, cloned into `.build/` and never committed:

* its modules are `TinyTitan*` and its tools are `prepare_*.py` / `*_reference.py`; there is **no shared
  module name** and no shared file layout beyond directory conventions;
* **four** files share a basename, and **none** shares more than **5 %** of its non-comment lines — the
  measured overlap at the threshold that would matter is zero;
* the identifiers in common are **format and model vocabulary** (`hiddenSize`, `numLayers`, `vocabSize`,
  `safetensors`, `quantized`, `dequantize`, `tieWordEmbeddings`), several of which come from the model's own
  `config.json`;
* no file carries a port marker, an upstream copyright line or an attribution comment.

The positive evidence is stronger than the absence of matches: the install reader exists **twice in this
repository**, in Swift and in Python, written independently and checked against each other, and its layout
rules were established from the artifact rather than the source (`D29` — the `padded_columns = 0` rule was
learned by refusing a reader that demanded a padded width).

**What is shared, and why it carries nothing.** The install **format** is an interface, and interfaces and
file formats are not the licensed expression; the repository **conventions** are ideas, kept deliberately
and documented as such; the **model weights** are Apache-2.0 and are never redistributed, with attribution
kept in `docs/reference-qwen36-35b-a3b.md`. Apache-2.0 §4(d) attaches a `NOTICE` only when a distribution
*includes* the licensed material, and the sister project's `NOTICE` — which names *turbo-fieldfare* — covers
material they include, not material anyone here does.

**Recorded, so the answer is re-derivable.** `THIRD_PARTY_NOTICES.md` states the position and the three
conditions that would change it (a port from an Apache-2.0 source, redistributed weights, a vendored
dependency), and `tools/check_provenance.py` checks offline what can be checked: that the file exists, that
it still names the relationships it describes, and that no source file has acquired a third-party copyright
header — the earliest visible symptom of code arriving without its obligations.

## D37 — What the cluster cost, per node: the all-reduce is accounted for

`DC-081` asked for per-node figures, an all-reduce time and a cache hit rate. The forward already reported
expert traffic, payload bytes and cache hits (`D32`, `D33`); what was missing was what the *cluster* cost,
and a way to see any of it **per node** rather than per process.

`ExchangeLedger` counts what each all-reduce actually did — reduces, terms sent and received, bytes sent
and received, and the wall seconds inside the exchange. It counts **every attempt**, not only the successful
one, because a retry that cost a round trip is part of what the run cost. It is a class inside
`ShardExecution`, which is a struct — the same shape as `ReadState` and the payload cache, and for the same
reason.

Each node writes its own `metrics.json`, and the gate harness prints them as a table. The two-node local run
on the fixture, which is the deterministic one:

```
[4/5] per node, from each node's own metrics.json
      node 0: 10 request(s), 30,208 B read, 0 B dense (0 cache hit(s)), 2 reduce(s) 7/5 terms, 0.000 s
      node 1: 8 request(s), 27,840 B read, 0 B dense (0 cache hit(s)), 2 reduce(s) 5/7 terms, 0.000 s
```

**The table is self-checking, which is the property worth having**: node 0 sent seven terms and received
five, node 1 sent five and received seven. An all-reduce that dropped or duplicated a peer's terms shows up
here as an asymmetry, before any trace comparison.

A sharded **generation** across two machines reports the same shape per node — 10 reduces, 10 terms each
way, the dense payload read once (18,368 B) and served from cache 120 times — and its tokens and digests are
still identical to the single-node reference, so this round's accounting changed no arithmetic.

A node that wrote no metrics is reported **NOT REPORTED** rather than as zeroes, the same distinction the CI
job makes about the wiki's tables.

**What these numbers are not.** `step_seconds` and `exchange_seconds` are raw observations of one run on a
shared farm. They are instrumentation, not a throughput claim: `DC-053` is where the ≥3× claim lives and it
needs a quiet farm. Every number above is measured and labelled as an observation rather than a result.

## D38 — M3's gate is built, and it refuses to run on a busy farm

`DC-053` asks for ≥3× the single-node throughput on four nodes. The number cannot be taken on a shared farm
and still mean anything, so the gate's first step is a **measurement of the farm itself**: it asks each node
for its one-minute load average and stops if any is above the threshold, naming what it saw. Today:

```
M3 GATE REFUSED: the farm is not quiet — node4 at 3.00, node1 at 5.58, node2 at 2.44, node3 at 2.25.
A throughput figure from a busy farm is a different measurement, not a weak one.
```

Note `node1 at 5.58`: the gate detected the load my own install staging was putting on that node, which is
the instrument working rather than a coincidence.

**The gate is built and not passed.** That distinction is the whole point of writing it down: `--allow-busy-farm`
records an **observation** and cannot pass the gate — the ratio is printed, the threshold is not asserted, and
the exit status depends only on bit-identity. A quiet farm is a decision someone makes, not a flag to work
around.

Three rules are built into it:

* **Correctness is asserted before speed, and never traded for it** — every node's tokens *and* trace digest
  must equal the single-node baseline's, or the run fails no matter how fast it was;
* **the ratio uses the slowest node**, because a cluster step finishes when its slowest member does; an
  average over nodes would report a speed no user of the cluster can obtain;
* **a node nobody could ask is NOT CHECKED, not quiet** — the same distinction the CI job makes about the
  wiki's tables and the harness makes about a missing `metrics.json`.

Its own numbers come from each node's `metrics.json` (`D37`), so the join and the staging are not counted as
compute. The tests cover the gate's judgement — which node decides the ratio, what counts as quiet, and that
`None` is not zero — because the measurement itself needs four idle machines and a 21.7 GB install.

**The farm, provisioned.** The gate needs the verified install on every node, so it was staged on the two
that lacked it: node1 had **91 GB** free and node2 **70 GB**, against a 21.7 GB install, so neither came near
the 5 GB floor. The copies run over the LAN in the background while the rest of this round proceeds.

## D39 — The sister project's int4 is not this repository's int4, and the evidence is in the code

`DC-085` is a standing task — *report cluster-relevant defects upstream* — so this round audited the sister
project's quantisation rather than waiting for a defect to appear. Reading it settled a claim this repository
had been repeating: that the sister project "holds the install format and the repacker".

**It does not hold this one.** Its on-disk format is the `GTurbo*V1` family (`GTurboFormatV1`,
`GTurboManifestV1`, `GTurboPackedExpertsLayoutV1`), and its int4 validation reference
(`Quantization.quantizeInt4Affine` / `dequantizeInt4Affine`) is **unsigned with a bias**:

```
q = max(0, min(15, round((w - bias) / scale)))    // encoder
w = Float(nibble) * scale + bias                  // decoder, bf16 scale and bias
```

This repository's install container is its own — `install.json` plus `data.bin`, **signed** four-bit codes
(`value >= 8 ? value - 16 : value`), fp32 scales, int8 zero points, and `w = (code - zero) * scale`. Both are
internally consistent, so **there is no defect to report upstream**; what there was is a wrong assumption in
our own documents, and `AGENTS.md` is corrected.

The distinction is not cosmetic. The signed case is the one where reading `0b1000` as 8 instead of −8 shifts a
whole group by sixteen steps *and still looks like plausible weights* — which `tools/quantize.py` says in a
comment and `D34` confirmed by hand — so a reader has to know which convention a file uses. Ours is signed,
and the container is this repository's own.

**A defect in our own tooling, found by using it on the farm.** Verifying the staged installs on node1 and
node2 was the first time `tools/verify_install.py` ran anywhere but this checkout, and it died: the manifest
records the policy's **absolute path on the building machine**
(`/Users/<builder>/Downloads/…/tools/quant_policy.json`), and the fallback looked in a *repository* layout
that does not exist on a node that received only an install and two scripts. It now prefers the file **beside
the script** — where a copied tool finds its own data — and when nothing is found it says which file to copy
instead of raising a `FileNotFoundError`. The first run after that fix failed again, because the tool had
travelled without the policy: the sharpened message is what named the missing file.

**And a harness of mine hid the failure**: the ssh command ended in `| tail -8`, so the remote crash reported
exit 0. A pipeline returns the *last* command's status; when the exit code is what matters, it must not be
behind a pipe.

**The farm, provisioned and verified.** node1 (91 GB free) and node2 (70 GB) each received the 21.7 GB install
at **115.8 MB/s — 3m07s each** — and each was then verified **on the node itself**:

```
  payload: 693 tensor(s) tiling 21,700,655,616 byte(s)
  hashing 5 payload(s), 2.97 GB, streamed
INSTALL VERIFIED: 693 tensor(s), structure and policy, 5 payload digest(s)
```

Every node now holds the same install, tiling and digests included, and the figures are the ones this
checkout reports.

## D40 — The baselines are data, and the counts in them are asserted

`DC-084` asked for the sister project's practice — recorded baselines, re-checked — and this repository's
figures lived only in prose. They are now `tools/baselines.json`, read by `tools/check_baselines.py`.

**Two kinds, and the split is the decision.** A **count** is deterministic given its input (expert requests,
payload bytes read, cache hits), so it is asserted **exactly** and can be re-checked at any time — on a
shared farm, in the middle of other work. An **observed** value depends on the machine and the moment
(seconds, memory), so it is **reported with its delta** and asserted only with `--assert-observed`, which
belongs on a quiet farm. The first real check shows why: the recorded throughput baseline is **0.108 tok/s**
and the fresh run measured **0.1761**, +63.1%, on a farm that happened to be quieter. Asserting that would
have failed a correct run.

**What the demonstration verified.** A trace and a cached generation on the real 35 B install, at the
recorded conditions (prompt `760,6511,314,9338,369`, three steps), reproduced every recorded count exactly:

| baseline | recorded | this run |
| --- | --- | --- |
| `expert_requests` | 2218 | 2218 |
| `expert_hit_rate` | 0.0 | 0.0 |
| `install_bytes_read_this_forward` | 3,060,562,432 | 3,060,562,432 |
| `dense_payload_bytes_read` | 1,043,708,416 | 1,043,708,416 |
| `dense_payload_cache_hits` | 4277 | 4277 |
| `dense_payload_bytes_held` | 1,043,708,416 | 1,043,708,416 |

The trace also re-produced digest `b0d382dbabf36df0…` — the M2 digest — without being asked to.

**Applicability is checkable, because prose is not.** `applies_to` says in words what a figure was measured
on, and the first version treated it as documentation. It compared the **two-node fixture's**
`exchange_reduces = 10` against a **single-node** real run, whose correct value is 0, and reported a failure.
Every baseline now carries a `requires` map keyed on something its producer **actually records** — `layers`
for the trace, `steps` for generation, `nodes` for the fixture — and a baseline whose requirements are not
met is NOT CHECKED rather than compared. That last clause matters: the first attempt keyed the trace
baselines on `nodes`, which no trace metrics file writes, so they could never have been checked at all — and
a baseline that can never run is a comment.

**The test had invented the artifact.** `steps_per_second` is derived from what a run reports, and the unit
test used `step_seconds_0` scalars — a shape no run writes. The real file has a `step_seconds` **list**, so
the tool crashed while 19 tests passed. The test now uses the recorded shape, and an unexpected shape is
reported NOT CHECKED with the command that produces it rather than becoming a traceback.
