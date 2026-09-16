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
