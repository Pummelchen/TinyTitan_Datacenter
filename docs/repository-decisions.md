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

## D41 — One command for the gates, and the counts come from the runs

`DC-036` asked for "a Swift gate that actually runs on a clean machine". GitHub's hosted runners cannot meet
the toolchain (their `macos-26` image carries Xcode 26.x, below the manifest's 6.4 floor), and that failure is
the point rather than the problem — but it left the *substance* of the task unmet: the battery was a chain of
commands run by hand, and a chain that is remembered rather than executed eventually skips a link.

`tools/run_all_gates.py` is that battery in one command, and it does two things a list of commands cannot:

* **the counts come from the runs.** It runs the Python and Swift suites first, parses their own output, and
  hands those numbers to `check_status_claims.py`. For three rounds those numbers were typed in beside the
  command, and twice they had gone stale. The class of mistake stops being possible instead of being watched
  for — and on its very first use on another machine it **caught one**: the Python suite reported **288** where
  the documentation said 276, because this round had just added twelve tests to it;
* **a gap is named, not fatal.** A Python test file whose package is missing is reported NOT CHECKED by name,
  and when that happens the documentation's counts are **not** compared at all — an incomplete run's total is
  smaller than a complete one's, so comparing them would blame the documentation for what the machine lacks.

**Run on a clean machine, and it passes.** The whole tree was staged on another farm node — one that has never
built this checkout — and the battery ran there:

```
  toolchain        OK
  markdown tables  OK
  markdown links   OK
  provenance       OK
  python tests     288 test(s)
  swift tests      186 test(s), 0 skipped, 0 failure(s)
  status claims    OK
```

That is `DC-036`'s criterion met: **the Swift gate runs on a clean machine**, because the farm nodes carry the
toolchain the manifest requires. What GitHub's hosted runner cannot do is now precisely stated rather than
implied, and the same one command runs in both places — `--skip-swift` is the half a machine without Xcode 27
can still do, and it is what the link job runs.

**Two bugs the tool found in itself, both of the same kind.** It read the Swift summary with `search`, which
returns the *first* `Executed …` line — a suite's six tests — and reported **13 of 186**; XCTest prints one
line per suite and the total last, so it now parses the last. And its per-file Python invocation used
`-m unittest tools.<file>`, which does not put `tools/` on the path, so nine **first-party** modules were
reported as missing packages; it now runs each file the way `unittest discover` does. Both are the same
mistake this project keeps meeting: an instrument that agrees with itself and not with the thing it measures.

## D42 — `D19`'s mid-frame rule did not hold on a real socket: the run hung instead of failing

`D19` says a peer that stops part-way through a frame **fails the run, and is not retried**, because the
stream is desynchronised. `ShardExchangeTests` asserts that, with a counting transport whose peer is a Swift
object that can be told to go quiet. This round added the three tests that use a **real socket** instead — a
descriptor that really closes, and a length prefix that really promises more than arrives — and one of them
did not fail. It **hung**, and the run had to be killed and sampled to find out where:

```
SocketContributionTransport.receive()  ContributionTransport.swift:120
SocketContributionTransport.readExactly(_:midFrame:)  ContributionTransport.swift:131
NSFileHandle.read(upToCount:) -> -[NSConcreteFileHandle readDataOfLength:] -> read (blocking)
```

**The poll and the read disagreed about how much they were asking for.** `readExactly` polls until the
descriptor is readable and then calls `FileHandle.read(upToCount: remaining)`. That method is
`readDataOfLength:`, which blocks until it has **every** byte it was asked for or the peer closes — so after
`poll` correctly reported that three bytes had arrived, it waited for the other sixty-one. A peer that stops
mid-frame therefore produced an **infinite hang** rather than the documented failure, on the one path where
the deadline exists to prevent exactly that. The count of retries, and the failure itself, never happened.

`read(2)` is the call that matches the poll: it returns what is **there**. The descriptor has just been
reported readable and nothing else reads from it, so it cannot block, and the loop's next `poll` is what
enforces the deadline. The test that hung now completes in **0.061 s** against a 60 ms deadline — the
deadline is the thing being measured, so a passing test with the wrong duration would have been no evidence
at all.

**Why the unit tests could not find it.** Every `D19` rule is asserted in `ShardExchangeTests`, and those
tests are right — but their transport is a Swift object that returns immediately, so the *shape* of the
underlying read never enters them. Only a real socket can block, and only a real socket produced this. It is
the same lesson as `DC-087`'s diagnostic and the metrics shape that `check_baselines.py` assumed: an
instrument that agrees with itself rather than with the thing it measures.

**Verified after the change, not before it.** `swift test --no-parallel` is **189 tests, 0 skipped, 0
failures**, and bit-identity was re-established on four real machines with the fixed transport: every node
produced the baseline's tokens, `[11751, 11, 264, 3177]`, against a single-node baseline of 6.112 s/step.
The throughput ratio, 1.06x, is recorded as an **observation** because the farm was busy and four steps is
not a measurement (`D38`).

**A harness that looked like a hang was a data migration.** The mesh run appeared to hang for ten minutes
with no output. It was `scp`-ing the 20 GB install to every peer, because `--remote-install` was not given —
the harness documents that flag, and its `stage_remote` says in so many words that "a 20 GB install is not
copied here". It is right; the caller was wrong. It now **announces what it is about to copy** before it
starts, which is the difference between a mystery and a message. And the run's output was invisible because
it was piped into `tail`, which buffers: the pipeline lesson recorded in the previous round, applied one
round late.

## D43 — A killed node, and what the survivor actually says

`D19`'s failure rules were asserted with a transport whose peer is a Swift object that can be told to go
quiet. Last round showed that an in-process peer can be wrong about a real socket, so the rules were finally
tested the way a cluster fails: a **two-node run on the real install, with one node's process killed
mid-exchange**.

**The run failed the way it should.** The survivor exited with a named failure and did not hang — which is
last round's fix doing its work under a genuine `kill` rather than a simulated one:

```
node 0 failed (exit 2):
datacenter-generate: generation failed: connection reset by peer: the peer's process is gone, not merely quiet
node 1 failed (exit 255):
no output: exit 255, which is what `ssh` reports when the remote command was killed
```

**Three things were wrong, and the first run said so.**

* **A reset had no name.** A killed process does not close politely: the kernel answers with RST, and the
  transport reported it as a generic `socket error` carrying an errno. `ContributionTransportError.reset`
  now exists for it, `readChunk` maps `ECONNRESET` and `send` maps `EPIPE` and `ECONNRESET`, and the
  exchange's decision logic is unchanged — a reset fails the run exactly as a closed connection does. What
  improved is the *diagnosis*: "the peer's process is gone, not merely quiet" is a sentence an operator can
  act on, and a socket error is not.
* **An empty failure is not a failure message.** The harness printed `node 1 failed:` followed by a blank
  line, because a killed process has no stderr. It now says what it knows, including what `255` means —
  and that number is `ssh`'s, not "128 + signal 127", which was the first version of the explanation and
  arithmetic on a number that never meant it.
* **Using a closed transport crashed the process.** `close()` was added so a test could make the kernel send
  RST rather than a FIN, and a test of its idempotence found that a second use raises an Objective-C
  exception from `FileHandle.fileDescriptor` — "Operation now in progress" — which **Swift cannot catch**.
  A closed transport now refuses with `.closed` before touching the handle, which is the same reasoning that
  put `SO_NOSIGPIPE` on the descriptor: this project cannot afford a failure that presents as a crash.

**Two tests were wrong, and they were removed or rewritten rather than weakened.**

* Asserting *which* of `.reset` and `.closed` a killed peer produces asserts the kernel's choice, and not
  every path chooses the same one: the same two tests passed in a filtered run and failed in a full one.
  They now assert the contract the exchange acts on — the peer is **gone**, so the run fails and does not
  retry, and `.timedOut` is the one answer that would mean something else. The platform decides the wording;
  the rule is ours.
**A transient failure this round, stated as one.** During one battery run the Swift suite reported **one
failure**, and three consecutive runs afterwards reported none. Which test it was cannot be recovered: the
runner summarised it as "1 failure(s)" and threw the name away, so a flake became a mystery instead of a bug
report. That is fixed first — the runner now reads the failing test's name and reason out of XCTest's own
line and prints them — and the failure itself is recorded here as **unexplained**, not as absent. If it
recurs it will arrive with a name attached.

* The write-side twin cannot be a test. A small frame is buffered successfully before the RST arrives, so
  "the write eventually fails" is a property of how long the kernel takes, and a test that loops until it
  does is asserting a race. It was removed with that reason written where it stood; the mapping it found
  (both `EPIPE` and `ECONNRESET`) stayed.

## D44 — The memory floor and the one-job lock are enforced too, not just the disk floor

`D41`'s runner made the gates reproducible. This is about the other half of the same problem: the project's
worst incident was not a gate failing, it was a **machine panicking**. On 2026-09-16 a 35 B engine at ~4.5 GB
resident ran on a machine with about 4.5 GB usable, *alongside* a 20 GB install build; macOS grew swap to
thirteen swapfiles, the machine stopped responding for 90 seconds, and the hardware watchdog panicked it.
Disk had a floor enforced by `check_disk_headroom.py`. Memory and concurrency were **rules in a document**,
and a rule in a document is what that incident is worth.

`tools/heavy_job.py` adds both, and the split between them is the design:

* **A refusal is deterministic.** A job that declares a peak larger than what the machine can use is
  refused, with the arithmetic in the message: physical memory less the reserve the project measured
  (~4.5 GB usable of 8 GB, stated as a number so a reader can disagree with it rather than with a mystery).
  Free disk cannot make such a job safe.
* **A warning is not.** What looks *reclaimable right now* is reported, and when it is already below the
  declared need the job is warned about loudly — but not refused, because macOS's notion of available
  memory is fuzzy, and a guard that acts on one fuzzy reading is the mistake `disk_watchdog.py` already made
  once and recorded. What the operator does with the warning is the human-approval rule.
* **Being alone is a lock, with stale detection.** "One heavy job at a time, never concurrent" is now a file
  with the holder's purpose, pid and time. A second job refuses and is told how to inspect it. A claim whose
  process is gone is taken over with a printed notice — because a lock that a crashed job leaves behind
  would refuse every future run and be deleted by the next person to hit it, which is how a guard becomes a
  formality.

**Where the claim belongs, learned by getting it wrong.** The first version of the wiring put the preflight
inside `quantize.build_install` — a *library* function — which made every caller a heavy job. The tests that
build two-tensor fixtures then took the **production** lock, and refused each other: one test claimed it, the
next one in the same process was told another heavy job was running. Four tests errored and one failed, and
the guard was right about all five. The claim belongs to the **entry point**, where a deliberate run begins;
a library function builds what it is asked to build. The lock's location is also overridable through
`HEAVY_JOB_LOCK`, read per call — and that moves the lock rather than switching it off, because an
environment variable that disables a safety check is a foot-gun, and this round is the demonstration.

It runs, on this machine, as:

```
$ python3 tools/heavy_job.py --needs-gb 4.2 --purpose "the M1 checkpoint path"
memory: physical 8.59 GB, usable 5.09 GB, reclaimable now 3.73 GB, swap 1.49 of 2.05 GB used
heavy job warning: the M1 checkpoint path declares 4.20 GB and only 3.73 GB looks reclaimable right now;
other work is using this machine
```

The job fits — 4.20 GB against 5.09 GB usable — so it is allowed, and the warning says what the operator
needs to know instead: the machine is already swapping, because of this round's own ten test runs. And a
job that cannot fit is refused with the number that decided it:

```
$ python3 tools/heavy_job.py --needs-gb 9
refusing to start this check: it declares 9.00 GB and this machine can use 5.09 GB (8.59 GB less a 3.5 GB
reserve for macOS).
```

**It is wired into the heavy paths, not left as a tool to remember.** `quantize.py` (the install build),
`run_m1_gate.py` (the checkpoint path, declaring its **measured** 4.2 GB peak) and `run_m3_gate.py` (the
install path, declaring its **measured** 0.35 GB) now call it before they load anything. The install build
declares no memory figure at all, deliberately: what it holds depends on the checkpoint, and inventing a
number would be a number with nothing behind it. The **lock** is the part that matters there — the machine
panicked while an install build ran beside an engine.

## D45 — The M1 gate is not current, and verifying against a stored artifact is not verifying

`DC-111` came from the habit this project uses on everything else: check the instrument before believing the
finding. The verification was cheap — a fresh engine trace on today's install, compared with `trace_diff`
against the stored contract — and it does not pass:

```
the stored contract vs the stored engine trace (2026-09-16 01:5x)   IDENTICAL — 83 tensors, 0 elements, 40 discrete
a fresh engine trace vs the stored engine trace                     DIFFERENT — 40 discrete, 1 float
a fresh engine trace vs the stored contract                         DIFFERENT — 40 discrete, 1 float
```

The first row is the one that makes the other two evidence: the two stored traces have **byte-identical**
`data.bin` (`a7c77b63…`), so they agreed when they were written. Today's has `f52ca4c3…` and prints digest
`b0d382dbabf36df0…` where the stored pair prints `b8c976c5e7ba8816…`.

**Which side moved, and how I know it was not a guess.** The reference's only change since the stored
contract is **two lines** adding a headroom guard (`d26b419`) — its arithmetic is untouched — and it reads
the **checkpoint**, which has not changed at all. The engine reads an install that was **rebuilt at 04:50**,
and the dequantiser's zero-and-NaN canonicalisation landed with `D34` at 23:36. So the drift is on the
install path, and the candidates are enumerable.

**Why it stops here rather than being settled.** Re-running the reference is a **GB-scale streaming read of
the 67 GB checkpoint on this node**, which is the operation that took free disk from 17 GB to 2.96 GB in
half a minute and helped panic it once. The checkpoint exists on no other node, and no node can hold it
beside a rebuild. Saying "not current" and naming the discriminating step is worth more than a run that
risks the machine for a diagnosis — and the discriminating step is exact: run the **engine on the
checkpoint** and the **reference on the checkpoint**, which separates an install-path drift from a
checkpoint-path one.

**The lesson, which is the general one.** A stored trace is evidence only while **both** sides that produced
it are unchanged. It is a photograph of an agreement, not the agreement itself; here the contract side was
frozen and the engine side was not, and three rounds of recorded digests quietly became history. The
project's habit of re-running a gate rather than trusting its report is what caught it, and `docs/m1-gate.md`
keeps the old pass on the page with a status update above it rather than deleting it.

**What did not change.** M2 and M3 remain current: their comparisons are between nodes reading the **same**
install, and all four produce `b0d382db…`. The recorded baseline counts still match exactly — the install's
traffic did not move even though its arithmetic did. What is in question is one claim: that the engine's
quantised install path reproduces the reference's checkpoint path bit for bit.

## D46 — Milestones are re-checked, and two claims that look alike are kept apart

`DC-111` was found by hand: a manual trace, a manual comparison, thirty seconds of work that nothing in this
repository was doing. `tools/check_milestones.py` is that work made repeatable, and it exists because the
failure was not a wrong number — it was a **right number nobody re-checked**.

**Two claims, deliberately not conflated.** `tools/milestones.json` records, for each milestone, the digest
the **engine** produces on this install and whether that was ever checked against the **contract**:

* *the engine is stable* — it still produces the digest its own record says it does. A single-node trace on
  the install answers this in twenty seconds, and any new divergence fails.
* *the engine matches the reference* — the milestone's actual claim. On the 35 B model that means reading the
  checkpoint, which this machine cannot safely do. So it is reported `stale` or `not checked` **with the
  reason**, never inferred from the first.

Treating the first as the second is exactly how the M1 confusion happened: the engine has been stable at
`b0d382dbabf36df0…` — a later table in its own gate document says so, `D34`'s record confirms it, and the M2
and M3 gates report it — while the status section still quoted `b8c976c5e7ba8816…` from the run that
preceded the change. Both statements were true at different times, and neither was checked against the other.

**A declared divergence does not fail; an undeclared one does.** A stale agreement must name the task that
tracks it, or `milestones.json` is refused. A tool that is permanently red is a tool people stop reading, and
the alternative — reporting the staleness without owning it — is how a fact becomes nobody's. A divergence
that is *not* declared fails the check, so the next one arrives loudly.

Its first full run, on this repository: **M1** engine reproduces, contract stale (`DC-111`); **M2** and
**M3** engine reproduces, contract current, because their comparison is between machines reading one install
and does not depend on a stored artifact at all; **M0**, **M4** and **M5** not checkable here, each with the
reason. It runs as `python3 tools/run_all_gates.py --milestones`.

## D47 — The reference can read a checkpoint without mapping it, and that was not the whole problem

The M1 gate could not be re-established here for a reason that had never been tested: the reference reads the
checkpoint through `safetensors.safe_open`, which **memory-maps** the shard, and a 67 GB mapping on an 8 GB
node is the mechanism behind this project's two panics. That was an inference from the documented incident, so
it was checked — `safe_open` maps, `SafetensorsSource` has no other path — and then **fixed**.

`tools/uncached_safetensors.py` has the same interface (`tensor`, `rows`) and serves both by `pread`-ing
exactly the bytes asked for through the same `F_NOCACHE` path the install reader uses. Nothing is mapped, so
nothing accumulates in the page cache, and the peak cost of a read is the array it returns rather than the
file it came from. It is not taken on trust: **byte-identity with `safe_open` is tested on a real 4 GB shard
of this checkpoint** — a row slice of the 1074 MB expert tensor and a one-dimensional norm — as well as on a
file the tests build themselves, where every dtype, a NaN, both infinities and a signed zero are constructed
and carried through. The reference takes it behind `--uncached`, so the authority's default reader changes
only deliberately.

**And the gate still cannot be re-established here, which the fix is what revealed.** With the uncached reader
active, one attempt drove swap from 2048 MB to **5120 MB** and took free disk from 11.8 GB to **8.0 GB** in
sixty seconds. The page cache was out of the picture, so the growth was the **process's own anonymous
memory**: one `gate_up_proj` is 1074 MB of bf16 that the reference materialises as **2 GB of fp32**, before
the layer's other tensors and its activations, against a node with about 4.5 GB usable. It was stopped
deliberately at 8 GB free rather than letting the disk floor stop it, and it recovered to 10 GB.

**Two lessons, and the second is the one worth keeping.** A documented hazard is a hypothesis about *which*
resource runs out first, and fixing the named one does not fix the run: the page cache was real, and removing
it exposed the memory underneath. The way to settle a blocker is to attempt the thing and watch it, not to
reason about which of its parts is fatal — the attempt cost ninety seconds and produced a better blocker than
three rounds of inference.

## D48 — The M1 gate fails on today's artifacts, and the divergence is in layer 0

`DC-111` asked whether the engine still reproduces the contract. It does not, and now that is a measurement
rather than an inference — because two blockers had to be removed before the question could be asked at all.

**The blockers were different, and the first hid the second.** The first was a page cache: the reference read
the checkpoint through `safe_open`, which memory-maps a 67 GB file on an 8 GB node (`D47`). Removing it — with
a `pread` reader proven byte-identical to `safe_open` on a real shard — did **not** make the run work, and
that is how the second was found: the reference materialised a whole layer's experts, 3.2 GB in fp32, and the
attempt drove swap to 5.1 GB and free disk from 11.8 GB to 8.0 GB in a minute. The routed experts are now
fetched **by index**, which is the same fact `DC-032` used on the engine side: the checkpoint's leading axis
*is* the expert, so one expert is one row. The kernel already accumulated in ascending expert index and
already indexed `gate_up[expert]`; only the materialisation was whole.

**The streaming is not a hypothesis.** `tools/test_ordered_moe.py` asserts that the array path and the
provider path produce byte-identical output, and that the provider is asked only for the experts that were
chosen. And on the **real model**, the contract produced with the streamed reference is **IDENTICAL** to the
contract produced before these changes with the stacked one — 83 tensors, 0 elements, 40 decisions. A change
that reads less and computes the same thing is worth making and worth proving.

**What the re-run found.** The fresh contract and today's engine differ by **40 discrete decisions and 1
float** — the same divergence as against the stale contract, so it is real. In the differ's order:

```
float     layer.00.hidden_out  element 0: reference -0.006576654966920614,
                                          candidate -0.005132569000124931, 3101151 ULP apart
discrete  layer.00.router.topk missing [231, 71], unexpected [72, 19]
```

`layer.00.hidden_in` matches, so the embedding is not the cause: the difference is **inside layer 0**, at
**token 0, channel 0**. It is 28% relative and 0.0014 absolute — a boundary or a rounding difference, not a
structural one — and layer 0 of this family is a **Gated DeltaNet** layer, whose causal convolution has its
boundary exactly there. The router then flips marginally and the discrete decisions cascade through all forty
layers: the failure mode `I3` exists to catch, arriving through a single element.

**What this changes about the objective.** M1 is not "passing, historically" and it is not "unverifiable
here": it is **failing, reproducibly, with the first divergence localised to one layer and one token**, and
the instrument that shows it now runs on this node in two minutes. The next step is a bisect inside layer 0 —
the trace captures layer boundaries, so the reference needs internal capture points — with the convolution's
boundary and the recurrent state's first step as the two named candidates.

## D49 — Localising M1's divergence: the convolution is out, and my own reading of the differ was wrong

Two rounds of localisation have produced one elimination, one correction, and a sharper set of candidates.

**The correction first, because it was mine.** The differ prints `DIFFERENT — 40 discrete, 1 float`, and I read
the second number as one *element*. It is one float **tensor**, and **all 10240 values** of
`layer.00.hidden_out` differ: the value quoted in `docs/m1-gate.md` is only the first of them. That changes
what the evidence suggests — a *systematic* difference inside layer 0, not a boundary rounding at one
position — and the earlier text said the wrong thing. `layer.00.hidden_in` matching is unaffected, and the
conclusion that the divergence is *inside layer 0* is unaffected; what is withdrawn is the reason for
expecting the convolution.

**A probe that disagreed with its own reference, and was wrong.** Before adding capture points, I tried to
discriminate cheaply: recompute the reference's router on the **engine's** layer-0 output and see whether it
chooses what the engine chose. The probe disagreed with the reference's *recorded* decision — which is not a
finding about the engine, it is a broken instrument: `layer.00.hidden_out` is the layer's **final** output,
while the router ran earlier, on the value the residual had after attention. Feeding the final output to a
router that never saw it produced a number that meant nothing. The habit that caught it is the one this
project keeps relearning: when an instrument and the record disagree, the instrument is what is wrong until
proven otherwise.

**What that leaves.** The trace captures layer boundaries only, so the router's input — and every other
intermediate — is not in it: a bisect needs capture points inside the layer, in the reference at least, which
is a small change to a place that is otherwise the authority. The named candidates are now: the **delta
rule's first step** (token 0 is the recurrence's first update from an all-zero state), the **gates'
exponentiation** (`a_log` and `dt_bias` reach an exponential, where a 4-bit or a bf16 rounding is not a small
error), and the **gated RMSNorm's ordering** — the engine's own comment records that its bf16 oracle rounds
the normalised value *before* the weight multiply and that an all-fp32 contract does not, which is exactly the
kind of difference that is invisible on a tiny fixture and systematic on a real one.

**Eliminated by reading, which is cheaper than running.** The convolution: reference
`depthwise_causal_conv` and engine `depthwiseCausalConv` index `position + tap - (kernel - 1)`, accumulate
`tap · x[source]` in ascending tap order, zero-pad on the left, and apply silu. Same arithmetic, same order,
same boundary. Both cite `causal_conv1d_fn:270`, and for once the two implementations agree with each other
and with the comment.

## D50 — The divergence is in layer 0's attention half, and the instrument that showed it

`D49` left a bisect needing capture points inside the layer, because the trace records boundaries only. Both
sides now have them, **opt-in**: `SHARD_TRACE_INTERNALS=1` for the engine (the same pattern as
`SHARD_PROFILE=1`), `--capture-internals` for the reference. They record `attn_out` — the mixture's input,
after the attention residual — and `ff_out`, the feed-forward's output, per layer: 163 tensors instead of 83.

**Opt-in is not a courtesy, it is a requirement.** The trace's digest covers its tensor list, so a trace with
extra tensors is a *different artifact* — every recorded digest would move. The check that the switch is
genuinely off by default is not the code, it is the output: the default trace is still **83 tensors with
digest `b0d382dbabf36df0…`**, byte for byte.

**What one comparison bought.** With 163 tensors on each side:

```
layer.00.hidden_in   identical
layer.00.attn_out    differs — all 10240 values
layer.00.ff_out      differs
```

The divergence is in the **attention half** of layer 0 — the input norm and the Gated DeltaNet — and
everything after it, including the router flip and all forty layers of discrete decisions, is a
**consequence**. That is the whole mixture path eliminated as a cause, from one run and one diff.

**Eliminated by reading, and one near miss.** The convolution (identical index arithmetic, order and
padding). The decay gate: the reference computes `-exp(A_log) · softplus(a + dt_bias)` and the engine computes
`-Ops.exp32(aLog) * softplus(a + dtBias)` — same formula, same order, negation outside the exponential in both,
which the reference's own comment flags as the subtle part. What remains inside the attention half is the
chunked delta rule (a five-token prompt is one chunk), the projections and their L2 norms, and the **gated
RMSNorm** — where the engine's own comment records that its bf16 oracle rounds the normalised value before the
weight multiply and that an all-fp32 contract does not. That comment is the most specific lead the codebase
contains, and it is now the next thing to read.

**One process note worth keeping.** My first attempt to add the reference's capture points patched the
*wrong call site* — the non-streamed forward — so the run produced 83 tensors and a digest identical to the
contract, looking exactly like a successful capture of nothing. What caught it was comparing the tensor
count against the one the engine had produced for the same change. A capture that silently captures nothing
is the same failure as a diagnostic narrower than its gate.

## D51 — Reading the attention half: four stages eliminated, and the two-path structure found

`D50` put the divergence in layer 0's attention half — the input norm and the Gated DeltaNet. This round read
that half rather than running it, and eliminated four of its stages by comparison, which is cheaper than any
instrument.

**Eliminated, with the reason each time.**

| stage | reference | engine | verdict |
| --- | --- | --- | --- |
| the convolution | `position + tap - (kernel - 1)`, zero-padded | same | identical (`D49`) |
| the decay gate | `-exp(A_log) · softplus(a + dt_bias)` | same | identical (`D50`) |
| the input norm | `1/sqrt`, fp32, ordered sum ascending | same | identical |
| `silu` / `sigmoid` | `x · sigmoid(x)`, branch at zero | same branch | identical |

The norm was worth checking closely because it is the half's **first** operation, and a difference there would
produce exactly the systematic pattern seen — every element of `attn_out` differing. It checks out: both sides
take `1.0` over `np.sqrt`/`squareRoot` of `variance + eps` (both correctly rounded, and the contract says in
so many words that `1/sqrt` is chosen over a hardware rsqrt because the two differ in the last bits), both
compute the mean of squares as an **ascending one-addition-per-step** ordered sum, and both apply the
family's weight-**offset** norm as `(1 + weight) · (x · inverse)`. The multiply order swaps between them,
which is harmless: IEEE multiplication is commutative, so `a · b` and `b · a` round identically.

`silu`/`sigmoid` matter more than they look. The contract's sigmoid **branches at zero** — `1/(1+exp(-x))`
for `x ≥ 0`, `exp(x)/(1+exp(x))` otherwise — and both sides do. A single-expression sigmoid would agree for
positive inputs and differ in the last bits for negative ones, which is the kind of difference that hides on
a fixture.

**What the reading found instead: the algorithm has two implementations, and the contract names one.** The
reference calls `chunk_gated_delta_rule(..., chunk_size=64)` — `torch_chunk_gated_delta_rule:301` — and the
engine has a `chunkedDeltaRule` that transcribes the same function. It *also* has a **recurrent**, one-token
form (`torch_recurrent_gated_delta_rule:440`) reached from a path that projects with `rows: 1`, which is the
decode case. A five-token trace takes the chunked path, so both sides are on the contract's algorithm and the
divergence is inside the chunked rule or the ops immediately around it.

That is where the next capture goes, and the chunk of code is not small: the intra-chunk attention with its
`exp(-inf)` masking, the cumulative decay prefix sum, the per-chunk state decay, and the state update. It is
also code that has been wrong before — `DC-038` records the grouped-query repeat being fixed in this very
path, and the repeat is a stage the reference spells out with `np.repeat(..., axis=2)`.

## D52 — Reading the chunked rule to the end: every stage corresponds, and reading has hit its limit

The hunt is inside `chunk_gated_delta_rule` — one function, five suspects. This round read the whole function
against the engine's `chunkedDeltaRule`, stage by stage, and **every stage corresponds**.

| stage | reference | engine | verdict |
| --- | --- | --- | --- |
| query scale | `query * np.float32(d ** -0.5)` (float64, then rounded) | `query * (1 / Float(d).squareRoot())` | equal for every dimension the model uses |
| `l2norm` | ordered sum of squares, `1/sqrt`, scale | same | identical |
| triangular solve | forward substitution, rows in order, `acc - (lower·sol)` | same order, same fused step | identical |
| `v_new`, `inter`, output | `… - ordered_matmul(key_cumdecay, stateᵀ)` | same | identical |
| state update | `state · chunk_decay + ordered_matmul(keyᵀ, v_newᵀ)` | same | identical |
| grouped-query repeat | `np.repeat(q, factor, axis=2)` | `(head · repeats + i) · headK` | interleaved in both |
| projection / conv order | `matmul(hidden, in_proj_qkv)` then conv | same | identical |

The query scale was worth checking by **number** rather than by eye, because the two spellings genuinely can
differ in the last bit: `d ** -0.5` is evaluated in float64 and then rounded, while `1 / sqrt(d)` rounds twice.
For every dimension this model uses they agree — and for a power of two the answer is exact — so it is out,
and the check cost a second. The triangular solve was the other one worth a close look, because its docstring
says in as many words that **the order the rows are eliminated in is what the contract fixes**; both eliminate
in ascending row order with one subtraction of a product per step.

**What this means, and it is the useful part.** After the convolution, the decay gate, the input norm,
`silu`/`sigmoid`, the scale, `l2norm`, the solve, the state update, the output, the repeat and the ordering,
there is no stage left that reads as different — and the outputs differ systematically. Two conclusions
follow, and both point the same way:

* the difference is in something a **side-by-side read cannot see**: a rounding that both spellings produce
  "the same way" while the values differ (a cast, a fused multiply-add, a `Float`-vs-`np.float32` evaluation
  order), or an input to this function that I have only verified *by reading* rather than *by number*;
* therefore the next step is **numeric capture inside `chunk_gated_delta_rule` and around the gated norm**,
  which is the instrument `D50` built and which has already turned "inside layer 0" into "the attention half"
  in one run.

Reading eliminated ten candidates at the cost of no runs at all. Knowing when it has stopped paying is the
other half of that: the next comparison has to be of numbers, not of code.

## D53 — The divergence is inside the delta rule, and it is not a rounding difference

`D52` said the next comparison had to be of numbers. It was, and it inverted the expectation.

Two more capture points on each side — `delta_core`, the rule's output before the gated norm, and
`gated_norm_out`, after it — split the DeltaNet's tail in four. Both sides produce **223 tensors**, and they
even agree on which layers are DeltaNet layers (30 of 40, so layer 3 and its kind have neither point). The
comparison:

```
layer       hidden_in  delta_core  gated_norm   attn_out   ff_out
layer.00       ok      DIFFERS     DIFFERS      DIFFERS    DIFFERS
layer.01    DIFFERS    DIFFERS     DIFFERS      DIFFERS    DIFFERS
```

`layer.00.hidden_in` matches and **`layer.00.delta_core` differs in all 20480 values** — so the divergence is
inside `chunk_gated_delta_rule` itself, before the gated norm, before the projection, with the input norm
already eliminated. The hunt is now one function's inputs and body.

**And it is not a rounding difference, which is the part that matters.** The character of the difference:

| measure | value |
| --- | --- |
| max abs value in the reference | 0.758 |
| max absolute difference | 0.0126 |
| **median relative difference** | **4.3%** |
| 90th percentile relative | 29% |
| relative difference on values above 1e-3 | 2 – 4.7% |

A 1-ULP difference is about `1e-7` relative. **4.3% is not a rounding difference** — it is a structural one,
of exactly the size a slightly different `beta` or `decay` would produce, or a slightly different value
entering the rule. That rules out the whole family of explanations `D52` was left with — a cast, a fused
multiply-add, a `Float`-versus-`np.float32` evaluation order — all of which are last-bit effects. Something
being computed is *different*, not *rounded differently*.

**Which is a strange place to be, because the body reads as identical.** Every stage of
`chunk_gated_delta_rule` was compared line by line in `D52` and again here: the l2 norm and the scale, the
strictly-upper mask (`column > row` masked to `-inf`, the diagonal kept), `ut_system` and `intra_chunk_attn`
as `ordered_matmul(...) · pairwise`, `key_beta`/`value_beta`, `decayed_key_beta`, the two triangular solves,
the row/column rotations by `exp(cum_decay)` and `exp(cum_last − cum_decay)`, `chunk_decay`, `v_new`, `inter`,
the output and the state update. With the initial state zero and a five-token prompt — one chunk — the core is
`intra_chunk_attn · solved` and nothing else, so the inputs to that product are the whole story.

**The next comparison is therefore the rule's inputs**: `query`, `key`, `value`, `beta` and `decay` as the
rule receives them, which the reference computes offline from the identical `hidden_in` and the engine can
record at the same seam. If they differ, the cause is in the convolution, the projections, the gates or the
repeat — all of which read as matching, which is the same trap as before: **a line-by-line read has now
twice agreed with itself against a measurement.**

**The instrument is what made this cheap.** Four capture points per side turned a milestone-sized question
into four runs and four diffs, and the *character* of a difference — median relative, not maximum absolute —
is what said "structural" rather than "rounding". That distinction is the one that changed the next step.

## D54 — The memory budget is enforced, not advised

`disk_watchdog.py` guards the disk and earned its place: on 2026-09-16 the engine's memory grew swap, swap is
disk, and the node panicked. What was missing was the guard upstream of that — **resident memory** — and on
2026-09-17 its absence bit me twice in one round.

Two comparison scripts of mine grew to gigabytes on an 8 GB node. The OS killed the first. The second filled
the page cache until `disk_watchdog.py` tripped the 5 GB disk floor and wrote its marker, with free space at
**3.42 GB** — the same shape as the panic, caught by the other guard. Three stale disk watchdogs from earlier
rounds were also still running, because a `pkill` with a partial pattern had matched none of them.

Neither script *meant* to be heavy, and the two shapes are worth naming because both look innocent:

* `list(install.rows(tensor))` — wrapping an iterator that was **already** streaming row blocks. It
  materialises the whole tensor, and the streaming property was the entire point of the iterator.
* `install.dequantize(tensor)` for a large tensor — it returns a Python `list[float]`, which is twenty-four
  bytes per value, so an 8.4 M-element tensor costs 200 MB before numpy sees it.

So the budget is now enforced by `tools/memory_watchdog.py`, mirroring the disk one:

* the ceiling is **4.0 GB of real memory** — `rss`, not virtual size, which is the quantity a page cache, swap
  and a panic are actually made of;
* **three readings in a row** above it, not one, for the reason the disk guard already records: a single
  reading of a fuzzy number once killed a read-only verification that had done nothing wrong;
* it writes `.build/MEMORY_STOP`, so a run that tripped the limit cannot resume by itself;
* seven tests cover the detector, the exclusion of the watchers themselves, the limit, the killer against a
  child the test owns, and the `--once` no-op.

**What a process-name pattern cannot catch is the important half.** Ad-hoc scripts run as `python - <<EOF` have
no command line to match, so no watchdog sees them. The rule for those is self-imposed and cheap: a script
that loads model data asserts its own peak — `resource.getrusage(RUSAGE_SELF).ru_maxrss` — and aborts above
its budget. The experiments in `D55` do exactly that and peak at 336 MB.

## D55 — M1's divergence is the install's int4 projections, proved bit-exactly

`D50`–`D53` localised M1's divergence to `layer.00`, then to the attention half, then inside
`chunk_gated_delta_rule` — and found every stage of that function corresponding, twice over, while the
measured difference was **4.3% median relative**, which is far too large to be a rounding difference. The
resolution is that the two sides are not running the same weights, and it can be shown exactly.

The install holds `linear.in_qkv`, `linear.in_z` and `linear.out` — the Gated DeltaNet's three projections —
as **int4-affine**, and `attn.q/k/v/o` likewise. That is deliberate: it is what `tools/quant_policy.json`
says, and the policy's rationale protects the small sensitive tensors (`conv`, `in_a`, `in_b`, `a_log`,
`dt_bias`, `norm`, the shared expert, the router) while the bulk projections are quantised. The contract, on
the other hand, is generated from the **checkpoint**, where those tensors are bf16. Everything else in layer 0
is byte-identical between the two.

Take the reference's layer 0, feed it the identical `hidden_in`, and run it twice: once with the checkpoint's
bf16 projections and once with **the install's own int4 values** for those three tensors, read back through
the install reader so they are literally the weights the engine uses.

```
reference bf16-proj vs engine (install)    max abs 0.0155785   median rel 0.245   identical=False
reference int4-proj vs engine              max abs 0           median rel 0       identical=True
reference bf16 vs reference int4-proj      max abs 0.0155785   median rel 0.245   identical=False
```

**The engine is correct.** Given the same weights it reproduces the reference's attention output byte for
byte, through the convolution, the gates, the l2 norms, the chunked delta rule, the triangular solves, the
gated norm and the projection. There is no engine bug in this path, and the forty differing discrete
decisions are a consequence of a 24.5% median difference at layer 0 compounding through forty layers.

**Two consequences, and the second is a correction to the record.**

1. **M1's gate and the quantisation policy are in conflict, and the conflict must be settled deliberately**
   (`DC-112`). Either the DeltaNet's and attention's projections stay at bf16 — roughly +3 GB on a 21.7 GB
   install, since the routed expert stacks are about 19 GB of it and stay int4 — or M1's claim is restated
   with this measurement as its evidence. `rule 3` forbids the quiet version of either.
2. **The M1 pass recorded on 2026-09-16 cannot be reconciled with these artifacts.** It claimed a trace
   byte-identical to a checkpoint-based contract (`b8c976c5e7ba8816…`) while the engine read int4 projections
   that a checkpoint-based contract does not have. The evidence says the claim was stale or wrong when it was
   written; the mechanism is now explicit rather than suspected, and the engine's correctness is evidenced
   *positively* — byte-identical under matched weights — which is a stronger statement than the original gate
   ever made.

The rebuild that would test the bf16 policy needs roughly 25 GB free and this node has 9.5 GB, so it is
**blocked here** and must not be attempted on this machine in any case: it is a GB-scale job of the kind that
panicked it. `DC-112` carries that.

## D56 — The gate had two inputs, so it could not be tested; now it has one

`D55` ended with an awkward fact: the claim M1's gate makes — *the engine reproduces the contract* — was being
checked with **two different inputs**. The engine read an install and the contract read a checkpoint, and the
entire divergence turned out to be that difference. A claim of that shape cannot be falsified by the
comparison it was using, because any disagreement is ambiguous between "the arithmetic differs" and "the
weights differ", and the second is not a bug.

`tools/install_source.py` removes the ambiguity. It exposes the same two methods the contract's forward asks a
source for — `tensor(name)` and `rows(name, start, end)` — and fills them from the **install**, through the
same dequantiser the Swift reader mirrors. An install-backed contract run and an install-backed engine run
therefore differ only if the arithmetic differs, which is what the milestone is actually about. Two deliberate
refusals keep it honest: an expert stack is never materialised whole (the streaming path exists for it, and a
dequantised stack is gigabytes), and an absent name is an error rather than a default.

**The finding worth keeping is the row space.** An install flattens a tensor's trailing dimensions into the
row length, so the real `expert.stack_gate_up` is stored as **262,144 rows of 2,048** — geometry
`(262144, 2048)` — and **one expert is 1,024 consecutive install rows**, not one. `D32`'s "one expert is one
row" is true of the **checkpoint**; it is not true of the install, and a source that assumed it returned
2,048 values where an expert needs 2,097,152. That is a thousandth of an expert, and it failed **loudly** on
the reshape rather than silently producing a plausible trace, which is the only reason it was cheap to find.
The test that guards it now uses a stack whose padded width is *narrower* than its row — the case a naive
fixture misses, and the shape the real install has.

`Install.row_range` came with it: `rows()` walks from the beginning, which is right for a scan and wrong for a
shard, so fetching expert 255 no longer dequantises the 254 experts in front of it. Ten tests cover the
mapping, the refusals, the padding trim, the three-dimensional tensor that is *not* a stack (`linear.conv` is
`(8192, 1, 4)` and is read whole) and the equivalence `row_range(a, b) == rows()[a:b]` that keeps the fast
path and the scan path from becoming two implementations of one thing.

**What is not claimed yet.** The full-model run is in flight, and it is not a quick one: the contract fetches
**1,600 experts** (40 layers, 8 per token, 5 tokens) through a per-value Python dequantiser. Measured at
about twenty minutes in: **19.5 minutes of CPU, 878 MB of real memory**, disk flat at 9 GB free, with both
guards live. It is safe and it is running; its outcome is the gate's outcome, and it will be recorded when it
is measured rather than before. `memory_watchdog.py --once` was run against it while it ran and reported "no
heavy job over 4.0 GB", which is the first live use of `D54`'s guard.

**Result.** The run finished and it settles the milestone. `trace_diff` between the engine and the contract,
both reading the install:

```
IDENTICAL — 83 tensor(s), 0 element(s), 40 discrete decision(s) checked (matching digests)
```

All 83 tensors, all 40 router decisions, and the two digests agree (`b0d382dbabf36df0…`). Against the
checkpoint contract the same engine is `DIFFERENT — 40 discrete, 1 float`, which is the quantisation effect
`D55` measured and `DC-112` tracks. **The engine implements the contract exactly**; what differed was never
the arithmetic. The cost was real but affordable: 1,600 expert fetches, and at the twenty-minute mark 878 MB
of real memory and 19.5 minutes of CPU, with the disk flat at 9 GB and both guards live.

## D57 — I6 is closed: the artifact can name its source, and the repair touches metadata only

`I6` was the last `partly`. The mechanism existed — `tools/quantize.py` digests every source weight file and
records the repo and revision as inputs — but the shipped M1 install **predated it**: `source.files` was
empty, `source.repo` held a commit hash, and `source.revision` said `"local"`. So the artifact that every
gate reads could not answer "which weights is this".

Both missing values turned out to be **discoverable** rather than needing a human, which the audit had not
assumed: the cache path `…/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96e…/` names the repo and the
revision. `tools/repair_install_provenance.py` repairs the metadata in place, with three rules that make it a
repair rather than a rewrite:

1. **The payload is provably untouched.** The manifest is rebuilt with only `source` and `passes` changed, and
   every other key is compared in canonical JSON before anything is written. A repair that could move a tensor
   digest would not be one.
2. **A repair is recorded as a pass.** `provenance-repair` is appended, so the artifact describes its own
   history instead of pretending it was built that way.
3. **A disagreement is refused, not overwritten.** An install that already carries file digests is compared
   against the recomputed ones: equal is a no-op, different is an error for a human, because it means the
   artifact and the weights on disk are not the same pair.

**Result.** `files` 0 → **26 source-shard digests**, `repo` → `Qwen/Qwen3.6-35B-A3B`, `revision` → the commit.
`tools/verify_install.py` now reports `provenance: source.files records 26 file digest(s)` and the three
complaints are gone, and **the engine's trace is unchanged** — still `b0d382dbabf36df0…` after the repair —
so every digest recorded in `docs/` and on the wiki still stands. The cost was 1m19s with the disk flat at
9 GB the whole way, because the digest path is the uncached one; the tool has joined `disk_watchdog.py`'s
`HEAVY_PATTERNS` so a future run is guarded like the other multi-gigabyte reads.

`I6` therefore moves from `partly` to **`verified`**, and the audit says so with the evidence.

## D58 — `DC-087` is fixed, the GPU unpack is wired in behind a flag, and it is not yet faster

`DC-033` had one clause left — "**Metal remains**" — and the reason it had not landed was a documented
defect. `MetalUnpack` has existed since `D10` and was called by **nothing but its own tests**, because it
disagreed with the scalar path on a row whose final group is partly filled (`DC-087`). Its test file said so
rather than hiding it, and the grid it had to pass was written down.

**The defect is gone, and the acceptance test is a grid rather than a case.** `DC-087`'s reproducer passes,
and the main grid — which used to `continue` past every shape with `columns % group != 0` — now covers
**17, 33, 65 and 129** as well as the round numbers. Every shape is bit-identical on synthetic payloads, and
`testTheGpuUnpackMatchesTheWholeInstallTensor` does the same on a **real fixture install tensor**. One case
was missing and is now there: every other test hands the decoder a whole-tensor payload, while the engine's
row path assembles **three byte ranges for the requested entries** and tells the decoder `rowCount: R`. That
is the shape the wiring actually depends on, and `testPartialRowPayloadsDecodeIdentically` covers it for
partial ranges at the start, in the middle and across a boundary.

**The wiring is a switch, not a replacement.** `SHARD_GPU_UNPACK=1` selects the GPU decoder for int4 rows and
the default is unchanged, because every digest recorded in `docs/` came from the scalar path and a changed
default would move all of them at once. The check is the strongest one available: the real 35 B trace is
**`b0d382dbabf36df0…` with the flag off and on**, and `trace_diff` between the two runs reports **IDENTICAL —
83 tensors, 0 differing elements, 40 discrete decisions**. The GPU path is exercised by everything int4 in
the model — every routed expert fetch, plus `linear.in_qkv`, `in_z`, `out` and `attn.q/k/v/o`.

**And it is slower, which is the useful part.** `18.2 s` scalar against `38.2 s` on the GPU for the same
five-token trace. The kernel is not the cost: `rows(named:range:)` concatenates `codes + scales + zeros`
into one `Data` and `unpack` then `subdata`s it again, so a one-expert fetch copies its payload twice and
allocates a fresh `MTLBuffer` per call — at 2,218 fetches that dominates, and the comment in the row path
already says the concatenation is inside the unpack timing "on purpose". So the path stays opt-in and the
remaining work is the plumbing, not the shader: fetch into a reusable buffer, or dispatch over a payload
already in one. That is `DC-107`'s next target with its measurement attached.

**A second lesson, and it cost the disk.** I had killed the disk watchdogs while tidying up in the previous
round, so this round's GPU run — which uses more memory and grew swap to 1.8 GB — drove free disk to **0 GB**
with nothing watching. It recovered on its own when the run ended, `du` showed nothing oversized, and the
sample was almost certainly APFS purgeable space, which is exactly why the watchdog acts on three readings
and not one. But a guard that is tidy to remove is the guard that was needed: the watchdog is running again,
and the rule is that it is started with the heavy job and stopped by the marker, not by hand.

## D59 — The GPU unpack's cost was the plumbing, and fixing it made it faster than the scalar path

`D58` ended with a precise diagnosis and no fix: the GPU unpack was bit-identical and **slower** (38.2 s
against 18.2 s for one five-token trace), because the row path concatenated `codes + scales + zeros` into a
`Data`, `unpack` sliced it twice more, each section was copied into an `[UInt8]` before reaching a buffer, and
— the dominant term — a **fresh output buffer was allocated per fetch**, 8 MB for a single expert and 2,218
fetches in that trace.

Two changes, both inside `MetalUnpack` so the public API does not move:

* **A one-slot buffer cache.** The four buffers are reused and only reallocated when a shape needs a bigger
  one. Reuse is safe because the lock is held across the dispatch **and** its completion, so no kernel can
  still be reading a buffer that the next call is about to overwrite — and it costs nothing, because work on
  one GPU is serialised in any case. That is what `@unchecked Sendable` stands on here, and the comment says
  so where the annotation is.
* **One copy per section, straight into the buffer the GPU reads.** `copyBytes(to:from:)` for the codes and
  the zeros, and one `copyMemory` of the flushed scales, replacing the `subdata`/`[UInt8]`/`Data` chain.

**Result: 16.3 s**, against 38.2 s before the change and **18.2 s for the scalar path** — so the GPU unpack is
now faster than the CPU one, with `trace_diff` against the scalar run reporting **IDENTICAL — 83 tensors, 0
differing elements, 40 discrete decisions**. The target `DC-107` set is met: *faster, with the same digest*.

**These are wall-clock observations of one trace each, not a benchmark.** The timing phase is deferred, and
this is the same kind of number `testGpuUnpackThroughputReport` reports rather than gates.

**Why it stays behind the flag anyway.** A default is a policy, and the evidence for changing a default
should be the frozen prompt set rather than one prompt; the path also holds a persistent GPU buffer, which is
a memory decision on a node that has panicked for less. `SHARD_GPU_UNPACK=1` makes it a deliberate act, and
the end-to-end check — the same trace digest with the flag off and on — is what any future default change
would have to repeat. What is *not* deferred is the correctness: `testTheBufferCacheDoesNotLeakBetweenCalls`
unpacks small, large, small and large again, each compared against the scalar path, because the cache's real
risk is the second call rather than the first.

## D60 — The GPU unpack is the default, and the profile corrects `DC-107`'s priority list

`D59` left the verified-faster GPU unpack behind a flag, on the argument that a default is a policy. That
argument is sound for a change that could move a digest. It does not apply to this one, because this one
cannot: the decoder is asserted bit-identical over the `DC-087` grid, a real fixture tensor, the partial-row
payload the row path assembles, and buffer-cache reuse; and end to end the real 35 B trace is
`b0d382dbabf36df0…` with the decoder either way. Leaving a path that is verified identical *and* verified
faster switched off would mean measuring M3 against a slower engine than the repository has.

**So it is on where there is a GPU**, and `SHARD_GPU_UNPACK=0` restores the scalar path — which is how the
two are compared rather than which one is trusted. A host with no Metal falls back on its own, so CI runners
and a machine without a GPU are unaffected. Verified after the switch: **193 Swift tests, 0 failures** (their
fixture installs now run through the GPU path), the default trace is `b0d382dbabf36df0…` in **16.1 s**, the
scalar one is the same digest in **17.3 s**, and `trace_diff` against the stored scalar trace reports
**IDENTICAL — 83 tensors, 0 differing elements, 40 discrete decisions** for *both*. One trace each: these are
wall-clock observations, not a benchmark, and the timing phase is still deferred.

**A profile run corrects the record.** `SHARD_PROFILE=1` on the current build gives, over a 19 s trace:

| phase | s | what it is |
| --- | --- | --- |
| `mix.read` | **7.23** | the install read (3.24) plus the unpack (4.53) |
| `attn.core` | 4.05 | the attention/DeltaNet core |
| `load` | 2.22 | loading a layer's weights |
| `head` | 1.76 | the lm head |
| `mix.gateup` | 1.22 | the routed experts' gate/up projection |
| `mix.down` | 0.53 | their down projection |

`DC-107` recorded the unpack at 3.45 s as "at its measured limit". It is now **4.53 s**, it is the largest
single sub-cost in the forward, and it was **not** at a limit — the GPU path takes about two seconds off the
whole trace, which `D59` measured and this decision acts on. What *is* at its limit is the **read**: 3.24 s
for 3.06 GB is about 0.95 GB/s, the sequential floor the earlier measurement established, and no kernel
changes it.

That leaves the compute targets, and the order is not the one the record implied: after `mix.read`, the
largest are **`attn.core` 4.05 s**, **`load` 2.22 s** and **`head` 1.76 s**. The head is a plain GEMM and the
easiest of the three to make bit-exact, which is a better first kernel than the DeltaNet's chunked rule.

## D61 — `D10`'s open question, measured: no math mode is enough, and `fma(x, w, 0)` is

The comment on `MetalUnpack`'s pipeline recorded an open question in as many words: the GEMM that follows
"must not assume that `.relaxed` is enough, and will have to **measure** whether `.relaxed` is enough or
whether the accumulation needs to be guarded differently". This is that measurement, and the answer is that
no mode is enough.

**What is at stake.** The contract's matmul is `ordered_matmul`: "one multiply and one add per output, so the
rounding sequence is fully determined" — **no FMA** — and `Ops.orderedMatmulScalar` materialises the product
before the add because "the contract forbids fusing the two". A GPU that contracts `a * b + acc` into one
`fma` produces a different number one ULP away, and a bit-exactness claim built on it would be false.

**The method is a discriminator, not a shape.** A triple where the fused and separate roundings differ —
found by search, `fma(a,b,c) != round(round(a*b)+c)`, and both bit patterns asserted in the test so the
probe cannot pass by being unable to tell the difference — compiled under every `MTLMathMode` and four
spellings of the accumulation:

| | `plain` | `product-first` | `fma(x,w,0)` |
| --- | --- | --- | --- |
| `.fast` | fused ✗ | fused ✗ | **separate ✓** |
| `.relaxed` | fused ✗ | fused ✗ | **separate ✓** |
| `.safe` | fused ✗ | **separate ✓** | **separate ✓** |

**`.fast` and `.relaxed` both contract**, so `D10`'s worry was right. **`.safe` does not fix it on its own** —
it keeps a product in its own statement apart from the following add, but still fuses a single expression —
so a GEMM written plainly would be wrong under every mode the unpack might choose. **`accumulator +
metal::fma(x, w, 0.0f)` reproduces the contract under all three**, and that is the spelling a bit-exact GPU
GEMM has to use. It also means the GEMM can share `.relaxed` with the unpack instead of forcing a second,
slower pipeline.

**The trap is worth more than the answer.** The same probe run over a 257-term dot **matched in all nine
mode-and-formulation combinations**, including the ones that contract, because over a long sum the
contraction differences cancelled. A test written against a long dot — the obvious shape to test a matmul
with — would have concluded that the GPU was exact and been wrong. The discriminator has to be **one
multiply and one add**, with the addend nonzero, which is why the probe is shaped that way rather than like
the operation it is clearing the way for.

Two tests carry it: `testWhichModeAndFormulationReproduceTheContractsRounding` asserts that at least one
combination reproduces the contract and prints which, and
`testTheFormulationThatMatchesAlsoKeepsTheOrder` checks a 257-term dot against
`Ops.orderedMatmulScalar` — the engine's own definition of the order, not a loop written in the test.

## D62 — The head's matmul on the GPU: bit-exact over 288 shapes, and 26% off the phase

`D61` settled the arithmetic, so the kernel could be written. It is deliberately simple: **one thread per
output element, `k` accumulated in ascending order**, one rounding for the product and one for the add, with
the product spelled `metal::fma(x, w, 0.0f)` because that is the form `D61` measured to survive every math
mode. There is no reduction and nothing to reassociate, which is exactly why a GPU kernel can be bit-exact
against a sequential loop.

**The proof is a grid, not an example.** `MetalMatmulTests` asserts bit-identity against
`Ops.orderedMatmul` over **288 shapes** — `rows` 1, 2, 3, 5; `k` 1, 2, 3, 4, 5, 8, 17, 64, 257; `out` 1, 2, 3,
4, 5, 8, 17, 129 — chosen so that output counts which are **not** multiples of four are in it, because that is
where the CPU's four-wide vector takes its tail. Plus the head's real shape at `k = 2048`, buffer-cache reuse
across shapes, and refusals for mismatched buffers. **One boundary the grid does not reach was found later
(`D108`):** an Apple GPU flushes a denormal *product* to zero where the CPU keeps it, so "bit-identical" means
"bit-identical whenever no intermediate product is denormal" — which is every value the 288 shapes and the real
model produce, and is now asserted as a named test rather than left implicit.

**The choice lives in the forward, not in `Ops`.** `Ops` **is** the definition of the contract and should not
know about a device; `Qwen3_5Forward` is the wiring, which is what its own comment says. So a small chooser
sits there, with a **work threshold** of a million multiply-adds — deliberately conservative, since the CPU
measures about 1.4 G of them a second and a dispatch costs roughly a hundred thousand — and a fallback to
`Ops.orderedMatmul`. Because the two are bit-identical, that fallback cannot change a trace, which is the
property that makes a silent fallback safe rather than sloppy. `SHARD_GPU_MATMUL=0` turns it off.

**End to end: identical, and the phase moved.** The real 35 B trace is `b0d382dbabf36df0…` with the head's
matmul on the GPU and on the CPU, and `trace_diff` against the stored trace reports IDENTICAL for both.

| phase | CPU head | GPU head |
| --- | --- | --- |
| `head` | **1.74 s** | **1.28 s** |
| `attn.core` | 4.08 s | 4.05 s |
| `mix.read` | 5.76 s | 5.86 s |

**The totals would have hidden this.** The two runs take 16.3 s and 16.4 s — the wrong way round — because
the reads vary by more than the saving. A phase profile resolves what a total cannot, which is the same lesson
as `D60` from the other side.

**What the head still spends is I/O, and it is the same floor.** The head is the embedding — 248,320 rows of
2,048 bf16, **1.02 GB** — and at 1.28 s that is about **0.8 GB/s**, the sequential rate the expert reads
already established. So the kernel removed the compute and left the floor, and no further kernel changes it.
That is `DC-107`'s "documented as at its limit with the measurement that says so" for the head.

**Next is `attn.core` at 4.05 s, which did not move** because its matmuls still go through `Ops`. It is the
opposite case to the head: its activations are small and its arithmetic is the cost, so the kernel is the
right instrument there rather than the wrong one.

## D63 — The GPU matmul is bit-exact and slower, so it is opt-in; and round 45's head number was drift

`D62` put the head's matmul on the GPU and measured `head` falling from 1.74 s to 1.28 s. This round routed
the **core's** matmuls through the same chooser as well — sixteen call sites in `GatedDeltaNet`, fifteen in
the forward, seven in the mixture, which is every contract matmul in the engine — and the result was slower
everywhere. Re-measuring the head the same way showed why the earlier number was wrong.

**Correctness is not in question.** The kernel is bit-identical to `Ops.orderedMatmul` over the 288-shape
grid, the head's real shape and cache reuse; the real trace is `b0d382dbabf36df0…` with the matmul on the GPU
and on the CPU; `trace_diff` between them is IDENTICAL. 199 tests pass. What follows is about speed only.

**The measurement had to be controlled, and the first one was not.** Comparing a "CPU" run with a "GPU" run
taken afterwards gave `attn.core` 4.08 against 5.20 s and looked like machine drift — because *every* phase
was up, including `mix.read`, which no matmul touches. Alternating the conditions on one build settled it:

| phase | off | on |
| --- | --- | --- |
| `attn.core` | 4.05, 4.06 | **5.52, 5.64** |
| `mix.gateup` | 1.17, 1.18 | **1.65, 1.60** |
| `head` | 1.74, 1.74 | **1.88, 1.85** |
| `mix.read` (no matmul) | 6.30, 6.34 | 6.38, 6.33 |

Reproducible to a tenth of a second, and the untouched I/O phase is identical, so this is the kernel and not
the machine. **The lesson is the method**: conditions that are run in sequence drift together, and the earlier
head figure — CPU first, GPU second — was measuring a warming machine. Alternating them is what makes the
difference visible; that is how this table was taken.

**The cause is the access pattern, not the arithmetic.** With one thread per output the inner loop is over
`k`, so thread `column` and thread `column + 1` read `w` rows **`k * 4` bytes apart** — 8 KB at the head's
width. A warp therefore touches thirty-two separate cache lines to use four bytes of each: roughly
thirty-two-times the memory traffic the arithmetic needs. The CPU's vectorised path walks `k` **contiguously**
inside one output and reuses `x` across outputs, which is why it wins. This is the textbook naive-GEMM
pattern, and it is the one thing the kernel's simplicity bought that it should not have.

**The fix is known and does not threaten bit-exactness.** A **threadgroup-tiled** kernel loads a tile of `w`
into shared memory cooperatively — coalesced, one line per warp instead of thirty-two — and each thread still
accumulates over `k` **ascending** inside its own output. Tiling changes *which thread* does the accumulating,
not the order, so the grid's assertion carries over unchanged.

**So the default is off.** `SHARD_GPU_MATMUL=1` turns it on; the kernel, its grid and the chooser stay, so the
tiled version has somewhere to land. A kernel that is slower is not a default, and `D62`'s conclusion that the
head was at its I/O floor does **not** hold either: the head's compute was never removed, and the 1.28 s was a
warmer machine rather than a faster matmul. What *does* hold from `D62` is the part that was measured twice:
the head reads 1.02 GB of its own weights, and that is the floor under it.

## D64 — The tiled kernel is bit-exact and still slower: the engine is I/O-bound, and kernels are not M3's lever

`D63` diagnosed the GPU matmul's slowness as the access pattern — one thread per output walking `w` rows 8 KB
apart — and specified a threadgroup-tiled kernel as the fix. The kernel is written, and the diagnosis was
**wrong**.

**The kernel is right.** Each threadgroup owns 32 consecutive outputs for one row of `x`, and the tile is
loaded as a **transpose**: lane `l` fills column `l` of every row, so a warp reads `w[row*k + base + 0..31]`
— 32 contiguous floats, one 128-byte line — instead of each lane walking its own row. Each thread still
accumulates its output **ascending in k**, tile after tile, so the sequence of roundings is unchanged and
the grid's assertion carries over untouched: bit-identical over the 288 shapes (including `k = 257` and
`out = 17, 129`, both tiles-and-a-half), the head's real shape, and cache reuse. End to end the real trace is
`b0d382dbabf36df0…` with the tiled kernel and without it, `trace_diff` IDENTICAL.

Two traps were found by that grid rather than by reasoning, and both are worth keeping:

* **A `return` before a barrier is not a shortcut.** The first version returned early for lanes past
  `columns`; the grid failed with the first element of every short row correct and the rest wrong, because
  the surviving lane read a tile row whose owner had exited. A barrier that some lanes have left is
  undefined. It is predicated now, and the load is guarded on the column rather than on activity, because a
  lane past `span` simply has no column to fill.
* **The load must be a transpose, not each lane's own row.** Loading `tile[lane][0..span-1]` is correct and
  is exactly the 8 KB stride the kernel exists to remove; a warp has to cover one row of `w` together.

**And it is still slower.** Alternating the conditions on one build, as `D63` established:

| phase | CPU | GPU (tiled) |
| --- | --- | --- |
| `attn.core` | 4.05, 4.07 | 5.88, 5.60 |
| `head` | 1.74, 1.74 | 1.98, 1.81 |
| `mix.gateup` | 1.18, 1.17 | 2.12, 2.08 |
| `mix.down` | 0.53, 0.53 | 1.10, 1.04 |

Coalescing the loads changed nothing — the naive and tiled versions cost the same — so the access pattern was
not the bottleneck. What is left is the **path around the kernel**, and it is expensive: a command buffer and
encoder per call, two `copyMemory` calls to move `x` and `w` into the cached buffers, and a
`waitUntilCompleted` that serialises every dispatch. `mix.gateup` gives the clearest number: about 210 calls,
**0.9 s of overhead, ~4 ms per dispatch** — orders of magnitude more than the dispatch itself, and consistent
with a GPU that has to be woken for each one.

**The strategic reading is what matters.** A profile of the same run says where a forward actually goes:

```
mix.read + load + head   12.73 s of 18.75 s   68%   (I/O, at the sequential floor)
attn.core + gateup + down 5.77 s                     (compute)
```

**The engine is I/O-bound.** The reads are already at ~1 GB/s sequential, which no kernel changes, and the
fix for that is not arithmetic — it is **sharding**, which is what `M2` built and `M3` demonstrated: four
nodes give four times the read bandwidth. So GPU compute kernels are **not on M3's critical path**, and
rounds of tuning them would be work done because it is interesting rather than because it moves the gate.

**Therefore: the kernels stay, opt-in, as a tested asset** — bit-exact, gridded, and reachable with
`SHARD_GPU_MATMUL=1` — and the work stops here rather than continuing to chase a lever that the measurement
says is the wrong one. `DC-033`'s Metal clause is recorded with its measurement and the task is closed; its
done-when, the tiny model bit-for-bit against the contract with router decisions included, has been met since
long before this.

**One more method note, because it nearly went wrong twice.** The first tiled measurement showed *every*
phase identical between "on" and "off" — because the flag's default had been inverted by the round before, so
both runs were the CPU. A comparison that shows no difference at all is as suspicious as one that shows the
wrong direction: **check that the conditions differ before believing either**, in the same way that
alternating them is what stops a warming machine from being read as a speedup.

## D65 — M1's "identical generated tokens" is now verified and gate-checked, not a memory

`M1`'s record has said since it was written that generation is "**0.108 tok/s** cached and **0.0374**
uncached **with identical generated tokens**". The rate was measured; the token equality was observed by hand
and never checked again — and the engine has changed since (the GPU unpack became the default, the matmul
chooser was added). A claim checked by hand is a claim that drifts, so it was re-run and then made part of
the gate.

**Re-run, on the install, eight tokens from the frozen prompt.** Both paths produce

```
11751,11,264,3177,34756,364,1141,8807
```

with **identical top-2 margins at every step** — `1.6400, 0.1339, 1.3402, 2.7697, 0.3323, 2.8134, 5.1502,
0.0445`. Two of those margins are narrow enough to be worth naming: 0.1339 and 0.0445 are calls that a
small numerical difference could move, and they did not move. The first token, `11751`, is also the argmax
`D40` measured independently at the trace's last position, which is a second, unrelated line of evidence
agreeing with the first.

**The gate now checks it.** `run_m1_gate.py` already ran the full-sequence generation for throughput; it now
runs the **cached** path as well and requires the two token lists to be **equal** — a discrete decision with
no tolerance, which is I3 — and records both in the report. Two details matter more than the change:

* **An empty parse fails.** If the `generated:` line cannot be read from either run, the gate fails rather
  than passing, because two failures to read produce two empty lists and `[] == []` is a perfectly true
  statement about nothing. That is the same shape as the diagnostic that once printed "identical" while the
  gate stayed red: an instrument has to be shown to have measured something.
* **It is verified by the gate's own test.** `test_run_m1_gate.py` drives the whole script on the 236 KB
  fixture — "so the gate is never untested code" — and that test now exercises the new path end to end.

**What this node cannot do is run M1's gate itself.** The gate hands the **checkpoint** to both sides, which
is what makes it a matched-weight comparison, and both sides map the whole 70 GB file; on an 8 GB node that
mapping is the mechanism behind two panics. So the gate's own full run belongs to a machine with more memory,
and it is part of the timing phase. What this node can establish, it now has: matched-weight traces that are
byte-identical (`D56`), and generated tokens that are identical between the two generation paths.

**One process note.** The first run of this verification lost the token line to `tail -3`, and the claim I was
checking was on that line. The repository already records what a `| tail` does to an exit status; this is the
same lesson one step further on — **a claim being verified has to be captured in full, not summarised**.

## D66 — The cluster claims were not gate-checked either: I2 and M2 re-established on the current engine

`D65` found one claim that had been checked by hand once and had drifted. Asking the same question of the rest
of the repository finds a larger one: **`check_milestones.py` re-runs the single-node trace and nothing else**,
so the cluster claims — `I2` ("sharding is semantically free"), M2 and M3's functional half — rest on manual
runs made **before** the engine changed. Since those runs the GPU unpack became the default and every contract
matmul went through a chooser, and both are asserted bit-identical, which is exactly the sort of assertion
that wants re-checking rather than trusting.

**Re-run, two machines, real model, current binaries.** `tools/run_m2_gate.py --remote node3@… --install
.build/m1-install --remote-install /Users/node3/Downloads/m1-install` — the peer's own install, so this is a
functional test and not a data migration:

```
[2/4] reference, one node here     83 tensors  40 discrete  digest b0d382dbabf36df0…   16.0 s
[3/4] node 1 on the peer machine   83 tensors  40 discrete  digest b0d382dbabf36df0…   14.4 s
      node 0 here                  83 tensors  40 discrete  digest b0d382dbabf36df0…   13.9 s
[5/5] node 0 IDENTICAL — 83 tensors, 0 differing elements, 40 discrete decisions
      node 1 IDENTICAL — 83 tensors, 0 differing elements, 40 discrete decisions
```

A 256-expert plan over two contiguous halves, one node on another machine reached over TCP, each node reading
**its own half** — 2,053,044,736 bytes on one and 2,051,226,112 on the other — with the reduction contract
exercised 40 times per node over 797 and 803 terms. Both traces are **byte-identical to the single-node
reference**, digests included, on binaries that contain the GPU unpack default and the matmul chooser. `I2` and
`M2` are therefore re-established rather than inherited, and the audit's `I2` evidence now cites this run.

**One observation, deliberately not a claim.** The all-reduce cost differed by a factor of three between the
two nodes — 1.497 s here against 0.477 s on the peer. That is not a defect and not a finding: the local node
is the one that *connects*, it was also running the single-node reference immediately beforehand, and this host
is the 8 GB machine of the pair. The numbers are recorded because a later timing phase will want them, not
asserted, because a functional gate is not a benchmark (`D40`).

**What deliberately did not run is the four-node mesh**, which is M3's distinctive functional claim. One peer
is at load 14, and borrowing a busy machine's time to re-check a claim whose two-machine half is now
re-established is not worth it. The mesh and the throughput gate both belong to the quiet window the operator
has already reserved for them, and `DC-053` carries them.

**The method is the point, and it is the second time it has paid.** The single-node claims are all
gate-checked — `check_milestones` re-runs the trace, the claims gate re-reads the numbers, `test_run_m1_gate`
drives the gate itself on a fixture. The cluster claims were the ones resting on memory. Asking "what has *not*
been re-checked?" is now two for two.

## D67 — "The form a reader copies" now includes paths, and the docs were already consistent

The repository's own trap says it: **check the form a reader copies, not only the prose.** A previous round
widened `check_status_claims.py` to read the `--swift-tests` flag beside the sentences, because that was the
number most likely to be pasted. The same argument applies to **paths**, and nothing checked them: the link
gate validates `[text](path)`, and a bare `tools/run_m1_gate.py` in a sentence or a code block is not a link.

**So the claims gate now checks them**, and this is a gate about *shape* rather than about numbers: every
`tools/…`, `docs/…`, `sources/…` or `tests/…` path with an extension in the six claim documents must exist.
Two rules in it are worth naming, because both are properties rather than lists:

* **A template is not a claim about a file.** `RELEASE.md` legitimately names
  `docs/release-notes-vX.Y.md`, and a rule that fired on it would have to be silenced with an exception —
  which is how a rule stops being true quietly. The test is that a path in this repository is **lower case**,
  so a mixed-case or angle-bracketed name is a shape and is skipped by rule.
* **An empty scan is not a pass.** The paths checked are counted into the gate's total, which went from
  **73 claims to 126**, so "no problems" means fifty-three paths were looked at rather than that the loop did
  nothing. That is the same distinction as `D65`'s empty-parse guard.

**And the result is negative, which is the good outcome.** All fifty-three exist: the documentation's paths
are consistent with the tree, in the README, `AGENTS.md`, the tracker, the roadmap, the testbed page and the
architecture page. That is now enforced rather than hoped for, and a deleted script will fail the gate instead
of leaving a command that cannot be run.

**The same audit turned up one stale thing in the gate itself.** Its `--help` example still read
`--swift-tests 184 --python-tests 200`. Those are example *values* rather than claims, so nothing checked
them — but they are printed to a reader who may copy them, so they are now placeholders. An example that
cannot go stale is better than an example that has to be maintained, which is the same reason the shard plan
is generated rather than hard-coded.

## D68 — M3's four-node mesh re-verified, and the functional half made runnable on its own

`D66` re-established the two-machine claim and deliberately left the four-node mesh alone, because one peer was
at load 14. The farm went quiet — node1 at 2.5, node2 at 1.4, node3 at 1.0 — and this round ran it.

**The mesh is M3's distinctive functional claim** — every node generates from the same prompt in a full mesh
driven by the loaded plan, and the result is compared with the single-node run. Four machines, current
binaries, each reading **its own** install:

```
baseline, one node here      tokens [11751, 11, 264, 3177]
node 0  IDENTICAL — tokens [11751, 11, 264, 3177]
node 1  IDENTICAL — tokens [11751, 11, 264, 3177]
node 2  IDENTICAL — tokens [11751, 11, 264, 3177]
node 3  IDENTICAL — tokens [11751, 11, 264, 3177]
```

Tokens **and** digests, on all four, over a plan staged to three peers that already held their own install
copies — so the only thing that moved across the network was the binary and the plan. The tokens are also the
first four that `D65` verified for the single node, which is a second line of evidence rather than a second
measurement of the same thing.

**The tool now expresses the split the operator asked for.** The timing half of M3 is a quiet-window
measurement; the functional half is a different question, answerable any time. `run_m3_gate.py
--functional-only` runs the cluster, checks tokens and digests, and **stops** — it does not compute, print or
record a speedup, and its report **nulls** the timing fields rather than omitting them, so a later reader
cannot mistake the output for a measurement. It also skips the quiet-farm precondition and says that it
skipped it, because a functional run is not making a throughput claim and should not pretend to have checked
a precondition it does not use.

**Two of my own defects, and the run caught both.**

* The functional path returns **before** the point where the output directory is created, so the first run
  wrote its report into a directory that did not exist and ended in a traceback — after printing four
  IDENTICAL lines. The fix creates the report's parent, because a report path whose parent does not exist is
  the caller's intent rather than an error.
* **I hid that traceback's exit code with a pipe to `tail`.** The gate crashed and my command reported `gate
  exit: 0`, because `$?` was the exit status of `tail`. This is the third time in this session that the same
  recorded lesson has bitten, and the fix is not care but shape: capture the output to a file and read `$?`,
  which is how the re-run was done.

## D69 — M1's gate can run against an install, which it never could, and the fixture found a latent shape bug

`D65` recorded that M1's gate cannot run on this node because it hands the checkpoint to both sides and both
map 70 GB. That is true of the *checkpoint* form and was never true of the install form: M1's claim was
restated in `D56` as the engine against a contract reading the **same install**, which needs no mapping at all.
The gate nonetheless could not do it, for a reason that is worth writing down precisely.

**The defect: the contract half never passed `--stream-experts`.** `install_source.py` refuses to materialise an
expert stack — a single expert layer is gigabytes, so it raises rather than trying — and its own docstring says
the streaming path "exists for those and the contract uses it (`stream_experts=True`)". The `InstallSource`
construction in the contract was right; the **gate's command line was not**, so the contract died with
`InstallSourceError` on the first expert stack it met. The intent was documented in the module and absent from
the call site, which is the same shape as `D65`'s manual claim: a statement about the code that no execution
had ever tested.

**The fix is small and the flag is two things at once.** The gate now passes `--stream-experts`, and it also
discovers whether it was handed a checkpoint or an install — from `install.json`'s `family` or `config.json`'s
`model_type` — rather than making the caller name the kind with another flag, because the path the caller
already chose says which they meant. The flag is **correctness** for the install, which refuses without it,
and **memory** for the checkpoint: `safetensors_source.py` and `uncached_safetensors.py` do *not* refuse an
expert stack, they materialise it. That is why this gate declares 4.2 GB, and streaming it should lower the
real figure — the declaration stays as it is, because a declaration is a promise of the worst case and lowering
a guard to match a measurement is how guards get weakened.

**And the fixture earned its keep.** With the flag in place the gate still failed, on the tiny install, with
`ValueError: cannot reshape array of size 512 into shape (1, 32, 32)`. `InstallSource._materialise` took its row
width from the *original* tensor's last axis instead of the install's own row width, `padded`. For the real
model those are equal — `expert.stack_gate_up` is 2,048 wide either way — and for the fixture they are 32
against 512, so the read was correct for the 35 B model and wrong in general. Widening a row to `padded` is
also right for the one-dimensional rows and for the convolution's `(8192, 1, 4)`, whose row is four values
rather than one, so the fix needed no special case. This is the first defect the tiny fixture has found in the
source rather than in a test, and it found it because the new install-mode case drives the path no checkpoint
case can reach.

**Two process notes, both mine.** The gate holds the heavy-job lock while it runs and I released it, because I
was treating the lock as mine to tidy; the next tool that tried to start — `quantize.py` on a fixture — refused
and printed the holder's purpose and pid back at me, which is how the mistake surfaced rather than hid. And
Python buffers stdout when it is redirected, so a 12-minute run's log held nothing but its first warning:
background runs of long jobs use `-u`, which is a property of the invocation rather than of the program.

**The fixture then found a third thing, and this one is not fixed.** With the shape bug corrected the
install-mode gate case runs end to end on the tiny install and reports **DIFFERS** on both fixture prompts,
in 0.04 s at 0.02 GB peak — so the streaming path works and the engine and the contract simply do not agree
there. The real 35 B install agrees byte for byte (`D56`), so this is a fixture-scale divergence, and it is
`DC-113` rather than a guess: either the tiny install's quantisation reaches a Swift/Python dequantiser
difference the real geometry avoids, or the fixture's install is built against a policy the fixture cannot
satisfy. The case stays in the suite as an **expected failure**, because an expected failure becomes an
unexpected *success* the moment the divergence is understood, and nothing else drives that path.

**And the declaration had to follow the source.** The gate claimed 4.2 GB unconditionally — the checkpoint
path's measured peak (`docs/m1-gate.md`: 4.16 GB), taken *before* the contract streamed expert stacks — and
this node's reclaimable headroom now sits below that, so the gate refused on an install it could have run. An
install with streaming held **1.21 GB** resident when measured during the real run, and the engine's own trace
is smaller, so the install path now declares **1.5 GB** and the checkpoint path keeps 4.2 GB. That asymmetry is
deliberate: the install figure is a measurement with a margin, and the checkpoint figure is a measurement
nobody has repeated since streaming landed. Lowering it would be a guard weakened to match a guess, and
re-measuring the checkpoint path is its own job.

## D70 — DC-113 is int4-specific, and the contract's install reads are the slow half of M1's gate

Two measurements this round, both narrowing rather than guessing.

**DC-113 runs in 0.04 s, so it can be interrogated directly.** Pointing both sides at the tiny **checkpoint**
instead of the tiny install gives

```
IDENTICAL — 7 tensor(s), 0 element(s), 2 discrete decision(s) checked (matching digests)
```

so the model code and the arithmetic agree, and the divergence lives in the quantised path. Pointing both at
the tiny **install** reproduces it: `embed.out` and `layer.00.hidden_in` agree, `layer.00.hidden_out` differs
by 18,676,984 ULP, and the router top-k follows (`layer.01.router.topk`, `[5, 0, 0, 3, 6, 2]` against
`[1, 0, 6, 0, 3, 4]`). A difference that large with a *matching* input is structural, not rounding.

Layer 0 is a **Gated DeltaNet** layer, and the tensors that carry its computation are exactly the ones the
install pads: `in_proj_qkv` is 32 wide stored at 64, `in_proj_z` and `out_proj` are 32 wide stored at 64, and
the expert stacks are 16 and 32 wide stored at 64 — with a **group size of 64**. So every one of them is a
**partial group on a padded row**, which is the `DC-087` family, and it is a geometry the real model's tensors
do not have: on the real install the two readers agree byte for byte (`D56`), which is why this could sit
undiscovered. The engine's reader and the contract's reader are each verified against their own goldens, and
the fixture is the only case that drives the padded path end to end. The next step is a bisection rather than a
guess: build the tiny install with a policy that quantises **one role at a time** — a second each — find the
tensor that first moves `hidden_out`, and compare the two readers' values for that tensor alone.

**The other half of the round was a measurement that killed my own hypothesis.** M1's gate could not finish a
**five-token** prompt in twenty-five minutes, so I suspected the recorded trap — `_materialise` accumulating a
Python `list[float]` at twenty-four bytes per value. Measured on one real expert (1,048,576 values):

| formulation | time |
| --- | --- |
| iterate the rows, build nothing | 0.320 s |
| today: `list[float]` | 0.315 s |
| candidate: preallocated fp32 array | 0.335 s |

All three are the same, so allocation is not the cost and my hypothesis was wrong. The read rate is **12.4 MB/s
at 16 KB per read and saturates at 17.7 MB/s at 2 MB per read**, so it is not syscall overhead either: it is the
**per-row crossing into Python**, 2,048 of them per expert. `.fast`, `.relaxed` and `.safe` were not involved
and neither is `F_NOCACHE`; the Swift path reaches ~1 GB/s on the same hardware (`D64`) because it never
materialises rows one at a time. The fix is a bulk read in `install_reader` — `np.frombuffer` over one `pread`
— and that is a **timing-phase** item, deliberately not done here, because the operator has deferred timing
work and this is performance rather than correctness. It is `DC-114`.

**What this means for M1's gate is a cost, not a verdict.** The gate is *functionally* able to run against an
install now (`D69`); on this node it needs hours per prompt because the contract's read path is 80× off the
engine's, and the engine's own trace is not the bottleneck. That is why the milestone's own gate remains a
quiet-window, or bigger-node, item rather than something I can report as passed.

## D71 — DC-113 fixed: the install source mapped an expert to half its rows, and my D69 fix was wrong

`D70` ended with the next step named: bisect the tiny install by quantising **one role at a time**. It took one
round and it worked.

**The bisection.** With int4 on exactly one role and everything else at its declared precision, **every role is
IDENTICAL except the two expert stacks** — `expert.stack_gate_up` and `expert.stack_down` both DIFFER, and
`attn.*`, `linear.*` and `mlp.*` all agree. So the divergence is not quantisation in general and not the Gated
DeltaNet projections; it is the *stacks*, which are exactly the tensors whose `shape.last` is not their
`padded_columns`.

**The decisive number was in the manifest.** For the fixture's `expert.stack_gate_up`, shape `(8, 32, 32)` padded
to 64, the quantiser records `nbytes: 9472`. That is **256 rows** of 32 padded to 64 — the layout
`Install.swift` computes as `prod(shape.dropLast())` rows of `shape.last` — and it is **not** the 4,736 bytes
that 128 rows of 64 would be. The Swift reader was right and the Python reader was wrong.

**The bug is one formula, and it is off by exactly two.** `InstallSource._rows_per_index` asked how many install
rows make one checkpoint row and answered "the flattened trailing width divided by the padded row":
`(32 * 32) // 64 = 16`. The truth is the product of everything **between** the first and last axis — 32 — because
the install keeps the leading axis as the index and drops only the last axis into the row. Sixteen against
thirty-two, so the reader returned half of every expert and the reshape that followed was the
`cannot reshape array of size 512 into shape (1, 32, 32)` of `D69`.

**And `D69`'s fix was wrong, which the same evidence shows.** I read that error as an over-trim and changed
`columns` from `shape[-1]` to `min(padded, _width)`. The error was in the **row count**, not the column count:
16 rows of 32 is 512, and 32 rows of 32 is the 1,024 the reshape wanted. The correction is to take the row
count from the middle axes and the row content from the **last axis**, which is what the quantiser padded, and
to revert my change. What misled me was `_width`'s own docstring — "every trailing dimension flattened, because
`rows()` is per leading index" — which is **false of the quantiser**. The payload's `nbytes` is the authority,
and it was one command away.

**Why the real model never showed it.** On the real install `shape.last` and `padded_columns` are both 512 and
the middle product is 2,048, so the old formula returns 2,048 as well: the two agree, and `D56`'s byte-identical
result is not evidence for the wrong one. Every real-model geometry is **provably unchanged** by this fix:

| tensor | shape | padded | old `(factor, columns)` | new | |
| --- | --- | --- | --- | --- | --- |
| `expert.stack_gate_up` | (256, 2048, 512) | 512 | (2048, 512) | (2048, 512) | same |
| `expert.stack_down` | (256, 2048, 512) | 512 | (2048, 512) | (2048, 512) | same |
| `linear.in_qkv` | (2048, 512) | 512 | (1, 512) | (1, 512) | same |
| `norm` | (512,) | 0 | (1, 0) | (1, 0) | same |
| `linear.conv` | (64, 1, 4) | 0 | (4, 0) | (1, 4) | **changed** |

**The last row is a second, latent bug found by the same table.** The convolution was being read as `factor 4`
with `columns 0` — four install rows per index and no values from them. The correct reading is one row per index
and four values, which is the `(8192, 1, 4)` case `_width`'s docstring was written about. Nothing had compared
its output, because the checkpoint contract and the install contract both reached agreement on that path
another way.

**How the fix is verified.** The install-mode gate case that *was* `DC-113`'s expected failure now **passes**,
and the suite reports **373 tests, OK, expected failures 2** where it reported 3 — the count moving is the
evidence, and it is the same shape as `D68`'s unexpected success. The tiny install with the **full** policy is
`IDENTICAL` between the two sides, and so is each single-stack policy that used to differ. `DC-113` is closed.

## D72 — The class of `D71`, not just the instance: an index mapping now has a check

`D71` fixed a formula. This round asked where else that *kind* of defect could hide, and found that the check
which should have caught it did not exist anywhere.

**The manifest is provably consistent, which locates the bug precisely.** For every tensor in the real install
(693) and in the fixture (36), the recorded `nbytes` equals `prod(shape[:-1])` rows of `padded_columns`, or of
`shape.last` when the manifest's zero means "not padded": **0 mismatches**, and 20,695 MB accounted for against
a known install of about 20 GB. So the payload agrees with the layout, `Install.geometry()` already derives the
same rows and already refuses a payload that disagrees, and `verify_install.py` already calls it for every
tensor. The wrong number was in the **one derived quantity nothing checked** — `_rows_per_index`, which maps the
checkpoint's index space onto the install's rows and had no invariant and no test.

**So the fix is an invariant rather than another patch.** `InstallSource.check_index_mapping` asserts two things
that must both hold for any tensor with a shape, and both are arithmetic over the manifest:

* the install's rows are the checkpoint's indices times the rows one index takes — `shape[0] * factor ==
  geometry().rows`;
* the values one index contributes are the checkpoint's trailing dimensions — `factor * shape.last ==
  prod(shape[1:])`.

For `D71`'s tensor the first fails with 8 x 16 = 128 against 256 rows, which is the half-expert that cost two
rounds to find. It is called **for every tensor when an `InstallSource` is opened**, so it fails at the point of
use in every caller — the contract, the gate, a one-off script — rather than in a test that someone has to
remember to write. A check is worth more than the attention that missed it.

**And it is verified both ways, because a check that cannot fail is decoration.** The real install (693 tensors)
and the tiny one (36) open cleanly, so the invariant is not a false positive on the model it was written for.
A test pins the `D71` geometry — a `(2, 2, 4)` stack padded to 8, where an index is two install rows and the
padding is dropped rather than read — and then **reinstates the formula that was wrong** and asserts that
opening now refuses, with the message naming both numbers. The suite reports **374 tests, OK**.

**The process note is the point of the round.** Two rounds went into bisecting what twenty lines of arithmetic
catch at open. The generalisation is not "be more careful": it is that a quantity *derived* from a checked one
needs its own check, because the check on the input says nothing about the derivation. `geometry()` was right
and verified; `_rows_per_index` was wrong and unchecked; the fixture's whole value was that it made the
difference visible at all.

## D73 — M1's gate passes on the real checkpoint, on this node, and the other half of the safe pair was missing

`docs/m1-gate.md` records what made a real-model contract run survivable: **both** flags together. "With both, the
contract run finishes in about two minutes with swap flat at ~1.2 GB and disk steady at 11 GB, where the attempt
without them drove swap to 5.1 GB and disk down to 8.0 GB in a minute." The gate passed **neither** for the
checkpoint path. `D69` added the first, `--stream-experts`, because an install refuses to materialise a stack;
this round found the second missing.

**The checkpoint path was mapping 67 GB.** Without `--uncached` the contract falls through to `SafetensorsSource`,
which uses `safe_open` and therefore **maps** the file — and the node's own rules say it plainly: a read that does
not go through `UncachedFile`/`open_uncached` "still is the old hazard". So the gate's checkpoint runs were the
hazardous variant the doc warns about, while `uncached_safetensors.py` sat there saying "never mmap, so a contract
run does not fill the page cache".

**Measured before changed.** The contract on the checkpoint, with both flags:

```
66.20 real   58.05 user   2.42 sys     397,344,768  maximum resident set size
```

0.397 GB, against the 4.16 GB `docs/m1-gate.md` recorded before either flag existed. Then the gate itself, one
prompt: `IDENTICAL — 83 tensor(s), 0 element(s), 40 discrete decision(s) checked (matching digests)`, engine
34.5 s, peak 3.71 GB.

**And then the whole thing, which is what this round is for.** All five frozen prompts, on the real checkpoint,
on this 8 GB node:

| prompt | tokens | engine | peak RSS | |
| --- | --- | --- | --- | --- |
| capital | 5 | 34.94 s | 3.79 GB | IDENTICAL |
| arithmetic | 33 | 89.65 s | 3.76 GB | IDENTICAL |
| code | 40 | 115.92 s | 3.63 GB | IDENTICAL |
| repeat | 60 | 128.13 s | 3.60 GB | IDENTICAL |
| long | 67 | 146.26 s | 3.67 GB | IDENTICAL |

`GATE PASSED`, 83 tensors and 40 discrete decisions on every one. The engine's digest for the first prompt is
`b8c976c5e7ba8816…` — **the figure `docs/m1-gate.md` recorded for this pair on 2026-09-16**, still reproduced
after the GPU unpack became the default and every contract matmul went through a chooser. Expert traffic runs
2,240 to 8,040 requests and 3.5 to 12.6 GB of elements read, with a cache hit rate of **0** on every prompt,
which is `D31`'s finding arriving again from a different direction.

**A correction I made within the hour, because the measurement said so.** I first re-based this path's
declaration on the contract's 0.397 GB and set it to 1.5 GB. The gate's own peak — **3.79 GB** — proved that
wrong immediately: streaming made the *contract* cheap, but a checkpoint run's peak is the **engine's** trace
over bf16 weights. The declaration went back to 4.2 GB with the measurement that now justifies it, because an
under-declared guard admits a job the machine cannot take, which is worse than a conservative one. It is `D71`'s
lesson in another register: measure the thing the number is about, not the thing that was easy to measure.

**What this supersedes.** `D65` concluded that M1's gate "belongs on a machine with more memory" because it maps
70 GB. For the **checkpoint** path that is no longer true: it ran here, in about eight minutes of engine time
plus the contracts, with both flags. What remains specific to this node is the **install** path, where the
contract's per-row Python reads make the same run take hours (`DC-114`) — a cost, not a correctness problem.

## D74 — The same question was answered in three places, so it now has one home

`D69` added `--stream-experts` to an invocation that had omitted it and `D73` added `--uncached` to the same
one. Two rounds, one shape: a contract was invoked without the flag its own documentation called for. So this
round asked the structural question instead of waiting for a third instance — *where else is "which reader, and
with which flag" decided independently?* — and audited every tool that invokes a contract.

**The finding is one model family down.** `ordered_qwen35_trace.py`, M0's contract, had **no `--uncached` flag
at all**: line 51 read `source = SafetensorsSource(args.snapshot)` and nothing else, so every M0 run mapped its
checkpoint and there was no way to ask it not to. `run_m0_gate.py` could not have passed the flag because the
flag did not exist. The 2B checkpoint is smaller than the 35 B one, which is why this never produced an
incident — but the rule the node's own instructions state does not have a size threshold: a multi-gigabyte
mapping is a hazard rather than a neutral operation.

**So the question has one home.** `tools/contract_source.py` holds the selection rule and the flag that goes
with it: `open_source(snapshot, uncached=…)` — an install if that is what the path is, else a checkpoint read
through `pread` or through `safe_open`'s mapping — and `add_uncached_argument(parser)` for CLIs to declare it.
Both trace CLIs call it, so the 2B contract gains both `--uncached` and the install branch the 36 B one already
had, and `run_m0_gate.py` and `check_engine_contract.py` now pass the flag for their checkpoint runs.

**What is deliberately not shared is `--stream-experts`, and the reason is a property of the model rather than
of the reader.** `qwen3_5` is dense — its `streamed_text_forward` has no `stream_experts` parameter and the
fixture's roles contain no `expert.*` — so a flag declared there would do nothing while reading like a
capability. A test asserts the flag is declared **exactly** where the family has experts, which encodes the
reason and not merely the current fact.

**One hidden coupling came out with it.** `test_ordered_qwen36_quant.py` reached `SafetensorsSource` *through*
the CLI module, which is what the `# noqa: F401` on that import was for. That test now calls `open_source`, so it
exercises the shared rule instead of a module attribute — and the re-export it depended on is gone rather than
silently restored.

**Seven tests, and the one that matters is the byte-identity case.** Both CLIs must accept `--uncached`;
`--stream-experts` must be declared exactly where experts exist; the rule must choose install, then pread, then
mapped — including that **an install wins over the flag**, which `run_m1_gate.py` relies on when it passes
`--uncached` only for a checkpoint; and the whole 2B contract CLI is run **twice** on the fixture, mapped and
through `pread`, with every tensor's `sha256` compared. That last one is the claim the flag rests on, and it is
now checked through the CLI the gates actually invoke rather than only at the reader. The suite reports **381
tests, OK**.

**One exclusion is named rather than skipped.** `ordered_reference.py` still maps its checkpoint. It reads the
**small** reference models (0.6B) rather than the pinned large ones, and its reader is its own rather than the
shared rule, so bringing it in would be a different change with a different risk. Saying so is the point:
the alternative is an omission that looks like an oversight.

## D75 — DC-114 fixed: the install dequantiser is vectorised, and M1's gate now passes on the install too

`DC-114` was recorded as a timing item and deferred twice, on the reading that the operator had postponed
timing work. What made it worth doing is not the number: **M1's restated claim** (`D56`) is that the engine's
trace is byte-identical to a contract reading the *same install*, and M1's own gate could not finish that
comparison — a five-token prompt had not completed in twenty-five minutes (`D70`). A gate that cannot run is a
claim that cannot be checked, so this is a correctness enabler with a performance shape.

**The cost was one loop, and it was measured before it was touched.** One real expert is 2,048 install rows of
512 values, and `install_reader._dequantize_row` visits **every value in Python** — nibble, sign, zero, denormal
flush, multiply — while `_materialise` rebuilt a `list[float]` a row at a time. Timed on one expert from the real
install: **0.253 s**. An eight-token forward fetches roughly 2,560 experts, so the expert reads alone were about
eleven minutes, which is the twenty-five-minute prompt.

**The fix splits the question along the line the modules already draw.** `install_reader.py` is
standard-library only on purpose — it gates the repository, so it has to run on any `python3` — and numpy
cannot live there. So the **offsets** stay in one place: `Install.int4_block_bytes(tensor, start, count)`
returns the `(codes, scales, zeros)` for a block, and `_int4_block` was refactored to call it, so there is now
one computation of the layout rather than two. The **arithmetic** moves to `install_source._dequantize_int4_block`,
which dequantises a whole block with numpy, and `_materialise` uses it for int4 in chunks of at least 1,024
rows. The streaming was right; the per-value loop was not.

**Bit-identical is argued and then measured.** `(code - zero)` is a small integer and the scale is a float32, so
their product needs at most 48 bits of significand: the float64 product the scalar path computes is **exact**,
and rounding it to float32 rounds once — which is what the float32 multiply does as well. The three layout
details are the layout's and not the arithmetic's: two four-bit codes share a byte with the **low nibble first**,
the codes are **signed** in four bits, and scale and zero are per **group** across a `padded` row. Denormal
scales are flushed, which `D11` requires of both readers.

Measured rather than trusted, on the real install: a full expert (`2048 x 512`), a padded `64 x 2048` matrix and
a `32 x 2048` matrix all come back **bit-identical** under `np.array_equal(....view(np.uint32))`, and the expert
drops from **0.253 s to 0.013 s**. A test pins the agreement permanently over **both fixtures' checked-in
installs** — 25 int4 tensors including the MoE fixture's expert stacks, the case the fast path exists for —
because two implementations that agree today is exactly the situation in which the second one is forgotten
(`D34`).

**And the gate that could not run now passes, on all five frozen prompts, against the install:**

| prompt | tokens | engine | peak RSS | |
| --- | --- | --- | --- | --- |
| `capital` | 5 | 17.40 s | 0.98 GB | IDENTICAL |
| `arithmetic` | 33 | 54.51 s | 1.30 GB | IDENTICAL |
| `code` | 40 | 67.63 s | 1.34 GB | IDENTICAL |
| `repeat` | 60 | 86.90 s | 1.41 GB | IDENTICAL |
| `long` | 67 | 98.45 s | 1.31 GB | IDENTICAL |

`GATE PASSED`, every comparison `IDENTICAL — 83 tensors, 0 elements, 40 discrete decisions`, with digests
`b0d382dbabf36df0…` on **both** sides — the install digest `tools/milestones.json` records. About an hour for
the whole gate, against a single prompt that would not finish. Both forms of M1's gate now pass on this node:
the checkpoint pair (`D73`) and the install pair, which is the restatement.

**One declaration moved, for the reason `D73` recorded.** The install path was declared at 1.5 GB from a
contract-only measurement of 1.21 GB; the whole gate peaks at **1.41 GB** on the `repeat` prompt, and 1.41
against 1.5 is 6%, which is a coincidence rather than a guard. It is **2.0 GB** now, measurement plus margin.

**A note on what this round was not.** The operator deferred *benchmark timings*, and no speed claim is made
here: the seconds above are what it takes for a gate to finish, and the correctness claim is byte-identity. The
change is recorded as removing an obstacle to a verification, not as an optimisation, because that is how it was
justified before it was made.

## D76 — M0's gate had no test, and a flag that disappears cannot be seen by a fixture

M1, M2 and M3's gate scripts each carry an instrument test — `test_run_m1_gate.py` says why in its first line,
"so the gate is never untested code" — and **M0's did not**. That was worth more than a coverage percentage when
`D74` changed the invocation M0's gate makes, adding `--uncached` to the contract it runs, with nothing watching
it. A gate whose invocation can drift is exactly the failure this repository spent three rounds on (`D69`, `D73`,
`D74`), so this round wrote the test that was missing.

**It drives the same script on the 236 KB fixture, so both halves of M0's gate run.** The gate does more than the
contract comparison: it also captures the **reference implementation** through `tools/trace_capture.py` and
asserts the discrete decisions against it, which is the half that would catch a wiring error every per-tensor op
test would miss. On the fixture it passes end to end — `contract comparison: IDENTICAL`, `oracle comparison:
discrete MATCH, worst relative 1.92e-07` — and the test asserts the contract comparison, the oracle match, and
that the **smallest margin** is greater than zero, because a margin of zero would be an argmax that happened to
land right rather than a decision that was made.

**The subtler half is that a fixture cannot see a missing flag.** The fixture is small enough that a mapped read
changes no number, and `--uncached` is byte-identical to `--mmap` by construction — so *no* byte-identity
assertion can notice the flag disappearing, which is how it went missing for a whole era in the first place. The
invocation had to become evidence. Both gates now define the flags **once**, use that list for the command, and
**record it in the report**, and the tests assert it:

| gate | source | flags asserted |
| --- | --- | --- |
| M0 | checkpoint | `--uncached` |
| M1 | checkpoint | `--stream-experts --uncached` |
| M1 | install | `--stream-experts` |

The install row is not an omission: an install is read through its own reader, which reads uncached by
construction, and the install branch is chosen first — so the assertion records *why* the flag is absent rather
than tolerating it. Defining the list once is the same move as `D74`'s: a flag named in two places is a flag that
can disagree with itself.

**Why this is worth a round rather than a footnote.** Three of the last eight rounds were spent finding
invocations that did not pass what their own documents required, and every one was found by reading code against
a document. After this round the class is assertable: a gate that stops passing a required flag fails a test
naming the flag. That is the difference between a defect that is discoverable and one that is caught.

## D77 — Three narrownesses hid a false status, each masking the next

`D76`'s follow-through was to bring the status surfaces current, and the wiki's landing page turned out to carry
one that was not stale but **false**: `.wiki/Home.md` said *"M1 has run on the real 35B model with all five
frozen prompts at exit 0 and two byte-identical re-runs, and **its gate is still open**"*, and quoted **105 Swift
tests** and **174 standard-library Python tests** — figures from the era it was written in, while the suites
report **200** and **384**. Two rounds earlier the same page class had been corrected in `AGENTS.md` and
`README.md`; this one had never been read by anything.

**It survived three separate narrownesses, and each hid the next.** That is the part worth recording, because
fixing any one of them alone would have left the claim exactly where it was.

1. **It was not a claim document.** `CLAIM_DOCUMENTS` was a hand-maintained list of six, and the wiki's landing
   page was not one of them. This is the trap this module already names — "*a gate whose configuration is a list
   will go stale*" — biting a **second time in the same file** that records it. The fix is the one the trap
   prescribes: **discover** the pages. A page is a claim document if it is the README, `AGENTS.md`, or any
   `.wiki/*.md` that is not the log and not GitHub's `_`-prefixed furniture. The exclusions are rules with
   reasons rather than exceptions: `News.md` records what was true when each entry was written, so a number that
   has since moved is *history* there, not an error.
2. **Its phrasing was not recognised.** The house-style patterns are `**N tests, M skipped**` and
   `**N** standard-library Python tests`; the page wrote "**105 Swift tests pass** (2 skipped" and "174
   standard-library Python tests" — so even once the page was read, neither count matched a pattern and neither
   was checked. Two prose patterns were added, deliberately narrow: a bolded count immediately before "tests",
   and an unbolded count before "standard-library Python tests".
3. **The claim was inside a blockquote, wrapped.** After both fixes the Python count was caught and the Swift one
   was still invisible, because the raw text is `"**105 Swift\n> tests pass**"` — a blockquote's continuation
   marker sits between two words of the same sentence. Reading line at a time cannot see a claim that spans a
   line break; collapsing whitespace cannot see one split by markup. `collapsed_matches` now does both, and
   keeps the line number so the message still points at a place.

**The instrument found the false claim before it was corrected**, which is the evidence that widening it was the
right move rather than a cosmetic one:

```
.wiki/Home.md:7: claims 105 Swift test(s) in prose; the suite reports 200
.wiki/Home.md:8: claims 174 Python test(s) in prose; the suite reports 384
```

Then the numbers were made true, the status was replaced with what rounds 56-61 established — M1's gate passes in
both forms on all five frozen prompts — and the gate reports **148 claims checked**, against 126 before this
round: twenty-two claims in three wiki pages that nothing had ever read. Two tests pin the widened instrument,
using the exact shape that hid the claim.

**What the round is really about.** Every one of the three defects is the same defect in a different costume: an
instrument narrower than the claim it is supposed to check. The repository already has that trap written down.
It has now been found four times — the flags (`D69`, `D73`, `D74`), the document list, the phrasing, the markup —
and the pattern in every case is that the *gate reported success*, because a claim it cannot parse is a claim it
does not disagree with.

## D78 — The install verifier was never run by a gate, and the first time it was, it was wrong

`I4` says the policy is data and `I6` says the artifact carries its own provenance, and both are checked by
`tools/verify_install.py` — which **only ever ran by hand**. So the milestone check, which is where this
repository re-checks its claims, now verifies the install **before trusting a digest computed from it**: a trace
that reproduces its recorded digest on an install whose policy coverage or provenance is broken is a green light
over a broken artifact, which is the shape of failure this repository keeps finding. The step covers schema,
roles, policy coverage, tiling and the provenance header, and deliberately not the payload digests, which are
`--digests` and cost a full read.

**Then the new step immediately failed, and it was right to.** The verifier reported the **checked-in fixtures**
as mis-tiled — and they are the installs every quantisation and install test uses. The cause is a rule stricter
than the format: `D13` measured that the payload is **64-byte aligned**, because that is what `SIMD4<Float>`
loads and a `pread` want, and `InstallWriter.add` pads each tensor up to it. The verifier demanded **exact
contiguity** instead, so every install with a tensor whose size is not a multiple of 64 was reported broken —
while the engine and the contract read it without complaint. It looked correct because it had only ever been
pointed at the real install, whose tensors are all far larger than the alignment and therefore tile exactly.
That is `D69` and `D73` again in a different tool: a check that agrees with the one case it has seen.

**The fix gives the alignment one home.** `install_reader.ALIGNMENT` is now the format's constant, used by the
writer that pads and by the verifier that checks; the rule became *no overlaps, and no gap at least as large as
the alignment*. A second correction came out of the tests: the **start** of a tensor is not required to be
aligned, because the writer guarantees that but a reader does not need it — and the hand-built fixtures in the
tests are not aligned, which is how that requirement was caught rather than shipped.

**Two tests encoded the stricter rule and were updated rather than deleted.** `test_a_gap_is_reported` used an
eight-byte gap and `test_a_payload_longer_than_its_tensors_is_reported` appended two bytes; both are now a whole
alignment wide, which preserves what each test is *for* — a gap that is not padding, a tail that is not padding —
and a new test pins the other side: a gap smaller than the alignment is the format, not a defect.

**Why a false positive in this tool matters more than it looks.** `verify_install.py` is the evidence for two
invariants, and a verifier that cries wolf about a valid artifact teaches its reader to ignore it. Both installs
now verify cleanly: the fixture at 55 tensors, and the real one at **693 tensors tiling 21,700,655,616 bytes**,
after which the milestone check reproduces M1's digest and exits 0.

## D79 — I6's strongest form had never been run: the payload digests were an assertion the tooling did not honour

`verify_install.py --digests` hashes **every payload** and compares it to the manifest's `sha256`. It had never
been run on the real install. The manifest has carried those digests, and `slab_sha256` beside them, since the
format was written — the point of a per-payload digest is that a reader can check one slab instead of a whole
tensor — and nothing in the repository had ever compared them to the bytes. `I6` said the artifact carries its
own provenance, and the strongest available evidence for it was unexercised.

**Measured, and then decided.** The whole install: **693 payload digests, 21,700,655,616 bytes, 26.69 s,
29,392,896 bytes peak RSS, zero problems**. Free disk stayed at 8 GB and swap did not move, because this is one
sequential `pread` pass and never a page-cached mapping — the envelope `verify_install.py`'s own docstring
describes, now confirmed by a run instead of by its author's intent.

**And then it became routine, which is the part that matters.** A check that runs when someone remembers is the
state this round found `I6` in. The **milestone check** now runs it — with the digest pass included rather than
left to a separate invocation, because its cost was measured *before* the decision rather than assumed, and
because the milestone check is already the mode that re-checks the claims with the artifacts in hand. `--sample
N` bounds the work for a quick run, and the verifier's output is **inherited rather than captured**: it prints
the payload coverage on success and the problems themselves on failure, so the evidence appears in the run
instead of behind a summary line.

**The check has teeth, and they are tested.** A copied install with **one byte flipped** in its payload is
reported, by the digest rather than by the geometry — which is the whole difference between checking that a
payload is the right *shape* and checking that it is the right *bytes*.

**What was verified here for the first time.** Not a claim about the engine, and not a claim about the
quantisation: that the artifact every M1 and M2 and M3 claim is computed *from* is the artifact the manifest
describes, byte for byte, across 693 tensors and 21.7 GB. Every earlier verification in this repository has been
downstream of that assumption.

## D80 — DC-085 re-audited: the format claim holds, and it now has file-level evidence

`DC-085` is the one open row whose action is *tracking* rather than timing or missing weights — "track its
releases, report cluster-relevant defects upstream" — so this round re-ran its audit rather than waiting for a
quiet farm.

**The checkout cannot answer the "releases" half.** `/Users/node4/TinyTitan` is a source drop with **no git
history** and files dated before the original audit of 2026-09-16, so there is nothing to compare against and no
release list to read. What *can* be audited is the snapshot itself, and the claim worth re-checking is the one
`AGENTS.md` makes about it: that the two install formats are not interchangeable.

**Re-verified, with the evidence the original review did not name.** The sister project's own reference states
its layout in the first line of the function that consumes it — `tools/qwen35_reference.py`, `dequantize()`:
*"`bits`-wide **unsigned** lanes packed low-first inside each uint32, one BF16 scale and **bias** per group"* —
and the arithmetic that follows is `grouped * scales + biases`. Its Swift side names the same thing:
`dequantizeInt4Affine`. **Ours** reads each four-bit code through `_signed()` — two's complement in four bits —
and computes `(code - zero) * scale`, with the zero stored as an **int8** and the scale as an **fp32**. The
container families differ too: `GTurboFormatV1`, `GTurboExpertV1`, `GTurboLayerV1`, `GTurboManifestArchV1` and
`GTurboManifestFileV1` against our `install.json` plus `data.bin`.

So the conclusion is arithmetic rather than opinion: the same bytes mean different numbers under the two rules,
and the bias types differ besides. **There is no defect to report upstream**, because its reader and its
converter agree with each other — the finding is that the formats are *different*, which is what makes the two
projects independent implementations rather than copies of one another.

**The evidence is now in the statement rather than only in a decision record.** `THIRD_PARTY_NOTICES.md`
recorded the *code* review of 2026-09-16 — the four coincident basenames, the measured overlap, the shared
vocabulary — and did **not** record what the format claim rests on. A position that a reader cannot check is a
position that has to be taken on trust, so the file now names the files, the docstring and the two formulas.

**One boundary, stated rather than glossed.** This is a *recorded* audit, not an automated check: the artifact it
audits lives in another repository and cannot be a gate here. What is automated is our own side — the provenance
gate inspects **158 files** and reports no third-party attribution to account for — and the third-party claims
themselves are re-read on a stated date and left in the file with their evidence.

## D81 — The last copyable surface: documented commands are now checked against the tools

The repository's trap says *check the form a reader copies, not only the prose*, and that check had been widened
three times: to the flags in the claims gate's own examples (`D67`), to the paths named in the claim documents
(`D67`), and to the invocations the gates make (`D76`). What was left is the surface a reader meets **first** and
copies **most often** — the commands in `docs/`, `README.md`, `AGENTS.md`, `RELEASE.md` and the wiki. A
documented command naming a flag its tool no longer accepts is a command that cannot be run, and nothing checked
one.

**The audit found nothing wrong, which is worth saying plainly.** Forty-five flags across every document, all of
them accepted by the tool they are written against, and no command naming a script that does not exist. Two
lessons came out of reaching the number: a first, narrower pass found only **thirty-two**, because it skipped the
wiki and did not join backslash continuations — the instrument's own reach decided the answer, again — and the
reason the count is worth publishing at all is that a *silent* surface looks identical whether it was checked or
not.

**So it became a gate rather than a paragraph.** `check_documented_commands.py` parses the commands out of the
documents, asks each tool what it accepts, caches the answers, and holds three properties that decide whether it
is honest:

* **A tool that cannot be asked is NOT CHECKED, not passed.** `install_reader.py` is a library with a
  self-description rather than an `argparse` CLI and exits non-zero for `--help`; the gate says so instead of
  counting its flags as fine. The one outcome it must never produce is silence dressed as success.
* **Prose that names a tool is not a command.** A sentence is not something a reader copies into a shell, and
  treating one as a command would invent flags to check — tested, because the first version of the pattern would
  have.
* **A wrapped command is still a command.** The gate docs break their examples across lines, and a flag on the
  continuation is one a reader copies.

**And the same pass answered a second question: there is no dead code.** All **fifty** non-test tools are
referenced outside their own tests — by another tool, a gate, a document or a fixture builder. The twelve files
that appear unreferenced are all `test_*.py`, which `unittest discover` finds by pattern rather than by name, so
the naive rule that flagged them was wrong about what a reference is. Nothing was removed, because nothing should
be.

**Why this is the last one of these, and why it was still worth doing.** Every widening in this series has found
a *different* way for a claim to be invisible — the flags, the list, the phrasing, the markup, the commands — and
each time the failure mode was silence: a gate reporting success over something it never read. This one closes
the surface a reader copies from, and it is the first widening whose audit came back clean. Both outcomes are
recorded because both are results: a check that finds nothing is evidence, and a check that is never run is not.

## D82 — The Swift CI job built and tested, but nothing in CI compared that to the documented counts

The surface this round examined had not been read in this session: `.github/workflows/`. Both files are in good
shape — the standard-library job runs `run_all_gates.py --skip-swift` rather than a list maintained beside the
tools, clones the wiki to check its tables, and reports a check it cannot run as *not checked* rather than as a
pass — and the Swift job selects the newest Xcode explicitly, prints the toolchain, and refuses a runner below
the standard with no skip branch. One hole was real.

**`swift.yml` ran `swift build` and `swift test --no-parallel` and never compared the result to the numbers the
documentation claims.** The other workflow cannot: it skips the Swift half, and the claims gate then honestly
reports that half *not checked*, and `run_all_gates.py` says so too. So **"200 tests, 0 skipped, 0 failures" was
compared to a real run nowhere in CI** — a test deleted from the suite would have left every document correct
about a suite that no longer existed, and both jobs green. That is the `D46` shape at the CI level: a claim no
instrument reads.

**The Swift job now runs the whole gate set**, which is what its own header already said it was — *"the engine's
gates, on a clean machine"* — and which it was not doing. `run_all_gates.py` reads the counts from the runs
themselves and hands them to the claims gate, so the step cannot go stale against them; the standard-library job
keeps the half a machine without the toolchain can run.

**Stated plainly, because it would be easy to imply otherwise: CI still fails today, by design.** `macos-26`
carries Xcode 26.x, below the manifest's 6.4 floor, so the toolchain gate stops the job before this step — which
is the correct signal and the reason `AGENTS.md` says to run the gates locally. This change is fidelity for the
day an image ships Xcode 27, not a green build now.

**A distinction that came out of it, and belongs in this record because it decides where the check lives.** The
commands in a *workflow* do not need the documented-command gate (`D81`), because a workflow's commands **run**:
a renamed flag fails the job, loudly and immediately. The commands in a *document* never run, which is exactly
why they were the surface worth gating. Same words, different instrument — because a different thing happens to
them.

## D83 — The documented cluster command could not work: it staged the install under one name and launched with another

The operator authorised a throughput run on the farm while it is busy, so the first thing I ran was the command
`run_m3_gate.py`'s own docstring shows a reader:

```
python3 tools/run_m3_gate.py --install .build/m1-install \
    --mesh node4@<addr>,node1@<addr>,node2@<addr>,node3@<addr> --steps 4
```

**It failed on every peer**, after copying the 21.7 GB install to each of them: `install.json` could not be
opened at `~/Downloads/m3-gate/install/install.json`. The cause is one name spelled in two places.
`stage_remote` copies the install into the remote directory **under its own name** — `.build/m1-install` becomes
`~/Downloads/m3-gate/m1-install` — while both gates launched the peer with the literal path **`./install`**:

    install = args.remote_install or "./install"          # run_m2_gate.py:197 and :339
    install = str(args.install) if index == 0 else (args.remote_install or "./install")   # run_m3_gate.py

That only matches when the local install happens to be *called* `install`. `--remote-install` worked all along and
is what every earlier cluster run used, which is exactly why the default was never exercised: **the flag that
worked hid the path that did not**, and the path that did not is the one a reader copies. It is `D67`'s family
again — check the form a reader copies — found this time by running that form on real machines rather than by
reading it. `stage_remote`'s docstring had drifted with it, claiming the install "is not copied here" while its own
code copies it whenever `--remote-install` is absent.

**The fix is one shared rule.** `remote_install_path(install, remote_install)` in `run_m2_gate.py`, beside
`stage_remote` that decides where the install lands, returns `remote_install` when one is named and
`f"./{install.name}"` otherwise; both gates use it at all three call sites, and the docstring now says what the
code does.

**Four tests pin it, and one of them is deliberately not a unit test.** Three assert the rule's behaviour — the
default is the staged name, an explicit path passes through untouched, and a differently named install does not
inherit the old default. The fourth reads both gates' sources and refuses the literal `"./install"` anywhere in
them. That is the one that matters here: **the defect was the two halves disagreeing**, so a test of either half
alone would have passed while the cluster stayed broken.

## D84 — M3's first real number: 0.93x, bit-identical, and the synchronisation budget that explains it

The same run, once the path was fixed (`--remote-install ./m1-install`, which is exactly what the default now
produces), gave the cluster its first measurement. The gate labels it itself: `observation_only: true`,
`busy_farm: true`, the loads recorded, and *"this is not a gate result and the threshold was not asserted"*. The
farm was busy — node4 2.53, node1 2.01, node2 3.18, **node3 9.31** — and the run was 4 steps.

**Correctness first, and it holds: every node's tokens and trace digest are identical to the single-node
baseline** (`[11751, 11, 264, 3177]`). **Throughput: 0.93x** — baseline 5.662 s/step against the slowest node's
6.086 s/step. A four-node cluster of 256 experts is *slower* than one node.

**The per-node metrics explain it, and this is the budget `DC-051` has been waiting for.**

| | baseline (1 node) | node (4-node plan) |
| --- | --- | --- |
| payload read per step | 1.570 GB | 0.592 GB |
| — dense, replicated | 0.261 GB | 0.261 GB (identical) |
| — experts, sharded | 1.309 GB | 0.331 GB (**3.95x fewer**) |
| step time | 5.662 s | 5.975 s |
| effective read rate | 277 MB/s | 99 MB/s |
| exchange | 0 s | 0.893 s/step (**15.0%** of the step) |
| exchange volume | 0 B | 4.49 MB/step at **5.0 MB/s** effective |
| exchange terms | 0 | 547/step at **1.63 ms** each |

Two conclusions follow, and both are arithmetic rather than opinion. First, **the exchange is latency-bound, not
bandwidth-bound**: 4.5 MB per step moving at 5 MB/s is one round trip per term at ~1.6 ms, which is a per-message
cost, not a link saturation — shrinking the messages would not help and batching them would. Second, and more
important, **the step is not expert-read-bound at all**: the node reads 2.65x less payload per step than the
baseline and is *slower* than it. The dense payload is read in full by every node because it is replicated, and
whatever costs the remaining ~3.8 s/step is likewise not distributed by an expert plan.

**So the >=3x target is not what sharding experts delivers on this design**, and this measurement says why: the
lever is the non-shardable part — the replicated dense and attention path, and the per-term exchange — which is
`DC-107`'s I/O-bound finding with numbers on it rather than an argument.

**What this does not say.** It is not a gate result: the threshold was not asserted, the farm was busy, and four
steps is a short measurement. A quiet window would improve the ratio — the baseline ran on a node at load 2.53
while the slowest peer sat at 9.31 — but it cannot turn 0.93x into 3x, because the per-node step times are all
within 2% of each other and none of them is expert-bound. What remains genuinely unmeasured is a **per-phase
breakdown of the ~3.8 s/step that is neither payload read across the plan nor exchange**. That is the instrument
`DC-051` and `DC-107` still need, and it is named in the tracker rather than estimated here.

## D86 — The profiler is real and two-layered; the path the speed work measures is the one path it does not reach

Looking for the instrument to answer "where do the ~3.8 s/step go", I found a good one already built.
`Qwen3_5Forward` marks nine phases around a full-sequence forward — `embed`, `load`, `attn.norm`, `attn.core`,
`attn.add`, `ff.norm`, `ff`, `head`, `trace.copy` — and `MixtureOfExperts` marks **ten** more inside the mixture:
`mix.read`, `mix.router`, `mix.experts`, `mix.gateup`, `mix.act`, `mix.down`, `mix.combine`, `mix.acc`,
`mix.gather`, `mix.shared`. It is turned on with `SHARD_PROFILE=1`, it is allocated only when on, and it has been
used before: `D62`'s record cites `mix.read`, `mix.gateup` and `head` by name when it established that the GPU
matmul was slower.

**What it is not wired to is the tool the throughput gate runs.** M3's gate measures `datacenter-generate
--cached`, and the cached decode loop (`ModelCache.decodeOne`) calls the mixture **without a profiler**, while
`Generation` — the struct that loop returns — **carries no profile at all**. So the one path the speed work needs
is the one path with no breakdown, which is why the first M3 observation could report only a step total,
`exchange_seconds` and byte counts. The absence was not a missing instrument; it was a missing wire.

**What changed here.** `ProfileMetrics.fields(_:)` now turns a profile into the metrics fields, in one place, and
the trace CLI uses it instead of spelling the two keys out inline. It takes the **optional** report rather than a
report, so that "the profiler was off" produces **no fields at all** instead of zeroes — the distinction
`ForwardResult.profile` already documents, and the reason it is optional: zero is a measurement and says a phase
took no measurable time, while nil says nobody looked. Four Swift tests pin the marks landing in the phase they
name, the fields carrying every phase and the layer count, an absent profile contributing nothing, and an empty
report still being a report (**measured and found nothing** is not the same state as **not measured**).

**What was deliberately not done, and is said in the code rather than left silent.** `datacenter-generate` does
not yet report `profile_seconds`, because `Generation` cannot carry one yet. The file says so where a reader
would look for the key, and names what comes next: pass a profiler from `decodeOne` into the mixture — the marks
are already there, so this is plumbing rather than new instrumentation — carry the report on `Generation`, and
the gate then reports phases per node. Printing zeroes under a phase name would have been the easier change and
the wrong one.

**Also decided by looking:** the gates require a **release** build (`"run swift build -c release first"`), so the
0.93x observation was taken on optimised binaries and is not a debug-build artefact. That was worth checking
before spending an hour profiling a build nobody measures.

## D87 — The first phase breakdown: two thirds of a forward is reading payload, and an expert plan can only reach a third of it

`SHARD_PROFILE=1` on the release trace binary, on the real install, with the frozen M1 prompt:

```
SHARD_PROFILE=1 .build/out/Products/Release/datacenter-trace \
    .build/m1-install .build/profile-trace 760,6511,314,9338,369
```

It produced the **same digest as ever** (`b0d382dbabf36df0…`, 83 tensors, 40 discrete decisions), so the
instrument does not perturb what it measures, and a breakdown over **40 layers, 17.363 s measured**:

| phase | seconds | share | |
| --- | --- | --- | --- |
| `mix.read` | 6.521 | **37.6%** | the expert slices, read from the install |
| `attn.core` | 4.102 | 23.6% | attention and the Gated DeltaNet |
| `head` | 2.339 | 13.5% | the LM head: 1.02 GB of weights (`D62`) |
| `load` | 2.338 | 13.5% | the layer's dense weights |
| `mix.gateup` / `mix.down` | 1.255 / 0.550 | 7.2% / 3.2% | expert matmuls |
| `mix.shared` | 0.202 | 1.2% | the shared expert |
| `mix.router`, `mix.act`, `mix.gather`, `mix.experts`, `mix.acc`, `mix.combine`, `attn.norm`, `ff.norm`, `ff`, `embed`, `final_norm`, `trace.copy` | 0.264 | 1.5% | everything else, together |

**Two thirds of the forward is reading.** `mix.read` + `load` + `head` is **11.198 s = 64.5%** of the measured
time. That is `DC-107`'s I/O-bound finding at phase resolution rather than as an argument, and it settles the
scaling question the 0.93x observation raised.

**And it explains the arithmetic of the target.** A four-node expert plan divides `mix.read` by four and leaves
everything else where it is. Perfect, instantaneous, cost-free sharding of the expert reads is therefore worth
`6.521 × 3/4 = 4.891 s` — a step of **12.472 s instead of 17.363 s, or 1.39x** — before any exchange cost, and
measured against the *forward* rather than the cached step. **A 3x target cannot be reached by sharding experts
on this design, and that is now arithmetic rather than opinion.** The measured 0.93x on the cluster is what is
left of that 1.39x after the exchange and the farm's load.

**Where the lever therefore is.** Not the expert plan: the *replicated* reads (`load` 13.5% + `head` 13.5%) which
every node pays in full, the attention core at 23.6%, and the read path itself. The effective read rate is the
number to attack — `mix.read` moves roughly 33 MB per layer in 163 ms, about **200 MB/s**, in the same range as
the whole-node figure, which is what a scattered uncached slice read costs rather than what the device can do.
`DC-052` (cache and prefetch) and `DC-107` (the I/O path) are where the speed target lives, and `DC-112`'s bf16
idea moves in the wrong direction for it: more bytes, no reads saved.

**What this breakdown is not.** It is a **full-sequence forward of five tokens**, while M3's gate measures a
**cached decode**, one token per step — so the shares are indicative and not the measured step. The cached path
has no marks of its own (`D86`: `decodeOne` passes no profiler into the mixture, and `Generation` cannot carry a
report), which is the next wiring step before the cluster's own per-phase numbers exist.

## D88 — The cached step's own breakdown: 83% of it is materialising weights, and only a third is really device I/O

`D86` found the wire missing: the cached decode passed no profiler into the mixture and `Generation` could not
carry a report. Both ends are connected now — `decodeOne` takes a profiler and marks the same phase names the
sequence path does, `generateCached` runs a **fresh profiler per step** (so a phase's seconds are that step's
rather than a running total with the gap between steps folded into whatever came first) and adds the reports with
`ProfileReport.combined`, `Generation` carries the result, and `datacenter-generate` writes `profile_seconds`.
`Profiler.requestedProfiler` answers "was profiling asked for?" in one place, and `profiling:` is a parameter so
the tests turn it on without the environment variable, which is read once at load time.

Then the measurement the whole exercise was for, on the real install, exactly the command M3's baseline runs:

```
SHARD_PROFILE=1 .build/out/Products/Release/datacenter-generate \
    .build/m1-install .build/profile-decode 760,6511,314,9338,369 4 --cached
```

**21.795 s over 4 steps = 5.449 s/step**, against a cluster baseline of 5.662 s/step from the M3 run, and the
phases sum to the wall exactly — the profiler's marks sit after their work, so nothing is unattributed:

| phase | seconds | share | per step |
| --- | --- | --- | --- |
| `mix.read` | 7.202 | **33.0%** | 1800 ms |
| `load` | 6.647 | **30.5%** | 1662 ms |
| `head` | 4.187 | **19.2%** | 1047 ms |
| `attn.core` | 2.099 | 9.6% | 525 ms |
| `mix.gateup` | 0.995 | 4.6% | 249 ms |
| `mix.down` | 0.442 | 2.0% | 111 ms |
| `mix.shared` | 0.173 | 0.8% | 43 ms |
| the other ten phases | 0.050 | 0.2% | |

**The three big phases are not the same kind of cost, and the metrics say which is which.** The dense payload was
read **once** for the whole run (`dense_payload_bytes_read` equals `dense_payload_bytes_held`, 1.0437 GB, with
4888 cache hits), so `load`'s 6.6 s is **not I/O**: it is `InstallFile.tensor` dequantising the same constants on
every step, for every layer, and releasing them — the docstring on `loadLayer` says so in as many words ("loaded
and then released"). The dequantiser is already SIMD-vectorised with a scalar definition beside it, so this is not
a loop that wants micro-optimising; it is **work that does not need doing at all**, recomputed 160 times per
generation. `head` is cached weights and a matmul. **`mix.read` is the only phase that is genuinely device-bound**
— `install_bytes_read_total` is 2.37 GB for the node over the run, 0.59 GB/step, of which the expert slices are
about 0.33 GB/step in 1.80 s, **184 MB/s** — and it is the only one of the three that a plan divides.

**So the arithmetic of the target holds, and gets slightly worse.** A four-node plan divides `mix.read` alone:
perfect, instantaneous, cost-free sharding is `7.202 × 3/4 = 5.402 s`, 21.795 → 19.995 s, **1.09×** on the
measured step. The forward's 1.39× (`D87`) and this 1.33%-of-nothing difference are the same conclusion from two
paths: **the expert plan is not the lever**, and 62% of the step is work that is identical on every node.

**What the attack is, then, and it is `DC-052` by name.** The largest single reducible cost is `load`: ~1 G
parameters of constants dequantised per step, per node, on the CPU, while the GPU sits idle and the payload it
comes from is already resident. The fix is a **decoded-weight budget** — decode each layer once and hold it while
memory allows, with the budget measured and recorded per node rather than guessed, which is exactly what
`DC-052`'s done-when asks for. That, not more sharding, is where 30% of the step is.

## D89 — The decoded-layer cache works, and on this node it loses: measured, so the default is off

`D88` left one clear target: `load` is 30.5% of a cached step, all of it the same constants dequantised again on
every token. The fix is a cache, and the first design question is what shape. **Not an LRU.** A decode sweeps
every layer in order, once per token, so a least-recently-used policy has a **zero** hit rate by construction —
the layer evicted is always the one about to be asked for. Every step revisits every layer, so holding *any*
layer pays on every step that follows, and the layers worth holding are simply the ones that fit.

`LayerWeightCache` does that: a budget in bytes, layers held as decoded fp32, bytes **counted from the arrays it
holds** rather than derived from a formula (the same arithmetic spelled out per expert is how a 14.5 GB change
once shipped past 98 green tests), a layer too large for the remaining budget returned but not kept, and metrics
— budget, bytes held, layers held, hits, misses — written into `metrics.json` per node, which is `DC-052`'s
done-when. `generateCached` creates one per generation and threads it through `decodeOne`; the sequence path and
every existing caller get `nil`, so nothing else changed. Seven new tests, and the one that matters asserts a
cached decode is **bit-identical** to an uncached one, tensor by tensor.

**Then the measurement, and it says no — on this node.** Budgets of 0, 256 MB, 1 GB and 2 GB, alternated on one
binary because conditions run in sequence drift together (`D62`):

| budget | layers held | step, run 1 | step, run 2 | `load` |
| --- | --- | --- | --- | --- |
| 0 MB | 0 | 5.327 s | 5.348 s | 1.670 / 1.642 |
| 256 MB | 2 | **5.322 s** | **5.328 s** | 1.606 / 1.593 |
| 1 GB | 8 | 5.459 s | 5.595 s | 1.578 / 1.625 |

The cache does exactly what it was built to do — 48 hits, which is 16 held layers times the three steps that
follow the first, and `load` falling as the budget rises — and **the step gets slower the more it holds**. The
saving is real and computable: about **0.035 s per layer per step**, which across all 40 layers would be **1.4 s
of a 5.42 s step, 26%**. What it costs is memory, and this node does not have it: at 2 GB the *other* phases rose
— `head` +0.20 s/step, `attn.core` +0.27, `mix.read` +0.17 — with swap already at 2.1-2.4 GB. **The idea is
sound where memory is free; this is an 8 GB node with 4.5 GB usable.**

**So the default is zero**, and it is a measurement rather than caution. `SHARD_LAYER_CACHE_MB` turns it on for a
machine with headroom, and the metrics record what each node actually held, so the choice is visible either way.
That is the same shape as `D31`'s expert bank — a knob whose default is what this hardware measured, not what
would be nice — and the same lesson as `D62`'s GPU matmul: **a mechanism that saves the work it was aimed at can
still lose to the resource it spends.**

**What it means for the target.** If 1.5× is to come, it is not from here. The step's shardable and reducible
parts are `mix.read` at 33.0% (the plan divides it), `head` at 19.2% (vocabulary-parallel, and identical on every
node), `mix.gateup`/`mix.down` at 6.6%, and the exchange at 0.89 s of a cluster step. That is where the next
rounds go.

## D90 — The cluster's cost is the exchange, and the exchange blocks on peers one at a time

`D88` and `D89` were both about one node. This round asked the cluster the same question, and to do that the gate
had to be able to ask it: `SHARD_PROFILE` and `SHARD_LAYER_CACHE_MB` are read by the engine from **its own**
environment, and the gate launched peers over `ssh` without them — so a profiled run would have profiled the local
node and reported the other three as unprofiled, and a cache budget would have reached one node of four. One
helper (`forwarded_environment`/`environment_prefix`, beside `remote_install_path` and for the same reason: one
rule, three launch sites) now passes them through, with four tests.

**Then the per-node profile of a real four-node run.** Bit-identity held on every node, and the phases say two
things at once:

| node | step | exchange | share | reads |
| --- | --- | --- | --- | --- |
| 0 | 5.946 s | **4.250 s** | **71.5%** | 2.402 GB |
| 1 | 5.954 s | **4.136 s** | **69.5%** | 2.369 GB |
| 2 | 5.989 s | 0.672 s | 11.2% | 2.313 GB |
| 3 | 5.925 s | **3.739 s** | **63.1%** | 2.328 GB |

**The plan works.** Against the baseline's 6.281 GB read per run, each node reads 2.31-2.40 GB, and the phases
follow: `mix.read` falls from 1.80 s/step to about 0.49, `mix.gateup` from 0.25 to 0.06. The expert sharding is
doing exactly what it was built to do.

**And the exchange eats all of it and more**, on a farm whose nodes sat at load 2.4-4.1 with one at 8 earlier in
the day. The tell is the **spread**: four nodes doing identical work, one of them waiting 0.67 s and three waiting
3.7-4.3 s.

**The cause is in the code, not in the network.** `ShardExchange.allReduce` sends its frame to every peer and then
receives **sequentially, in peer order, blocking**:

```swift
for peer in peers { try peer.send(frame) }
for peer in peers { let answer = try peer.receive() }
```

A peer that is slow to answer does not delay itself — it delays **everyone else in the list behind it**, and the
node that happens to be read last pays the whole sum of its peers' latencies. That is head-of-line blocking, and
the 6× spread between nodes doing the same arithmetic is its signature. One all-reduce per layer, forty per step,
so the latency is paid forty times per generated token: 4.25 s ÷ 40 ≈ **105 ms per layer**, which is not a LAN
round trip but a LAN round trip **behind a busy node**.

**What the fix is, and why it is worth more than the head shard.** Two contained changes, neither of which touches
the arithmetic — the received frames are merged by key with a deterministic order (`OrderedReduction`), so *when*
a frame arrives cannot change a result:

* **Do not block on one peer at a time.** Receive from peers concurrently, so the cost is the slowest peer rather
  than the sum of the ones queued behind it.
* **Do not talk to a peer that owns none of the chosen experts.** With 256 experts over four nodes and eight
  chosen per token, roughly six are remote — so one or two peers matter per layer, not all three.

With the exchange at its floor (node 2's 0.67 s shows the shape of that floor on a busy farm), the cluster step
becomes the work plus a small wait: ~3.4 s of dense work plus ~0.7 s of local mixture work plus a few hundred
milliseconds, against a 5.55 s baseline. **That is the 1.5×**, and the head shard (`head` is 1.05 s/step,
identical on every node) is the margin on top of it.

## D91 — What the sister project does for ANE prefill and GPU decode, and what it changes here

The direction is: **ANE for prefill, GPU for decode.** The sister checkout is the reference, and reading it
changes the plan in three concrete ways.

**What they actually run.** Prefill is chunked, and an **opt-in** path (`TINYTITAN_PREFILL_ANE=on`) runs the
**full-attention prefill blocks on the Neural Engine** from an exported Core ML sidecar
(`tools/export_ane_prefill.py` → `ane_prefill/layer_<L>.mlpackage`). Decode **stays on the GPU**: quantized
**GEMV** kernels (`Metal/Quant/bf16_gemv.metal`, `dequant_int4.metal`) with a tiled GPU Top-K for sampling
(`Metal/Sampling/logit.metal`), while their prefill matmuls use **Metal Performance Primitives tensor ops**
(`Metal/TensorCore/tensorops.metal`: `matmul2d_descriptor`, affine tiles, fp16 threadgroup tensors). Their
measurement method is ours: **one variable per arm, a fresh process, one discarded warmup, off/on/on/off
interleaved** — and `prefill_s` is the qualification metric.

**Three things they learned by measuring, which we would otherwise have learned the hard way.**

1. **The ANE path is not bit-identical, and they say so in the benchmark's docstring.** The sidecar is fp16 with
   a different reduction order — *"~1% per-layer deviation"* — so their A/B **expects the digests to differ
   between arms**. That is irreconcilable with our `I3` as it stands: a gate that asserts a byte-identical trace
   cannot have an ANE prefill in it. So an ANE path here is **opt-in, declared, and never inside a bit-identity
   claim** — the same shape as the Gated DeltaNet's declared tolerance in `D8`.
2. **The fused SDPA op produces NaN/inf on their M3's ANE from sequence length 2048**, so they build attention
   decomposed instead. A hardware/compiler defect that only a real run shows.
3. **Core ML exits 0 while dispatching to the CPU.** Their words: an export could *"succeed" into a sidecar that
   the runtime then runs on the CPU at ~38x the GPU prefill cost"*. Their fix is a flag —
   `aneCompileVerified` — that only a verified export writes, and a **runtime that refuses a sidecar without
   it**, written through a staging directory and an atomic replace. This is our own rule in someone else's code:
   *a check that cannot run must never look like a pass*, and it is worth copying as a discipline for **any**
   device path we add — a GPU kernel selection that silently falls back to the CPU would be the same defect with
   a different name.

**What it changes in our plan, and it is a real correction.** Decode is **M=1**, so the right GPU kernel for it is
a **quantized GEMV that dequantises in-kernel**, not a tiled matmul. That is not a detail: it means `D88`'s
`load` phase — 30.5% of a cached step, the same constants dequantised every token — **stops existing** rather
than being cached, which is what `D89` tried and lost to memory. The weights stay int4 on the device and are
consumed in the kernel. `head`, `attn.core` and the mixture matmuls get the same treatment.

**And it stays bit-exact**, which the ANE path cannot: `D63` already established the rule — one thread per output
element, accumulating `k` in the same order, changes *which thread* does the work and not the order of the sum —
and the existing opt-in GPU matmul was already proven byte-identical on a real trace before it was proven slow.
So the order of work is: **the decode GEMV first** (it removes the largest reducible phase and is bit-exact),
then the exchange (`D90`, the ratio's real lever), then the ANE prefill sidecar with its verification.

## D92 — The exchange is 99.6% *waiting*, so the cost is imbalance and not the protocol

`D90` measured the exchange at 4.25 s/step on three of four nodes and named the suspect in the code: `allReduce`
sends to every peer and then receives **sequentially, in peer order, blocking**. That is a real defect, and acting
on it would have been guessing, because a blocking receive and a peer that has nothing to send look identical
from outside. So the timer was split: `ExchangeMetrics` now carries **encode**, **send**, **receive** and
**merge** seconds beside the total, `allReduce` marks each segment, and `datacenter-generate` writes all four
into `metrics.json`. The segments **cover 100.0% of the total**, so the instrument is exact rather than
indicative — and it says something different from the hypothesis.

A four-node run, loads 1.8-4.5:

| node | step | exchange | encode | send | **receive** | merge | reduces |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 4.790 s | 2.323 | 0.002 | 0.007 | **2.311** | 0.002 | 90/step |
| 1 | 4.785 s | 2.023 | 0.002 | 0.007 | **2.011** | 0.003 | 90/step |
| 2 | 4.790 s | 1.128 | 0.002 | 0.009 | **1.114** | 0.002 | 90/step |
| 3 | 4.789 s | 1.578 | 0.002 | 0.007 | **1.567** | 0.002 | 90/step |

**Receive is 99.6% of the exchange.** Encoding costs 2 ms per *step*, sending 7 ms — the protocol and the wire
format are not the problem, and neither is the head-of-line blocking I suspected: making the receives concurrent
would shorten a *sum of latencies*, and there is no latency here to sum. Ninety reduces per step at 13-26 ms
each is one node waiting for its peers to **have** something to send. **That is compute imbalance**, and its
natural home is the plan: a **contiguous** split gives each node a *block* of expert ids, and if the router's
traffic is skewed across ids — which it is, that is what a learned router does — one block is hotter than the
others and everyone waits on whoever owns it.

**The same run also put the previous number in context.** With the farm quieter, the identical measurement is
**1.13×** — cluster 4.790 s/step against a 5.389 s/step baseline, bit-identity intact — where the busy farm gave
**0.93×**. So the engine's own cluster step was never below one node; the earlier figure was mostly the farm, and
`D90`'s four-second exchange was mostly a peer at load 8. **The measurement to trust is the one with the load
beside it**, which is exactly why the gate records it.

**The experiment ran, and it falsified the hypothesis.** Four runs, contiguous and round-robin alternated:
**1.06x / 0.96x / 0.86x / 0.88x** — the two distributions are indistinguishable — with a run-to-run spread of
**24%** (slowest node 5.135 → 6.359 s/step). The loads beside them are the explanation, and one is decisive:

| run | baseline | slowest node | ratio | loads at the start |
| --- | --- | --- | --- | --- |
| contiguous 1 | 5.465 | 5.135 | 1.06x | 3.7 3.0 2.6 2.7 |
| contiguous 2 | 5.495 | 6.359 | 0.86x | 4.2 3.4 3.6 3.4 |
| round-robin 1 | 5.452 | 5.657 | 0.96x | 4.0 2.8 3.1 3.1 |
| round-robin 2 | 5.453 | 6.193 | 0.88x | 5.0 **18.7** 4.3 2.8 |

**The run with the slowest cluster step is the run with a peer at load 18.7.** On this farm the cluster step
measures the farm, and no distribution can fix a peer that is doing something else. So: the per-id skew
hypothesis is **falsified** by four interleaved runs, the receive wait is a busy peer rather than a slow
protocol or an unbalanced plan, and a 1.5x figure cannot be *certified* here without a quiet window — which is
precisely what the gate's `--quiet-load 1.0` rule exists for (`D38`), and why every figure above carries its
loads.

**What that leaves in our hands.** Two things, and neither depends on the farm: make the **replicated** work
smaller, because the ratio is `(replicated + experts) / (replicated + experts/n + exchange)` and the head's
1.05 s/step is identical on every node; and **overlap the wait with work that is not on the critical path** —
a node waiting for a peer can prefetch the next layer's weights, which turns the exchange's seconds into time
already spent. Both are engine work.

**And the original experiment, for the record.** The engine already supports `roundRobin`, which interleaves
expert ownership and so spreads a hot id's traffic across nodes; the gate now exposes it as `--distribution`
with the two values validated and refused rather than defaulted, and four tests pin the owner lists
(`[0,0,1,1,2,2,3,3]` against `[0,1,2,3,0,1,2,3]`). Four runs follow — contiguous, round-robin, contiguous,
round-robin, alternating as `D62` requires — and the question they answer is whether the receive time falls. If
it does not, the imbalance is elsewhere (per-token routing variance rather than per-id skew) and the next lever
is overlapping the wait with compute rather than removing it.

**Why this matters to the target.** The cluster needs 4.79 → 3.59 s/step for 1.5×. The two levers are the
exchange, 1.1-2.3 s/step of which is *waiting*, and the head shard at −0.79 s/step. Neither is a protocol
rewrite; both are about who does which work.

## D93 — The head is vocabulary-parallel, and it took the cluster from 1.13x to 1.36x

The ratio is `(replicated + experts) / (replicated + experts/n + exchange)`, so the lever is the *replicated*
work — and the head was the largest piece of it: **1.05 s/step on every node**, 19.2% of the step, identical
everywhere (`D88`). It is also the most shardable thing in the model, because the head's rows are independent:
row `v`'s value is a dot product over the hidden width and depends on no other row.

**The design, and why it stays bit-exact.** Each node computes only its vocabulary slice, with the split
**derived from the node count** rather than declared in the plan (so no field can disagree with it), and applies
it *inside* the existing `headBlockRows` loop — so a node's rows are computed by exactly the same
multiplications in exactly the same order as before. Then the slices are gathered, so **every node ends with the
same full logits array**: the argmax, the margin and the captured trace are unchanged, which matters because the
gate compares tokens *and* the trace digest, and a design that only agreed on tokens would have quietly weakened
it. `VocabSlice` is a partition by construction and the test asserts it as one — every row owned exactly once,
for vocabularies from 0 to 248,320 over one to five nodes — because a row nobody computed stays zero and shows
up as neither an error nor a difference.

**The gather is chunked, and that is not an optimisation.** A full slice is about a megabyte, and if every node
sent a megabyte before reading anything the sends would fill the socket buffers and every node would block in
`send` waiting for a peer blocked in `send`. Small frames never reach that and a fixture-sized test never would
either. So the vocabulary is cut into fixed windows — the *same* windows on every node, so the message count
matches everywhere — and within a window each node sends one 16 KB frame and reads one from each peer. A node
whose slice misses a window sends an **empty frame rather than nothing**, which is what keeps the reads
aligned. `HeadSliceWire` is its own format (magic `TTDH`) rather than a reused contribution frame, because a
contribution is a keyed term that gets *summed* and a slice is a run of rows that gets *placed*; overloading the
one frame would have meant inventing a key and excluding it from the sum.

**The measurement.** Four nodes, `--allow-busy-farm`, bit-identity intact on all four:

| | baseline (1 node) | cluster, before | cluster, after |
| --- | --- | --- | --- |
| step | 5.487 s | 4.790 s | **4.021 s** |
| `head` | 1.083 s | 1.045 s | **0.26-0.32 s** |
| ratio | — | 1.13x | **1.36x** |

The head fell by a factor of four as intended, the step fell by 0.77 s, and **all four nodes are now identical
to the millisecond** (4.021 s each). The gap to 1.5x is **0.36 s/step** — the cluster needs 3.658.

**One number still does not reconcile, and it is recorded rather than used.** On every node the ledger's
`exchange_seconds` (1.4-1.9 s/step) is *larger* than the `ff` phase that contains it (0.575 s), even though the
marks are placed correctly — `ff` is marked after `mixtureOutput` returns — and the phases sum to the step. One
of the two is not saying what it appears to say, and `D90` and `D92` both reasoned from that counter. It is not
worth building on until it is resolved; the head shard was justified by the *phases*, which do add up.

## D94 — `load` was one core of eight: the dequantiser's rows are now spread across the machine, and the 1.5x target is met

`D88` measured `load` at **30.5% of a cached step**, `D93` at **47.9% of a cluster step**, and `D89` showed that
caching the decoded values loses to memory on an 8 GB node. What none of those said is the obvious thing: the
engine is **single-threaded** — a 64-98% CPU sample on an eight-core machine — and `load` is ~1 G parameters of
int4 constants dequantised with SIMD4 on one core, the same constants on every token and on every node. It is
replicated work, which is exactly what the cluster's ratio is made of, so making it faster raises the ratio *and*
the absolute speed.

**The change is one loop.** Rows of an int4 tensor are independent: a row's values are a function of that row's
codes, scales and zeros and of nothing else. So the row loop in `InstallFile.dequantizeInt4` is spread across the
machine's cores. That changes **which thread** computes a value and not how the value is computed — the same rule
the GPU matmul is held to (`D63`) — and `SHARD_DECODE_THREADS=1` restores the single-threaded path so the two can
be compared **on one binary** instead of across builds, which is `D62`'s lesson.

**Bit-exactness is already guarded, and by the strongest available check.** `Int4UnpackTests` compares the vector
path against `dequantizeInt4Scalar` — the definition the GPU is also held to — bit-for-bit over a grid of **more
than fifty shapes**, and the trace, generation and sharded-generation tests all still pass.

**Measured on one binary, alternated (single node):**

| threads | step | `load` |
| --- | --- | --- |
| 1 | 5.124 / 5.159 s | 1.307 / 1.350 s |
| 8 | **4.324 / 4.391 s** | **0.539 / 0.586 s** |

`load` falls **2.4x** and the step by 0.78 s, with `head` and `mix.read` untouched — the gain is where it was
aimed. (2.4x rather than 6x on eight cores says part of the work is serial or memory-bound; that is a fact about
the phase, not a disappointment about the change.)

**And in the cluster, four runs, alternated, every one bit-identical on all four nodes, loads 1.6-5.5:**

| threads | run | baseline | slowest node | ratio |
| --- | --- | --- | --- | --- |
| 8 | 1 | 4.281 s | 2.455 s | **1.74x** |
| 1 | 1 | 5.093 s | 3.074 s | 1.66x |
| 8 | 2 | 4.268 s | 2.513 s | **1.70x** |
| 1 | 2 | 5.095 s | 3.067 s | 1.66x |

**The objective is met: the 35 B model runs at 1.70-1.74x over a single node**, and at **1.66x even with the
parallel dequantiser disabled** — so the shard work alone clears 1.5x on this hardware. The farm was **busy**
throughout, and the day's sequence (0.93x at load 2.4-9.3, 1.13x quieter, 1.36x after the head shard, 1.74x now)
shows that load *hurts* the ratio, since four nodes are exposed to a spike and the baseline is one. These figures
are therefore a **lower bound**.

**What this does not say.** The gate's own roadmap target is **≥3x on a quiet farm** (`DC-053`), and nothing here
asserts it: every figure is an `observation_only` run with its loads beside it, and the gate — which refuses a
busy farm by design (`D38`) — remains the certification instrument. What is demonstrated is the operator's
objective: **≥1.5x, measured, bit-identical, on the real model.**

**And it retires a plan.** `DC-113`'s GPU GEMV was written down as the way to remove `load`. The CPU path took
the same phase down 2.4x with no device work, no protocol, no bit-exactness argument to make and none of the
silent-CPU-fallback risk `D91` found in the sister project. The GEMV remains a candidate for absolute speed; it is
no longer on the path to a number.

## D95 — The first release: an identity a build can enforce, and an archive that can be checked

`RELEASE.md` was written before there was anything to release, and it says so in its own Part 2: no
`VERSION`, no tag, no release script, no packaging step — while the compiled artifacts *do* exist and the
arm64 rules therefore already applied. Cutting `v1.0.0` is what turned that section from a description of an
absence into a process, and three of its demands shaped the work more than the rest.

**Identity is single-sourced and enforced (§1.3), and "enforced" needed measuring.** `VERSION` holds
`1.0.0`; `sources/DatacenterEngine/Version.swift` is **generated** from it by `tools/version.py --write`, so a
bump is one edit plus one command and the mirror is never hand-maintained. Every tool answers `--version`,
because §1.3 wants identity observable from the artifact and a number that lives only in an archive's
filename is not observable from the binary inside it. Two checks compare the mirror with the authority, and
the difference between them is the finding: `tools/version.py --check` fails on **every** run, while
`Package.swift` reads both files when the manifest is evaluated — and **SwiftPM caches the compiled
manifest**, so a mangled mirror slipped past an incremental `swift build` and only stopped it once the
manifest itself changed. That is written into the manifest's own comment, because a guard believed to be
stronger than it is, is worse than one known to be narrow. The release script runs the gate before it
packages anything, so a mismatch cannot reach a release at all.

**Gates run before packaging, and the scratch build is clean (§1.5).** `tools/release.py` refuses a dirty
tree, a machine without ~8 GB free, a competing build or model process (it names one and stops — never
terminates it), a `gh` account that is not the repository's owner, the repository's whole gate set, and a
clean scratch build whose log is scanned for **warnings** — a fresh scratch path, because a warning scan over
an incremental build compiles nothing and passes vacuously. It also asks the *plan* what it declares
(`swift package describe`) rather than checking for artifacts after the fact.

**The architecture is asserted on what ships (§1.2.2), and the notes cannot lie about the digest (§1.8).**
`lipo -archs` is checked on each binary **extracted from the archive** — checking the build directory would
assert the wrong thing — together with a smoke test that each shipped tool answers `--version` correctly. The
notes are the changelog's section for the version, and `--publish` **refuses** unless they carry the
checksum placeholder or quote the digest the script has just computed; a dry run's size is never copied
forward, because publishing rebuilds. Eight tests cover that refusal, since a check that has never been seen
to fail is not yet trusted.

**What the release measures, and what it does not.** Single Mac mini M2: **0.23 tok/s prefill, 0.23 tok/s
decode**. Four nodes: **0.41 tok/s decode — 1.7x** over one node, bit-identical on every node. The M3
throughput gate is still **not asserted**: it refuses a busy farm by design (`D38`) and the farm was shared,
so every figure is an observation with its loads and the README says so. The GPU matmul stays opt-in and this
release claims nothing about the tiled kernel written after `D63` measured its predecessor slower.

## D96 — Dense models are outside this design, and the 4 B measured why

The question was an overview across model sizes: Qwen3.5 4B, 9B and Qwen3.6 35B-A3B. The 4 B is a real
checkpoint and was built into a real install — 426 tensors, 4,204,789,760 weights, **3.34 GB** at 6.36
bits/weight — on a helper MacBook M3, because this node had neither the checkpoint nor the disk for it, and
the install was copied here to measure. **It is slower than the model seven times its size, and the reason is
structural.**

**A dense model cannot be distributed at all.** The shard plan is an *expert* plan: `ShardPlan`/`ExpertOwnership`
assign expert ids to nodes and the mixture all-reduces their contributions. A dense model has no experts, so
there is nothing for the plan to divide — the 4 B and the 9 B are single-node models in this engine by
construction, not by omission. That also means the cluster work of the last three rounds (`D92`–`D94`) does
nothing for them.

**And a dense model streams its whole self every token.** The engine holds a bounded payload cache —
`SHARD_DENSE_CACHE_MB`, **1 GiB by default** — and for the 35 B that is enough, because the dense backbone is
1.04 GB and the 18 GB of experts are read a row range at a time and never cached. For a dense 4 B the payload
*is* the model: 3.34 GB against a 1 GiB cache, so the run read **18,638,208,000 bytes in four steps with zero
hits**, and every step paid for most of the model again.

The measurement follows from those two facts rather than surprising us: **5.870 s/step = 0.170 tok/s** on one
Mac mini M2 at load 5.05 (`load` 2.707 s, `head` 1.908 s, `ff` 0.747 s, `attn.core` 0.506 s), against the
35 B MoE's **0.230 tok/s** on the same node. The 35 B wins because it is **sparse** — ~3 B active of 35 B — and
that is not a coincidence of this release: it is the property that makes a 35 B model runnable on an 8 GB
machine at all. The 9 B would be worse again: ~5.5 GB of payload against ~4.5 GB of usable RAM, still
undividable.

**So the design's model class is MoE, and the README now says so** — with the 4 B row kept in the table,
its prefill left unmeasured rather than invented, and the three MoE rows beside it. Two tooling gaps were
found on the way and are worth recording: `tools/quantize.py` needs `safetensors` and `torch` as *direct*
dependencies (it reaches them through `SafetensorsSource`), so a helper machine without the reference stack
fails on the first shard until both are installed — which is how this install was finally built.

## D97 — Single-node throughput first: 7 tok/s on this machine, and the five gaps between here and there

The operator set the order on 2026-09-17: **the engine reaches 7 tok/s decode on one Mac mini M2 before any
further network or cluster measurement.** The cluster work is not wrong, it is early — `D92`-`D94` bought 1.7x
on a design whose single-node step is 4.32 s, when the reference implementation runs the same class of model
seven times faster on the same class of machine. A network measurement taken against a 0.23 tok/s engine
optimises the wrong tail.

**The reference, measured by the sister project on a comparable 8 GB M-series node** (35 B-A3B at 4-bit, 46-token
prompt, 512 tokens generated, identical output digest `bfe8fa42b239e55b` across every configuration, so the
differences are purely speed):

| expert cache | decode tok/s | expert hit | I/O hidden | GPU busy | occupancy | host wait/token |
| --- | --- | --- | --- | --- | --- | --- |
| 1 GB | 5.164 | 60.5% | 8.1% | 34.5% | 32.9% | 55 ms (at 3 GB) |
| 2 GB | 6.019 | 72.3% | 11.1% | 40.1% | 37.9% | — |
| 3 GB | **7.075** | 79.8% | 21.4% | 45.7% | 42.8% | 55 ms |
| 4 GB | 2.756 | 84.6% | 46.6% | 17.9% | 17.4% | 219 ms |

The 4 GB row is the most useful one: the **highest** hit rate produced the **slowest** run, because a 4 GB wired
cache plus the dense weights, KV and prompt cache stop fitting in 8 GiB and the machine swaps against a cache
that its own pressure paged out. Its spread (3.167 → 2.756 → 2.429 run over run) is accumulated pressure, not
noise, and TTFT is flat across all four (4.91-5.36 s) — so the workload is **bandwidth-bound on expert I/O, not
prefill**, and the safe cap is about **30% of physical RAM**.

**Our engine, read against that table, has five gaps and they are structural.** In the order they cost time:

1. **The expert cache cannot outlive a layer.** `case mixture(MixtureWeights, provider: ExpertSlotCache)` is
   constructed in the layer-load path (`Qwen3_5Forward.swift`) and dies with the layer's weights. For a cached
   decode that makes cross-token reuse impossible and the hit rate **structurally 0** — `D31` measured 0 and was
   read as "experts are never reused"; the truth is worse and simpler: they could not be reused by construction.
   `SHARD_EXPERT_BANK_MB` (512 MB default) sizes a bank that is dropped before the next token asks for anything.
2. **The GPU is off.** `MetalMatmul` is opt-in because `D63` measured it slower — with the *older* kernel, whose
   access pattern it blames; the threadgroup-tiled kernel written since has never been re-measured (`DC-113`
   carries the question). The reference runs 33-46% GPU busy; ours is one CPU core for everything the unpack
   does not touch.
3. **Nothing overlaps I/O.** There is no asynchronous file I/O in the engine at all — no `DispatchIO`, no reader
   thread — so 0% of expert reads are hidden, against 8-21% there. On a bandwidth-bound workload that is the
   single largest structural difference in the list.
4. **The per-token matmul is one core with a strided access pattern.** `Ops.orderedMatmulVectorized` walks four
   output rows `k*4` bytes apart per `k` step — 8 KB at the head's width — so a cache line delivers four useful
   floats. Outputs are independent, so parallelising across them is bit-exact (`D63`'s rule), and the loop order
   for a one-row GEMV wants one contiguous weight row per output.
5. **The head materialises 2.03 GB of fp32 from bf16 every token** — 1.017 GB read, 508 M values converted
   single-threaded, 1.05 s of the 4.32 s step — when a bf16-weight GEMV could convert in-register and never
   write the fp32 array at all.

**The order of work** follows the table: make the expert bank live for the generation and instrument it (hit
rate, bytes read, read seconds — the columns the reference has and we do not), because that is the one gap the
reference quantifies and the only one that is free of arithmetic risk (caching decoded weights cannot change a
value, so the trace digest is the check). Then the head and the CPU matmul shape, then the tiled GPU kernel
re-measured and batched one command buffer per layer, then asynchronous expert prefetch on top of a bank worth
prefetching into. Every step is measured on one binary with the conditions alternated (`D62`) and the load
recorded, and **7 tok/s is the gate**: no cluster measurement resumes before it.

## D98 — The expert bank works, and it proved that capacity is not the lever on this node

`D97` read this engine's **0% expert hit rate** as a lifetime defect: `ExpertSlotCache` was constructed inside the
layer-load path and dropped with the layer's weights, so a generation asking for the same expert on the next
token asked a store that no longer existed. That was correct, and `DC-119` fixed it — `ExpertBank` holds slices
for the **whole generation**, byte-budgeted and least-recently-used, with a per-projection cap and peak (the
meaning the old metric had, and M1's gate reads it), and it reports hits, misses, elements read and resident
peak into `metrics.json` for the first time. Seven new tests pin the semantics, including the two that matter
for safety: a slice never crosses a layer and never crosses a projection. 233 Swift tests pass.

**Then the measurement refused the conclusion.** Five alternated 24-step decodes on the real 35 B-A3B, one
binary, the bank's budget as the only variable:

| bank | s/step | hits | elements read |
| --- | --- | --- | --- |
| 0 MB | 4.132 | 0.0% | 29,192.4M |
| 512 MB | 4.139 | 0.0% | 29,192.4M |
| 1024 MB | 4.335 | 0.0% | 29,192.4M |
| 0 MB (again) | 4.020 | 0.0% | 29,192.4M |
| 512 MB (again) | 4.236 | 0.0% | 29,192.4M |

**Zero hits at every size, and byte-for-byte the same bytes read.** The bank was demonstrably working — its
resident peak was 43 slices per projection at 512 MB — and the arithmetic says why it could not hit: a decode
token asks for **773 slices**, at **12.5 MB per slice**, so one token's working set is **387 slices per
projection = 4.83 GB**, about **9.66 GB for both projections**, against a 537 MB bank. The **reuse distance is
nine times the capacity**, so every reuse has long been evicted. A bank that could hit this model across tokens
would need 10-20 GB, and this node has 8.

So the default budget is **0**, exactly as `D89` set the layer cache to 0 when it measured that on this node it
loses — and the bank on was slightly *slower* in both alternated pairs, which is what 43 slices of residency buy
when nothing hits. The knob stays, because the prefetch ring will want storage and a node with headroom may
want to test it; the default does not pretend. `D31`'s "hit rate 0 at every size" was therefore right about the
**workload** as well as the lifetime, and the lifetime fix is what made the two distinguishable at all — the old
metric could not tell them apart.

**What this redirects the work to.** The bytes have to be read either way, so the levers are not capacity:

1. **Hide the reads** (`DC-121`): the reference's `ExpertPrefetchRing` stages the *next* token's likely experts
   into raw slots while the current token computes, and adopts them only if the router selects them. That is
   where its 8-21% hidden I/O comes from, and — as an inference, not a measurement — it is most likely what its
   60-85% "expert hit" column counts too, because a pure cache on this class of model at 1-3 GB gives the same
   ~0% that we measured.
2. **Cut the latency of the misses** (`DC-118`): 773 reads per step issued one at a time, where the reference
   fans them across threads.
3. **Cut the per-token compute** (`DC-120`): the dequantise-then-matmul path, on the CPU, with the GPU off.

The order of the remaining work is unchanged; what changed is that each of those three is now supported by a
measurement rather than by analogy with the sister project.

## D99 — Threading the element-wise passes, and reverting a matmul that moved bits

`D97` listed five gaps; the cheapest was not the expert bank (`D98` proved that) but the two passes still paying a
single core for work that is independent per element. `D94` had threaded the int4 unpack and measured 2.4x;
this extends the same idea to `decodeRaw` and records what happened when it was tried on the matmul.

**What is threaded.** `decodeRaw` converts bf16/fp16/fp32 payloads to fp32, and it is a **map**: every element
is a pure function of its own bytes. Splitting it into disjoint ranges cannot change a value, so it is the
cheapest parallelism in the engine — and it is the LM head's 508-million-element conversion on every token,
plus the dense layer's norms and routers. One knob now covers it and `D94`'s unpack (`DecodeThreads`;
`SHARD_DECODE_THREADS=1` restores the single-threaded behaviour, which is how the A/B attributes its win on one
binary rather than across builds, `D62`).

**Measured**, alternated, one binary, four runs on the real 35 B-A3B, 8-step decode:

| threads | s/step | `load` | trace digest |
| --- | --- | --- | --- |
| 1 | 4.119 / 4.162 | 1.097 / 1.096 | `89d654ff54b0fd03` |
| 8 | **3.448 / 3.465** | **0.360 / 0.356** | `89d654ff54b0fd03` |

**0.242 → 0.290 tok/s**, and the digest is identical in every arm — which is the property that matters, not the
seconds. The load average is recorded beside it (2.98), as this repository requires of any timing.

**What was reverted, and why that is the point.** The same treatment looks obviously applicable to
`Ops.orderedMatmulVectorized`: every output is its own dot product over `k`, so splitting `(row, column block)`
across threads changes *which thread* accumulates and not the order — the rule `D63` states for the GPU. The
version was written, and **it moved bits**. The repository's own comparison caught it — a reused-buffer test
that had been passing since `D9` — so the function is the original single-threaded body **verbatim** and the
attempt is recorded in `DC-122` rather than shipped. The likely cause is a hypothesis, not a finding: bounding
each block by its own `last` moves the four-wide grouping and the scalar tail relative to a single
`out`-bounded loop, for shapes where `out % 4 != 0` or where the tail lands inside a block. The next attempt
starts from the contract grid and a bit-for-bit test, not from the timings.

**Where the step stands.** `mix.read` is now the top phase at **1.71-1.77 s of 3.45 s (50%)**, followed by
`head` 0.51, `attn.core` ~0.5 and `mix.gateup` ~0.22. The distance to 7 tok/s is ~24x, and the remaining work is
the list `D97` and `D98` converged on: fuse the dequantise into the matmul so the fp32 slice is never
materialised (`DC-120`), hide the reads behind compute (`DC-121`), and thread the matmul once it can be shown
to preserve the contract order (`DC-122`).

## D100 — Code may be taken from the sister project; `mix.read` is read *and* unpack; and the GPU matmul needs memory this node has not got

Three things happened in one round, and the first is a policy change that outlives the round.

**Taking code is now allowed, and therefore attribution is now required.** The operator authorised taking code
from TinyTitan on 2026-09-18, having authorised reading it the day before. TinyTitan is **Apache-2.0** and this
repository is **MIT**, so the code may be used here — §4 of that licence, not a favour, is what makes it legal,
and it requires the NOTICE to travel (their notice names Copyright (c) 2026 André Borchert and a
`turbo-fieldfare` dependency), the licence text to be included, and modified files to be marked. `AGENTS.md` now
says so, `THIRD_PARTY_NOTICES.md` says what is required, and `tools/check_provenance.py` will be changed in the
same commit as the first code taken: today it **refuses** a third-party copyright line and asserts that no
third-party source is included, and that assertion becomes false the moment the permission is used. A gate that
forbids what the operator has allowed gets disabled in a hurry, so it is changed deliberately, with the reason,
in the commit that needs it — and **nothing has been copied yet**, which is why this record changes no code.

**`mix.read` is two costs, and neither is the device.** `SourceTiming` has existed since the install reader was
written — its own comment says "`mix.read` was 65% of a real forward and the disk is measured at ~1 GB/s, so the
caller needs to know whether those seconds are the device or the unpacking, a distinction arithmetic cannot
settle" — and no caller had ever asked for it. Now `datacenter-generate` writes all three, and on an 8-step
decode of the 35 B-A3B:

| quantity | per step | share of the step |
| --- | --- | --- |
| step | 3.342 s | — |
| `mix.read` | 1.585 s | 47.4% |
| source **read** (the device) | 1.298 s | 38.8% |
| source **unpack** (CPU) | 1.353 s | 40.5% |
| bytes read from disk | 1.08 GB | 0.8 GB/s |
| fp32 materialised | **6.5 GB** | — |

The read and unpack totals span every phase, which is why they sum past `mix.read` — the head's bf16 and the
dense path's tensors are read and unpacked in `head` and `load` as well. Two conclusions follow, and they are
measurements rather than arithmetic: **the disk is not the limit** (1.08 GB/step is 0.8 GB/s against a device
this repository has measured at ~1 GB/s and up), and **the unpack is as large as the read** — 1.35 s and 6.5 GB
of fp32 written per step, discarded, and written again on the next token. That is what `DC-120` is for, and it
is now the best-supported target in the list.

**The GPU matmul cannot run here, and the reason is the same one.** `MetalMatmul` is opt-in because `D63`
measured the *older* kernel slower; the threadgroup-tiled kernel written since had never been measured
(`DC-113`). It was measured this round and the run was **stopped by the disk watchdog** at 3.94 GB free with
swap at 1.71 GB: the head's weights are 1.017 GB of bf16, the path materialises them as **2.03 GB of fp32**, and
the kernel then needs an `MTLBuffer` of the same size — about **4 GB** on a node with ~4.5 GB usable. So the
GPU path is not blocked by the kernel, it is blocked by the fp32 array that `DC-120` exists to remove, and that
is one more reason to do `DC-120` first: after it, both the threaded CPU matmul (`DC-122`) and the GPU kernel
become addressable.

## D101 — The expert reads were latency-bound, and fanning them out is worth 1.35x

`D100` split the largest phase and found two halves: the device read (1.298 s/step) and the CPU unpack
(1.353 s). The read half is 1.08 GB per step at **0.83 GB/s**, a quarter of what the device can do — so it is
**latency**, not bandwidth: about 520 small `pread`s per step, issued one after another. `DC-118` fixes that
with a hint rather than a rewrite of the reader.

**The change.** `ExpertWeightProvider.preload(experts:shape:)` is called once per layer, at the one point where
the router's choices are known, and its default is to do nothing. The adapter in front of the
generation-scoped bank implements it by fanning the *misses* across threads through `DecodeThreads`, writing
into the bank as **staging**. Two details are load-bearing:

- **The warm reads do not count as requests.** The loop that follows is the requester, finds each slice
  resident and counts a *hit*; counting the warm-up too would double every request and every miss, and
  `ForwardResult.expertMetrics` is summed over layers by its callers.
- **The bank must have a budget and there must be more than one thread**, or the preloaded bytes are discarded
  and read again — twice the work rather than half the latency.

The reader's counters are now **write-locked**, because a `+=` from two threads loses one of them and a lost
read under-reports exactly the cost this change exists to reduce. The *reads* of those counters are
deliberately unlocked, which is a statement about when they happen: `sourceTiming` is asked between forwards,
when no reader is running.

**Measured**, alternated, one binary, 8-step decode, load average 2.90:

| configuration | s/step | `mix.read` | digest |
| --- | --- | --- | --- |
| bank 0 | 3.383 / 3.382 | 1.671 / 1.660 | `89d654ff54b0fd03` |
| bank 512 MB + fan-out | **2.503** | **0.761** | `89d654ff54b0fd03` |
| bank 512 MB, `SHARD_DECODE_THREADS=1` | 4.109 | 1.708 | `89d654ff54b0fd03` |

**0.296 → 0.400 tok/s**, digest identical in every arm. The third row is what makes the first two readable: the
bank **alone** is a *loss*, which is what `D98` measured and why it set the default to 0; what changed is what
the bank is *for*. It is not a cache that has to earn its keep by hitting — it cannot hit this model — it is the
**staging area the fan-out writes into**, and with the fan-out the same 512 MB is worth 1.35x. The default is
therefore 512 MB, and `SHARD_EXPERT_BANK_MB=0` disables both halves at once.

**One instrument caveat, recorded because it would mislead otherwise.** With concurrent reads the per-thread
counters sum **past wall time**: the same run reports read 3.867 s and unpack 5.206 s inside a 2.503 s step.
They measure *thread-time under overlap*, not elapsed time, and any later reading of them has to know that. The
honest fix, when it matters, is to attribute the segments once per phase rather than per read.

**Where this leaves the objective.** 2.503 s/step = **0.400 tok/s** against the operator's 7 — a factor of 17.5.
The remaining levers are unchanged in kind and now better ordered: the unpack is the largest single cost and
`DC-120` (fuse the dequantise into the matmul, never materialise the fp32 slice) removes it; `DC-122` (the
threaded CPU matmul) is still open with its failure on the record; and `DC-121` (prefetching the *next* token's
experts) is now cheap to try, because the fan-out it would feed already exists.

## D102 — The head's blocks fan out, and why that is legal where the general matmul was not

`head` was 0.43 s/step, and it is the one matmul in the engine whose decomposition is *obviously* sound. `DC-122`
had already tried threading the general `Ops.orderedMatmulVectorized` and it **moved bits** — bounding each
thread's work by its own `last` regrouped the four-wide columns and moved the scalar tail, and a comparison that
had held since `D9` caught it. That function is therefore the original single-threaded body, and the experiment
lives in the tracker rather than in the source.

**The distinction is structural, not a matter of luck.** In the general kernel the unit of work is a *column
range inside one row*, so a decomposition has to decide where one thread's columns end — and that is where the
grouping moves. In the head the unit is a **whole vocabulary block of rows**: `Ops.orderedMatmul` is called
exactly as it was, with the same `k` order and the same internal grouping, for a set of rows that a given thread
owns outright. Nothing is split, so nothing can be regrouped, and the only thing the decomposition changes is
**which thread** runs a block — precisely what `D63` allows. That is an argument, and an argument is not evidence,
which is why `HeadLogitsTests` walks block widths **1, 3, 7, 64 and 512** and compares bit patterns: `1`, `3` and
`7` are the `out % 4 != 0` regime that broke the earlier attempt. Two more tests carry the rest of the contract:
a node's vocabulary slice is bit-identical to the same rows of the full array with the remainder left at zero, and
a block that throws is **raised** rather than swallowed — a fan-out that dropped an error would hand back zeros
that look like logits.

**Measured**, alternated, one binary, 8-step decode, load average 2.84:

| `head` | step | digest |
| --- | --- | --- |
| 0.434 / 0.439 s (one thread) | 4.142 / 4.077 s | `89d654ff54b0fd03` |
| **0.259 / 0.255 s** (fan-out) | **2.270 / 2.246 s** | `89d654ff54b0fd03` |

The `head` column is the isolated measurement: **1.7x on that phase**, 0.43 → 0.26 s, digest identical in every
arm. The step column is *not* the head's alone — `SHARD_DECODE_THREADS=1` switches off `D99`'s element-wise
threading and `D101`'s expert fan-out as well as this one — so the honest attribution is the phase, not the step.
With all three on, the step is **2.246 s = 0.443 tok/s**, from 0.400 before this change and 0.230 when the
operator re-scoped the work.

**What is still open** is the general matmul, and it now has a menu rather than a mystery: a case split per output
so no thread ever owns a partial group, `x`-major blocking where each thread keeps a private accumulator, or
accepting the four-wide grouping as part of the contract and asserting it. The third is a contract change and
would need the reference re-derived, so it is the last resort; the first two are `DC-122`.

## D103 — `DC-124` was a double release in the checkpoint reader's streaming handle, and the discriminator found it

`DC-124` had been an unexplained intermittent abort for two rounds. It is now found, fixed, and closed on a
failure rate — with the honest caveat that a rate is a measurement and not a test, which is why `DC-125` exists.

**The bug.** `Safetensors.StreamHandle.file` was a bare `var`, the "open on first use" cache behind
`rowsStreaming`. It is read *and written* from the reader's worker threads — `D94`'s row-parallel unpack, `D99`'s
element-wise map, `D101`'s expert fan-out — so two workers can see `nil` at the same moment, each open a
descriptor, and the **racing assignments release one `UncachedFile` twice**. The Swift runtime reports precisely
that as "deallocated with non-zero retain count 2 … may have created a strong reference to self which outlived
deinit, resulting in a dangling reference", and the process aborts with `-6`.

**What found it was the discriminator, not the message.** The line looked like teardown noise for two rounds,
and it was actually the fatal error itself — it appeared in **every failing run and no passing one**. What
narrowed it to this cache was the pattern of *which* runs failed: **every failure was on a checkpoint path**
(`test_run_m1_gate` and `test_sharded_checkpoint`) while **every install run was clean**, including all of this
session's A/B measurements and the trace CLI. That is the signature of the one difference between the two
readers: `InstallFile` holds its handle in a `let`, and this one cached it in a mutable slot with no lock.
Two rounds of looking at the message got nowhere; one look at *who fails* named the file.

**The fix** is a lock in `StreamHandle` and a double-checked open, with the read itself outside it so concurrent
`pread`s still overlap. The loser of a race drops its ordinary local reference, which is safe; what was unsafe
was a second release through the shared slot.

**The evidence, and its limits.** The reproduction is
`python3 -m unittest tools.test_run_m1_gate tools.test_sharded_checkpoint`, which failed **3 of 4** times before
and **0 of 5** times after, with the retain-count line appearing **zero** times in the five. Against a 75% prior,
five clean runs by luck is about 0.1%, which is why this is treated as fixed rather than as quieter. But a race
that needs timing pressure has no deterministic reproduction, so what is asserted is a rate, and a rate is not a
test: `DC-125` is the row for the stress test that will fail loudly if the lock is ever removed, and until it
exists this decision rests on the rate alone. The regression is also recorded as a trap for the next reader:
`ExpertSlotCache.metrics` had the same shape — fan-out writes with no lock — and was fixed in the same round
before it could produce its own version of this.

**What it cost the objective.** Nothing measurable, and it is worth saying plainly: the *speed* work of this
round is one guarded counter. What the round bought is a gate that can be trusted — the Python suite was failing
75% of the time for a reason that had nothing to do with any measurement in this repository, and every claim
made this session was made on runs that happened to be in the clean 25%.

## D104 — The contract matmul is threaded, on width-aligned chunks: `DC-122` is solved on the second attempt

`DC-122` had been open since the first attempt moved bits: the threaded `Ops.orderedMatmulVectorized` was caught
by a comparison that had held since `D9`, reverted verbatim, and left with a *hypothesis* about a block-local
`last`. The hypothesis was right, and writing it down precisely is what made the second attempt possible.

**The rule.** The serial body runs a four-wide loop while `column + 4 <= out`, so its vector groups are the fixed
partition `{0-3}, {4-7}, …` and its **scalar tail is exactly the last `out % 4` columns**. A decomposition that
gives a thread a range such as `[0, 6)` and lets it run its own `column + 4 <= last` loop **re-cuts the grouping
at 6 and puts the scalar part where the vector part used to be** — a different sequence of additions, hence
different bits. The rule is therefore: **every chunk boundary is a multiple of four.** Then a non-final chunk
covers whole groups and reaches its end with *no tail at all*, and the final chunk ends at `out` and performs the
same `out % 4` tail the serial body performs. Each thread's additions are the same additions in the same order;
only the thread differs, which is what `D63` permits.

**Two other decisions fell out of it.** The chunk size is rounded **down** to a multiple of four, so the remainder
lands in the last chunk rather than in an unaligned boundary. And the work **threshold lives at the call site**,
not in the fan-out: a dispatch costs more than it saves below `DecodeThreads.minimumWork` (`D59`), and keeping
that policy out of the mechanism is what lets `OrderedMatmulThreadTests` drive the chunking directly on
deliberately awkward shapes instead of only on shapes large enough to trip the threshold.

**Measured**, alternated, one binary, 8-step decode, load average 2.53. The phase columns are the isolated
evidence — `SHARD_DECODE_THREADS=1` switches off `D99`, `D101` and `D102` as well as this — and the step is the
sum:

| phase | one thread | fanned out | factor |
| --- | --- | --- | --- |
| `attn.core` | 0.493 / 0.495 s | **0.209 / 0.202 s** | 2.4x |
| `mix.gateup` | 0.224 / 0.225 s | **0.077 / 0.073 s** | 3.0x |
| `mix.down` | 0.102 / 0.102 s | **0.049 / 0.047 s** | 2.1x |
| step | 4.124 / 4.103 s | **1.866 / 1.797 s** | — |

Digest `89d654ff54b0fd03` in **all four arms**. The step is now **~0.55 tok/s**, from 0.230 when the operator
re-scoped the work and 0.443 before this change — 2.4x overall, every step of it bit-identical.

**What the tests assert, and why they are the point.** `OrderedMatmulThreadTests` compares bit patterns against
*both* the definition and the serial fast path, over `out ∈ {1,2,3,4,5,6,7,8,9,11,15,16,17,33,64,65,129}` (every
residue mod 4), `out < 4` where the vector loop never runs, `k = 1`, `rows ∈ {1,2,3}`, thread counts 0, 1, 2, 3,
5, 8 and 16, and the two shapes the decode path actually spends its time in. It also repeats a call on the same
buffers, because **a reused buffer changing the answer is how the first attempt was caught**. A faster
formulation is only trustworthy while something independent says it agrees — and the argument above is a reason,
not that something.

## D105 — The predictive prefetch loses, because the reads are no longer latency-bound

`DC-121` was the reference's `ExpertPrefetchRing` idea, and it was the item this repository's own notes had called
the measured priority. It is implemented, measured, and **off by default** — and the measurement is worth more
than the change.

**Why it should have worked.** A layer's routing is strongly correlated between consecutive tokens. `D101` had
taken the expert reads off the *latency* problem by fanning the misses across threads, which left the read itself
— about 1.08 GB per step, irreducible, because it is the weights — sitting synchronously on the critical path just
before the loop that needs it. Issuing the *previous token's* choices early, on a background thread, before the
layer's attention runs, should have hidden them behind compute. A wrong guess would cost only the read the loop
would have made anyway.

**What it did instead.** Measured on one binary, alternated, 8-step cached decode, load average 3.14:

| | step | `mix.read` | digest |
| --- | --- | --- | --- |
| prediction on | 1.859 / 1.816 s | 0.830 / 0.813 s | `89d654ff54b0fd03` |
| prediction off | 1.854 / 1.721 s | 0.790 / 0.760 s | `89d654ff54b0fd03` |

No gain, and `mix.read` is consistently **larger** with the prediction on. The digest is identical in every arm,
so nothing about correctness is in question — only the arithmetic of where the seconds go.

**The reading, which is the actual result.** `D101`'s fan-out did not just make the reads faster, it **removed the
latency problem** — so the device is now **saturated**: at roughly 1.08 GB in about half a second it is moving
something like 2 GB/s (a derived figure, not a measured one, from the phase and the byte count), and a second
stream of reads does not fill an idle gap, it takes bandwidth from the first. **A prefetch can only help a
latency-bound resource; a saturated one is made worse by asking it to do more at once.**

**What follows.** The lever moves from *hiding* the read to **reading fewer bytes**. The bank holds slices as
`[Float]`, so 512 MB buys about 61 slices while a token asks for roughly 773; the same bytes in their **packed
int4 form** buy four times as many, and at 1.5 GB that is approximately the whole per-token working set. That is
`DC-120`, and it needs the dequantise to happen at *consumption* rather than at load — the fused kernel this
round's plan started from, now with a measurement saying why it is the only remaining CPU-side lever.

The switch is `SHARD_EXPERT_PREDICT=1`, the default is off (`D89` and `D98` set the same precedent: a measured
loss is not a default), and `prefetchPredicted(force:)` plus `waitForPrefetch()` exist so the mechanism stays
tested rather than rotting behind a switch nobody can turn on in a test.

## D106 — The packed slab cache works, and loses: this node's constraint is memory

`D105` ended with the lever stated precisely: the device is saturated, so a read must be **avoided** rather than
hidden, and the way to avoid it is to hold the slabs in their **packed** form — four bits per weight rather than
the thirty-two the decoder produces, which is four times as many slabs per byte. `InstallFile.SlabCache` is that
cache: expert row ranges keyed by tensor and range, holding the concatenated `codes + scales + zeros` exactly as
they were read, with a hit skipping all three `pread`s. It is built, tested and measured. **It works, and the
default is zero, because on this node it loses.**

**The measurement**, one binary, 8-step cached decode, load average 2.66–3.01, digest `89d654ff54b0fd03` in
**every** arm:

| slab budget | slab hit rate | disk per step | `mix.read` | `load` | `head` | step |
| --- | --- | --- | --- | --- | --- | --- |
| 0 MB (default) | 0.0% | 1.08 GB | 0.754 s | **0.363 s** | **0.261 s** | **1.735 s** |
| 512 MB | **0.0%** | 1.08 GB | 0.867 s | 0.65 s | — | 2.097 / 2.130 s |
| 768 MB | 36.1% | 0.73 GB | 0.628 s | 0.650 s | 0.348 s | 2.022 s |
| 1024 MB | 45.0% | **0.65 GB** | **0.544 s** | 0.591 s | 0.411 s | 1.931 s |

**The cache does exactly what it was designed to do.** At 1 GB it serves **45%** of the expert slabs from memory
and takes the read from **1.08 GB to 0.65 GB per step**, which is a **0.21 s** cut in `mix.read` — the first time
anything in this session has moved the *bytes* rather than the seconds. And the step still gets **worse**.

**Why, and this is the result.** The saving is more than spent elsewhere: `load` nearly doubles (0.363 → 0.591) and
`head` grows by half (0.261 → 0.411), on a machine swapping 1.94 GB of 3.07. It is the same signature as `D89`
("the resident fp32 arrays cost more elsewhere than they saved: `head` +0.20, `attn.core` +0.27"), and it is now
the **fourth** time this engine has reached this verdict — `D89`'s decoded-weight cache, `D98`'s fp32 expert bank,
`D105`'s prefetch, and this. Four different mechanisms, four losses, one cause:

> **This node's binding constraint is memory.** Buying a saturated device with RAM costs more elsewhere than it
> saves, so every "cache more of it" route on an 8 GB Mac mini has now been measured and closed.

That does not make the cache wrong; it makes it wrong *here*. The mechanism is validated by its own hit rate and
byte counters, the values are bit-identical, and `SHARD_SLAB_CACHE_MB` is the knob for a node with headroom — the
same disposition `D89` and `D98` were given. What it closes is the *route*: **a bigger cache is not how this
engine reaches 7 tok/s on this machine.**

**What it leaves open, and where the next work goes.** The read can still be *avoided* without spending memory —
by not materialising the fp32 slab in the first place (`DC-120`), which is the unpack and the 6.5 GB/step of
`Float` this repository allocates and discards per token. That is the remaining CPU-side lever and it costs no
residency. Beyond it the arithmetic is unchanged and worth restating plainly: the reference reaches 7.075 tok/s
with a 3 GB cache on comparable hardware, which cannot fit beside this engine's **fp32** weights on an 8 GB node —
so the difference between the two designs is the format the weights are held in, and that is a GPU/int4 change
rather than a caching one.

One incidental fix in the same round: `ReadState.addUnpack` was written in `D94` and **nothing called it**, so the
unpack counter was still an unguarded `+=` shared with the fan-out. It is routed through the guarded method now,
and the cache's hit path uses it too.

## D107 — The fused int4 matmul is 4.7x slower than the split, and that closes the CPU side

`DC-120` was the last CPU-side lever standing: dequantise **inside** the matmul's inner loop so the fp32 slab never
exists. The arithmetic looked good — `mix.read` writes **6.5 GB of `Float` per token** and the matmul reads it
straight back — and `D106` had just shown that avoiding the *read* costs memory this node does not have, which left
avoiding the *materialisation* as the only move that costs nothing.

**It was built, and correctness was never the question.** `InstallFile.int4Matmul` was proved **bit-identical** to
`dequantizeInt4` followed by `Ops.orderedMatmul` over a grid of every residue mod 4, `k = 1`, groups of 1, 4, 8 and
64, output widths 1 to 8, **and the model's real shapes** (512×2048, 4096×2048). That is the only reason the speed
result is worth anything: a fast wrong kernel would have been discarded for the wrong reason.

**Measured**, release build, one 512×2048 slab, 20 iterations after warm-up:

| path | unpack | matmul | total | ratio |
| --- | --- | --- | --- | --- |
| split (what the engine does) | 0.23 ms | 0.10 ms | **0.33 ms** | 1.00x |
| fused (`int4Matmul`) | — | — | **1.55 ms** | **0.21x** |

**4.7x slower**, and 0.14x in a debug build. The reason is structural rather than incidental: dequantising per
element means per-element group-index arithmetic, a scale and zero load, and a signed-nibble extraction **inside**
the accumulation loop — which destroys the vectorisation that `dequantizeInt4`'s eight-wide path and the matmul's
four-wide path each enjoy on their own. Two tight loops beat one clever loop.

**And the probe bounded the prize**, which matters more than the loss. A 512×2048 slab unpacks in **0.23 ms**, so
the ~520 slabs a token asks for cost about **0.12 s per step — roughly 7%**. Even a fused kernel that had been
*faster* had almost nothing to win on this workload. The engine's split of dequantise-then-multiply is not a
compromise it settled for; it is the right shape for a CPU.

**Reverted, not kept** — the code, the test and the probe — on the `DC-122` precedent that a measured failure is
worth more on the record than in the tree. Nothing is left behind to rot.

**What this closes, stated plainly.** With `D105` (the expert read is saturated at ~2 GB/s, so hiding it with a
prefetch makes things worse), `D106` (a 4x denser packed cache is a *loss* because the memory costs more than the
bytes save) and now this, **every CPU-side lever in this engine has been measured and closed**:

| lever | verdict |
| --- | --- |
| thread the element-wise passes and the unpack (`D94`, `D99`) | kept, 0.230 → 0.443 tok/s |
| fan the expert misses (`D101`) | kept, 0.443 → 0.55 |
| thread the contract matmul on aligned chunks (`D104`) | kept, → ~0.55 |
| the head's vocabulary blocks (`D102`) | kept |
| predictive prefetch (`DC-121`/`D105`) | measured, loses, default off |
| packed slab cache (`DC-126`/`D106`) | measured, loses, default off |
| fused int4 matmul (`DC-120`/`D107`) | measured 4.7x slower, reverted |

The engine stands at **~0.55 tok/s**, up from 0.230, with every step bit-identical. The remaining distance to the
operator's 7 tok/s is **not** more of this work: the reference reaches 7.075 with a 3 GB cache on comparable
hardware, and this node cannot hold that beside **fp32** weights — so what separates the two designs is the format
the weights are held in and the device that consumes them. `DC-113` is therefore no longer a candidate but the
route: packed int4 weights resident in GPU buffers, dequantised in the kernel, with the residency table and
prefetch ring the reference uses.

## D108 — The fused int4 matmul is 1.2–2.4x faster on the GPU, and the unpack was never the bottleneck

`D107` closed the **CPU** side of `DC-120`: dequantising inside the inner loop destroys the vectorisation the
split path has, and it measured 4.7x slower. The same fusion on the **device** is a different question in kind,
not degree. The unpack already runs there (`D59`), so the fp32 slab is not a compute cost — it is **traffic**.
`mix.read` materialises about **6.5 GB of `Float` per token** into a Metal buffer, copies it back to the CPU as a
Swift array, and the contract matmul reads it a second time. A fused kernel removes the write and the read-back:
it reads the packed codes it needs and writes `out` floats.

**Built and proved first.** `MetalInt4Matmul` (`sources/DatacenterEngine/MetalInt4Matmul.swift`) computes
`x @ dequantizeInt4(w)ᵀ` in one dispatch — one thread per output, `k` ascending, the weight built with the same
one-multiply rounding, the same `D34` zero-sign/NaN normalisation and the same `D11` denormal-scale flush the
unpack kernel uses, accumulated as `accumulator + metal::fma(x, w, 0.0f)` because that is the only spelling that
keeps the contract's product and add apart (`D61`). `MetalInt4MatmulTests` asserts it **bit for bit** against
`InstallFile.dequantizeInt4` followed by `Ops.orderedMatmul` over the `D107` grid — every residue mod 4, `k = 1`,
groups of 1/4/8/64, a partly-filled final group, a padded width wider than the real one, and one and two tokens —
and against the engine's own path (the GPU unpack followed by the contract matmul) on the real expert shapes.

**Measured**, release build, one binary, 20 iterations after warm-up, one token:

| shape `out × k` | payload | unpack | matmul | split | fused | fused/split |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 × 2048 (`gate_up`) | 1,212,416 B | 0.83 ms | 0.21 ms | 0.940 ms | **0.763 ms** | 1.23x |
| 2048 × 512 (`down`) | 606,208 B | 0.49 ms | 0.10 ms | 0.582 ms | **0.402 ms** | 1.45x |
| 4096 × 2048 (the head's width) | 4,849,664 B | 2.62 ms | 0.79 ms | 2.972 ms | **1.258 ms** | 2.36x |

A token asks for 8 experts in each of 40 layers, i.e. **320 slices of each projection** (the install's own
geometry): split **0.487 s**, fused **0.373 s**, so the fusion is worth **0.114 s per step — about 6.6%** of the
1.74 s step `D104` left. That is the same order as the 7% `D107` bounded from the CPU and it says the same thing
from the other side: **the unpack and the matmul together are not where the step goes.** `mix.read` is the *disk
read* — 1.08 GB a step at the device's floor — and no kernel helps with bytes that have not arrived. The route to
the operator's 7 tok/s remains **residency**, not arithmetic; this kernel is what makes residency possible (a
resident slab never has to become fp32), not what makes it fast on its own.

**The denormal boundary, found by the grid and pinned rather than hidden.** The first grid run failed exactly one
output in seventeen, by 128 ULP. The weights were bit-identical on both sides; the difference was a single term
where `x * w ≈ -5.8e-39`, a **denormal**, which the CPU keeps and the GPU returns as zero. It is the device, not
the kernel and not the math mode: `MetalMatmul.matmul` — the kernel the tree already calls bit-exact — gives the
same flushed answer on the same input, and switching the fused pipeline from `.relaxed` to `.safe` changed
nothing. Weights themselves can never be denormal, because the smallest non-zero code is 1 and a denormal scale
is flushed to zero first, so the property cannot be reached through the unpack; it needs a product of two small
normals, which real activations and real weights never produce. `MetalInt4MatmulTests` asserts the boundary
directly — the CPU keeps the denormal, both GPU kernels flush it — and `MetalMatmul`'s claim is narrowed to the
form that is true.

**Disposition.** Kept as a tested, measured asset and **not wired into the engine yet**. Wiring it means the
provider hands over a **packed payload** instead of a `[Float]` slice and the slot bank stops holding fp32 — the
residency change `DC-113`/`DC-123` describe — and that change is the next row, gated by the M1 install trace
rather than by a kernel test. `D63` is the precedent: a bit-exact, faster kernel sat opt-in until something
called it, and the switch it needs is a call site, not a flag.

## D109 — The LM head comes off the device every step, and the first fixes were losses as much as wins

The round opened by reading the sister project's decode path (`D100` authorised it) and by taking a fresh
profile of this engine's step, because the operator's objective is **7 tok/s on one node** and every earlier
conclusion had to be re-checked against the tree rather than trusted. What follows is in the order it was
measured, including the three ideas that **lost**, because the losses set the budget for the rest.

**The step, measured before anything changed** (`SHARD_PROFILE=1`, 6 cached steps, digest `ed5e0328c087e4db…`):

| phase | ms/step | what it is |
| --- | --- | --- |
| `mix.read` | **805** | expert slabs off the device (582 MB/step at ~4 GB/s) plus the unpack |
| `load` | **361** | the dense backbone decoded to fp32, every layer, every token |
| `head` | **255** | bf16 weights read and widened to `Float` |
| `attn.core` | **202** | the attention and DeltaNet matmuls |
| `mix.gateup` + `mix.down` | 119 | the routed projections |
| step | **1769** | **0.565 tok/s** |

**Losses first, because they bound the prize.**

1. **The GPU unpack for the dense `tensor(named:)` path is slower than the CPU one.** It looked like a
   three-line win — `rows` uses `MetalUnpack`, `tensor` did not — and it cost **`load` 353 → 534 ms/step**.
   A dense projection is a *few large* tensors, so the device path pays a payload copy in and an fp32 result
   out where `dequantizeInt4` is already threaded across the cores (`D94`) on data that is in cache. Reverted,
   with the reason in the code so it is not "fixed" again.
2. **The fused int4 expert path is a wash, and it is only not a loss because of a cache.** `D108` proved the
   kernel and measured it 1.23-2.36x faster per slab in isolation. In the engine, with the slab cache off, it
   is **2.29 s/step against the control's 1.69** — the per-slab dispatch and payload copy (640 dispatch-and-wait
   pairs per token) cost more than the unpack-plus-threaded-matmul it replaces. With `SHARD_SLAB_CACHE_MB=1024`
   the reads fall **8.46 → 5.73 GB** (71% slab hits) and the step comes back to **1.575 s** — level with the
   control, not ahead of it. Default **off**; the wiring (`packedRows`, `gateUpProduct`/`downProduct`,
   `preloadPacked`) stays as the tested foundation for batching, which is where the arithmetic says the win is:
   one dispatch per layer instead of eight per expert.
3. The first attempt at head residency **took the node into the same swap-discipline failure `D106` recorded**,
   twice. Two bugs, and both are now fixed and pinned:
   - `headLogits` fans out over 31 blocks, and the natural first cut asked `storedRows` for the head *per block*.
     The payload cache is populated only by the **first** read to finish, so all 31 missed and each read the
     whole 1.017 GB — 31 GB of transient allocations. Fixed by fetching the whole stored tensor **once**, before
     the fan-out. The disk watchdog stopped that run at 4.48 GB free with **6 GB of swap in use**.
   - `UncachedFile.readData` was `Data(try read(...))`, so **every** read existed twice at its peak: 2.03 GB of
     transient memory for the head's 1.017 GB block. It now fills `Data(count:)` in place. That is a general
     fix — every payload read in the engine paid it.

**Wins, all alternated on one binary with the digest unchanged.**

**The LM head runs on the GPU from its stored bf16** (`MetalBf16Matmul`). The head is `[248320, 2048]` of bf16
— **1.017 GB stored, 2.034 GB decoded, per token** — and `rows` read it with `F_NOCACHE` through a path that
never touched the payload cache, so it came off the device on every step. `storedRows` fetches it once through
the payload cache (whose default budget moves **1024 → 2048 MiB**, because the head and the layer payload must
both fit or an LRU trades them back and forth), and the kernel widens each bf16 in a register — a shift of the
top sixteen bits, which cannot round. Measured:

| arm | `head` | step | tok/s |
| --- | --- | --- | --- |
| CPU head | 255.3 ms | 1.7178 s | 0.582 |
| GPU bf16 head | **117.8 ms** | **1.5804 s** | **0.633** |

**And three small ones that were kept because they measured:** `MetalBufferCache` is now **per slot and
grow-only** — it replaced *every* buffer as soon as one slot was too small, and gate-up and down alternate
sizes, so the whole set including the 8 MB output was reallocated ~320 times a step (1.77 → 1.75 s, small but
strictly right); `GatedDeltaNet.decodeStep` no longer allocates a `kernel`-element array inside an 8,192-iteration
loop (**245,760 allocations a step**, `attn.core` 201 → 180 ms); and `headLogits` calls the **serial** matmul
body rather than the chooser, because it is already a fan-out and was nesting 31 × 8 tasks on 8 cores.

**One more bug, found by the suite rather than by a measurement.** A `Data` slice keeps its parent's indices, so
the head's per-block window `storedHead.data[start..<end]` was out of bounds the moment a vocabulary window did
not start at zero — which is exactly the sharded case, and `ShardedGenerateTests` trapped on it. Indexed from
`startIndex` now. The full suite is **266 tests, 0 failures** (7 new: 4 `MetalBf16MatmulTests`, 3
`InstallStoredFormTests`).

**Where the step stands, and what the reference says is left.** Default configuration: **1.567 s/step,
0.638 tok/s, peak RSS 3.10 GB**, digest `ed5e0328c087e4db…` unchanged — from 0.565 at the top of the round. The
study of the sister project is unambiguous about the remaining distance, and it is not arithmetic: its
per-token traffic is **~1.5-1.8 GB of dense weights plus ~0.17 GB of experts, all of it packed, none of it ever
widened to fp32**, and its head is **int4 at 286 MB** against this install's bf16 at 1017 MB. Its 141 ms is
~60 ms of GPU, **~55 ms of host blocked on the routing readback and the miss reads**, and ~26 ms of CPU encode,
with per-layer slot caches (48 of 256 experts per layer at 3 GiB) and layer L's MoE running in its own command
buffer while layer L+1's attention runs. The measured order of attack here is therefore: **batch the experts
per layer into one dispatch** (the fused kernel is already bit-exact and the per-slab overhead is what eats
it), **residency sized from one budget** with a packed slab cache, and only then the overlap.

## D110 — The int4 kernel was loading one byte at a time; fixing that made the packed expert path win

`D109` left the fused expert path as a wash and named the reason: the per-slab dispatch and copy cost what
the kernel saves. This round measured the kernel itself, and the answer was not dispatch at all.

**The probe, in release, one shape per call:** a bf16 dispatch with 32 KB of weights takes **0.858 ms**, so a
synchronous command buffer plus its wait is of that order; the bf16 head block (33.5 MB) takes 3.36 ms; the
fused int4 kernel takes **0.514 ms for 1024×2048** and **8.734 ms for 32768×2048**. The last figure is
**38.8 MB in 8.7 ms = 4.4 GB/s**, on hardware whose memory does roughly **100 GB/s**. The kernel was not
dispatch-bound, it was **mis-loading**: lane `l` and lane `l + 1` walked rows `padded / 2` bytes apart and
each element was a separate `uchar` load, so every fetch used one byte of a cache line.

**The fix is a load shape, not an arithmetic one.** Each thread now reads its own row as **`uint4`** — 16
bytes, 32 four-bit codes, little-endian within each word — and the nibble order, the `D34` normalisation and
`D61`'s `accumulator + metal::fma(x, w, 0)` are untouched. The tail (a group whose start is not 16-byte
aligned, or fewer than 32 codes left in it) keeps the scalar loop, so a shape the wide path cannot take is
slower and not different. Measured on the same probe:

| shape | before | after | |
| --- | --- | --- | --- |
| 1024 × 2048 | 0.514 ms | **0.303 ms** | |
| 2048 × 512 | 0.328 ms | **0.202 ms** | |
| 8192 × 2048 | 2.327 ms | **0.652 ms** | 3.6x |
| 32768 × 2048 | 8.734 ms | **2.196 ms** | 4.0x, 38.8 MB at **17.7 GB/s** |

**And the grid caught a trap that had nothing to do with loads.** The first wide-load version failed one
output in a 33-column test by **1 ULP**. The cause was `#pragma unroll`: with the inner chain written out,
the optimiser is free to **reassociate** the float additions under `.relaxed`, and it did. An accumulator
chain written in source order is the contract; a tree-reduced one is a different sum. The unroll is gone and
the pipeline is `.safe` — `D61` measured that `.safe` still contracts `a*b+c` into one `fma`, which is why the
`fma(x, w, 0)` spelling stays exact, so nothing is given up but the licence to reorder.

**The cache size is the other half, and smaller is better.** With the fused path on, the packed slab cache
was swept: **256 MiB 1.465 s, 512 1.467, 768 1.483, 1024 1.667** per step. Above 256 the resident bytes cost
more in memory pressure than the reads they save — `D106`'s verdict, a third time — so the default is the
smallest size that pays rather than the largest that fits. Five alternated pairs on one binary against the
split path, same digest `ed5e0328c087e4db…` in all ten runs:

| arm | per-step, five runs | median |
| --- | --- | --- |
| split (fused off) | 1.539, 1.544, 1.534, 1.524, 1.564 | 1.539 s |
| fused, 256 MiB slab cache | 1.464, 1.550, 1.463, 1.466, 1.465 | **1.465 s** |

**−4.8%**, and at **lower** peak RSS (3.04 against 3.13 GB), because the fp32 slab is never built. Both
defaults are now the measured ones: `SHARD_GPU_INT4_EXPERTS=0` and `SHARD_SLAB_CACHE_MB=<n>` restore the
other arms.

**Three things were found by the suite rather than by a benchmark, and all three are fixed rather than
excused.**

1. **A units bug that made the default look like a cache that never held anything.** `slabCacheBudget`
   returns **bytes**, and the new default was written as the bare literal `256` — a 256-**byte** budget,
   which refused every 1.2 MB slab. The env-var path, which multiplies, worked, so the symptom was a default
   that behaved as zero (`slab_cache_bytes_held` 0, step **2.52 s**) beside an explicit 256 that behaved
   correctly (1.46 s). `SlabCacheTests` caught it. Absent and invalid are now different answers: unset takes
   the default, unparseable is refused as zero.
2. **`ExpertSlotCache.preload` was choosing a destination by switch rather than by capability.** It took the
   packed branch whenever the fused path was on, so an array-backed provider — every test fixture — silently
   lost the bank path it has always had. There is now a `servesPacked` capability on the provider, and an
   install answers it only when it actually has a packed cache.
3. **The fused path was over-reporting its traffic.** `countFused` added the weights it walked to
   `elementsRead` on every request, so a forward whose slabs were all served from memory reported
   **13,824 elements read**. It now counts the **request** and leaves the volume to the source's own
   `bytesReadFromSource` and the slab counters, which is where the disk was actually touched — the same rule
   `ExpertSlotCache.warm` learned in `D101`.

**The step is 1.465 s, 0.682 tok/s**, from 0.638 at the top of the round and 0.565 before `D109`. 266 Swift
tests, 0 failures; the three tests that asserted the old bank-only mechanism now assert the saving rather
than the configuration. The objective is still 7 tok/s (143 ms), so what remains is the same list `D109`
left: the expert read, `load`'s fp32 decode, and the per-layer dispatch structure.

## D111 — The page cache is worth having, and the dense int4 projections should never have been decoded

Two changes, both measured, and one bug found between them that would have made the second a trade rather
than a win.

**The install can now be read through the kernel's buffer cache** (`SHARD_INSTALL_CACHED`, on by default;
`=0` restores `F_NOCACHE`). `UncachedFile`'s rule was written for a **sequential scan of a file larger than the
machine** — the 20 GB verification that filled the page cache, grew swap, and is why `D58` exists. A decode step
is the opposite access pattern: a **582 MB working set of expert slabs that repeats every token**, whose pages
are read-only, clean and evictable. Measured, `mix.gather` (the preload read):

| state | `mix.gather` | step |
| --- | --- | --- |
| cold cache, `F_NOCACHE` | 469 ms | 1.456 s |
| warm cache | **266-286 ms** | **1.31-1.38 s** |

A clean cold A/B is **not available on this node**: `purge` needs root, and once any run has populated the
cache the `F_NOCACHE` arm reads the warm pages too — so the honest statement is the cold/warm contrast above,
and that the switch is what *populates* the cache on a fresh node. Peak RSS (2.9-3.0 GB), free disk and swap
were all unchanged across the runs, which is the part that matters: the pages are being reclaimed, not swapped.

**The three Gated DeltaNet int4 projections now come from their stored form.** `linear.in_qkv`, `linear.in_z`
and `linear.out` are **581 MB of the install's 738 MB of dense int4** — the largest per-step decode in the
engine — and `MetalInt4Matmul` widens them in registers, so the fp32 array is never built. `MetalInt4Matmul`
was already bit-exact (`D108`, `D110`), so the trace digest is unchanged. `GatedDeltaNetWeights` carries the
packed handles beside the arrays, and when one is set its `[Float]` twin is **empty**: `GatedDeltaNet.projection`
is the only reader of either and refuses to multiply an empty array, falling back to decoding the stored bytes
on the CPU rather than multiplying nothing onto a `-0.0` accumulator. `load` fell **350 → 119 ms**.

**The bug between them is the one worth recording.** The first version of `packedTensor` went through
`int4RowsPayload` — the row-range reader — which is right for an expert slab and wrong for a dense tensor whose
whole payload is already in the cache. It sent **7.5 GB a step** back to the device for bytes that were
resident, and the only symptom was a counter: reads went **8.46 → 14.28 GB** while the step still looked
slightly better, because a cached-page disk read is faster than a CPU decode. `packedTensor` now slices
`payload(entry)`, the same cached whole-tensor bytes the CPU path decodes. The lesson is the one `D101` wrote
down about volume counting: a re-read disguised as an optimisation shows up in the byte counter and nowhere
else.

**Result.** Three runs, digest `ed5e0328c087e4db…` in all of them: **1.133, 1.087, 1.079 s/step — median
1.087 s, 0.920 tok/s**, from 0.682 at the top of the round. 266 Swift tests, 0 failures. The step is now:

| phase | ms/step | what is left in it |
| --- | --- | --- |
| `mix.gather` | 299 | the expert read, at ~1.5-2 GB/s with a 256 MiB slab cache |
| `mix.read` + `mix.down` | 355 | 640 synchronous fused dispatches at ~0.3-0.5 ms each |
| `attn.core` | 156 | the recurrence plus the attention projections still decoded |
| `head` | 132 | 1.017 GB bf16 through the GPU GEMV |
| `load` | 119 | the attention projections and the bf16 tensors, still decoded per step |

The attention projections (`attn.q/k/v/o`, a further 1.02 GB of fp32 a step) are the same change as the GDN
three and are the next one; after that the dispatch count and the expert read are what stand between this and
the objective.

## D112 — The attention projections were the same change, and the slab cache got smaller once the kernel cached too

**The four int4 attention projections now come from their stored form**, exactly as the GDN three did in
`D111`. `attn.q`, `attn.k`, `attn.v` and `attn.o` are **1.02 GB of fp32 a step** across the ten
full-attention layers, and `Qwen3_5Forward.projection` — the sequence path and the cached path both — now
multiplies the packed bytes with `MetalInt4Matmul`. `DecodedLayer` carries a `packedWeights` map beside the
arrays, a role in that map has an **empty** `[Float]` twin, and the projection helper is the only reader of
either: if the device refuses the shape it decodes the stored bytes on the CPU rather than letting an empty
array reach `Ops.orderedMatmul`, where it would index out of bounds rather than answer zeros.

| | `load` | step | tok/s |
| --- | --- | --- | --- |
| after `D111` | 119 ms | 1.087 s | 0.920 |
| after the attention four | **54 ms** | **1.025 s** | 0.975 |

`load` is now only the bf16 and fp32 tensors — norms, `in_a`/`in_b`/`conv`, the router and the shared
experts — which are small and are read directly by the CPU.

**And the slab cache's optimum moved down, because the kernel is now caching the same slabs.** `D110` swept
it with the device bypassed and found 256 MiB best. `D111` turned the **buffer cache** on, which holds those
slabs in *clean, evictable* pages rather than anonymous memory, and the trade changed: three alternated pairs
put **128 MiB at 0.930 s against 256 MiB at 0.957**, and a finer sweep found 64/96/128/192 at 0.951, 0.943,
0.944, 0.953 — flat between 96 and 128. The default is **128 MiB**. **Zero is worse than every non-zero size**
(1.595 s), which is worth stating because it looks like it should be the cheapest: with no cache at all
`preloadPacked` declines, so the *fan-out* disappears and the loop reads every slab itself, one at a time.

**Result.** Three runs, digest `ed5e0328c087e4db…` in all: 0.997, 0.984, 0.957 s/step — median **0.984 s,
1.016 tok/s**, from 0.682 at the top of `D111`'s round and 0.230 when the operator re-scoped the work. Peak
RSS is **2.57-2.67 GB**, lower than before either change, because less is resident and less is built. 266
Swift tests, 0 failures.

The step is now 63% expert path:

| phase | ms/step |
| --- | --- |
| `mix.gather` | 274 — the read, at ~1.9 GB/s |
| `mix.read` + `mix.down` | 346 — 640 synchronous fused dispatches and their payload copies |
| `attn.core` | 150 |
| `head` | 118 |
| `load` | 55 |

The next lever is the dispatch count: one command buffer per layer instead of one wait per expert is the
structure the sister project uses, and the kernel can now support it.

## D113 — The preload was fanning out over half the work, and a partial payload cache is catastrophic

Two measurements this round, one a win and one a warning that protects a default.

**One preload task per (expert, projection), not per expert.** A slab is **three `pread`s** — codes, scales,
zeros — so fanning over eight experts gave eight threads each issuing **six sequential reads**. The fan-out is
the only concurrency the device sees, and the layer was asking for sixteen slabs while offering eight
workers. Sixteen tasks of three reads each is the same bytes with twice the requests in flight. Five runs,
digest `ed5e0328c087e4db…` in all:

| arm | per-step | median |
| --- | --- | --- |
| `D112` (8 tasks) | 0.997, 0.984, 0.957 | 0.984 s |
| one task per (expert, projection) | 0.949, 0.894, 0.965, 0.919, 0.868 | **0.919 s** |

**1.016 → 1.089 tok/s**, and `mix.gather`'s median moved 274 → 260 ms. The spread across five runs is ±5%,
which is why the arm is five runs rather than three — the farm is shared and the medians are the only thing
that separates the arms at this size.

**And the payload cache cannot be sized partially.** The idea was to shrink the anonymous payload cache and
let the kernel's page cache hold the same bytes in clean, evictable pages, freeing memory for the expert slabs.
It does not work, and the way it fails is worth recording. One run each, same command and binary:

| `SHARD_DENSE_CACHE_MB` | step | peak RSS | `head` | `load` |
| --- | --- | --- | --- | --- |
| **0** (no cache) | 1.359 s | 1.71 GB | 273 ms | 242 ms |
| 256 | **2.341 s** | 2.00 GB | 711 ms | 624 ms |
| 1024 | **2.579 s** | 2.32 GB | 797 ms | 740 ms |
| **2048** (default) | **0.950 s** | 2.64 GB | 113 ms | 53 ms |

Zero is *better than 256 or 1024* — and the reason is the whole finding. At 256 MiB the LRU holds **part** of
the dense payload and **not** the head, so both are re-read from the device on every step: the budget is
large enough to fill and small enough to evict, which is the worst of both. At zero nothing is held and
everything streams, uniformly. At 2048 the dense payload (995 MiB) **and** the head (970 MiB) both fit at
once, which is the only configuration in which either is ever reused. **2048 is therefore near-minimal and
load-bearing, not a tuning knob**: 1965 MiB of content against a 2048 MiB ceiling, and anything below it
converts two resident streams into two device streams. `PayloadCache` evicts one key at a time rather than by
working set, so a budget between the two content sizes has no good behaviour to fall back on.

**Result.** Five runs: median **0.919 s/step, 1.089 tok/s**, from 0.230 when the operator re-scoped the work.
266 Swift tests, 0 failures. The expert path is still ~65% of the step: `mix.gather` 260 ms of device reads
that the page cache is not holding (it is being squeezed by the 2.6 GB of anonymous payload we need), and
`mix.read` + `mix.down` ~340 ms of **640 synchronous fused dispatches**. One command buffer per layer instead
of one wait per expert is the next change, and the kernel is now fast enough for batching to be the whole
question rather than a rounding error on top of a slow kernel.

## D114 — A layer's experts are independent, so they are asked for in one dispatch

The decode step issued **640 fused calls per token** — 8 experts x 2 projections x 40 layers — and each one
built a command buffer, committed it and **blocked** on it. At these shapes the kernel is a few microseconds
of the call: `D110`'s probe put `1024x2048` at 0.303 ms and `2048x512` at 0.202 ms, and the layer's arithmetic
is microseconds. The wait was the cost, and the GPU was idle for the handoff on every one of the 640. A
layer's experts are independent and in decode share `x`, so they are now encoded back to back into **one**
command buffer and waited for once — 80 batch dispatches a step instead of 640 waits.

`MetalInt4Matmul.matmulBatch` encodes `items.count` dispatches into one buffer, then waits, then reads all the
outputs back. Every item must have the **same shape**, because the outputs are read back together and a
mixture layer's experts are one shape by construction; anything else is **refused**. Each item's arithmetic is
`matmul`'s, unchanged.

| | `mix.read` | `mix.down` | step | tok/s |
| --- | --- | --- | --- | --- |
| `D113` | 179 ms | 151 ms | 0.919 s | 1.089 |
| batched | **84 ms** | **52 ms** | **0.722 s** | **1.384** |

Three runs, digest `ed5e0328c087e4db…` in all: 0.712, 0.722, 0.731 s/step. 266 Swift tests, 0 failures.

**The batch is taken only when every chosen expert was routed the same number of rows** — `xs[i].count /
hiddenSize` equal across the layer, which is **every decode step and not every prefill**, where the experts'
assignment counts differ. When it is not uniform the single-expert calls run exactly as before, so a
checkpoint, the tiny fixtures and an A/B control keep the path they had.

**Two bugs on the way, and the first is the one worth remembering.**

The batch asks its buffer cache for `5 x experts` buffers; the single form and the dense projections ask for
5. Both went to the **same** `MetalBufferCache`, whose contract is that it keeps a *set* of slots and
reallocates **all of them** whenever the request **count** changes. So every alternation between a dense
projection and an expert batch rebuilt the whole set — measured at **2.47 s/step against 0.92**, with
`mix.gather` alone at 1079 ms, which is nowhere near the code that was changed. The batch now has its own
cache (`MetalInt4Matmul.batchBuffers`). This is the `D101` shape again: a cost that appears in a phase that
has nothing to do with it, and a shared cache whose sizing assumption was never written down.

The second was that the batch's `rows` was taken from `current.count` — the gathered vector, which is
`rows * hiddenSize` long — rather than from the assignment count. `matmulBatch`'s shape check **refused** it
(`x has 2048 values, 2048x2048 needs 4194304`) instead of multiplying a mis-shaped matrix, which is the whole
argument for validating the batch's shape rather than trusting the caller.

**What is left.** The step is 722 ms and the largest single phase is now `mix.gather` at **245 ms**, which is
the expert read — 582 MB a step that is a device read because the page cache is not holding it. `attn.core` is
147 ms and `head` 110 ms. The lever the sister project uses is **residency**: its 3 GB expert cache is what
takes it to 7 tok/s, and this node's usable memory is why `D106` and `D113` both refused an anonymous cache
that size. Getting there needs the expert working set to be resident in *clean, evictable* pages rather than
in the process, which is a change to how the install is read, not to how it is multiplied.
