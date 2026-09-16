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
