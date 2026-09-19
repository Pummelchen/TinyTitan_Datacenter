# TinyTitan Datacenter 1.1

**ttd 1.1 is the first release under this project's own identity, and the number is its own.**
TinyTitan Datacenter is a dedicated repository - not a fork, with no upstream - built from
scratch and licensed under MIT, and its release line starts from `v1.0.0` rather than from any
number inherited with the code. The version is `1.1`, two components: there is no patch
component in this project.

## What is in it

The repository ships one runtime: a native Swift and Metal engine that streams routed experts
from SSD so a model larger than RAM still runs, plus the cluster layer that runs it across
machines.

**Single node, measured.** Four Mac minis, one binary (`md5 ec6710cd...` on all four), 128
tokens at temperature 0, prompt *"The capital of France is"*:

| node | tok/s |
| --- | --- |
| node1 | 7.935 |
| node2 | 7.937 |
| node3 | 7.884 |
| node4 | 7.146 / 7.448 / 7.396 |

Mean 7.73, and every machine produces the same text.

**Four nodes, measured.** A four-stage layer pipeline runs one model across all four machines
and produces the reference text at **6.016 tok/s** at 128 tokens. It is correct and balanced
within 9% - and it is *slower* than a single node, which this release states rather than
dresses up.

## What this release does not claim

`21 tok/s` is not reachable on this hardware as configured, and the reason is measured rather
than asserted:

- the step is **141.6 ms/token**: **47% expert I/O** and **50% GPU wait**, and the expert I/O is
  already **fully overlapped** - `exposed_io` reads **0.0 ms** on every node;
- a layer pipeline divides **memory, not time**: the head's period of **166.2 ms/token** sits at
  the serial prediction (the sum of the stage times, 194.8 ms) and nowhere near the pipelined
  one (48.7 ms);
- every single-node tuning axis is at its optimum - prefetch depth, the prefetch ring, and the
  expert cache, where **40 slots is the best of seven points** and 64 and above collapse into
  swap;
- the wire is **1 GbE**: 118 MB/s and a 765 us round trip, and UDP measures the same as TCP, so
  it is the link and not the protocol. Pooling expert residency is **8.6x slower than local
  disk**, and tensor parallelism spends **61 ms of synchronisation** against a 47.6 ms target.

**The route to the target is a faster link, not more engine work.** The Thunderbolt and 10GbE
ports on these machines report `status: inactive`; connected, an RTT near 70 us puts the step
near 26.6 ms.

## Verification

- The four single-node runs and the four-stage chain above, each with its timing footer.
- `swift test --no-parallel` on the runtime.
- The release gates: force-cast **ok**, func-length **ok** (0 baselined, 0 new, 2186 scanned),
  unchecked-sendable **ok**, arch-path **ok**.
- A clean scratch release build, and the archive
  `tinytitan-1.1-macos-arm64.tar.gz` at 25,830,098 bytes,
  sha256 `b44e151f3f76b5b92609e7a68ad3b9d1cd2e2defd72e429f3bea7904360c71b4`.
- **No model, dataset or dependency was fetched to make anything pass.**

## Not checked, and named here as the gate requires

- **Every golden baseline is NOT CHECKED.** No install was present under `models/` on the
  machine that cut this release, and none may be fetched to change that:
  `ornith-8`, `ornith-4`, `qwen36-4`, `qwen36-8`, `qwen38-4`, `qwen38-8`, `agentworld-4`,
  `agentworld-8`, `katcoder-4`, `katcoder-8`, `qwen35-2b-4`, `qwen35-2b-8`, `qwen35-4b-4`,
  `qwen35-4b-8`, `qwen35-9b-4`, `qwen35-9b-8`. **This release therefore ships no golden-baseline
  evidence.** The benchmarks quoted above were measured through the CLI against an install
  outside `models/`, which is a measurement, not a golden gate.
- The **converter-expert-order** gate reports `SKIP: No module named 'numpy'` — the converter's
  dependencies are unavailable here, so that gate is not checked either.
