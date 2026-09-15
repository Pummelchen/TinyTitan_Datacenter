# Contributing to TinyTitan Datacenter

Thanks for looking. The engine runs, M0's gate is recorded as passing, and M1's
correctness claim holds on the real 35B checkpoint for one prompt of the frozen set — so
the most useful contributions right now are design review, corrections to the
documentation, measurements from hardware this project does not have, and anything that
closes a gap `docs/m1-gate.md` names as unmeasured.

## Before you write code

Open an issue or a [discussion](https://github.com/Pummelchen/TinyTitan_Datacenter/discussions)
first. The plan is deliberately gated: every phase ends with a measurement, and the
next phase assumes the previous gate passed. A pull request that adds a feature
without the phase it belongs to is hard to review, and one that adds a capability
whose gate cannot be run is harder still.

The [Roadmap](https://github.com/Pummelchen/TinyTitan_Datacenter/wiki/Roadmap) says
what each phase is and what closes it; the
[Project Tracker](https://github.com/Pummelchen/TinyTitan_Datacenter/wiki/Project-Tracker)
says what is planned, in progress, blocked or open.

## What a pull request must contain

1. **What and why.** The change, and the problem it solves.
2. **The task and phase.** The `DC-nnn` id from the tracker, or a new one proposed in
   the PR description.
3. **How it was verified.** The command you ran and its output. "Works" is not
   evidence; a pasted result is. If a claim is not measured, say that it is not.
4. **The documentation update.** A change to behaviour, the plan or a decision
   updates the README, the affected wiki page and the tracker in the same pull
   request. Reviews ask for this, and it is the most common reason a PR is sent back.

Use the pull request template; it mirrors those four points.

## Gates

Run these before opening a pull request:

```bash
python3 tools/check_markdown_links.py --verbose
python3 -m unittest discover -s tools
```

Once code exists, the gates grow to match the sister project's: a warning-free
release build, the lint gates, the full test suite, and byte-identical golden
baselines. A gate is never quietly weakened — if one cannot be met, say so in the PR
and in the tracker, with the measurement that forced it. That record is the point.

## Style

- **Swift 6.4 on Xcode 27** (when code lands): `swift-tools-version:6.4` and the
  Swift 6 language mode. `sources/` holds one directory per target, `tests/` mirrors
  it path for path, and no architectural change is made to an imported model — all
  speed comes from sharding, expert repacking and quantization.
- **Markdown**: wrapped, tables for structure, no trailing whitespace.
- **Commits**: an imperative subject line, and a body that explains *why*.

## Licence

This project is MIT, and contributions are accepted under the same terms
(inbound = outbound). Code reused from the sister project
[TinyTitan](https://github.com/Pummelchen/TinyTitan) is Apache-2.0: it keeps its
licence, and a `NOTICE` entry is required. `DC-013` tracks that review; if your
change reuses anything from there, say so in the pull request.

## Security

Do not open a public issue for a suspected vulnerability. See
[SECURITY.md](SECURITY.md).
