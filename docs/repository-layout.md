# Repository layout

The sister project's convention, kept deliberately: `sources/` holds one directory per
SwiftPM target, `tests/` mirrors it path for path, `tools/` holds the gates and the
harness, `docs/` holds the reference contracts and decision records, and the wiki holds
the plan and the status.

```
Package.swift            swift-tools-version 6.4, Swift 6 language mode
sources/
  DatacenterIR/          the model IR: roles, shape contracts, policy, validation,
                         and the per-family importers
tests/
  DatacenterIRTests/     mirrors sources/DatacenterIR; Fixtures/ holds a real
                         checkpoint's tensor inventory as a declared resource
tools/                   the M0 harness and the repository gates, standard-library
                         Python only (trace_format, trace_diff, make_synthetic_trace,
                         trace_capture, check_markdown_links) plus pinned requirements
docs/                    m0-decisions, trace-format, ir-schema, and one
                         reference-<family>.md contract per model family
.github/                 CI: the tool suite and the link gate on every push
```

## Rules that came out of building it

- **Target paths are declared explicitly** in `Package.swift`. SwiftPM's conventional
  directories are `Sources/` and `Tests/` with capitals; this repository uses lower case
  to match the sister project, which resolves silently on a case-insensitive filesystem
  and fails on a case-sensitive one. If it is declared, it cannot surprise anyone on a
  different disk format or on CI.
- **Fixtures are declared resources**, loaded through `Bundle.module` rather than from a
  source path, so they travel with the test bundle.
- **A target is added when its milestone needs it.** There is no engine target yet
  because there is no engine: the IR and its importer are what M0 needs first, and
  speculative structure is the main way a project like this fails.
- **Kernel comments cite a contract.** Every numeric kernel cites its
  `docs/reference-<family>.md` entry (file and line at a pinned revision), not a memory
  of how a model works.

## Naming

The project is **TinyTitan Datacenter** (D7). The package is `TinyTitanDatacenter`,
targets carry the `Datacenter` prefix (`DatacenterIR` today), and the engine's targets
will follow the same prefix so a reader can tell at a glance which project a type
belongs to when both are open.
