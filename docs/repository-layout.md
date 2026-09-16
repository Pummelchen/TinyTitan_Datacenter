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
  DatacenterEngine/      the numeric contract's ops, the safetensors reader, the
                         qwen3 forward driven by the IR, and the trace writer
  DatacenterTrace/       the `datacenter-trace` executable: run a checkpoint and
                         write a trace the differ can compare
tests/
  DatacenterIRTests/     mirrors sources/DatacenterIR; Fixtures/ holds a real
                         checkpoint's tensor inventory as a declared resource
  DatacenterEngineTests/ op-level contract vectors (bit patterns) and the
                         safetensors reader against a 296-byte fixture
tools/                   the M0 harness and the repository gates, standard-library
                         Python only (trace_format, trace_diff, make_synthetic_trace,
                         check_markdown_links, check_markdown_tables, check_toolchain,
                         check_disk_headroom, disk_watchdog, run_m2_gate) plus the
                         venv-run pieces
                         (trace_capture, ordered_reference, make_contract_vectors,
                         make_safetensors_fixture, check_engine_contract) and pinned
                         requirements
docs/                    m0-decisions, m1-decisions, m2-decisions, trace-format,
                         ir-schema, and one reference-<family>.md contract per model
                         family. Decision records are per phase and are written before
                         the code that depends on them.
.github/                 CI: the tool suite, the two Markdown gates, and — on a runner
                         that meets the toolchain requirement, failing otherwise — the
                         Swift build and test
```

## Rules that came out of building it

- **Target paths are declared explicitly** in `Package.swift`. SwiftPM's conventional
  directories are `Sources/` and `Tests/` with capitals; this repository uses lower case
  to match the sister project, which resolves silently on a case-insensitive filesystem
  and fails on a case-sensitive one. If it is declared, it cannot surprise anyone on a
  different disk format or on CI.
- **Fixtures are declared resources**, loaded through `Bundle.module` rather than from a
  source path, so they travel with the test bundle.
- **A target declares its dependencies.** `DatacenterEngine` imported `DatacenterIR`
  without declaring it, which the debug build resolved by accident through the module
  search path and the release build refused outright (`unable to resolve module
  dependency`). A debug-only build is not a build.
- **A target is added when its milestone needs it.** The engine target arrived with M0, and each later addition followed one: the trace and generate executables with M0, `UncachedFile` and the install reader with the streaming work, `MetalUnpack` with the first Metal kernel.
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
