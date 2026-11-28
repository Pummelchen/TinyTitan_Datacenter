# TinyTitan Datacenter

TinyTitan Datacenter (**TTDD**) is between architectures. The previous implementation was retired on 2026-11-28 and is preserved only as a reference under `legacy/pre-redesign/`. A replacement architecture has not yet been approved or implemented.

## Naming

- **TT** means TinyTitan, the single-Mac sister project.
- **TTDD** means TinyTitan Datacenter, this repository.
- Use these exact abbreviations in new code, documentation, tasks, and discussions. Historical `ttd` references inside the archive are preserved, not current naming.

## Repository state

- There is intentionally no root `Package.swift` and no active product, build, test, or release target.
- `legacy/pre-redesign/` is a complete snapshot of the former repository layout. Its historical instructions, manifests, tests, tools, benchmarks, and documents remain in their original relative locations.
- Do not add features, optimizations, or routine maintenance to the legacy implementation. Consult it for measurements, design lessons, protocol details, and prior art only.
- Run historical code only when a task explicitly requires a reference experiment. Run it from `legacy/pre-redesign/` and follow `legacy/pre-redesign/AGENTS.md`; model-dependent checks still require already installed models.
- New design work belongs in `docs/redesign/`. It must not be presented as implemented behavior.
- Do not create a replacement scaffold or choose a new architecture unless the user asks for that work.

## Human-owned README

`README.md` is written and maintained by the human owner. No AI agent may change it, including its badges. If it becomes inconsistent with the repository state, report that and leave it untouched.

## Related project

The single-Mac TinyTitan repository (**TT**) is a separate sister project. It may be studied for design lessons, but TTDD remains standalone unless a future approved design explicitly decides otherwise.

## Completion and publication

For implementation work: code, test, audit, document, and update the project tracker. Keep changes on `main`, commit and push completed major tasks, and synchronize the GitHub wiki after publishing. Do not create release tags or publish legacy release artifacts as part of redesign work.
