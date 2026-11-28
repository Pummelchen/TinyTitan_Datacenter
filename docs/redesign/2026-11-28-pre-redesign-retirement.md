# Pre-redesign retirement

## Intent

Retire the existing TinyTitan Datacenter implementation so a new architecture can be designed without inheriting the old runtime as active code. Preserve the old implementation, tests, tools, benchmarks, and engineering records as reference material.

## Decision

Move the existing project into `legacy/pre-redesign/` without rewriting its source. The repository root will contain project-wide identity and legal files, the human-owned `README.md`, the new agent instructions, and `docs/redesign/`. There will be no root `Package.swift` and no active product target until a separate redesign is approved.

The reference remains buildable from `legacy/pre-redesign/` with its historical constraints. It is not supported as the current product and must not receive new feature or optimization work. Historical documents retain their original context; new status belongs in `docs/redesign/` and the wiki.

## Alternatives considered

- Keep the old tree at the root and mark it deprecated: rejected because it leaves the retired implementation indistinguishable from active code.
- Delete the old tree and rely on a Git tag: rejected because the owner explicitly wants the code kept as an easily browsable reference.
- Move the complete old tree under `legacy/pre-redesign/`: selected because it preserves relative paths and build behavior while making retirement explicit.

## Constraints

- Do not edit the human-owned root `README.md`.
- Preserve all tracked reference files and their contents.
- Do not begin the replacement architecture or implementation in this change.
- Keep the unrelated TinyTitan sister checkout and its uncommitted work untouched.
- Publish the retirement to `main` and synchronize the wiki, including its project tracker.

## Verification

Compare every tracked path from the retirement commit's parent with its counterpart under `legacy/pre-redesign/`, excluding intentionally new or changed root instructions and redesign documentation. Confirm the archived manifest still describes the historical package, the root has no active package manifest, and no unrelated files changed. Do not run model-dependent checks or download models.
