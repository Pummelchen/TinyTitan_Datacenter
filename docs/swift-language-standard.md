# The Swift language standard

What this tree is written to, what the compiler enforces, and what was deliberately left for its
own pass. Measured on 2026-09-16 with the toolchain below; the numbers come from builds, not from
reading release notes.

## The baseline

| | |
| --- | --- |
| Toolchain | Swift **6.4** on **Xcode 27** (the manifest requires `swift-tools-version:6.4`) |
| Platform | `macOS(.v26)`, arm64 |
| Language mode | **6**, which is the tools-version default — strict concurrency and the graduated 6.0–6.3 features are already errors, not warnings |
| Diagnostics on a clean tree | **0** warnings, **0** errors, measured with `swift test` |

## Enforced: the four that are still upcoming, declared once

`Package.swift` declares `shardLanguageStandard` and applies it to **all six targets**, so a target
added later cannot quietly opt out:

```swift
.enableUpcomingFeature("InferIsolatedConformances")
.enableUpcomingFeature("ImmutableWeakCaptures")
.enableUpcomingFeature("MemberImportVisibility")
.enableUpcomingFeature("NonisolatedNonsendingByDefault")
```

Each was measured by building with the flag and counting diagnostics. All four cost what is
recorded here, and the two that were not free were paid rather than waved through:

| feature | measured cost |
| --- | --- |
| `InferIsolatedConformances` | 0 diagnostics |
| `ImmutableWeakCaptures` | 0 diagnostics |
| `NonisolatedNonsendingByDefault` | 0 diagnostics |
| `MemberImportVisibility` | **one import, in one test file** — `Qwen3_5ForwardTests.swift` needed `import DatacenterIR` |

`MemberImportVisibility` is worth its own paragraph because it was briefly declared free on a
measurement that was wrong, then declared expensive on a pass that was interrupted. Neither was a
number. The count above came from iterating `swift test` under the flag and adding exactly the
imports the compiler named — one round, one file, then exit 0. **A feature that is not free is not
the same as a feature that is expensive**, and both claims need the loop run to the end.

## Deliberately not enabled: `ExistentialAny`

Measured cost: **fourteen warnings**. It requires `any` on every existential, which is worth doing
and is not free — it is a mechanical pass across the engine's protocol types, and doing it in the
same change as the four above would bury both. It stays off until it is its own commit with its own
count, which is the same rule the rest of this file follows.

## Graduated, and therefore already enforced by the language mode

Probed rather than assumed: building with `-enable-upcoming-feature StrictConcurrency` produces no
diagnostics and the feature is no longer separable, because language mode 6 already enforces it.
The sister project's list of retired names — `ConciseMagicFile`, `ForwardTrailingClosures`,
`BareSlashRegexLiterals`, `IsolatedDefaultValues`, `DisableOutwardActorInference`,
`GlobalActorIsolatedTypesUsability`, `InferSendableFromCaptures` — is the same class of thing.

## The method, so the next person can repeat it

```bash
# `swift test`, not `swift build`: the test targets carry the same swiftSettings, and a feature
# that only troubles them is invisible to `swift build`. That mistake is why this section exists.
swift test --no-parallel > /tmp/log 2>&1; echo $?          # baseline, must be 0
for f in ExistentialAny InferIsolatedConformances ImmutableWeakCaptures \
         MemberImportVisibility NonisolatedNonsendingByDefault; do
  swift test --no-parallel -Xswiftc -enable-upcoming-feature -Xswiftc "$f" > /tmp/log 2>&1
  echo "$f exit=$?"
done
```

Parse the diagnostic paths carefully: this repository's path contains a space, so a regex that
stops at whitespace truncates it and the loop then edits a file that does not exist.

A feature is enabled with a **measured** count, never a confident one — and the trap is that CI
cannot do this for you: the `macos-26` runner's Xcode is below the manifest's floor, so the Swift job
warns and skips. These numbers come from the farm, and they have to be re-measured there.
