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

## Enforced: the five that are still upcoming, declared once

`Package.swift` declares `shardLanguageStandard` and applies it to **all six targets**, so a target
added later cannot quietly opt out:

```swift
.enableUpcomingFeature("InferIsolatedConformances")
.enableUpcomingFeature("ImmutableWeakCaptures")
.enableUpcomingFeature("ExistentialAny")
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
| `ExistentialAny` | **four `any` keywords**, all in one file, the Metal kernel |

`MemberImportVisibility` is worth its own paragraph because it was briefly declared free on a
measurement that was wrong, then declared expensive on a pass that was interrupted. Neither was a
number. The count above came from iterating `swift test` under the flag and adding exactly the
imports the compiler named — one round, one file, then exit 0. **A feature that is not free is not
the same as a feature that is expensive**, and both claims need the loop run to the end.

## A correction that matters more than the feature

`ExistentialAny` was recorded here as costing **fourteen warnings** and was left for its own pass.
That number was wrong. The real count is **four** — four `any` keywords, in one file — and it is
enabled.

The mistake was not arithmetic. It was that **an incremental build does not re-emit warnings**: the
fourteen came from a `swift build` whose output happened to include unrelated diagnostics, and every
measurement after it was taken against a warm build that printed nothing at all, which read as a
clean tree. A diagnostic count is only meaningful after touching the sources, so the method below
now does that first — and the same trap will hide a regression from anyone who reads a quiet build
as a clean one.

Worth noting for the next feature too: this is the second time a cost in this register was wrong in
the same direction — first overstated (`MemberImportVisibility` at "expensive", actually one import),
then understated (a quiet build read as zero). Both were guesses wearing numbers.

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
touch sources/DatacenterEngine/*.swift sources/DatacenterIR/*.swift   # or the build re-emits nothing
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
