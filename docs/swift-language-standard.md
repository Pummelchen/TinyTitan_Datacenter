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

## Enforced: the three that are still upcoming, declared once

`Package.swift` declares `shardLanguageStandard` and applies it to **all six targets**, so a target
added later cannot quietly opt out:

```swift
.enableUpcomingFeature("InferIsolatedConformances")
.enableUpcomingFeature("ImmutableWeakCaptures")
.enableUpcomingFeature("NonisolatedNonsendingByDefault")
```

Each was measured by building with the flag and counting diagnostics. All three cost **zero**, which
is why they are on: an "upcoming feature" that costs nothing today is a migration paid for now
instead of at a compiler upgrade.

**`MemberImportVisibility` was in this list for part of one commit, on a measurement that was
wrong.** It was probed with `swift build`, which does not compile the test targets — and that is
where it bites, because a test file that uses `DatacenterIR`'s properties has to say so. `swift
test` fails to build until those imports are added, and one pass did not converge. It is off, its
cost is *not yet counted*, and the correction is recorded here rather than the claim being quietly
narrowed. The method below now says `swift test` for exactly this reason.

## Deliberately not enabled: `MemberImportVisibility`

Cost so far: `swift test` fails to build in the test targets until their `DatacenterIR`-defined
uses are imported explicitly, and the count of those sites is **not yet measured** — the feature was
dropped mid-pass rather than left in a state where the suite does not build. It is a mechanical pass,
like the one below, and it belongs to its own commit.

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

A feature is enabled with a **measured** count, never a confident one — and the trap is that CI
cannot do this for you: the `macos-26` runner's Xcode is below the manifest's floor, so the Swift job
warns and skips. These numbers come from the farm, and they have to be re-measured there.
