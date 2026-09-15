// swift-tools-version:6.4
// TinyTitan Datacenter — the cluster engine's tooling and runtime.
//
// Swift 6.4 on Xcode 27, Swift 6 language mode (the tools version is 6.4, so the
// default language mode is 6 — the language-feature register is DC-015).
// `sources/` holds one directory per target and `tests/` mirrors it, which is the
// sister project's convention (docs/repository-layout.md is DC-012).

import PackageDescription

/// The Swift language standard for this tree (`DC-015`).
///
/// Declared once and applied to **every** target, so a target added later cannot quietly opt out.
/// Measured on this toolchain with `swift test`, which is the only command that compiles the test
/// targets too: the tree is clean of diagnostics, and each of these four costs what is recorded
/// beside it in `docs/swift-language-standard.md`. `MemberImportVisibility` cost **one import in one
/// test file** and `ExistentialAny` cost **four `any` keywords**, both counted by iterating the
/// build rather than estimated — and both counts were wrong the first time, for the same reason:
/// an incremental build does not re-emit warnings.
let shardLanguageStandard: [SwiftSetting] = [
    .enableUpcomingFeature("InferIsolatedConformances"),
    .enableUpcomingFeature("ImmutableWeakCaptures"),
    .enableUpcomingFeature("ExistentialAny"),
    .enableUpcomingFeature("MemberImportVisibility"),
    .enableUpcomingFeature("NonisolatedNonsendingByDefault"),
]

let package = Package(
    name: "TinyTitanDatacenter",
    platforms: [
        // The farm runs macOS 27; requiring 26 leaves room for an older node without
        // pretending the newer APIs are available.
        .macOS(.v26)
    ],
    products: [
        .library(name: "DatacenterIR", targets: ["DatacenterIR"]),
        .library(name: "DatacenterEngine", targets: ["DatacenterEngine"])
    ],
    targets: [
        // Paths are declared rather than inferred. The conventional SwiftPM names are
        // `Sources/` and `Tests/` with capitals, and this repository uses lower case
        // to mirror the sister project — which resolves silently on a case-insensitive
        // filesystem and fails on a case-sensitive one. Being explicit removes the
        // question instead of relying on the developer's disk format.
        .target(name: "DatacenterIR", path: "sources/DatacenterIR",
            swiftSettings: shardLanguageStandard
        ),
        .testTarget(
            name: "DatacenterIRTests",
            dependencies: ["DatacenterIR"],
            path: "tests/DatacenterIRTests",
            resources: [.copy("Fixtures")],
            swiftSettings: shardLanguageStandard
        ),
        .target(name: "DatacenterEngine", dependencies: ["DatacenterIR"], path: "sources/DatacenterEngine",
            swiftSettings: shardLanguageStandard
        ),
        .testTarget(
            name: "DatacenterEngineTests",
            dependencies: ["DatacenterEngine"],
            path: "tests/DatacenterEngineTests",
            resources: [.copy("Fixtures")],
            swiftSettings: shardLanguageStandard
        ),
        .executableTarget(
            name: "datacenter-trace",
            dependencies: ["DatacenterEngine", "DatacenterIR"],
            path: "sources/DatacenterTrace",
            swiftSettings: shardLanguageStandard
        ),
        .executableTarget(
            name: "datacenter-generate",
            dependencies: ["DatacenterEngine", "DatacenterIR"],
            path: "sources/DatacenterGenerate",
            swiftSettings: shardLanguageStandard
        )
    ]
)
