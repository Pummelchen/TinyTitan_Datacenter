// swift-tools-version:6.4
// TinyTitan Datacenter — the cluster engine's tooling and runtime.
//
// Swift 6.4 on Xcode 27, Swift 6 language mode (the tools version is 6.4, so the
// default language mode is 6 — the language-feature register is DC-015).
// `sources/` holds one directory per target and `tests/` mirrors it, which is the
// sister project's convention (docs/repository-layout.md is DC-012).

import PackageDescription

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
        .target(name: "DatacenterIR", path: "sources/DatacenterIR"),
        .testTarget(
            name: "DatacenterIRTests",
            dependencies: ["DatacenterIR"],
            path: "tests/DatacenterIRTests",
            resources: [.copy("Fixtures")]
        ),
        .target(name: "DatacenterEngine", path: "sources/DatacenterEngine"),
        .testTarget(
            name: "DatacenterEngineTests",
            dependencies: ["DatacenterEngine"],
            path: "tests/DatacenterEngineTests",
            resources: [.copy("Fixtures")]
        )
    ]
)
