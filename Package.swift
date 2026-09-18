// swift-tools-version: 6.4
import PackageDescription

/// The language standard this package is written to.
///
/// Swift 6 language mode is set below (`swiftLanguageModes: [.v6]`); these are
/// the upcoming features that are not yet default in that mode and that the
/// tree is clean under. Enforced here rather than documented, so a target
/// added later cannot quietly opt out. The ones deliberately *not* adopted
/// (and why, with their measured diagnostic counts) are recorded in
/// `docs/swift-language-standard.md`.
let tinytitanLanguageStandard: [SwiftSetting] = [
    .enableUpcomingFeature("InferIsolatedConformances"),
    .enableUpcomingFeature("ImmutableWeakCaptures"),
    .enableUpcomingFeature("MemberImportVisibility"),
]

let package = Package(
    name: "TinyTitan",
    platforms: [
        .macOS(.v26),
    ],
    products: [
        .library(name: "TinyTitan", targets: ["TinyTitan"]),
        .library(name: "TinyTitanFormat", targets: ["TinyTitanFormat"]),
        .library(name: "ContinuityCore", targets: ["ContinuityCore"]),
        .executable(name: "TinyTitanRepack", targets: ["TinyTitanRepack"]),
        .executable(name: "TinyTitanCLI", targets: ["TinyTitanCLI"]),
        .executable(name: "TinyTitanMac", targets: ["TinyTitanMac"]),
        .executable(name: "TinyTitanDecodeService", targets: ["TinyTitanDecodeService"]),
        .executable(name: "TinyTitanServer", targets: ["TinyTitanServer"]),
        .executable(name: "TinyTitanBench", targets: ["TinyTitanBench"]),
        .executable(name: "ContinuityDemo", targets: ["ContinuityDemo"]),
        .executable(name: "tinytitan-memory", targets: ["TinyTitanMemoryTool"]),
    ],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
        .package(url: "https://github.com/apple/swift-nio.git", exact: "2.99.0"),
    ],
    targets: [
        .target(
            name: "TinyTitanFormat",
            path: "sources/TinyTitanFormat",
            swiftSettings: tinytitanLanguageStandard
        ),
        // C99 + NEON for the inner loops where Swift's vector types do not
        // lower well. Kept deliberately small: one file, one entry point,
        // covered by the same tests as the Swift path it replaced. No custom
        // flags -- -O3 measured the same as SwiftPM's release default (0.675
        // vs 0.680 ms), so it is not worth the unsafeFlags constraint.
        .target(
            name: "TinyTitanKernelsC",
            path: "sources/TinyTitanKernelsC",
            swiftSettings: tinytitanLanguageStandard
        ),
        .target(
            name: "TinyTitan",
            dependencies: [
                "TinyTitanFormat",
                "TinyTitanKernelsC",
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "sources/TinyTitan",
            resources: [
                .copy("Metal"),
            ],
            swiftSettings: tinytitanLanguageStandard
        ),
        .target(
            name: "TinyTitanRepackCore",
            dependencies: ["TinyTitanFormat"],
            path: "sources/TinyTitanRepack/Core",
            swiftSettings: tinytitanLanguageStandard
        ),
        .executableTarget(
            name: "TinyTitanRepack",
            dependencies: ["TinyTitanRepackCore"],
            path: "sources/TinyTitanRepack/Command",
            swiftSettings: tinytitanLanguageStandard
        ),
        .target(
            name: "TinyTitanCLICore",
            dependencies: ["TinyTitan", "TinyTitanDecodeProtocol"],
            path: "sources/TinyTitanCLI",
            exclude: ["Command"],
            swiftSettings: tinytitanLanguageStandard
        ),
        .executableTarget(
            name: "TinyTitanCLI",
            dependencies: ["TinyTitanCLICore"],
            path: "sources/TinyTitanCLI/Command",
            swiftSettings: tinytitanLanguageStandard
        ),
        .target(
            name: "TinyTitanAppCore",
            dependencies: ["TinyTitan", "TinyTitanRepackCore", "TinyTitanDecodeProtocol"],
            path: "sources/TinyTitanApp/Core",
            resources: [
                .copy("Resources/app-prompts.json"),
            ],
            swiftSettings: tinytitanLanguageStandard
        ),
        .target(
            name: "TinyTitanMacPresentation",
            dependencies: ["TinyTitanAppCore"],
            path: "sources/TinyTitanApp/MacPresentation",
            swiftSettings: tinytitanLanguageStandard
        ),
        .target(
            name: "TinyTitanDecodeProtocol",
            path: "sources/TinyTitanDecodeProtocol",
            swiftSettings: tinytitanLanguageStandard
        ),
        .executableTarget(
            name: "TinyTitanDecodeService",
            dependencies: ["TinyTitanAppCore", "TinyTitanDecodeProtocol"],
            path: "sources/TinyTitanDecodeService",
            swiftSettings: tinytitanLanguageStandard
        ),
        // Continuity: sessions, task memory and context assembly, in this
        // process. Depends on nothing at all, not even NIO, so it cannot
        // reach the network and cannot be reached from one.
        .target(
            name: "ContinuityCore",
            path: "sources/ContinuityCore",
            // Documentation that lives next to the code it describes. SwiftPM
            // treats any undeclared file under a target path as unhandled and
            // warns on every clean plan; excluding it says so explicitly and
            // leaves the file where it is. (`sources/TinyTitanCLICore`'s
            // `exclude: ["Command"]` is the same mechanism.)
            exclude: ["README.md"],
            swiftSettings: tinytitanLanguageStandard
        ),
        // Worked examples and a scale check for ContinuityCore. Not part of
        // the server; it exists so the package's claims can be run.
        .executableTarget(
            name: "ContinuityDemo",
            dependencies: ["ContinuityCore"],
            path: "sources/ContinuityDemo",
            swiftSettings: tinytitanLanguageStandard
        ),
        // Agent memory: the model-facing surface (keys, tools, prompt
        // fragment, journal filter) over ContinuityCore. Depends on nothing
        // in the engine, so the serving path can use it without the memory
        // subsystem being able to reach back into inference, and on no
        // networking, so it cannot reach off the machine.
        .target(
            name: "TinyTitanMemory",
            dependencies: ["ContinuityCore"],
            path: "sources/TinyTitanMemory",
            swiftSettings: tinytitanLanguageStandard
        ),
        // See and correct what the server remembers: list, show, delete.
        // Reads take no lock; writes need the workspace.
        .executableTarget(
            name: "TinyTitanMemoryTool",
            dependencies: ["TinyTitanMemory", "ContinuityCore"],
            path: "sources/TinyTitanMemoryTool",
            swiftSettings: tinytitanLanguageStandard
        ),
        .target(
            name: "TinyTitanServerCore",
            dependencies: [
                "TinyTitan",
                "TinyTitanMemory",
                .product(name: "NIOCore", package: "swift-nio"),
                .product(name: "NIOPosix", package: "swift-nio"),
                .product(name: "NIOHTTP1", package: "swift-nio"),
            ],
            path: "sources/TinyTitanServer/Core",
            swiftSettings: tinytitanLanguageStandard
        ),
        .executableTarget(
            name: "TinyTitanServer",
            dependencies: ["TinyTitanServerCore"],
            path: "sources/TinyTitanServer/Command",
            swiftSettings: tinytitanLanguageStandard
        ),
        .executableTarget(
            name: "TinyTitanBench",
            dependencies: ["TinyTitan"],
            path: "sources/TinyTitanBench",
            swiftSettings: tinytitanLanguageStandard
        ),
        .executableTarget(
            name: "TinyTitanMac",
            dependencies: ["TinyTitanAppCore", "TinyTitanMacPresentation"],
            path: "sources/TinyTitanApp/Mac",
            resources: [
                .copy("Resources/tinytitan-app-icon.png"),
            ],
            swiftSettings: tinytitanLanguageStandard
        ),
        .target(
            name: "TinyTitanValidationSupport",
            dependencies: ["TinyTitan"],
            path: "sources/TinyTitanValidation/Support",
            swiftSettings: tinytitanLanguageStandard
        ),
        .testTarget(
            name: "TinyTitanTests",
            dependencies: ["TinyTitan", "TinyTitanValidationSupport", "TinyTitanRepackCore", "TinyTitanCLICore"],
            path: "tests/TinyTitan",
            resources: [.copy("Tokenization/Fixtures"),
                        .copy("Runtime/qwen38_tensor_names.txt"),
                        .copy("Runtime/ple_golden.json")],
            swiftSettings: tinytitanLanguageStandard
        ),
        .testTarget(
            name: "TinyTitanRepackTests",
            // `TinyTitanFormat` directly: the manifest and resident-index validation
            // tests assert on those types rather than on JSON dictionaries.
            dependencies: ["TinyTitanRepackCore", "TinyTitanFormat"],
            path: "tests/TinyTitanRepack/Core",
            resources: [.copy("Support/qwen38_tensor_names.txt")],
            swiftSettings: tinytitanLanguageStandard
        ),
        .testTarget(
            name: "TinyTitanAppCoreTests",
            dependencies: ["TinyTitanAppCore", "TinyTitan", "TinyTitanRepackCore", "TinyTitanDecodeProtocol"],
            path: "tests/TinyTitanApp/Core",
            swiftSettings: tinytitanLanguageStandard
        ),
        .testTarget(
            name: "TinyTitanDecodeServiceTests",
            dependencies: ["TinyTitanDecodeService", "TinyTitanAppCore", "TinyTitanDecodeProtocol"],
            path: "tests/TinyTitanDecodeService",
            swiftSettings: tinytitanLanguageStandard
        ),
        .testTarget(
            name: "TinyTitanMacPresentationTests",
            dependencies: ["TinyTitanAppCore", "TinyTitanMacPresentation"],
            path: "tests/TinyTitanApp/MacPresentation",
            swiftSettings: tinytitanLanguageStandard
        ),
        .testTarget(
            name: "ContinuityCoreTests",
            dependencies: ["ContinuityCore"],
            path: "tests/ContinuityCore",
            swiftSettings: tinytitanLanguageStandard
        ),
        .testTarget(
            name: "TinyTitanMemoryTests",
            dependencies: ["TinyTitanMemory", "ContinuityCore"],
            path: "tests/TinyTitanMemory",
            swiftSettings: tinytitanLanguageStandard
        ),
        .testTarget(
            name: "TinyTitanServerTests",
            dependencies: [
                "TinyTitanServerCore",
                "TinyTitanMemory",
                // `GenerationDefaults.Sampling`, so the mapper tests can pin
                // that an omitted field follows the served model's profile
                // rather than a hardcoded house default.
                "TinyTitan",
                .product(name: "NIOEmbedded", package: "swift-nio"),
            ],
            path: "tests/TinyTitanServer",
            resources: [.copy("Fixtures")],
            swiftSettings: tinytitanLanguageStandard
        ),
    ],
    swiftLanguageModes: [.v6]
)
