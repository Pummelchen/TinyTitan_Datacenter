import XCTest

@testable import DatacenterEngine

/// The measured-traffic counter that M1's gate depends on.
///
/// `datacenter-trace` reported `expertElementsRead * 2` as "bytes from the SSD" — an estimate that
/// assumes bf16 on disk and therefore overstates a **4-bit** install by about 3.5× (this model
/// stores experts at 0.578 bytes per weight: 4-bit codes with group-64 scales and zeros). The
/// install had been counting what it actually read all along, so the fix was to report that.
///
/// The second test is the one worth having beyond the fix: a forward must read a **slice** of the
/// install, not the file. That is the streaming claim, and it is also the class of bug this project
/// has hit before — a guard that made the reader touch a whole entry to answer a question about one
/// expert (`DC-088`).
final class SourceBytesTests: XCTestCase {
    private func install() throws -> URL {
        try XCTUnwrap(
            Bundle.module.url(forResource: "tiny-qwen36", withExtension: nil, subdirectory: "Fixtures")
        ).appendingPathComponent("install")
    }

    func testAForwardReportsTheBytesItActuallyRead() throws {
        let forward = try ModelLoader.open(snapshot: try install())
        let before = forward.sourceBytesRead
        let result = try forward.forwardWithDecisions(tokens: [1, 2, 3])
        let read = forward.sourceBytesRead - before

        XCTAssertGreaterThan(
            result.expertElementsRead, 0, "the fixture must route experts, or this test proves nothing"
        )
        XCTAssertGreaterThan(read, 0, "the install counts the payload bytes it hands out")
        XCTAssertEqual(
            forward.sourceBytesRead, before + read,
            "the counter only grows, so a forward's traffic is measurable as a difference across it"
        )
    }

    func testAForwardDoesNotReadTheWholeInstall() throws {
        let install = try install()
        let forward = try ModelLoader.open(snapshot: install)
        let before = forward.sourceBytesRead
        _ = try forward.forwardWithDecisions(tokens: [1, 2, 3])
        let read = forward.sourceBytesRead - before
        let whole = (try FileManager.default.attributesOfItem(
            atPath: install.appendingPathComponent("data.bin").path
        )[.size] as? Int) ?? 0

        XCTAssertGreaterThan(whole, 0, "the fixture install has a payload to read")
        XCTAssertLessThan(
            read, whole,
            "a forward read \(read) of \(whole) bytes: streaming means reading a slice, not the file"
        )
    }

    /// A family that does not count must say so with a zero rather than pretending, and the
    /// protocol requirement is what makes that visible through `any ForwardPass`.
    func testAnUncountedSourceReportsZeroRatherThanAFabricatedNumber() throws {
        let checkpoint = try XCTUnwrap(
            Bundle.module.url(forResource: "tiny-qwen36", withExtension: nil, subdirectory: "Fixtures")
        )
        let forward = try ModelLoader.open(snapshot: checkpoint)
        XCTAssertEqual(
            forward.sourceBytesRead, 0,
            "the safetensors path does not count reads, and zero means not counted"
        )
    }

    /// Slab verification is **opt-in**, and this pins both halves of the decision.
    ///
    /// On the real 35 B model it cost **21.04 s of a 39.8 s forward** — 82% of the expert fetch —
    /// because it re-hashes every expert payload the model reads, while the disk itself was 0.82 s.
    /// Integrity is established out of band (per-tensor and per-slab digests in the manifest,
    /// `tools/quantize.py verify` for the whole install), so the default reader must not hash, and a
    /// caller who asks must be able to see what it cost.
    func testSlabVerificationIsOptInAndItsCostIsVisible() throws {
        let install = try install()
        let manifest = try JSONSerialization.jsonObject(
            with: Data(contentsOf: install.appendingPathComponent("install.json"))
        ) as? [String: Any]
        let tensors = try XCTUnwrap(manifest?["tensors"] as? [[String: Any]])
        let stacked = try XCTUnwrap(
            tensors.first { $0["role"] as? String == "expert.stack_gate_up" },
            "the fixture must have a stacked expert tensor"
        )
        let name = try XCTUnwrap(stacked["name"] as? String)

        let unverified = try InstallFile(url: install)
        _ = try unverified.rows(named: name, range: 0..<1)
        XCTAssertEqual(
            unverified.sourceTiming.digestSeconds, 0,
            "the default reader must not hash the hot path; that cost is what made a forward 40 s"
        )
        XCTAssertGreaterThan(unverified.bytesReadFromSource, 0, "it still read the payload")

        let verified = try InstallFile(url: install, verifySlabs: true)
        _ = try verified.rows(named: name, range: 0..<1)
        XCTAssertGreaterThan(
            verified.sourceTiming.digestSeconds, 0,
            "asking for verification must verify, and the cost must be visible rather than asserted"
        )
    }
}
