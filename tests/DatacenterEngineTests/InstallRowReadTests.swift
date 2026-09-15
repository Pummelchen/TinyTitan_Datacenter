import Foundation
import XCTest

import DatacenterIR

@testable import DatacenterEngine

/// Reading **one expert** of a stacked tensor, without decoding the stack.
///
/// The real model's expert tensor is `[256, 1024, 2048]`: decoding it to hand back one expert is
/// 537 M parameters and two gigabytes of `Float` on a node with four and a half. The payload's
/// layout is section-major — every row's codes, then every row's scales, then every row's zeros —
/// so a row range is three bounded reads and one decode, and the tests below pin both halves of
/// that: the values must be the same as the whole-tensor decode, and the bytes must not be.
final class InstallRowReadTests: XCTestCase {
    private func install() throws -> InstallFile {
        let fixture = try XCTUnwrap(
            Bundle.module.url(forResource: "tiny-qwen36", withExtension: nil, subdirectory: "Fixtures")
        )
        return try InstallFile(url: fixture.appendingPathComponent("install"))
    }

    /// The stacked ones, where the leading axis is experts rather than rows.
    private func stackedExperts(_ install: InstallFile) -> [String] {
        install.manifest.tensors
            .filter { $0.dtype == "int4" && $0.shape.count == 3 && $0.shape[0] > 1 }
            .map(\.name)
    }

    func testOneExpertEqualsItsSliceOfTheWholeTensorDecode() throws {
        let install = try install()
        let names = stackedExperts(install)
        XCTAssertFalse(names.isEmpty, "the fixture must carry a stacked int4 tensor")
        for name in names {
            let entry = try install.entry(name)
            let width = entry.shape.dropFirst().reduce(1, *)
            let whole = try install.tensor(named: name)
            for expert in 0..<entry.shape[0] {
                let one = try install.rows(named: name, range: expert..<(expert + 1))
                XCTAssertEqual(
                    one, Array(whole[(expert * width)..<((expert + 1) * width)]),
                    "\(name) expert \(expert): the row read disagreed with the whole decode"
                )
            }
        }
    }

    /// A rank-2 quantized tensor is one leading entry per row, so a one-row range is one row. The
    /// translation that rank-3 needs must not change rank-2.
    func testRankTwoRowReadsStillMatchTheWholeDecode() throws {
        let install = try install()
        let names = install.manifest.tensors
            .filter { $0.dtype == "int4" && $0.shape.count == 2 }
            .map(\.name)
        try XCTSkipIf(names.isEmpty, "the fixture has no rank-2 int4 tensors")
        for name in names.prefix(3) {
            let entry = try install.entry(name)
            let width = entry.shape[1]
            let whole = try install.tensor(named: name)
            let rows = try install.rows(named: name, range: 1..<3)
            XCTAssertEqual(rows, Array(whole[(1 * width)..<(3 * width)]), name)
        }
    }

    /// The measurement. One expert of an eight-expert stack must not cost the stack.
    ///
    /// This test used to warm the entry first, because a cold read hashed the whole payload for
    /// `I6` — 537 MB of expert stack for the real model. With a digest per leading-axis slab the
    /// check covers only the bytes being read, so the warm-up is gone and the assertion is
    /// stronger: **this is the test that catches a guard which stops matching**, because a skipped
    /// slab branch still returns the right values, only dearer.
    func testOneExpertCostsOneExpertNotTheStack() throws {
        let install = try install()
        let name = try XCTUnwrap(stackedExperts(install).first)
        let entry = try install.entry(name)
        let experts = entry.shape[0]

        let before = install.bytesRead
        _ = try install.rows(named: name, range: 1..<2)
        let cost = install.bytesRead - before

        XCTAssertGreaterThan(cost, 0, "the read must actually read")
        XCTAssertLessThan(
            cost, entry.nbytes / 3,
            "one expert of \(experts) read \(cost) of \(entry.nbytes) bytes: the whole stack was decoded"
        )
        // And it should be close to the fair share, not merely under a third.
        XCTAssertLessThan(cost, 2 * entry.nbytes / experts, "the read cost more than twice its share")
    }

    /// The whole-tensor read is unchanged, which is what the row read is measured against.
    func testTheWholeTensorReadStillReturnsEveryValue() throws {
        let install = try install()
        let name = try XCTUnwrap(stackedExperts(install).first)
        let entry = try install.entry(name)
        let whole = try install.tensor(named: name)
        XCTAssertEqual(whole.count, entry.shape.reduce(1, *))
    }

    /// A slab digest catches a tampered expert **and** leaves its neighbours readable.
    ///
    /// The byte flipped is the payload's own **last** byte, which lies in the last section of the
    /// last slab — so the offset comes from the entry rather than from `inner * codesPerRow`
    /// computed by hand. Every hand-computed offset in this task was eventually wrong, and the last
    /// byte of the entry is the one offset that cannot be.
    func testASlabDigestCatchesTheExpertItBelongsToAndNotTheOthers() throws {
        let fixture = try XCTUnwrap(
            Bundle.module.url(forResource: "tiny-qwen36", withExtension: nil, subdirectory: "Fixtures")
        ).appendingPathComponent("install")
        let directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("slab-tamper-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        for file in ["install.json", "data.bin"] {
            try FileManager.default.copyItem(
                at: fixture.appendingPathComponent(file), to: directory.appendingPathComponent(file)
            )
        }
        let clean = try InstallFile(url: fixture)
        let name = try XCTUnwrap(stackedExperts(clean).first)
        let entry = try clean.entry(name)
        let slabs = try XCTUnwrap(entry.slab_sha256, "the fixture must carry per-slab digests")
        XCTAssertGreaterThan(slabs.count, 1)

        var bytes = try Data(contentsOf: directory.appendingPathComponent("data.bin"))
        let at = entry.offset + entry.nbytes - 1
        bytes[at] = bytes[at] ^ 0xFF
        try bytes.write(to: directory.appendingPathComponent("data.bin"))

        let tampered = try InstallFile(url: directory)
        _ = try tampered.rows(named: name, range: 0..<1)  // the first expert is untouched
        XCTAssertThrowsError(try tampered.rows(named: name, range: (slabs.count - 1)..<slabs.count)) { error in
            guard case InstallFile.Error.digestMismatch = error else {
                return XCTFail("expected a digest mismatch, got \(error)")
            }
        }
    }

    func testARowRangeOutsideTheTensorIsRefused() throws {
        let install = try install()
        let name = try XCTUnwrap(stackedExperts(install).first)
        let experts = try install.entry(name).shape[0]
        XCTAssertThrowsError(try install.rows(named: name, range: (experts - 1)..<(experts + 1)))
        XCTAssertThrowsError(try install.rows(named: name, range: -1..<1))
    }
}
