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
    /// The first read of an entry also hashes its payload — that is `I6`, and it is a deliberate
    /// one-time cost — so the entry is warmed before the measured read.
    func testOneExpertCostsOneExpertNotTheStack() throws {
        let install = try install()
        let name = try XCTUnwrap(stackedExperts(install).first)
        let entry = try install.entry(name)
        let experts = entry.shape[0]

        _ = try install.rows(named: name, range: 0..<1)  // warms the digest for this entry

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

    func testARowRangeOutsideTheTensorIsRefused() throws {
        let install = try install()
        let name = try XCTUnwrap(stackedExperts(install).first)
        let experts = try install.entry(name).shape[0]
        XCTAssertThrowsError(try install.rows(named: name, range: (experts - 1)..<(experts + 1)))
        XCTAssertThrowsError(try install.rows(named: name, range: -1..<1))
    }
}
