import Foundation
import Metal
import XCTest

@testable import DatacenterEngine

/// The GPU matmul, checked the way `D10` requires: **bit for bit**, against the op it replaces.
///
/// The shapes matter more than the size. `Ops.orderedMatmul` runs its vector across the **output** dimension
/// four at a time, so any output count that is not a multiple of four exercises a tail, and any `k` at all
/// exercises the accumulation order. Both are in the grid deliberately: a kernel that is right for
/// `out = 64, k = 64` and wrong for `out = 17, k = 3` would be useless, because the head's blocks and the
/// DeltaNet's contractions are not round numbers.
final class MetalMatmulTests: XCTestCase {
    private func values(_ count: Int, seed: UInt64 = 0x2545F4914F6CDD1D) -> [Float] {
        var state = seed
        return (0..<count).map { _ in
            state = state &* 6364136223846793005 &+ 1442695040888963407
            let unit = Float((state >> 40) & 0xFFFFFF) / Float(1 << 24)
            return (unit - 0.5) * 4
        }
    }

    func testTheGpuMatmulIsBitIdenticalOverAGrid() throws {
        try XCTSkipUnless(MetalMatmul.isAvailable, "no Metal device (CI runners have none)")
        var compared = 0
        for rows in [1, 2, 3, 5] {
            for k in [1, 2, 3, 4, 5, 8, 17, 64, 257] {
                for out in [1, 2, 3, 4, 5, 8, 17, 129] {
                    let x = values(rows * k, seed: 0x1234 &+ UInt64(rows))
                    let w = values(out * k, seed: 0x5678 &+ UInt64(out))
                    let expected = Ops.orderedMatmul(x: x, w: w, rows: rows, k: k, out: out)
                    let gpu = try MetalMatmul.matmul(x: x, w: w, rows: rows, k: k, out: out)
                    XCTAssertEqual(
                        gpu.map(\.bitPattern), expected.map(\.bitPattern),
                        "rows \(rows) k \(k) out \(out): the GPU moved a bit"
                    )
                    compared += 1
                }
            }
        }
        XCTAssertGreaterThan(compared, 200, "the grid must actually cover the shapes")
    }

    /// The head's real shape, smaller in the vocabulary dimension: a single block of it, at the width the
    /// engine actually uses for `k`.
    func testTheHeadShapeMatchesTheScalarOp() throws {
        try XCTSkipUnless(MetalMatmul.isAvailable, "no Metal device")
        let rows = 5, k = 2048, out = 512
        let x = values(rows * k, seed: 0xABCD)
        let w = values(out * k, seed: 0xEF01)
        let expected = Ops.orderedMatmul(x: x, w: w, rows: rows, k: k, out: out)
        let gpu = try MetalMatmul.matmul(x: x, w: w, rows: rows, k: k, out: out)
        XCTAssertEqual(gpu.map(\.bitPattern), expected.map(\.bitPattern), "the head's shape moved a bit")
    }

    /// The buffer cache is shared with the unpack and its risk is the **second** call: a buffer grown for a
    /// large shape must not leak into the next small one, and a small one must not be reused for a large
    /// shape without growing.
    func testTheSharedBufferCacheDoesNotLeakBetweenShapes() throws {
        try XCTSkipUnless(MetalMatmul.isAvailable, "no Metal device")
        // Deliberately small, large, small, large: the repeat is the point.
        for (rows, k, out) in [(1, 3, 2), (5, 2048, 512), (2, 5, 3), (5, 2048, 512)] {
            let x = values(rows * k, seed: 0x1111)
            let w = values(out * k, seed: 0x2222)
            let expected = Ops.orderedMatmul(x: x, w: w, rows: rows, k: k, out: out)
            let gpu = try MetalMatmul.matmul(x: x, w: w, rows: rows, k: k, out: out)
            XCTAssertEqual(
                gpu.map(\.bitPattern), expected.map(\.bitPattern),
                "rows \(rows) k \(k) out \(out): a reused buffer changed the answer"
            )
        }
    }

    /// The chooser must be indistinguishable from the op it chooses between. Under the default it is
    /// `Ops.orderedMatmul` called directly — `MetalMatmul.enabled` is a `static let` read once, so the test
    /// process cannot flip it — and with `SHARD_GPU_MATMUL=1` in the environment this same test exercises the
    /// GPU branch instead. Either way the assertion is the only one that matters: the *result* is the same
    /// bits, which is what makes a fallback safe rather than merely convenient.
    func testTheChooserIsIndistinguishableFromTheOpItChooses() throws {
        for (rows, k, out) in [(1, 3, 2), (5, 2048, 512), (2, 17, 4), (5, 2048, 8192)] {
            let x = values(rows * k, seed: 0x3333)
            let w = values(out * k, seed: 0x4444)
            let chosen = MetalMatmul.ordered(x: x, w: w, rows: rows, k: k, out: out)
            let expected = Ops.orderedMatmul(x: x, w: w, rows: rows, k: k, out: out)
            XCTAssertEqual(
                chosen.map(\.bitPattern), expected.map(\.bitPattern),
                "rows \(rows) k \(k) out \(out): the chooser changed the answer"
            )
        }
    }

    /// A degenerate shape is refused rather than answered: a silent zero for a mismatched buffer would look
    /// like a numerical result.
    func testShapesAreCheckedRatherThanTrusted() throws {
        try XCTSkipUnless(MetalMatmul.isAvailable, "no Metal device")
        XCTAssertThrowsError(try MetalMatmul.matmul(x: [1, 2, 3], w: [1, 2], rows: 2, k: 2, out: 1))
        XCTAssertThrowsError(try MetalMatmul.matmul(x: [1, 2], w: [1, 2, 3], rows: 1, k: 2, out: 2))
        // An empty result is empty, not an error: the engine asks for zero rows on an empty prompt.
        XCTAssertEqual(try MetalMatmul.matmul(x: [], w: [1, 2], rows: 0, k: 2, out: 1), [])
    }
}
