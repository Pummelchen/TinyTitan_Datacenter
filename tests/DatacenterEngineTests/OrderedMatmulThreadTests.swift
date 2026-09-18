import XCTest

@testable import DatacenterEngine

/// The contract matmul fans out across output columns (`DC-122`, second attempt).
///
/// The first attempt moved bits and was reverted. The rule that makes this one safe is stated in
/// `Ops.orderedMatmulThreaded` and checked here: **every chunk boundary is a multiple of four**, so a non-final
/// chunk covers whole four-wide groups and reaches its end with no scalar tail, while the final chunk ends at
/// `out` and performs exactly the `out % 4` tail the serial body performs.
///
/// The shapes below are chosen for the regimes a regrouping would show in: `out % 4 != 0` (where a mis-aligned
/// boundary moves the tail), `out < 4` (where the vector loop never runs), `k = 1`, `rows > 1`, and sizes large
/// enough that the production threshold fans out.
final class OrderedMatmulThreadTests: XCTestCase {
    /// Deterministic values with nothing special about them, so agreement is a property and not a coincidence.
    private func values(_ count: Int, seed: Int) -> [Float] {
        var state = UInt64(seed &* 2_654_435_761 &+ 1)
        return (0..<count).map { _ in
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Float(Int32(truncatingIfNeeded: state >> 33)) / Float(1 << 24)
        }
    }

    private func check(rows: Int, k: Int, out: Int, threads: Int, line: UInt = #line) {
        let x = values(rows * k, seed: rows &* 31 &+ k &* 7 &+ out)
        let w = values(out * k, seed: out &* 131 &+ k &* 17 &+ rows)
        let definition = Ops.orderedMatmulScalar(x: x, w: w, rows: rows, k: k, out: out)
        let serial = Ops.orderedMatmulVectorized(x: x, w: w, rows: rows, k: k, out: out)
        let fanned = Ops.orderedMatmulThreaded(x: x, w: w, rows: rows, k: k, out: out, threads: threads)
        XCTAssertEqual(
            serial.map(\.bitPattern), definition.map(\.bitPattern),
            "rows=\(rows) k=\(k) out=\(out): the fast path and the definition already disagree", line: line
        )
        XCTAssertEqual(
            fanned.map(\.bitPattern), serial.map(\.bitPattern),
            "rows=\(rows) k=\(k) out=\(out) threads=\(threads): the fan-out moved a bit", line: line
        )
    }

    func testEveryAwkwardOutputWidthIsBitIdenticalToTheDefinition() {
        // `out` walks every residue mod 4, then sizes where the four-wide loop runs many times with a tail.
        for out in [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 15, 16, 17, 33, 64, 65, 129] {
            for rows in [1, 2, 3] {
                for k in [1, 2, 5, 16] {
                    check(rows: rows, k: k, out: out, threads: 8)
                }
            }
        }
    }

    func testThreadCountsAgreeWithEachOtherAndWithTheSerialBody() {
        for threads in [0, 1, 2, 3, 5, 8, 16] {
            check(rows: 3, k: 12, out: 37, threads: threads)
            check(rows: 1, k: 64, out: 6, threads: threads)
        }
    }

    func testTheShapesTheModelActuallyUsesAreIdentical() {
        // The two shapes the decode path spends its time in: a projection and an expert slice.
        check(rows: 1, k: 2048, out: 4096, threads: 8)
        check(rows: 8, k: 2048, out: 512, threads: 8)
    }

    func testRepeatingTheCallOnTheSameBuffersGivesTheSameBits() {
        // The property that caught the first attempt: a caller that reuses its buffers must not see a different
        // answer. The fan-out reads only, so two calls must agree bit for bit.
        let rows = 2, k = 32, out = 19
        let x = values(rows * k, seed: 5)
        let w = values(out * k, seed: 9)
        let first = Ops.orderedMatmulThreaded(x: x, w: w, rows: rows, k: k, out: out, threads: 8)
        let second = Ops.orderedMatmulThreaded(x: x, w: w, rows: rows, k: k, out: out, threads: 8)
        XCTAssertEqual(first.map(\.bitPattern), second.map(\.bitPattern))
    }

    func testThePublicEntryPointStillHonoursTheWorkThreshold() {
        // Below `DecodeThreads.minimumWork` the public entry point takes the serial body; this asserts the
        // *result* is the definition either way, which is all a caller can observe.
        let rows = 1, k = 8, out = 4
        let x = values(rows * k, seed: 11)
        let w = values(out * k, seed: 13)
        XCTAssertEqual(
            Ops.orderedMatmul(x: x, w: w, rows: rows, k: k, out: out).map(\.bitPattern),
            Ops.orderedMatmulScalar(x: x, w: w, rows: rows, k: k, out: out).map(\.bitPattern)
        )
    }
}
