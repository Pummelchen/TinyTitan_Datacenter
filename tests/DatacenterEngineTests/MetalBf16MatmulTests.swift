import Foundation
import Metal
import XCTest

@testable import DatacenterEngine

/// The bf16 matmul, checked the way every kernel here is: **bit for bit**, against the op it replaces.
///
/// The kernel exists because the LM head is stored bf16 and `decodeRaw` widens it to `Float` — 1.017 GB
/// becomes 2.034 GB per token (`D109`). Widening a bf16 is a shift of the top sixteen bits of an fp32, so it
/// cannot round and the only thing that could move a bit is the accumulation, which is `D61`'s spelling. This
/// file is what turns that argument into an assertion.
///
/// CI runners have no GPU, so every test skips when there is no device (`DC-036`).
final class MetalBf16MatmulTests: XCTestCase {
    private func floats(_ count: Int, seed: UInt64) -> [Float] {
        var state = seed
        return (0..<count).map { _ in
            state = state &* 6364136223846793005 &+ 1442695040888963407
            let unit = Float((state >> 40) & 0xFFFFFF) / Float(1 << 24)
            return (unit - 0.5) * 4
        }
    }

    /// bf16 bits for each value, built the way the install stores them: the top half of the fp32, rounded to
    /// nearest-even, which is what `tools/quantize.py` writes for a bf16 role.
    private func bf16(_ values: [Float]) -> Data {
        var data = Data(capacity: values.count * 2)
        for value in values {
            let bits = value.bitPattern
            let rounded = bits &+ 0x7FFF &+ ((bits >> 16) & 1)
            var half = UInt16(truncatingIfNeeded: rounded >> 16)
            withUnsafeBytes(of: &half) { data.append(contentsOf: $0) }
        }
        return data
    }

    /// Decode the same bytes the way the engine's CPU path does, so the reference is the engine's own op.
    private func widened(_ data: Data, count: Int) throws -> [Float] {
        var values = [Float](repeating: 0, count: count)
        try data.withUnsafeBytes { raw in
            let base = raw.baseAddress!
            for index in 0..<count {
                let word = base.loadUnaligned(fromByteOffset: index * 2, as: UInt16.self)
                values[index] = Float(bitPattern: UInt32(UInt16(littleEndian: word)) << 16)
            }
        }
        return values
    }

    func testTheGpuBf16MatmulIsBitIdenticalOverAGrid() throws {
        try XCTSkipUnless(MetalBf16Matmul.isAvailable, "no Metal device (CI runners have none)")
        var compared = 0
        for rows in [1, 2, 3, 5] {
            for k in [1, 2, 3, 4, 5, 8, 17, 64, 257] {
                for out in [1, 2, 3, 4, 5, 8, 17, 129] {
                    let x = floats(rows * k, seed: 0x1234 &+ UInt64(rows))
                    // Round through bf16 first, so the reference and the kernel read identical weights.
                    let weights = try widened(
                        bf16(floats(out * k, seed: 0x5678 &+ UInt64(out))), count: out * k
                    )
                    let expected = Ops.orderedMatmul(x: x, w: weights, rows: rows, k: k, out: out)
                    let gpu = try MetalBf16Matmul.matmul(
                        x: x, w: bf16(weights), rows: rows, k: k, out: out
                    )
                    XCTAssertEqual(
                        gpu.map(\.bitPattern), expected.map(\.bitPattern),
                        "rows \(rows) k \(k) out \(out): widening moved a bit"
                    )
                    compared += 1
                }
            }
        }
        XCTAssertGreaterThan(compared, 200, "the grid must actually cover the shapes")
    }

    /// The head's real shape, a block at a time: `k = 2048` and a block of vocabulary rows.
    func testTheHeadBlockShapeMatchesTheScalarOp() throws {
        try XCTSkipUnless(MetalBf16Matmul.isAvailable, "no Metal device")
        let rows = 1, k = 2048, out = 512
        let x = floats(rows * k, seed: 0xABCD)
        let weights = try widened(bf16(floats(out * k, seed: 0xEF01)), count: out * k)
        let expected = Ops.orderedMatmul(x: x, w: weights, rows: rows, k: k, out: out)
        let gpu = try MetalBf16Matmul.matmul(x: x, w: bf16(weights), rows: rows, k: k, out: out)
        XCTAssertEqual(gpu.map(\.bitPattern), expected.map(\.bitPattern), "the head's shape moved a bit")
    }

    /// The buffer cache is shared with the unpack and the other matmul, and its risk is the **second** call
    /// across shapes: a buffer grown for a large block must not leak into the next small one.
    func testTheSharedBufferCacheDoesNotLeakBetweenShapes() throws {
        try XCTSkipUnless(MetalBf16Matmul.isAvailable, "no Metal device")
        for (rows, k, out) in [(1, 3, 2), (1, 2048, 512), (2, 5, 3), (1, 2048, 512)] {
            let x = floats(rows * k, seed: 0x1111)
            let weights = try widened(bf16(floats(out * k, seed: 0x2222)), count: out * k)
            let expected = Ops.orderedMatmul(x: x, w: weights, rows: rows, k: k, out: out)
            let gpu = try MetalBf16Matmul.matmul(x: x, w: bf16(weights), rows: rows, k: k, out: out)
            XCTAssertEqual(
                gpu.map(\.bitPattern), expected.map(\.bitPattern),
                "rows \(rows) k \(k) out \(out): a reused buffer changed the answer"
            )
        }
    }

    /// A shape mismatch is refused rather than read past the end — a wrong width is the one failure that
    /// multiplies cleanly and means nothing.
    func testAShortWeightBufferIsRefused() throws {
        try XCTSkipUnless(MetalBf16Matmul.isAvailable, "no Metal device")
        XCTAssertThrowsError(
            try MetalBf16Matmul.matmul(x: [1, 2], w: Data([0, 0, 0, 0]), rows: 1, k: 2, out: 2)
        )
    }
}
