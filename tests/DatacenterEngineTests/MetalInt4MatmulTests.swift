import Foundation
import XCTest

@testable import DatacenterEngine

/// The fused int4 matmul, checked the way every kernel here is: **bit for bit**, against the op it
/// replaces.
///
/// `D107` built the CPU version of this fusion and it lost on speed, but its correctness was never in
/// doubt and that is the order this file keeps: a fast wrong kernel is discarded for the wrong reason.
/// The reference is `InstallFile.dequantizeInt4` followed by `Ops.orderedMatmul` — the engine's own split
/// path, and the definition the contract names.
///
/// **One boundary is pinned rather than hidden.** Apple GPUs flush a denormal *result* to zero, and no
/// `MTLMathMode` changes it: `MetalMatmul` has the same property, and this file asserts both kernels
/// against each other on the case, so a later reader does not discover the difference by being surprised
/// by it. Weights themselves are never denormal (the smallest non-zero code is 1 and a denormal scale is
/// flushed to zero first), so the property cannot be reached through the unpack — only through a product
/// of two small normals, which is the model's regime never.
///
/// CI runners have no GPU, so every test skips when there is no device (`DC-036`).
final class MetalInt4MatmulTests: XCTestCase {
    private struct Random {
        var state: UInt64
        mutating func next() -> UInt64 {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return state
        }
        mutating func unit() -> Float { Float((next() >> 40) & 0xFFFFFF) / Float(1 << 24) }
    }

    /// Codes random, zero points small, and scales drawn from **[0.001, 1)** — normal, and far enough from
    /// the boundary that no product of one with the `x` values below can be denormal. The range is the
    /// point: a random `Float` scale is tiny often enough to reach the one case the GPU cannot reproduce,
    /// and that case has its own test instead of being mixed into this one.
    private func payload(rows: Int, padded: Int, group: Int, seed: UInt64 = 0x2545F4914F6CDD1D) -> Data {
        let groups = rows * (padded / group)
        var random = Random(state: seed)
        var data = Data((0..<(rows * (padded / 2))).map { _ in UInt8((random.next() >> 33) & 0xFF) })
        for _ in 0..<groups {
            let scale: Float = 0.001 + random.unit() * 0.999
            withUnsafeBytes(of: scale.bitPattern.littleEndian) { data.append(contentsOf: $0) }
        }
        for _ in 0..<groups { data.append(UInt8((random.next() >> 33) & 0xFF)) }
        return data
    }

    private func entry(rows: Int, columns: Int, padded: Int, group: Int) throws -> InstallFile.Entry {
        let json = """
        {"name":"synthetic","role":"expert.stack_gate_up","quant":"int4","shape":[\(rows),\(columns)],
         "padded_columns":\(padded),"group":\(group),"dtype":"int4","offset":0,"nbytes":0,"sha256":""}
        """
        return try JSONDecoder().decode(InstallFile.Entry.self, from: Data(json.utf8))
    }

    private func values(_ count: Int, seed: UInt64) -> [Float] {
        var random = Random(state: seed)
        return (0..<count).map { _ in (random.unit() - 0.5) * 4 }
    }

    private func padded(columns: Int, group: Int) -> Int {
        var padded = columns + (group - columns % group) % group
        if padded % 2 == 1 { padded += group }
        return padded
    }

    /// The grid `D107` used, plus the token dimension: every residue mod 4, `k = 1`, groups of 1, 4, 8 and
    /// 64, a partly-filled final group and a padded width wider than the real one, because a kernel that is
    /// right for round shapes is useless for the model's own.
    func testTheFusedGpuMatmulIsBitIdenticalOverAGrid() throws {
        try XCTSkipUnless(MetalInt4Matmul.isAvailable, "no Metal device (CI runners have none)")
        var compared = 0
        for columns in [1, 4, 16, 17, 33, 64, 65, 128, 129] {
            for group in [1, 4, 8, 64] {
                for out in [1, 3, 4, 5, 17] {
                    for tokens in [1, 2] {
                        let width = padded(columns: columns, group: group)
                        let entry = try entry(rows: out, columns: columns, padded: width, group: group)
                        let body = payload(rows: out, padded: width, group: group)
                        let x = values(tokens * columns, seed: 0x1234 &+ UInt64(columns * 31 + out))
                        let weights = try InstallFile.dequantizeInt4(body, entry: entry)
                        let expected = Ops.orderedMatmul(
                            x: x, w: weights, rows: tokens, k: columns, out: out
                        )
                        let fused = try MetalInt4Matmul.matmul(
                            payload: body, entry: entry, x: x, rows: tokens
                        )
                        XCTAssertEqual(
                            fused.map(\.bitPattern), expected.map(\.bitPattern),
                            "columns \(columns) group \(group) out \(out) tokens \(tokens): the kernel moved a bit"
                        )
                        compared += 1
                    }
                }
            }
        }
        XCTAssertGreaterThan(compared, 200, "the grid must actually cover the shapes")
    }

    /// The same assertion against the path the engine actually runs today: the GPU unpack followed by the
    /// contract matmul. A fused kernel that agreed with the definition but not with the pipeline in the
    /// tree would be answering a question nobody asked.
    func testTheFusedKernelAgreesWithTheEnginePath() throws {
        try XCTSkipUnless(MetalInt4Matmul.isAvailable, "no Metal device")
        for (out, columns, group) in [(1024, 2048, 64), (2048, 512, 64), (17, 129, 4)] {
            let width = padded(columns: columns, group: group)
            let entry = try entry(rows: out, columns: columns, padded: width, group: group)
            let body = payload(rows: out, padded: width, group: group)
            let x = values(columns, seed: 0x9876)
            let unpacked = try MetalUnpack.unpack(payload: body, entry: entry)
            let split = Ops.orderedMatmul(x: x, w: unpacked, rows: 1, k: columns, out: out)
            let fused = try MetalInt4Matmul.matmul(payload: body, entry: entry, x: x, rows: 1)
            XCTAssertEqual(
                fused.map(\.bitPattern), split.map(\.bitPattern),
                "out \(out) columns \(columns): the fused kernel disagrees with unpack + matmul"
            )
        }
    }

    /// A zero must carry no sign on this path too, and the case where the code and the zero point cancel is
    /// exactly where the two branches live: `D34` is a rule, not an implementation detail.
    func testAZeroCarriesNoSign() throws {
        try XCTSkipUnless(MetalInt4Matmul.isAvailable, "no Metal device")
        let out = 4, columns = 8, group = 8
        let entry = try entry(rows: out, columns: columns, padded: columns, group: group)
        // All codes zero, all zeros zero, unit scale: every weight is a zero, so every output is the sum
        // of `x * (+0.0)` and must be `+0.0` rather than `-0.0` whatever the sign of x.
        let groups = out * (columns / group)
        var body = Data(repeating: 0, count: out * columns / 2)
        for _ in 0..<groups { withUnsafeBytes(of: Float(1).bitPattern.littleEndian) { body.append(contentsOf: $0) } }
        body.append(Data(repeating: 0, count: groups))
        let x: [Float] = [-1, -1, -1, -1, -1, -1, -1, -1]
        let fused = try MetalInt4Matmul.matmul(payload: body, entry: entry, x: x, rows: 1)
        XCTAssertEqual(
            fused.map(\.bitPattern), [UInt32](repeating: 0, count: out),
            "a zero came back with a sign"
        )
    }

    /// `D11`'s flush, on this path: a denormal scale reads as zero, and the two sides must agree about it.
    func testADenormalScaleReadsAsZero() throws {
        try XCTSkipUnless(MetalInt4Matmul.isAvailable, "no Metal device")
        let out = 1, columns = 8, group = 8
        let entry = try entry(rows: out, columns: columns, padded: columns, group: group)
        var body = Data([0x01]) // code 1 in column 0, so a scale that survives is visible
        body.append(Data(repeating: 0, count: columns / 2 - 1))
        let denormal = Float.leastNormalMagnitude / 2
        withUnsafeBytes(of: denormal.bitPattern.littleEndian) { body.append(contentsOf: $0) }
        body.append(0)
        let x: [Float] = [1, 0, 0, 0, 0, 0, 0, 0]
        let expected = Ops.orderedMatmul(
            x: x, w: try InstallFile.dequantizeInt4(body, entry: entry), rows: 1, k: columns, out: out
        )
        let fused = try MetalInt4Matmul.matmul(payload: body, entry: entry, x: x, rows: 1)
        XCTAssertEqual(fused.map(\.bitPattern), expected.map(\.bitPattern), "the flush rule drifted")
        XCTAssertEqual(fused[0].bitPattern, 0, "a denormal scale must read as zero")
    }

    /// **The boundary, asserted rather than assumed.** A product of two small normals can be denormal, and
    /// an Apple GPU flushes it while the CPU keeps it — for the fused kernel *and* for the kernel that is
    /// already in the tree. The test pins the disagreement so that it is a known property with a name, and
    /// asserts the fused kernel against `MetalMatmul` rather than against the CPU here, because on this
    /// input the device is the common behaviour and the CPU is the outlier.
    func testADenormalProductFlushesOnTheGpu() throws {
        try XCTSkipUnless(MetalInt4Matmul.isAvailable, "no Metal device")
        let tiny = Float.leastNormalMagnitude // a normal, so it survives the flush rule
        let weights: [Float] = [0.5, 0] + [Float](repeating: 0, count: 14)
        let x: [Float] = [tiny, 1]
        let cpu = Ops.orderedMatmul(x: x, w: weights, rows: 1, k: 2, out: 8)
        XCTAssertEqual(cpu[0].bitPattern, (tiny * 0.5).bitPattern, "the CPU must keep the denormal")
        XCTAssertNotEqual(cpu[0].bitPattern, 0)

        let gpuMatmul = try MetalMatmul.matmul(x: x, w: weights, rows: 1, k: 2, out: 8)
        XCTAssertEqual(gpuMatmul[0].bitPattern, 0, "the device flushes a denormal product")

        // The same product through the fused path: code 1 against a scale of 0.5, with x as above.
        let out = 1, columns = 8, group = 8
        let entry = try entry(rows: out, columns: columns, padded: columns, group: group)
        var body = Data([0x01])
        body.append(Data(repeating: 0, count: columns / 2 - 1))
        withUnsafeBytes(of: Float(0.5).bitPattern.littleEndian) { body.append(contentsOf: $0) }
        body.append(0)
        let fused = try MetalInt4Matmul.matmul(
            payload: body, entry: entry, x: [tiny, 0, 0, 0, 0, 0, 0, 0], rows: 1
        )
        XCTAssertEqual(fused[0].bitPattern, 0, "the fused kernel shares the device's flush")
    }

    /// The buffer cache is shared with the unpack and the matmul, and its risk is the **second** call: a
    /// buffer grown for a large shape must not leak into the next small one.
    func testTheSharedBufferCacheDoesNotLeakBetweenShapes() throws {
        try XCTSkipUnless(MetalInt4Matmul.isAvailable, "no Metal device")
        for (out, columns, group) in [(1, 4, 4), (256, 2048, 64), (3, 17, 4), (256, 2048, 64)] {
            let width = padded(columns: columns, group: group)
            let entry = try entry(rows: out, columns: columns, padded: width, group: group)
            let body = payload(rows: out, padded: width, group: group)
            let x = values(columns, seed: 0x5555)
            let weights = try InstallFile.dequantizeInt4(body, entry: entry)
            let expected = Ops.orderedMatmul(x: x, w: weights, rows: 1, k: columns, out: out)
            let fused = try MetalInt4Matmul.matmul(payload: body, entry: entry, x: x, rows: 1)
            XCTAssertEqual(
                fused.map(\.bitPattern), expected.map(\.bitPattern),
                "out \(out) columns \(columns): a reused buffer changed the answer"
            )
        }
    }
}
