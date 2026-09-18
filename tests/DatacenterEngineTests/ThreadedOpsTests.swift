import XCTest

@testable import DatacenterEngine

/// The threaded element-wise passes (`D99`).
///
/// `D94` threaded the int4 unpack; this threads `decodeRaw`, which is the LM head's bf16 conversion — 508 M
/// elements, single-threaded, on every token, for 1.0 s of a 4.05 s step.
///
/// It is a **map**: every element is a pure function of its own bytes, so the test that matters is a
/// bit-for-bit comparison against the sequential result, in the same form `Int4UnpackTests` uses for the
/// unpack.
///
/// **A threaded matmul was attempted here and reverted.** The parallel version moved bits — the repository's
/// own reused-buffer comparison caught it — and a numeric change that cannot be justified is not a speed-up.
/// `Ops.orderedMatmulVectorized` is the original single-threaded body verbatim, and `DC-122` carries the
/// question with the failure on the record.
final class ThreadedOpsTests: XCTestCase {
    func testAThreadedRawDecodeIsABitForBitMap() throws {
        let count = 300_000
        // A payload with the values a "close enough" decoder loses, at both ends.
        var words: [UInt16] = (0..<count).map { UInt16(truncatingIfNeeded: $0 &* 2654435761) }
        words[0] = 0x0000                     // +0.0
        words[1] = 0x8000                     // -0.0
        words[2] = 0x7F80                     // +inf
        words[3] = 0x7FC1                     // a NaN with a payload
        let bf16 = words.withUnsafeBufferPointer { Data(buffer: $0) }
        let decoded = try InstallFile.decodeRaw(bf16, dtype: "bf16", elementCount: count)
        let expected = words.map { Float(bitPattern: UInt32(UInt16(littleEndian: $0)) << 16) }
        XCTAssertEqual(
            decoded.map(\.bitPattern), expected.map(\.bitPattern),
            "the threaded bf16 decode is a map; it must not move a bit, including on zeros and NaNs"
        )

        // And the fp32 identity path, which is a pure copy.
        var floats: [UInt32] = (0..<count).map { UInt32(truncatingIfNeeded: $0 &* 2246822519) }
        floats[0] = 0x8000_0000
        let fp32 = floats.withUnsafeBufferPointer { Data(buffer: $0) }
        let copied = try InstallFile.decodeRaw(fp32, dtype: "fp32", elementCount: count)
        XCTAssertEqual(copied.map(\.bitPattern), floats.map { Float(bitPattern: $0) }.map(\.bitPattern))
    }

    func testTheKnobIsSaneAndSmallWorkNeverFansOut() {
        XCTAssertGreaterThanOrEqual(DecodeThreads.count, 1)
        XCTAssertFalse(
            DecodeThreads.wantsParallelism(work: DecodeThreads.minimumWork - 1),
            "a dispatch costs more than the work below the threshold"
        )
        XCTAssertTrue(DecodeThreads.wantsParallelism(work: DecodeThreads.minimumWork))
    }
}
