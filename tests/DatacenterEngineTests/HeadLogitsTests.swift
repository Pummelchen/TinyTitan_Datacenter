import XCTest

@testable import DatacenterEngine

/// The head's vocabulary blocks fan out (`DC-122`, the safe half).
///
/// The general contract matmul was threaded once and it **moved bits**: bounding each thread's work by its own
/// `last` regrouped the four-wide columns and moved the scalar tail. The head is different in kind — the unit of
/// work is a **whole vocabulary block**, so every block runs the same `Ops.orderedMatmul` call with the same
/// internal grouping and only the *thread* differs. These tests are the evidence for that claim rather than the
/// argument for it, and they deliberately walk block widths where `out % 4 != 0`, which is the regime that
/// caught the earlier attempt.
final class HeadLogitsTests: XCTestCase {
    /// Deterministic weights whose every value differs, so a regrouped accumulation cannot coincide.
    private struct Grid: WeightSource {
        let width: Int

        func tensor(named name: String) throws -> [Float] { [] }

        func rows(named name: String, range: Range<Int>) throws -> [Float] {
            var values: [Float] = []
            values.reserveCapacity(range.count * width)
            for row in range {
                for index in 0..<width {
                    let mixed = (row &* 7919 &+ index &* 104_729) % 1013
                    values.append(Float(mixed) * 0.001953125 - 0.5)
                }
            }
            return values
        }
    }

    /// A source that fails for one row range only, so the fan-out's error path is exercised rather than assumed.
    private enum ReadFailure: Swift.Error { case forThisTest }
    private struct Failing: WeightSource {
        let width: Int
        let failAt: Range<Int>

        func tensor(named name: String) throws -> [Float] { [] }

        func rows(named name: String, range: Range<Int>) throws -> [Float] {
            if range.overlaps(failAt) {
                throw ReadFailure.forThisTest
            }
            return [Float](repeating: 0.25, count: range.count * width)
        }
    }

    private func input(width: Int) -> [Float] {
        (0..<width).map { Float($0 % 11) * 0.0625 - 0.25 }
    }

    func testThreadingTheHeadBlocksIsABitForBitMap() throws {
        let width = 64, vocabulary = 512
        let source = Grid(width: width)
        let x = input(width: width)
        // Block widths above and below four, and one that equals the whole vocabulary: `out % 4 != 0` is exactly
        // where the threaded general matmul moved a tail, so it is the case that has to be covered.
        for blockRows in [1, 3, 7, 64, 512] {
            let serial = try Qwen3_5Forward.headLogits(
                x: x, source: source, name: "head", rows: 0..<vocabulary, vocabulary: vocabulary,
                hiddenSize: width, blockRows: blockRows, threads: 1
            )
            let parallel = try Qwen3_5Forward.headLogits(
                x: x, source: source, name: "head", rows: 0..<vocabulary, vocabulary: vocabulary,
                hiddenSize: width, blockRows: blockRows, threads: 8
            )
            XCTAssertEqual(
                serial.count, vocabulary, "the array is always vocabulary wide, so a gather sees the whole slice"
            )
            XCTAssertEqual(
                serial.map(\.bitPattern), parallel.map(\.bitPattern),
                "blockRows=\(blockRows): the decomposition changed a value"
            )
        }
    }

    func testASliceLandsInPlaceAndTheRestStaysZero() throws {
        let width = 32, vocabulary = 256, lower = 64, upper = 160
        let source = Grid(width: width)
        let x = input(width: width)
        let full = try Qwen3_5Forward.headLogits(
            x: x, source: source, name: "head", rows: 0..<vocabulary, vocabulary: vocabulary,
            hiddenSize: width, blockRows: 16, threads: 8
        )
        let sliced = try Qwen3_5Forward.headLogits(
            x: x, source: source, name: "head", rows: lower..<upper, vocabulary: vocabulary,
            hiddenSize: width, blockRows: 16, threads: 8
        )
        XCTAssertEqual(
            Array(sliced[lower..<upper]).map(\.bitPattern), Array(full[lower..<upper]).map(\.bitPattern),
            "a node's own slice must be the same values the whole vocabulary produced"
        )
        XCTAssertEqual(
            sliced[0..<lower].allSatisfy { $0 == 0 } && sliced[upper...].allSatisfy { $0 == 0 }, true,
            "and the rest stays zero for the gather to fill"
        )
    }

    func testAFailedBlockIsRaisedRatherThanSwallowed() throws {
        let width = 16, vocabulary = 128
        let source = Failing(width: width, failAt: 32..<64)
        let x = input(width: width)
        XCTAssertThrowsError(
            try Qwen3_5Forward.headLogits(
                x: x, source: source, name: "head", rows: 0..<vocabulary, vocabulary: vocabulary,
                hiddenSize: width, blockRows: 8, threads: 8
            ),
            "a fan-out that dropped an error would report zeros as if they were logits"
        )
    }
}
