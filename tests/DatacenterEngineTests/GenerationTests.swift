import XCTest

@testable import DatacenterEngine

/// The sampler is a discrete decision, so these are exactness tests rather than
/// closeness tests, and the tie-break is the point of half of them.
final class GenerationTests: XCTestCase {
    func testArgmaxPicksTheLargestValue() {
        XCTAssertEqual(Greedy.argmax([0.1, 0.5, -0.2, 0.4], offset: 0, width: 4), 1)
    }

    func testTiesGoToTheLowestIndex() {
        // The tie-break is part of the contract (D5) and matches numpy's argmax, which
        // returns the first occurrence. A `>=` comparison here would silently pick the
        // last, and quantisation makes ties more likely rather than less.
        XCTAssertEqual(Greedy.argmax([0.5, 0.5, 0.5], offset: 0, width: 3), 0)
        XCTAssertEqual(Greedy.argmax([-1.0, -1.0, -2.0], offset: 0, width: 3), 0)
    }

    func testWindowSelectsTheRightRow() {
        // Three rows of width 4; the answer is relative to the row, not the buffer.
        let logits: [Float] = [
            9.0, 8.0, 7.0, 6.0,  // row 0 — index 0 would win if the offset were ignored
            0.0, 0.1, 5.0, 0.2,  // row 1
            1.0, 1.0, 1.0, 1.0,  // row 2 — all tied
        ]
        XCTAssertEqual(Greedy.argmax(logits, offset: 4, width: 4), 2)
        XCTAssertEqual(Greedy.argmax(logits, offset: 8, width: 4), 0)
        XCTAssertEqual(Greedy.argmax(logits, offset: 0, width: 4), 0)
    }

    func testArgmaxIsDeterministicOnTheSameInput() {
        let values = (0..<64).map { Float(($0 * 7919) % 101) / 101 }
        let first = Greedy.argmax(values, offset: 0, width: values.count)
        for _ in 0..<8 {
            XCTAssertEqual(Greedy.argmax(values, offset: 0, width: values.count), first)
        }
    }

    /// The trace's discrete section is what makes a generated token comparable as an
    /// index set, so it has to carry the tokens and their count.
    func testGeneratedTokensAreRecordedAsADiscreteDecision() throws {
        var writer = TraceWriter(producer: "test")
        writer.discrete = [TraceWriter.Discrete(name: "generated.tokens", shape: [3], values: [7, 8, 9])]
        let root = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("dsh-discrete-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }

        let manifest = try writer.write(to: root, tensors: [TraceWriter.Tensor(name: "logits", shape: [1, 2], values: [0, 1])])
        XCTAssertEqual(manifest.discrete.count, 1)
        XCTAssertEqual(manifest.discrete[0].values, [7, 8, 9])
        XCTAssertEqual(manifest.discrete[0].shape, [3])

        // And it must be part of the digest: two traces that generated different tokens
        // are not the same trace, however identical their tensors are.
        var other = TraceWriter(producer: "test")
        other.discrete = [TraceWriter.Discrete(name: "generated.tokens", shape: [3], values: [7, 8, 10])]
        let otherManifest = try other.write(
            to: root.appendingPathExtension("other"),
            tensors: [TraceWriter.Tensor(name: "logits", shape: [1, 2], values: [0, 1])]
        )
        XCTAssertNotEqual(manifest.digest, otherManifest.digest)
    }
}
