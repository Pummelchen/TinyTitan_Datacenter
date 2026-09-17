import XCTest

@testable import DatacenterEngine

/// The vocabulary split, and the frame that carries it (`D93`).
///
/// The split has to be a partition — every row owned exactly once, by exactly one node — because the gather
/// assumes it: a row nobody computed stays zero, and a row two nodes computed is written twice. Neither shows
/// up as an error, which is why it is asserted here rather than discovered later.
final class VocabSliceTests: XCTestCase {
    func testTheSlicesPartitionTheVocabularyExactly() {
        for vocabSize in [0, 1, 7, 8, 9, 128, 248_320] {
            for nodes in 1...5 where nodes <= max(vocabSize, 1) {
                var seen: [Int] = []
                for node in 0..<nodes { seen += Array(VocabSlice(node: node, nodes: nodes, vocabSize: vocabSize).range) }
                XCTAssertEqual(
                    seen, Array(0..<vocabSize),
                    "vocab \(vocabSize) over \(nodes) node(s) must be a partition, in order"
                )
            }
        }
    }

    func testTheSlicesDifferByAtMostOneRow() {
        let sizes = (0..<4).map { VocabSlice(node: $0, nodes: 4, vocabSize: 10).range.count }
        XCTAssertEqual(sizes.max()! - sizes.min()!, 1, "a remainder goes to the first nodes, not to one of them")
        XCTAssertEqual(sizes.reduce(0, +), 10)
    }

    func testAOneNodeSliceIsTheWholeHead() {
        let slice = VocabSlice(node: 0, nodes: 1, vocabSize: 1_000)
        XCTAssertTrue(slice.isWholeHead)
        XCTAssertEqual(slice.range, 0..<1_000)
    }

    func testAHeadSmallerThanTheClusterLeavesNodesWithNothingRatherThanFailing() {
        // 2 rows over 4 nodes: two nodes own a row each and two own none, which the gather must carry as an
        // empty frame rather than as an absence.
        let slices = (0..<4).map { VocabSlice(node: $0, nodes: 4, vocabSize: 2) }
        XCTAssertEqual(slices.map(\.range.count), [1, 1, 0, 0])
    }
}

final class HeadSliceWireTests: XCTestCase {
    func testARoundTripPreservesEveryBit() throws {
        // The values that a "close enough" encoder loses: a negative zero and a NaN with a payload.
        let values: [Float] = [0, -0.0, 1.5, -2.25, .infinity, -.infinity, Float(bitPattern: 0x7FC0_1234)]
        let decoded = try HeadSliceWire.decode(HeadSliceWire.encode(start: 42, values: values))
        XCTAssertEqual(decoded.start, 42)
        XCTAssertEqual(
            decoded.values.map(\.bitPattern), values.map(\.bitPattern),
            "a slice crosses the wire as bits, exactly as contributions do"
        )
    }

    func testAnEmptySliceIsAFrameNotAnAbsence() throws {
        // A node whose slice misses a window must still send something, or its peers read the next window's
        // frame and every later row lands in the wrong place.
        let decoded = try HeadSliceWire.decode(HeadSliceWire.encode(start: 7, values: []))
        XCTAssertEqual(decoded.start, 7)
        XCTAssertTrue(decoded.values.isEmpty)
    }

    func testAFrameThatIsNotOneIsRefused() {
        XCTAssertThrowsError(try HeadSliceWire.decode(Data("TTDC".utf8))) { error in
            XCTAssertEqual(error as? HeadSliceWire.Error, .tooShort(bytes: 4))
        }
        XCTAssertThrowsError(try HeadSliceWire.decode(Data(repeating: 0, count: 14))) { error in
            XCTAssertEqual(error as? HeadSliceWire.Error, .badMagic)
        }
    }

    func testAFrameWhoseLengthDisagreesWithItsCountIsRefused() throws {
        var data = try HeadSliceWire.encode(start: 0, values: [1, 2, 3])
        data.append(contentsOf: [0, 0, 0, 0])
        XCTAssertThrowsError(try HeadSliceWire.decode(data)) { error in
            XCTAssertEqual(error as? HeadSliceWire.Error, .wrongLength(declared: 3, found: 4))
        }
    }

    func testAnAbsurdCountIsRefusedBeforeAnythingIsAllocated() {
        var bytes = HeadSliceWire.magic
        bytes += [1, 0]                                   // version 1
        bytes += [0, 0, 0, 0]                             // start
        bytes += [0xFF, 0xFF, 0xFF, 0x7F]                 // a count nobody could hold
        XCTAssertThrowsError(try HeadSliceWire.decode(Data(bytes))) { error in
            XCTAssertEqual(error as? HeadSliceWire.Error, .absurdCount(0x7FFF_FFFF))
        }
    }
}
