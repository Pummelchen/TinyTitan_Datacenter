import Foundation
import XCTest

import DatacenterIR

@testable import DatacenterEngine

/// The streaming read: same values, different pages.
///
/// `DC-086`'s second half. The install path was fixed first; this is the checkpoint reader, which
/// still memory-maps its shard. An expert slab is read once per token and would evict everything
/// useful from the page cache, while the embedding and the head are read every token and are
/// exactly what the cache is for — so the split is by **consumer**, not by tensor, and the thing
/// that must not change is the arithmetic.
final class StreamingReadTests: XCTestCase {
    private func checkpoint() throws -> URL {
        try XCTUnwrap(
            Bundle.module.url(forResource: "tiny-qwen36", withExtension: nil, subdirectory: "Fixtures")
        )
    }

    private func file() throws -> SafetensorsFile {
        try SafetensorsFile(url: try checkpoint().appendingPathComponent("model.safetensors"))
    }

    /// Every row-addressable tensor in the fixture, read both ways. Names are not hardcoded: the
    /// point is that no tensor differs, so the test should not depend on which ones exist.
    func testEveryTensorReadsTheSameThroughBothPaths() throws {
        let file = try file()
        var compared = 0
        for name in file.names {
            guard let info = file.tensors[name], info.shape.count >= 2 else { continue }
            let rows = 0..<min(2, info.shape[0])
            let mapped = try file.float32(name, rows: rows)
            let streaming = try file.rowsStreaming(named: name, range: rows)
            XCTAssertEqual(mapped, streaming, "\(name): the uncached read changed the values")
            compared += 1
        }
        XCTAssertGreaterThan(compared, 0, "the fixture must have row-addressable tensors to compare")
    }

    /// A rank-1 tensor is not row-addressable, so the streaming path hands back the whole tensor —
    /// the same fallback the ordinary row read makes.
    func testRankOneTensorsFallBackToTheWholeTensor() throws {
        let file = try file()
        let rankOne = file.names.filter { (file.tensors[$0]?.shape.count ?? 0) == 1 }
        try XCTSkipIf(rankOne.isEmpty, "the fixture has no rank-1 tensors")
        for name in rankOne.prefix(3) {
            XCTAssertEqual(try file.rowsStreaming(named: name, range: 0..<1), try file.float32(name))
        }
    }

    func testARowRangeOutsideTheTensorIsStillAnError() throws {
        let file = try file()
        let name = try XCTUnwrap(file.names.first { (file.tensors[$0]?.shape.count ?? 0) >= 2 })
        let total = try XCTUnwrap(file.tensors[name]).shape[0]
        XCTAssertThrowsError(try file.rowsStreaming(named: name, range: (total - 1)..<(total + 1)))
        XCTAssertThrowsError(try file.rowsStreaming(named: name, range: -1..<1))
    }
}

/// Who asks for a streaming read, which is the half of the change that is easy to get wrong: a
/// reader that exists but is never called is a page cache that still thrashes.
final class StreamingConsumerTests: XCTestCase {
    private final class SpySource: WeightSource {
        var mappedCalls: [String] = []
        var streamingCalls: [String] = []
        var sizes: [String: Int] = [:]

        func tensor(named name: String) throws -> [Float] {
            [Float](repeating: 0.5, count: sizes[name] ?? 0)
        }

        func rows(named name: String, range: Range<Int>) throws -> [Float] {
            mappedCalls.append(name)
            return [Float](repeating: 0.5, count: sizes[name] ?? 0)
        }

        func rowsStreaming(named name: String, range: Range<Int>) throws -> [Float] {
            streamingCalls.append(name)
            return [Float](repeating: 0.5, count: sizes[name] ?? 0)
        }
    }

    func testTheExpertProviderAsksForStreamingReadsAndNeverTheMappedOnes() throws {
        let spy = SpySource()
        let shape = MixtureShape(hiddenSize: 8, experts: 4, topK: 2, intermediate: 16, sharedIntermediate: 16)
        spy.sizes["gate_up"] = 2 * shape.intermediate * shape.hiddenSize
        spy.sizes["down"] = shape.hiddenSize * shape.intermediate

        let provider = StackedExpertProvider(source: spy, gateUpName: "gate_up", downName: "down")
        _ = try provider.gateUp(expert: 1, shape: shape)
        _ = try provider.down(expert: 1, shape: shape)

        XCTAssertEqual(spy.streamingCalls, ["gate_up", "down"], "the streaming consumer must stream")
        XCTAssertTrue(spy.mappedCalls.isEmpty, "an expert read must not populate the page cache")
    }

    /// And the dense consumer keeps the mapping, because the embedding and the head are re-read
    /// every token and their pages are exactly what the cache is for. `rowsStreaming` exists on
    /// the protocol, so this asserts the default rather than a call site.
    func testTheDefaultStreamingReadIsTheOrdinaryRead() throws {
        let spy = SpySource()
        spy.sizes["embed"] = 4
        _ = try spy.rowsStreaming(named: "embed", range: 0..<1)
        XCTAssertEqual(spy.streamingCalls, ["embed"])
        XCTAssertTrue(spy.mappedCalls.isEmpty)
    }
}
