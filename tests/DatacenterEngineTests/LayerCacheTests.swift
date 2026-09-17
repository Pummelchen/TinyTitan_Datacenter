import XCTest

import DatacenterIR

@testable import DatacenterEngine

/// The decoded-layer cache, which exists because `D88` measured `load` at 30.5% of a cached step.
///
/// Two properties matter and they pull in opposite directions. It must **not** change a single value — a cache
/// that alters arithmetic is not a cache — and it must actually hold something, or it is a comment. The bit
/// identity is asserted where it belongs, against a real generation, in `ModelCacheTests`.
final class LayerCacheTests: XCTestCase {
    private func layer(_ name: String, count: Int) -> DecodedLayer {
        var weights: [TensorRole: [Float]] = [:]
        weights[.attnNorm] = [Float](repeating: 0.5, count: count)
        return DecodedLayer(
            weights: weights, gdn: nil, feedForward: .dense(gate: [], up: [], down: [])
        )
    }

    func testAHeldLayerIsServedWithoutLoadingItAgain() throws {
        let cache = LayerWeightCache(budgetBytes: 1_000_000, layerCount: 2)
        var loads = 0
        _ = try cache.layer(0) { loads += 1; return self.layer("a", count: 100) }
        _ = try cache.layer(0) { loads += 1; return self.layer("a", count: 100) }
        _ = try cache.layer(1) { loads += 1; return self.layer("b", count: 100) }

        XCTAssertEqual(loads, 2, "the second request for a held layer must not decode it again")
        XCTAssertEqual(cache.metrics.hits, 1)
        XCTAssertEqual(cache.metrics.misses, 2)
        XCTAssertEqual(cache.metrics.layersHeld, 2)
        XCTAssertGreaterThan(cache.metrics.bytesHeld, 0)
    }

    func testALayerTooLargeForTheBudgetIsReturnedButNotHeld() throws {
        // The budget is a limit, not a suggestion: returning the layer is required and keeping it is not, and
        // silently exceeding the budget to keep it would be the memory bug this whole area has history with.
        let cache = LayerWeightCache(budgetBytes: 64, layerCount: 1)
        let big = try cache.layer(0) { self.layer("a", count: 1000) }
        XCTAssertEqual(big.weights[.attnNorm]?.count, 1000)
        XCTAssertEqual(cache.metrics.bytesHeld, 0)
        XCTAssertEqual(cache.metrics.layersHeld, 0)
        XCTAssertEqual(cache.metrics.misses, 1)
    }

    func testAZeroBudgetHoldsNothingAndIsNotAnError() throws {
        let cache = LayerWeightCache(budgetBytes: 0, layerCount: 2)
        _ = try cache.layer(0) { self.layer("a", count: 10) }
        XCTAssertEqual(cache.metrics.layersHeld, 0)
        XCTAssertEqual(cache.metrics.hits, 0)
        XCTAssertEqual(cache.metrics.misses, 1)
    }

    func testTheCountedBytesAreTheArraysItHolds() throws {
        // Counted, not derived from a formula: the same arithmetic spelled out per expert is how a 14.5 GB
        // change once shipped past 98 green tests (`SlotBudgetTests`).
        let layer = self.layer("a", count: 512)
        XCTAssertEqual(layer.bytes, 512 * MemoryLayout<Float>.size)

        let withDense = DecodedLayer(
            weights: [.attnNorm: [Float](repeating: 0, count: 10)],
            gdn: nil,
            feedForward: .dense(
                gate: [Float](repeating: 0, count: 4), up: [Float](repeating: 0, count: 4),
                down: [Float](repeating: 0, count: 4)
            )
        )
        XCTAssertEqual(withDense.bytes, (10 + 12) * MemoryLayout<Float>.size)
    }
}
