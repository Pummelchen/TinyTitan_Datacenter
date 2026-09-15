import XCTest

import DatacenterIR

@testable import DatacenterEngine

/// `DC-032`: a layer's experts are read **by the ones the router chose**, not as a stack.
///
/// The claim this file has to make measurable is the one M1's gate asks for: reads proportional
/// to the chosen experts, and a hit rate that is a number rather than an assurance. A 35 B
/// layer's experts are 805 M parameters — 3.2 GB in fp32 against about 4.5 GB of usable memory
/// per node — so "it streams" is not a performance nicety, it is the difference between running
/// and not running.
final class ExpertProviderTests: XCTestCase {
    private struct Decisions: Decodable {
        var shape: [Int]
        var values: [Int]
    }

    private struct Vector: Decodable {
        var shape: [Int]
        var bits: [UInt32]
        var floats: [Float] { bits.map { Float(bitPattern: $0) } }
    }

    private struct Golden: Decodable {
        var tokens: [Int]
        var tensors: [String: Vector]
        var discrete: [String: Decisions]
    }

    private func checkpoint() throws -> URL {
        try XCTUnwrap(
            Bundle.module.url(forResource: "tiny-qwen36", withExtension: nil, subdirectory: "Fixtures")
        )
    }

    private func golden() throws -> Golden {
        try JSONDecoder().decode(
            Golden.self, from: Data(contentsOf: try checkpoint().appendingPathComponent("golden.json"))
        )
    }

    private let shape = MixtureShape(hiddenSize: 32, experts: 8, topK: 2, intermediate: 16, sharedIntermediate: 16)

    /// A stacked tensor of `[experts, width]` rows with a recognisable value per (expert, index),
    /// so a provider that returned the wrong expert's slice would be visible rather than merely
    /// different.
    private func stack(experts: Int, width: Int) -> [Float] {
        var values = [Float](repeating: 0, count: experts * width)
        for expert in 0..<experts {
            for index in 0..<width {
                values[expert * width + index] = Float(expert * 1000 + index) * 0.001
            }
        }
        return values
    }

    // MARK: - The provider reads the expert it was asked for

    func testTheArrayProviderReturnsTheRequestedExpertAndNothingElse() throws {
        let gateUp = stack(experts: 8, width: 2 * 16 * 32)
        let down = stack(experts: 8, width: 32 * 16)
        let provider = ArrayExpertProvider(gateUp: gateUp, down: down)
        for expert in 0..<8 {
            let rows = try provider.gateUp(expert: expert, shape: shape)
            XCTAssertEqual(rows.count, 2 * 16 * 32)
            XCTAssertEqual(rows[0], Float(expert * 1000) * 0.001, accuracy: 1e-7, "expert \(expert)'s own first value")
            XCTAssertEqual(rows[0], gateUp[expert * 2 * 16 * 32], accuracy: 1e-7)
            let downRows = try provider.down(expert: expert, shape: shape)
            XCTAssertEqual(downRows.count, 32 * 16)
            XCTAssertEqual(downRows[0], down[expert * 32 * 16], accuracy: 1e-7)
        }
    }

    /// The streaming provider reads the real file, one row per expert, and produces exactly what
    /// the array provider produces for the same expert.
    func testTheStreamingProviderAgreesWithTheArrayProviderOnEveryExpert() throws {
        let file = try SafetensorsFile(url: try checkpoint().appendingPathComponent("model.safetensors"))
        let stacked = StackedExpertProvider(
            source: file,
            gateUpName: "model.language_model.layers.0.mlp.experts.gate_up_proj",
            downName: "model.language_model.layers.0.mlp.experts.down_proj"
        )
        let gateUp = try file.tensor(named: "model.language_model.layers.0.mlp.experts.gate_up_proj")
        let down = try file.tensor(named: "model.language_model.layers.0.mlp.experts.down_proj")
        let arrays = ArrayExpertProvider(gateUp: gateUp, down: down)

        for expert in 0..<shape.experts {
            let streamed = try stacked.gateUp(expert: expert, shape: shape)
            let resident = try arrays.gateUp(expert: expert, shape: shape)
            XCTAssertEqual(streamed.map(\.bitPattern), resident.map(\.bitPattern), "gate/up expert \(expert)")
            let streamedDown = try stacked.down(expert: expert, shape: shape)
            let residentDown = try arrays.down(expert: expert, shape: shape)
            XCTAssertEqual(streamedDown.map(\.bitPattern), residentDown.map(\.bitPattern), "down expert \(expert)")
        }
    }

    /// A source whose rows are the wrong width must fail loudly: every row would be the wrong
    /// matrix, and a wrong matrix multiplies cleanly.
    func testAStackOfTheWrongShapeIsRefused() throws {
        let file = try SafetensorsFile(url: try checkpoint().appendingPathComponent("model.safetensors"))
        let wrong = MixtureShape(hiddenSize: 32, experts: 8, topK: 2, intermediate: 8, sharedIntermediate: 16)
        let stacked = StackedExpertProvider(
            source: file,
            gateUpName: "model.language_model.layers.0.mlp.experts.gate_up_proj",
            downName: "model.language_model.layers.0.mlp.experts.down_proj"
        )
        XCTAssertThrowsError(try stacked.gateUp(expert: 0, shape: wrong)) { error in
            guard case ExpertProviderError.unexpectedWidth(_, _, let got, let expected) = error else {
                return XCTFail("expected unexpectedWidth, got \(error)")
            }
            XCTAssertEqual(got, 2 * 16 * 32)
            XCTAssertEqual(expected, 2 * 8 * 32)
        }
    }

    // MARK: - The kernel asks for the chosen experts only

    /// The streaming path, driven by the real router, asks for the experts the router chose and
    /// for no others — and reads their rows, not the stack's.
    func testOnlyTheChosenExpertsAreRead() throws {
        let file = try SafetensorsFile(url: try checkpoint().appendingPathComponent("model.safetensors"))
        let counting = CountingExpertProvider(
            upstream: StackedExpertProvider(
                source: file,
                gateUpName: "model.language_model.layers.0.mlp.experts.gate_up_proj",
                downName: "model.language_model.layers.0.mlp.experts.down_proj"
            )
        )
        let golden = try golden()
        let chosen = Set(golden.discrete["layer.00.router.topk"]!.values)
        let hidden = try XCTUnwrap(golden.tensors["layer.00.hidden_in"]).floats

        let weights = MixtureWeights(
            router: try file.tensor(named: "model.language_model.layers.0.mlp.gate.weight"),
            sharedGate: try file.tensor(named: "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight"),
            sharedUp: try file.tensor(named: "model.language_model.layers.0.mlp.shared_expert.up_proj.weight"),
            sharedDown: try file.tensor(named: "model.language_model.layers.0.mlp.shared_expert.down_proj.weight"),
            sharedScalarGate: try file.tensor(named: "model.language_model.layers.0.mlp.shared_expert_gate.weight"),
            experts: counting
        )
        _ = try MixtureOfExperts.block(hidden: hidden, tokens: golden.tokens.count, weights: weights, shape: shape)

        XCTAssertEqual(
            Set(counting.requested), chosen,
            "the kernel must ask for exactly the experts the router chose"
        )
        XCTAssertLessThan(
            Set(counting.requested).count, shape.experts,
            "a top-\(shape.topK) of \(shape.experts) must not read them all"
        )
        // Two projections per distinct expert, each one row of its stack: the elements read are
        // the gate/up slice plus the down slice of every chosen expert. Naming the unit matters —
        // this counted rows in one counter and elements in another, and the two disagreed by the
        // row width while both looked like plausible numbers.
        let expectedElements = chosen.count * (2 * shape.intermediate * shape.hiddenSize
            + shape.hiddenSize * shape.intermediate)
        XCTAssertEqual(counting.elementsRead, expectedElements)
    }

    // MARK: - The cache, and the number it produces

    func testTheCacheServesRepeatsAndEvicts() throws {
        let arrays = ArrayExpertProvider(
            gateUp: stack(experts: 8, width: 2 * 16 * 32), down: stack(experts: 8, width: 32 * 16)
        )
        let counting = CountingExpertProvider(upstream: arrays)
        let cache = ExpertSlotCache(upstream: counting, capacity: 2)

        // Expert 0, twice: the second is a hit that never reaches the source.
        _ = try cache.gateUp(expert: 0, shape: shape)
        _ = try cache.gateUp(expert: 0, shape: shape)
        XCTAssertEqual(cache.metrics.requests, 2)
        XCTAssertEqual(cache.metrics.hits, 1)
        XCTAssertEqual(cache.metrics.misses, 1)
        XCTAssertEqual(counting.requested, [0], "a hit must not touch the source")
        // Only the gate/up slice has been asked for at this point: the down projection comes
        // later in the test, so counting it here is the kind of off-by-one-slice that a
        // mislabelled unit makes easy to miss.
        XCTAssertEqual(cache.metrics.elementsRead, 2 * 16 * 32, "one expert's gate/up slice, once, in elements")

        // Two more experts evict the least recently used, and the capacity is never exceeded.
        _ = try cache.gateUp(expert: 1, shape: shape)
        _ = try cache.gateUp(expert: 2, shape: shape)
        XCTAssertEqual(cache.metrics.peakResidentExperts, 2, "the slot bank is bounded")
        XCTAssertLessThanOrEqual(cache.metrics.peakResidentExperts, 2)
        // Expert 0 was evicted, so asking again goes to the source.
        let before = cache.metrics.misses
        _ = try cache.gateUp(expert: 0, shape: shape)
        XCTAssertEqual(cache.metrics.misses, before + 1)
    }

    /// The two paths must agree bit for bit — with a cache in the middle, and with a cache too
    /// small to hold two experts. A cache that changed the arithmetic would be a scheduling
    /// artifact masquerading as a speedup, which is exactly what I1 exists to prevent.
    func testTheCachedPathIsBitIdenticalToTheArrayPath() throws {
        let golden = try golden()
        let hidden = try XCTUnwrap(golden.tensors["layer.00.hidden_in"]).floats
        let file = try SafetensorsFile(url: try checkpoint().appendingPathComponent("model.safetensors"))
        let gateUp = try file.tensor(named: "model.language_model.layers.0.mlp.experts.gate_up_proj")
        let down = try file.tensor(named: "model.language_model.layers.0.mlp.experts.down_proj")
        let router = try file.tensor(named: "model.language_model.layers.0.mlp.gate.weight")
        let sharedGate = try file.tensor(named: "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight")
        let sharedUp = try file.tensor(named: "model.language_model.layers.0.mlp.shared_expert.up_proj.weight")
        let sharedDown = try file.tensor(named: "model.language_model.layers.0.mlp.shared_expert.down_proj.weight")
        let sharedScalar = try file.tensor(named: "model.language_model.layers.0.mlp.shared_expert_gate.weight")

        func run(capacity: Int?) throws -> [Float] {
            let provider: any ExpertWeightProvider
            if let capacity {
                provider = ExpertSlotCache(
                    upstream: ArrayExpertProvider(gateUp: gateUp, down: down), capacity: capacity
                )
            } else {
                provider = ArrayExpertProvider(gateUp: gateUp, down: down)
            }
            let weights = MixtureWeights(
                router: router, sharedGate: sharedGate, sharedUp: sharedUp, sharedDown: sharedDown,
                sharedScalarGate: sharedScalar, experts: provider
            )
            return try MixtureOfExperts.block(
                hidden: hidden, tokens: golden.tokens.count, weights: weights, shape: shape
            ).output
        }

        let plain = try run(capacity: nil)
        for capacity in [1, 2, 8] {
            XCTAssertEqual(
                try run(capacity: capacity).map(\.bitPattern), plain.map(\.bitPattern),
                "a cache of \(capacity) slot(s) changed the arithmetic"
            )
        }
    }

    /// The forward pass reports the metrics, because a hit rate nobody can read is not a
    /// measurement.
    func testTheForwardPassReportsItsExpertTraffic() throws {
        let forward = try Qwen3_5Forward(snapshot: try checkpoint())
        let result = try forward.forwardWithDecisions(tokens: [3, 1, 4, 1, 5, 9, 2, 6, 5])
        XCTAssertEqual(result.expertMetrics.count, 2, "one entry per mixture layer")

        // The traffic is not a guess: it is fixed by the router's own decisions, which the same
        // result carries. Two projections are read per *distinct* chosen expert per layer, and
        // nothing else — a prompt's positions usually reuse experts, so the distinct count is
        // below `tokens × topK` and the assertions below say so.
        let requests = result.expertMetrics.reduce(0) { $0 + $1.requests }
        let expected = result.discrete.reduce(0) { total, decision in
            // Distinct *experts*, not distinct top-k tuples: `[5, 0]` and `[0, 5]` ask for the
            // same two experts, and the kernel asks per expert.
            total + Set(decision.values).count * 2
        }
        XCTAssertEqual(requests, expected, "reads are proportional to the chosen experts, not to the stack")
        XCTAssertLessThan(requests, shape.experts * 2 * result.expertMetrics.count, "not the whole stack")
        XCTAssertGreaterThan(result.expertElementsRead, 0)
        XCTAssertTrue((0.0...1.0).contains(result.expertHitRate))
    }
}
