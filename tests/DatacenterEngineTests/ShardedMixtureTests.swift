import XCTest

@testable import DatacenterEngine

/// M2's gate at fixture scale: partition the fixture's **real** experts across simulated nodes, reduce
/// them, and compare with the single-node forward — bit for bit.
///
/// The synthetic tests in `OrderedReductionTests` pin the contract's arithmetic. These pin the thing
/// M2 actually has to demonstrate: that the production expert path, given **real** quantized weights, a
/// **real** router decision and a partition the run does not control, produces the same bits as one
/// node does. If this can fail, so can the cluster.
final class ShardedMixtureTests: XCTestCase {
    private struct Source {
        var state: UInt64 = 0x9E37_79B9_7F4A_7C15
        mutating func next() -> Float {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Float(Int64(bitPattern: state >> 12) % 2_000) / 256
        }
    }

    private struct Fixture {
        let weights: MixtureWeights
        let provider: any ExpertWeightProvider
        let shape: MixtureShape
    }

    private func fixture() throws -> Fixture {
        let root = try XCTUnwrap(
            Bundle.module.url(forResource: "tiny-qwen36", withExtension: nil, subdirectory: "Fixtures")
        )
        let forward = try Qwen3_5Forward(install: root.appendingPathComponent("install"))
        let layer = try forward.loadLayer(0)
        guard case .mixture(let weights, let provider) = layer.feedForward else {
            throw XCTSkip("the fixture's first layer is not a mixture, so there is nothing to shard")
        }
        return Fixture(weights: weights, provider: provider, shape: try forward.mixtureShape())
    }

    private func hidden(tokens: Int, width: Int) -> [Float] {
        var source = Source()
        return (0..<(tokens * width)).map { _ in source.next() }
    }

    private func assertBitIdentical(_ one: [Float], _ other: [Float], _ what: String) {
        XCTAssertEqual(one.count, other.count, "\(what): different lengths")
        for index in 0..<min(one.count, other.count) where one[index].bitPattern != other[index].bitPattern {
            XCTFail(
                "\(what): element \(index) is \(one[index].bitPattern) on one node and "
                + "\(other[index].bitPattern) on the other"
            )
            return
        }
    }

    /// The gate: 1 node against 2 and against 4, over real weights.
    func testTheShardedForwardIsBitIdenticalToTheSingleNodeForward() throws {
        let fixture = try fixture()
        let tokens = 3
        let shape = fixture.shape
        let input = hidden(tokens: tokens, width: shape.hiddenSize)

        // The real router decides, exactly as it does in a forward; the output is discarded so the
        // routed part can be compared on its own. The shared expert is replicated and not reduced
        // (`D17`), so it is not part of this comparison.
        let (_, indices, chosen) = try MixtureOfExperts.block(
            hidden: input, tokens: tokens, weights: fixture.weights, shape: shape
        )
        let single = try MixtureOfExperts.experts(
            hidden: input, tokens: tokens, provider: fixture.provider,
            indices: indices, weights: chosen, shape: shape
        )
        let selected = OrderedReduction.selectedKeys(indices)
        XCTAssertEqual(selected.count, tokens * shape.topK, "the router must select top-k per token")

        for nodes in [2, 4] {
            let ownership = ExpertOwnership(nodes: nodes)
            var terms: [ExpertContribution] = []
            // Nodes are visited in reverse and each node's experts arrive in its own order: arrival
            // order is the one thing the contract promises not to depend on.
            for node in (0..<nodes).reversed() {
                let owned = OwnedExpertProvider(base: fixture.provider, ownership: ownership, node: node)
                terms += try MixtureOfExperts.expertContributions(
                    hidden: input, tokens: tokens, provider: owned,
                    indices: indices, weights: chosen, shape: shape
                )
            }

            XCTAssertTrue(
                OrderedReduction.isComplete(terms, indices: indices),
                "\(nodes) nodes: the reduction must see every selected term exactly once before it sums"
            )
            XCTAssertEqual(terms.count, selected.count, "\(nodes) nodes: one term per selection")
            let sharded = OrderedReduction.accumulate(terms, tokens: tokens, hiddenSize: shape.hiddenSize)
            assertBitIdentical(single, sharded, "\(nodes) nodes")
        }
    }

    /// The ownership map is total and disjoint, which is what makes the reduction complete.
    func testEveryExpertHasExactlyOneOwner() throws {
        let fixture = try fixture()
        for nodes in [1, 2, 3, 4] {
            let ownership = ExpertOwnership(nodes: nodes)
            var seen = [Int: Int]()
            for node in 0..<nodes {
                for expert in ownership.experts(ownedBy: node, of: fixture.shape.experts) {
                    seen[expert, default: 0] += 1
                }
            }
            XCTAssertEqual(seen.count, fixture.shape.experts, "\(nodes) nodes: every expert must be owned")
            XCTAssertTrue(seen.values.allSatisfy { $0 == 1 }, "\(nodes) nodes: no expert may be owned twice")
            XCTAssertEqual(ownership.map(of: fixture.shape.experts).count, fixture.shape.experts)
        }
    }

    /// A node asked for an expert it does not own must fail, not return zeros: the reduction cannot
    /// tell a missing term from a term that is zero, so absence has to be loud at this end.
    func testAnUnownedExpertIsRefusedRatherThanZeroed() throws {
        let fixture = try fixture()
        let owned = OwnedExpertProvider(
            base: fixture.provider, ownership: ExpertOwnership(nodes: 2), node: 0
        )
        // Served: the experts this node owns.
        for expert in ExpertOwnership(nodes: 2).experts(ownedBy: 0, of: fixture.shape.experts) {
            XCTAssertNoThrow(try owned.gateUp(expert: expert, shape: fixture.shape))
        }
        // Refused: one it does not. A zero here would be a silently smaller sum.
        let foreign = ExpertOwnership(nodes: 2).experts(ownedBy: 1, of: fixture.shape.experts).first
        let expert = try XCTUnwrap(foreign)
        XCTAssertThrowsError(try owned.gateUp(expert: expert, shape: fixture.shape)) { error in
            guard case OwnedExpertProvider.Error.notOwned(let asked, let node) = error else {
                return XCTFail("expected a refusal, got \(error)")
            }
            XCTAssertEqual(asked, expert)
            XCTAssertEqual(node, 0)
        }
        XCTAssertThrowsError(try owned.down(expert: expert, shape: fixture.shape))
    }
}
