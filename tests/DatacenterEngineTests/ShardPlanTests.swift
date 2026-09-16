import XCTest

@testable import DatacenterEngine

/// `DC-040`: the shard plan is an artifact the cluster agrees on, so what it must be true of is checked
/// before a run rather than discovered during one.
///
/// The format matters as much as the validation: `owners` is a flat array indexed by expert id, so an
/// expert cannot be owned twice and cannot be quietly unowned — the property the reduction needs is
/// **structural** rather than something a validator has to catch after the fact.
final class ShardPlanTests: XCTestCase {
    private struct Fixture {
        let weights: MixtureWeights
        let provider: any ExpertWeightProvider
        let shape: MixtureShape
        let input: [Float]
        let indices: [[Int]]
        let chosen: [[Float]]
        let tokens: Int

        func singleNode() throws -> [Float] {
            try MixtureOfExperts.experts(
                hidden: input, tokens: tokens, provider: provider,
                indices: indices, weights: chosen, shape: shape
            )
        }
    }

    private func fixture() throws -> Fixture {
        let root = try XCTUnwrap(
            Bundle.module.url(forResource: "tiny-qwen36", withExtension: nil, subdirectory: "Fixtures")
        )
        let forward = try Qwen3_5Forward(install: root.appendingPathComponent("install"))
        let layer = try forward.loadLayer(0)
        guard case .mixture(let weights, let provider) = layer.feedForward else {
            throw XCTSkip("the fixture's first layer is not a mixture")
        }
        let shape = try forward.mixtureShape()
        var state: UInt64 = 0x0FED_CBA9_8765_4321
        let tokens = 2
        let input = (0..<(tokens * shape.hiddenSize)).map { _ -> Float in
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Float(Int64(bitPattern: state >> 15) % 2_500) / 256
        }
        let (_, indices, chosen) = try MixtureOfExperts.block(
            hidden: input, tokens: tokens, weights: weights, shape: shape
        )
        return Fixture(
            weights: weights, provider: provider, shape: shape,
            input: input, indices: indices, chosen: chosen, tokens: tokens
        )
    }

    // MARK: - generating and validating

    func testAGeneratedPlanIsTotalAndAsBalancedAsTheDivisionAllows() throws {
        for (experts, nodes) in [(256, 4), (256, 3), (8, 2), (8, 8), (256, 1)] {
            let plan = ShardPlan.generate(family: "tiny", experts: experts, nodes: nodes)
            try plan.validate()
            XCTAssertEqual(plan.owners.count, experts, "every expert must be owned")
            let counts = (0..<nodes).map { plan.count(ownedBy: $0) }
            XCTAssertEqual(counts.reduce(0, +), experts, "the counts must add up to every expert")
            XCTAssertLessThanOrEqual(
                (counts.max() ?? 0) - (counts.min() ?? 0), 1,
                "\(experts) experts over \(nodes) nodes: blocks may differ by at most one"
            )
        }
    }

    func testTheContiguousDistributionReallyIsContiguous() throws {
        let plan = ShardPlan.generate(family: "tiny", experts: 10, nodes: 3, distribution: .contiguous)
        XCTAssertEqual(plan.owners, [0, 0, 0, 0, 1, 1, 1, 2, 2, 2])
        try plan.validate()

        let interleaved = ShardPlan.generate(family: "tiny", experts: 10, nodes: 3, distribution: .roundRobin)
        XCTAssertEqual(interleaved.owners, [0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        try interleaved.validate()
    }

    func testAMislabelledDistributionIsRefused() throws {
        let plan = ShardPlan.generate(family: "tiny", experts: 8, nodes: 2, distribution: .contiguous)
        let mislabelled = ShardPlan(
            family: plan.family, experts: plan.experts, nodes: plan.nodes,
            distribution: .roundRobin, owners: plan.owners
        )
        XCTAssertThrowsError(try mislabelled.validate()) { error in
            guard case ShardPlan.Error.distributionMismatch = error else {
                return XCTFail("expected the label to be checked, got \(error)")
            }
        }
    }

    func testAPlanThatLeavesAnExpertUnownedIsRefused() throws {
        // The format cannot express double ownership, but a truncated file is the mistake an editor
        // makes, and it is caught on the count rather than on the first missing term at token one.
        let full = ShardPlan.generate(family: "tiny", experts: 8, nodes: 2)
        let truncated = ShardPlan(
            family: full.family, experts: full.experts, nodes: full.nodes,
            distribution: full.distribution, owners: Array(full.owners.dropLast())
        )
        XCTAssertThrowsError(try truncated.validate()) { error in
            guard case ShardPlan.Error.expertCountMismatch(let declared, let found) = error else {
                return XCTFail("expected an unowned expert to be refused, got \(error)")
            }
            XCTAssertEqual(declared, 8)
            XCTAssertEqual(found, 7)
        }
    }

    func testAPlanNamingANodeOutsideTheClusterIsRefused() throws {
        let plan = ShardPlan(
            family: "tiny", experts: 4, nodes: 2, distribution: .contiguous, owners: [0, 1, 2, 0]
        )
        XCTAssertThrowsError(try plan.validate()) { error in
            guard case ShardPlan.Error.unknownNode(let expert, let node, let nodes) = error else {
                return XCTFail("expected an out-of-range node to be refused, got \(error)")
            }
            XCTAssertEqual(expert, 2)
            XCTAssertEqual(node, 2)
            XCTAssertEqual(nodes, 2)
        }
    }

    func testAPlanForADifferentModelIsRefused() throws {
        let plan = ShardPlan.generate(family: "tiny-qwen36", experts: 8, nodes: 2)
        try plan.validate(forFamily: "tiny-qwen36", experts: 8)
        XCTAssertThrowsError(try plan.validate(forFamily: "qwen3_5_moe", experts: 8)) { error in
            guard case ShardPlan.Error.modelMismatch = error else {
                return XCTFail("expected a family mismatch to be refused, got \(error)")
            }
        }
        XCTAssertThrowsError(try plan.validate(forFamily: "tiny-qwen36", experts: 256)) { error in
            guard case ShardPlan.Error.modelMismatch = error else {
                return XCTFail("expected an expert-count mismatch to be refused, got \(error)")
            }
        }
    }

    func testAnUnsupportedSchemaIsRefused() throws {
        let plan = ShardPlan(
            family: "tiny", experts: 4, nodes: 2, distribution: .contiguous, owners: [0, 0, 1, 1],
            schema: 99
        )
        XCTAssertThrowsError(try plan.validate()) { error in
            XCTAssertEqual(error as? ShardPlan.Error, .unsupportedSchema(99))
        }
    }

    /// A plan that leaves a node out is caught by the shape check, not by a separate "owns nothing"
    /// rule: for a contiguous plan those are the same condition, and for `experts < nodes` an idle node
    /// is legitimate. The separate check was removed when this test could not reach it.
    func testAContiguousPlanThatSkipsANodeIsRefused() throws {
        let plan = ShardPlan(
            family: "tiny", experts: 4, nodes: 3, distribution: .contiguous, owners: [0, 0, 1, 1]
        )
        XCTAssertThrowsError(try plan.validate()) { error in
            guard case ShardPlan.Error.distributionMismatch = error else {
                return XCTFail("expected the shape to be checked, got \(error)")
            }
        }
    }

    /// Fewer experts than nodes is not an error: the idle node sends an empty frame.
    func testAnIdleNodeIsAllowedWhenThereAreFewerExpertsThanNodes() throws {
        let plan = ShardPlan(
            family: "tiny", experts: 2, nodes: 4, distribution: .roundRobin, owners: [0, 1]
        )
        try plan.validate()
        XCTAssertEqual(plan.count(ownedBy: 2), 0)
        XCTAssertEqual(plan.count(ownedBy: 3), 0)
    }

    // MARK: - identity

    func testTWOIndependentGenerationsProduceTheSameDigest() throws {
        let one = ShardPlan.generate(family: "tiny", experts: 64, nodes: 4)
        let two = ShardPlan.generate(family: "tiny", experts: 64, nodes: 4)
        XCTAssertEqual(try one.canonicalDigest(), try two.canonicalDigest())

        // And a plan that moves one expert to another node is a different plan, which is the whole
        // reason two nodes compare digests before a run.
        var owners = one.owners
        owners[0] = 1
        let moved = ShardPlan(
            family: one.family, experts: one.experts, nodes: one.nodes,
            distribution: .roundRobin, owners: owners
        )
        XCTAssertNotEqual(try one.canonicalDigest(), try moved.canonicalDigest())
    }

    func testAPlanSurvivesAFileRoundTrip() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("shard-plan-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("shard-plan.json")

        let plan = ShardPlan.generate(family: "tiny-qwen36", experts: 8, nodes: 3, distribution: .roundRobin)
        try plan.write(to: url)
        let loaded = try ShardPlan.load(from: url)
        XCTAssertEqual(loaded, plan)
        XCTAssertEqual(try loaded.canonicalDigest(), try plan.canonicalDigest())
    }

    func testALoadedPlanWithAMistakeIsRefusedAtLoadRatherThanAtTheFirstToken() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("shard-plan-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("bad.json")
        let broken = """
        {"schema":1,"family":"tiny","experts":4,"nodes":2,"distribution":"contiguous","owners":[0,0,1]}
        """
        try Data(broken.utf8).write(to: url)
        XCTAssertThrowsError(try ShardPlan.load(from: url)) { error in
            guard case ShardPlan.Error.expertCountMismatch = error else {
                return XCTFail("expected the loader to refuse it, got \(error)")
            }
        }
    }

    // MARK: - the property the plan exists for

    /// The M2 claim, driven by a **file** rather than by a rule in a test: generate a plan, write it,
    /// load it, and the sharded forward over the fixture's real weights still equals the single-node
    /// forward bit for bit — for two nodes and for three.
    func testARunDrivenByALoadedPlanIsBitIdenticalToTheSingleNodeForward() throws {
        let fixture = try fixture()
        let shape = fixture.shape
        let single = try fixture.singleNode()
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("shard-plan-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        for nodes in [2, 3] {
            let url = directory.appendingPathComponent("plan-\(nodes).json")
            try ShardPlan
                .generate(family: "tiny-qwen36", experts: shape.experts, nodes: nodes)
                .write(to: url)
            let plan = try ShardPlan.load(from: url)
            try plan.validate(forFamily: "tiny-qwen36", experts: shape.experts)
            let ownership = ExpertOwnership(plan: plan)

            var terms: [ExpertContribution] = []
            for node in 0..<nodes {
                let owned = OwnedExpertProvider(base: fixture.provider, ownership: ownership, node: node)
                terms += try MixtureOfExperts.expertContributions(
                    hidden: fixture.input, tokens: fixture.tokens, provider: owned,
                    indices: fixture.indices, weights: fixture.chosen, shape: shape
                )
            }
            let reduced = OrderedReduction.accumulate(
                try ShardExchange.merge(terms, indices: fixture.indices),
                tokens: fixture.tokens, hiddenSize: shape.hiddenSize
            )
            for (index, element) in reduced.enumerated() where element.bitPattern != single[index].bitPattern {
                XCTFail("\(nodes) nodes from a loaded plan: element \(index) differs by bits")
                return
            }
        }
    }
}
