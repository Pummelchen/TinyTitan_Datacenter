import Foundation

/// Which node owns which expert, at run time.
///
/// This is a thin view over a `ShardPlan` rather than a rule of its own, and that is the point: the
/// cluster agrees on a **file** (`DC-011`), and a node that computed ownership from its own arithmetic
/// would be one refactor away from disagreeing with its peers about who produces which term. Here the
/// file is the only source of truth, and `init(nodes:experts:)` merely generates one for a caller that
/// has no file yet.
///
/// The distribution is contiguous blocks by default, so a node's experts are adjacent in the stacked
/// tensor and its reads walk forward through the payload. Rounds 6 to 8 used a round-robin rule, and
/// changing it changed **no result** — which is `D17`'s whole claim: the reduction order is canonical,
/// so who computes a term never moves a bit.
public struct ExpertOwnership: Sendable, Equatable {
    /// The plan this ownership comes from.
    public let plan: ShardPlan

    public init(plan: ShardPlan) {
        self.plan = plan
    }

    /// Contiguous blocks over `experts`, for a caller with no plan file yet.
    public init(nodes: Int, experts: Int, distribution: ShardDistribution = .contiguous) {
        self.plan = ShardPlan.generate(
            family: "unspecified", experts: experts, nodes: nodes, distribution: distribution
        )
    }

    public var nodes: Int { plan.nodes }

    /// The single owner of an expert.
    public func owner(of expert: Int) -> Int { plan.owner(of: expert) }

    public func owns(_ expert: Int, node: Int) -> Bool { plan.owner(of: expert) == node }

    /// Every expert owned by one node, ascending, out of `count` experts.
    ///
    /// The count is taken and checked rather than trusted: a caller that asks about a different number
    /// of experts than the plan covers has a plan for the wrong model, and finding that out here is
    /// better than finding it out later as a missing term.
    public func experts(ownedBy node: Int, of count: Int) -> [Int] {
        precondition(
            count == plan.experts,
            "the ownership covers \(plan.experts) experts and was asked about \(count)"
        )
        return plan.experts(ownedBy: node)
    }

    /// The ownership map as data, so two nodes can compare what they each believe before a run starts.
    public func map(of count: Int) -> [Int] {
        precondition(
            count == plan.experts,
            "the ownership covers \(plan.experts) experts and was asked about \(count)"
        )
        return plan.owners
    }
}

/// One node's view of the experts: it can serve the ones it owns and nothing else.
///
/// The alternative — returning zeros for an expert this node does not have — would produce a number
/// that looks like a sum and is not one. `D17` requires the absence to be loud, because the reduction
/// cannot tell a missing term from a term that is zero; this is where that guarantee starts.
public struct OwnedExpertProvider: ExpertWeightProvider {
    public enum Error: Swift.Error, CustomStringConvertible {
        case notOwned(expert: Int, node: Int)

        public var description: String {
            switch self {
            case .notOwned(let expert, let node):
                return "node \(node) was asked for expert \(expert), which it does not own"
            }
        }
    }

    private let base: any ExpertWeightProvider
    private let ownership: ExpertOwnership
    private let node: Int

    public init(base: any ExpertWeightProvider, ownership: ExpertOwnership, node: Int) {
        precondition(node >= 0 && node < ownership.nodes, "node \(node) is outside the cluster")
        self.base = base
        self.ownership = ownership
        self.node = node
    }

    /// The experts this node owns are the ones it serves; the rest are skipped by the expert path and
    /// accounted for by the run's completeness check.
    public func serves(_ expert: Int) -> Bool { ownership.owns(expert, node: node) }

    public func gateUp(expert: Int, shape: MixtureShape) throws -> [Float] {
        try requireOwned(expert)
        return try base.gateUp(expert: expert, shape: shape)
    }

    public func down(expert: Int, shape: MixtureShape) throws -> [Float] {
        try requireOwned(expert)
        return try base.down(expert: expert, shape: shape)
    }

    private func requireOwned(_ expert: Int) throws {
        guard ownership.owns(expert, node: node) else {
            throw Error.notOwned(expert: expert, node: node)
        }
    }
}
