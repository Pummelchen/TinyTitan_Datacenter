import Foundation

/// Which node owns which expert.
///
/// `D17` makes the reduction order canonical, so ownership no longer affects any result — but it does
/// decide *who computes what*, and a run is only correct if every selected expert is served by exactly
/// one owner. That is why this is a data-shaped artifact and not a property of the network: it is the
/// thing that must be pinned for two runs at the same N to agree (`DC-011`).
///
/// The assignment is round-robin by expert id, which balances the count on any N and is trivial to
/// reproduce from the same inputs on every node. The planner that will replace it (`DC-040`) must
/// produce a map with the same properties: total, disjoint, and identical on every node.
public struct ExpertOwnership: Sendable, Equatable {
    public let nodes: Int

    public init(nodes: Int) {
        precondition(nodes >= 1, "a cluster with \(nodes) nodes cannot own anything")
        self.nodes = nodes
    }

    /// The single owner of an expert.
    public func owner(of expert: Int) -> Int {
        precondition(expert >= 0, "expert ids are non-negative")
        return expert % nodes
    }

    public func owns(_ expert: Int, node: Int) -> Bool {
        owner(of: expert) == node
    }

    /// Every expert owned by one node, ascending, out of `count` experts.
    public func experts(ownedBy node: Int, of count: Int) -> [Int] {
        precondition(node >= 0 && node < nodes, "node \(node) is outside 0..<\(nodes)")
        return (0..<count).filter { owns($0, node: node) }
    }

    /// The ownership map as data, so two nodes can compare what they each believe before a run starts.
    public func map(of count: Int) -> [Int] {
        (0..<count).map(owner(of:))
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
