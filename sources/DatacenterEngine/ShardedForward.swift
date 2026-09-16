import Foundation

/// What one node needs in order to run a forward as a member of a sharded cluster.
///
/// The node knows which experts it owns (from the plan, `D20`), who its peers are, and how long it will
/// wait for them. Everything else about the forward is unchanged, which is the point: sharding decides
/// *who computes a term*, never how the model is computed.
public struct ShardExecution {
    public let node: Int
    public let ownership: ExpertOwnership
    public let peers: [any ContributionTransport]
    public let policy: ExchangePolicy

    public init(
        node: Int, ownership: ExpertOwnership, peers: [any ContributionTransport],
        policy: ExchangePolicy = .default
    ) {
        precondition(node >= 0 && node < ownership.nodes, "node \(node) is outside the cluster")
        precondition(
            peers.count == ownership.nodes - 1,
            "node \(node) of \(ownership.nodes) needs \(ownership.nodes - 1) peer(s) and has \(peers.count)"
        )
        self.node = node
        self.ownership = ownership
        self.peers = peers
        self.policy = policy
    }
}
