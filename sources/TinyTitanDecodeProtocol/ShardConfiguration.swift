import Foundation

/// Everything a node needs to take part in a sharded run, and nothing it does not.
///
/// The engine is single-node by default and must stay that way: `ShardConfiguration` is absent unless a run is
/// explicitly sharded, so every existing path keeps the behaviour it has today and the ownership filter stays
/// **unset** rather than set to an identity (`D164`).
///
/// The plan is carried as data because it is the artifact that has to agree across nodes - a node that computed
/// its own ownership from a rule could disagree with a peer about who owns an expert, and `D166`'s digest exists
/// to make that disagreement detectable rather than silent.
public struct ShardConfiguration: Sendable, Equatable {
    /// The plan, as loaded. Its `canonicalDigest()` is what every node compares.
    public let plan: ShardPlan
    /// This node's index, 0..<plan.nodes.
    public let node: Int
    /// host and port for each peer index. A peer absent here is never asked.
    public let peers: [Int: PeerAddress]

    public struct PeerAddress: Sendable, Equatable {
        public let host: String
        public let port: UInt16
        public init(host: String, port: UInt16) {
            self.host = host
            self.port = port
        }
    }

    public enum Error: Swift.Error, Equatable, CustomStringConvertible {
        case nodeOutOfRange(node: Int, nodes: Int)
        case missingPeers([Int])
        case selfInPeers(Int)

        public var description: String {
            switch self {
            case let .nodeOutOfRange(node, nodes):
                return "shard node \(node) is outside 0..<\(nodes)"
            case let .missingPeers(peers):
                return "no address given for peer(s) \(peers.sorted()) that the plan gives experts to"
            case let .selfInPeers(node):
                return "peer address given for this node (\(node)); a node does not connect to itself"
            }
        }
    }

    public init(plan: ShardPlan, node: Int, peers: [Int: PeerAddress]) throws {
        guard node >= 0, node < plan.nodes else {
            throw Error.nodeOutOfRange(node: node, nodes: plan.nodes)
        }
        guard peers[node] == nil else { throw Error.selfInPeers(node) }
        // Every peer the plan gives experts to must be reachable, or this node would silently compute a
        // different answer from its peers - the failure `D166`'s digest exists to catch, caught earlier and
        // more cheaply here.
        let owners = Set(plan.owners).subtracting([node])
        let missing = owners.subtracting(peers.keys)
        // Sorted, because `missing` is a Set and its iteration order is not stable: an error whose payload
        // varies run to run cannot be compared in a test and cannot be matched in a retry, and the first
        // version of this made a test fail with two identical-looking messages.
        guard missing.isEmpty else { throw Error.missingPeers(missing.sorted()) }
        self.plan = plan
        self.node = node
        self.peers = peers
    }

    /// The peers this node will actually exchange with, which is every owner of an expert.
    public var peerIndices: [Int] { peers.keys.sorted() }

    /// A one-line summary for startup logs, so a run records what it was rather than what it was meant to be.
    public var summary: String {
        let digest = (try? plan.canonicalDigest()).map { String($0.prefix(12)) } ?? "?"
        return "shard node=\(node)/\(plan.nodes) peers=\(peerIndices) distribution=\(plan.distribution.rawValue) "
            + "replicated=\(plan.replicated.count) digest=\(digest)"
    }

    /// Build the runtime pieces for this configuration.
    public func makeTransport() -> ShardPeerSet {
        ShardPeerSet(plan: plan, node: node,
                     peers: peers.mapValues { ($0.host, $0.port) })
    }
}
