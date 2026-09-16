import CryptoKit
import Foundation

/// What every node in a run must agree about before it starts.
///
/// A cluster run is a claim that N machines are computing one model together, and every way that claim
/// can be false — a different model revision, a different plan, a different build — produces numbers
/// that look plausible. So the agreement is checked once, up front, and the fields are exactly the ones
/// whose disagreement would invalidate the run:
///
/// - `schema`: the bring-up protocol itself;
/// - `family`, `revision`: the same model, pinned the same way (`I6`);
/// - `experts`, `hiddenSize`, `topK`: the same geometry, because a plan for another shape is a plan for
///   another model;
/// - `planDigest`: the same ownership map (`D20`), which is the one artifact `D17` says must be pinned.
///
/// It deliberately does **not** include an install digest. Each node holds the experts it owns, so the
/// installs differ by construction and comparing them would refuse every correct sharded run.
public struct ClusterIdentity: Sendable, Equatable, Codable {
    public static let schema = 1
    public static let protocolName = "tinytitan-datacenter"

    public let schema: Int
    public let family: String
    public let revision: String
    public let experts: Int
    public let hiddenSize: Int
    public let topK: Int
    public let planDigest: String

    public init(
        family: String, revision: String, experts: Int, hiddenSize: Int, topK: Int,
        planDigest: String, schema: Int = ClusterIdentity.schema
    ) {
        self.schema = schema
        self.family = family
        self.revision = revision
        self.experts = experts
        self.hiddenSize = hiddenSize
        self.topK = topK
        self.planDigest = planDigest
    }

    /// Compare field by field, so a refusal can say **which** agreement failed instead of that one did.
    public func differences(from other: ClusterIdentity) -> [String] {
        var fields: [String] = []
        if schema != other.schema { fields.append("schema") }
        if family != other.family { fields.append("family") }
        if revision != other.revision { fields.append("revision") }
        if experts != other.experts { fields.append("experts") }
        if hiddenSize != other.hiddenSize { fields.append("hiddenSize") }
        if topK != other.topK { fields.append("topK") }
        if planDigest != other.planDigest { fields.append("planDigest") }
        return fields
    }

    public func canonicalJSON() throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return try encoder.encode(self)
    }

    public func canonicalDigest() throws -> String {
        SHA256.hash(data: try canonicalJSON()).map { String(format: "%02x", $0) }.joined()
    }
}

/// What one node says about itself: the shared identity, and where it sits in the cluster.
public struct NodeDeclaration: Sendable, Equatable, Codable {
    public let identity: ClusterIdentity
    public let node: Int
    public let nodes: Int

    public init(identity: ClusterIdentity, node: Int, nodes: Int) {
        self.identity = identity
        self.node = node
        self.nodes = nodes
    }
}

/// Where the other nodes are. Data, like the plan, because it is an agreement rather than a discovery
/// mechanism — mDNS and address negotiation need a real network and are part of `DC-008`.
public struct ClusterConfig: Sendable, Equatable, Codable {
    public let endpoints: [NodeEndpoint]

    public init(endpoints: [NodeEndpoint]) {
        self.endpoints = endpoints
    }

    public var nodes: Int { endpoints.count }

    public enum Error: Swift.Error, CustomStringConvertible, Equatable {
        case thisNodeNotInConfig(Int, nodes: Int)
        case duplicateEndpoint(node: Int, host: String, port: Int)
        case badEndpoint(node: Int)

        public var description: String {
            switch self {
            case .thisNodeNotInConfig(let node, let nodes):
                return "this node is \(node) and the config describes \(nodes) node(s)"
            case .duplicateEndpoint(let node, let host, let port):
                return "node \(node) is \(host):\(port), which another node already claims"
            case .badEndpoint(let node):
                return "node \(node) has an empty host or a port outside 1...65535"
            }
        }
    }

    public func validate(thisNode: Int) throws {
        guard thisNode >= 0, thisNode < endpoints.count else {
            throw Error.thisNodeNotInConfig(thisNode, nodes: endpoints.count)
        }
        var seen: Set<String> = []
        for (node, endpoint) in endpoints.enumerated() {
            guard !endpoint.host.isEmpty, endpoint.port > 0, endpoint.port <= 65_535 else {
                throw Error.badEndpoint(node: node)
            }
            let key = "\(endpoint.host):\(endpoint.port)"
            guard seen.insert(key).inserted else {
                throw Error.duplicateEndpoint(node: node, host: endpoint.host, port: endpoint.port)
            }
        }
    }

    public static func load(from url: URL, thisNode: Int) throws -> ClusterConfig {
        let config = try JSONDecoder().decode(ClusterConfig.self, from: Data(contentsOf: url))
        try config.validate(thisNode: thisNode)
        return config
    }
}

public struct NodeEndpoint: Sendable, Equatable, Codable {
    public let host: String
    public let port: Int

    public init(host: String, port: Int) {
        self.host = host
        self.port = port
    }
}

public enum ClusterBringUpError: Swift.Error, CustomStringConvertible, Equatable {
    case notADeclaration(String)
    case disagreement(node: Int, fields: [String])
    case nodeCountMismatch(ours: Int, theirs: Int)
    case unexpectedNode(Int, expected: [Int])
    case missingNode(Int)
    case noPeers

    public var description: String {
        switch self {
        case .notADeclaration(let why):
            return "a peer's bring-up frame is not a declaration: \(why)"
        case .disagreement(let node, let fields):
            return "node \(node) disagrees about \(fields.joined(separator: ", ")): the run would compare "
                + "numbers produced under different assumptions"
        case .nodeCountMismatch(let ours, let theirs):
            return "this node believes the cluster has \(ours) nodes and node \(theirs) says otherwise"
        case .unexpectedNode(let node, let expected):
            return "a peer claims to be node \(node) and the rest of the cluster is \(expected)"
        case .missingNode(let node):
            return "node \(node) never answered bring-up"
        case .noPeers:
            return "a cluster run with no peers is a single-node run with extra steps"
        }
    }
}

/// The handshake: exchange declarations, check them, and refuse the run if anything disagrees.
///
/// The frame carries its **own magic**, and that is not decoration. A declaration and a contribution
/// frame travel over the same connection, so a codec that guessed from the payload would eventually
/// read a declaration as terms — and a term decoded from JSON is exactly the kind of plausible nonsense
/// this project keeps having to eliminate.
public enum ClusterHandshake {
    public static let magic: [UInt8] = Array("TTDH".utf8)
    public static let version: UInt16 = 1

    public static func frame(_ declaration: NodeDeclaration) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        let payload = try encoder.encode(declaration)
        var bytes: [UInt8] = magic
        let little = version.littleEndian
        bytes.append(UInt8(truncatingIfNeeded: little))
        bytes.append(UInt8(truncatingIfNeeded: little >> 8))
        bytes.append(contentsOf: payload)
        return Data(bytes)
    }

    public static func decode(_ frame: Data) throws -> NodeDeclaration {
        let bytes = [UInt8](frame)
        guard bytes.count >= 6 else {
            throw ClusterBringUpError.notADeclaration("it is \(bytes.count) bytes long")
        }
        guard Array(bytes[0..<4]) == magic else {
            throw ClusterBringUpError.notADeclaration("its magic bytes are \(Array(bytes[0..<4]))")
        }
        let found = UInt16(bytes[4]) | (UInt16(bytes[5]) << 8)
        guard found == version else {
            throw ClusterBringUpError.notADeclaration("it is bring-up version \(found), not \(version)")
        }
        do {
            return try JSONDecoder().decode(NodeDeclaration.self, from: Data(bytes[6...]))
        } catch {
            throw ClusterBringUpError.notADeclaration("its payload does not decode: \(error)")
        }
    }

    /// Exchange declarations with every peer, and return theirs once all of them agree.
    ///
    /// Every check happens before a single token is computed. A disagreement that is found at the end of
    /// a run is a wrong answer with a plausible shape; the same disagreement found here is a message.
    @discardableResult
    public static func perform(
        ours: NodeDeclaration, peers: [any ContributionTransport],
        policy: ExchangePolicy = .default
    ) throws -> [NodeDeclaration] {
        guard !peers.isEmpty else { throw ClusterBringUpError.noPeers }
        guard ours.node >= 0, ours.node < ours.nodes else {
            throw ClusterBringUpError.unexpectedNode(ours.node, expected: Array(0..<ours.nodes))
        }
        let frame = try frame(ours)
        for peer in peers { peer.applyTimeout(milliseconds: policy.receiveTimeoutMilliseconds) }
        for peer in peers { try peer.send(frame) }

        var theirs: [NodeDeclaration] = []
        for peer in peers { theirs.append(try decode(try peer.receive())) }

        let expected = Array(0..<ours.nodes).filter { $0 != ours.node }
        var claimed: [Int] = []
        for declaration in theirs {
            guard declaration.nodes == ours.nodes else {
                throw ClusterBringUpError.nodeCountMismatch(ours: ours.nodes, theirs: declaration.nodes)
            }
            let fields = ours.identity.differences(from: declaration.identity)
            guard fields.isEmpty else {
                throw ClusterBringUpError.disagreement(node: declaration.node, fields: fields)
            }
            claimed.append(declaration.node)
        }
        // Three distinct failures, reported as themselves: a peer claiming a node twice, a peer claiming
        // a node that is not in this cluster, and a node this cluster expects that never answered. The
        // first version of this collapsed them and could index an empty array while doing it.
        var seenNodes: Set<Int> = []
        for node in claimed where !seenNodes.insert(node).inserted {
            throw ClusterBringUpError.unexpectedNode(node, expected: expected)
        }
        if let stranger = claimed.first(where: { !expected.contains($0) }) {
            throw ClusterBringUpError.unexpectedNode(stranger, expected: expected)
        }
        if let missing = expected.first(where: { !claimed.contains($0) }) {
            throw ClusterBringUpError.missingNode(missing)
        }
        return theirs.sorted { $0.node < $1.node }
    }
}
