import Foundation

/// A node's **persistent** connections to the peers it exchanges experts with.
///
/// The exchange happens once per layer, forty times per token, so a connection per exchange would pay a TCP
/// handshake per layer - and `D173` measured the whole per-step exchange at 17.3 ms, which a handshake per layer
/// would dwarf. This holds one `ShardPeerChannel` per peer for the life of the run.
///
/// It is deliberately **not** the thing that decides who to ask: `ShardExchangeParticipant` owns that, from the
/// plan, and calls `exchange(_:to:)` with the peer already chosen. This type only routes, and it fails loudly on
/// a peer it was never given rather than silently answering for another - the failure mode that would present as
/// a wrong token.
///
/// **Sendability.** The channel table is written once by `connect()` and read-only afterwards, and the decode
/// loop drives one step at a time from one task, which is the same contract `PreadExpertStreamer` and
/// `StreamingMTPDecoder` carry with `@unchecked Sendable`.
///
/// unchecked-invariant: the connections dictionary is written only by `connect()`, which runs once before any
/// request; every later access is a read of an already-published entry, and the per-peer `ShardPeerChannel`s are
/// each used by one task at a time because the decode loop drives one layer at a time from one task.
public final class ShardPeerSet: ShardTransport, @unchecked Sendable {
    public enum Error: Swift.Error, Equatable {
        /// A request was routed to a peer this node has no connection to. Named rather than defaulted, because
        /// answering from another peer's channel would produce a plausible wrong number.
        case unknownPeer(Int)
        case notConnected(Int)
    }

    public let plan: ShardPlan
    public let node: Int

    /// Where each peer listens. A peer absent from this map is never asked and `connect()` does not touch it.
    private let addresses: [Int: (host: String, port: UInt16)]
    private var channels: [Int: ShardPeerChannel] = [:]
    private let lock = NSLock()

    public init(plan: ShardPlan,
                node: Int,
                peers: [Int: (host: String, port: UInt16)]) {
        self.plan = plan
        self.node = node
        self.addresses = peers.filter { $0.key != node }
    }

    /// Which peers this node will ask at all, from the plan: a peer that owns none of the model's experts is
    /// never worth a connection.
    public func reachablePeers() -> [Int] {
        let owners = Set(plan.owners)
        return addresses.keys.filter { owners.contains($0) }.sorted()
    }

    /// Connect to every reachable peer, once.
    ///
    /// Sorted by peer so a failure names the first unreachable node in a stable order, and so two runs that fail
    /// differently are comparable.
    public func connect() throws {
        for peer in reachablePeers() {
            guard let address = addresses[peer] else { continue }
            let connected = try DecodeTCPSocket.connect(host: address.host, port: address.port)
            lock.lock()
            channels[peer] = ShardPeerChannel(input: connected.input, output: connected.output)
            lock.unlock()
        }
    }

    /// The peers currently connected, for a startup line that says what the node actually reached.
    public var connectedPeers: [Int] {
        lock.lock(); defer { lock.unlock() }
        return channels.keys.sorted()
    }

    public func exchange(_ request: ShardExchange.Request, to peer: Int) throws -> ShardExchange.Reply {
        lock.lock()
        let channel = channels[peer]
        lock.unlock()
        guard let channel else {
            throw addresses[peer] == nil ? Error.unknownPeer(peer) : Error.notConnected(peer)
        }
        try channel.send(try ShardExchange.encode(request))
        return try ShardExchange.decodeReply(from: try channel.receive())
    }

    public func close() {
        lock.lock(); defer { lock.unlock() }
        for (_, channel) in channels { channel.close() }
        channels.removeAll()
    }
}
