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
    /// What this node's all-reduces cost (`DC-081`). A class, so a `let` in this struct accumulates.
    public let ledger = ExchangeLedger()
    public var exchangeMetrics: ExchangeMetrics { ledger.metrics }

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

    /// Rows per head-slice frame. 4,096 fp32 rows is 16 KB — deliberately far inside a socket buffer, because
    /// the whole point of chunking is that nothing large is ever in flight (see `gatherHeadSlice`).
    public static let headSliceChunkRows = 4_096

    /// Exchange head-logit slices so that every node ends with the **same** full array.
    ///
    /// Each node sends the rows it computed and receives everyone else's, so the argmax, the margin and the
    /// captured trace stay bit-identical to the single-node run rather than forcing the gate to compare tokens
    /// only. That is worth about a megabyte per step: the head is 1.05 s/step of *replicated* work (`D88`), and
    /// a megabyte on a LAN is milliseconds.
    ///
    /// **The transfer is chunked, and that is not an optimisation.** A full slice is roughly a megabyte; if
    /// every node sent a megabyte before reading anything, the sends would fill the socket buffers and every
    /// node would block in `send` waiting for a peer that is itself blocked in `send`. Small frames never reach
    /// that and a test with a small fixture never would either. So the vocabulary is cut into fixed windows —
    /// the same windows on every node, so the message count is the same everywhere — and within a window each
    /// node sends one bounded frame and reads one from each peer. A node whose slice misses a window sends an
    /// empty frame rather than nothing, which is what keeps the reads matched.
    public func gatherHeadSlice(into logits: inout [Float], slice: VocabSlice) throws {
        guard ownership.nodes > 1, logits.count == slice.vocabSize else { return }
        let mine = slice.range
        if mine.isEmpty && peers.isEmpty { return }
        var window = 0
        while window < slice.vocabSize {
            let windowRange = window..<min(window + Self.headSliceChunkRows, slice.vocabSize)
            // The part of this window this node owns: empty when the window is not this node's business.
            let rows = windowRange.clamped(to: mine)
            let frame = try HeadSliceWire.encode(start: rows.lowerBound, values: Array(logits[rows]))
            for peer in peers { peer.applyTimeout(milliseconds: policy.receiveTimeoutMilliseconds) }
            for peer in peers { try peer.send(frame) }
            for peer in peers {
                let decoded = try HeadSliceWire.decode(try peer.receive())
                guard decoded.start >= 0, decoded.start + decoded.values.count <= logits.count else {
                    throw ShardExchangeError.headSliceOutOfRange(
                        start: decoded.start, count: decoded.values.count, vocabulary: logits.count
                    )
                }
                for (index, value) in decoded.values.enumerated() { logits[decoded.start + index] = value }
            }
            window += Self.headSliceChunkRows
        }
    }
}
