import Foundation

/// A transport the exchange can be driven over, so the merging logic is testable without a socket.
///
/// `ShardPeerChannel` is the real one — a length-prefixed frame over two `FileHandle`s, used across the LAN by
/// `DecodeTCPSocket`. This protocol exists so the *decision* logic (who to ask, what to send, how to merge) can
/// be tested on one machine, which is the part that can be wrong in a way a four-node run would only show as a
/// wrong token.
public protocol ShardTransport: Sendable {
    /// Send one request to **one named peer** and return its reply.
    ///
    /// The peer is an argument and not something the transport infers, because a node routes to several peers
    /// at once and the same request shape goes to each. An earlier version of this protocol omitted it, which
    /// worked only in the integration test - that test has a single peer, so `requests(...)` returning them in
    /// peer order was indistinguishable from the transport knowing the peer. With four nodes it is not: a
    /// transport that cannot be told which peer to talk to can only ever reach one.
    func exchange(_ request: ShardExchange.Request, to peer: Int) throws -> ShardExchange.Reply
}

/// One node's part in a layer's expert exchange.
///
/// The division of labour is deliberate and is what keeps the arithmetic exact:
///
/// * The **engine** computes this node's own partials, for the slots it owns, in the reference's layout.
/// * This type **asks** the peers that own the rest, and returns their replies.
/// * `ShardReduce` adds everything, zero-padded, in slot order.
///
/// Nothing here computes a mixture value. That matters: `moe_phase2_down_reduce_k8` is a fixed k = 8 fp32 sum
/// in slot order, and the only way to reproduce it across nodes is for every node's contribution to be a
/// partial in that same sum rather than a separately-reduced number. A peer that returned a *summed* answer
/// could not be merged exactly, because the association would already have been chosen.
public struct ShardExchangeParticipant {
    public enum Error: Swift.Error, Equatable {
        case layerMismatch(expected: Int, got: Int)
        case slotOutOfRange(Int)
        case dimensionMismatch(expected: Int, got: Int)
        case expertOutOfRange(Int)
    }

    public let plan: ShardPlan
    public let node: Int
    public let transport: ShardTransport

    public init(plan: ShardPlan, node: Int, transport: ShardTransport) {
        self.plan = plan
        self.node = node
        self.transport = transport
    }

    /// The peers that own at least one of `experts`, and which of them they own.
    ///
    /// A peer is asked **once per layer**, carrying every slot it owns, rather than once per expert: the reply
    /// is one frame either way, and a node routing eight experts to four peers would otherwise pay four times
    /// the latency for the same bytes (`D173` — the per-step frame beats the per-layer one, and within a step
    /// the same argument applies to the peers).
    public func requests(layer: Int,
                         experts: [Int],
                         slots: [Int],
                         activation: [Float]) -> [(peer: Int, request: ShardExchange.Request)] {
        precondition(experts.count == slots.count, "experts and slots are carried together by construction")
        var byPeer: [Int: (experts: [Int], slots: [Int])] = [:]
        for (index, expert) in experts.enumerated() {
            guard expert >= 0, expert < plan.experts else { continue }
            // `isLocal`, not `owner(of:) != node`: a REPLICATED expert is held by every node, so this node
            // must not ask anyone for it. Asking by ownership alone would put a replicated expert on the wire
            // - exactly the bytes replication exists to remove.
            guard !plan.isLocal(expert: expert, to: node) else { continue }
            let owner = plan.owner(of: expert)
            byPeer[owner, default: ([], [])].experts.append(expert)
            byPeer[owner, default: ([], [])].slots.append(slots[index])
        }
        // Sorted by peer so two runs produce the same request order, which keeps a failure reproducible.
        return byPeer.keys.sorted().map { peer in
            let group = byPeer[peer]!
            return (peer: peer,
                    request: ShardExchange.Request(layer: layer, slots: group.slots,
                                                   experts: group.experts, activation: activation))
        }
    }

    /// Ask every peer and return their replies, in **peer order**.
    ///
    /// The order is immaterial to the result — `ShardReduce.accumulate` is exact under any order — and that is
    /// the point of returning them as a list rather than folding here: the caller hands all of them to
    /// `ShardReduce` together with its own partial, so there is exactly one place the sum is formed.
    public func replies(layer: Int,
                        experts: [Int],
                        slots: [Int],
                        activation: [Float]) throws -> [ShardExchange.Reply] {
        // Asked in peer order, and each request carries the peer it goes to, so a transport with several
        // connections can route it and a failure names the node that failed.
        try requests(layer: layer, experts: experts, slots: slots, activation: activation)
            .map { try transport.exchange($0.request, to: $0.peer) }
    }

    /// The remote buffer the sharded phase-2 kernel reads, laid out `[d][8]` fp32 exactly as
    /// `moe_phase2_down_reduce_k8_remote` indexes it: `remote[d * 8 + sg]`.
    ///
    /// This is the whole CPU side of the call site. The kernel adds `remote[d*8+sg]` into `partial[sg]`
    /// **before** the ordered k = 8 sum, so what crosses the wire has to be a per-slot partial and not a
    /// reduced value - a peer that returned a sum could not be merged exactly, because the association would
    /// already have been chosen (`D180`).
    ///
    /// **This node's own slots are left at zero**, and that is not a placeholder: the kernel computes them
    /// itself from its own blobs, and adding a peer's zero for a slot it owns is exact (`ShardReduce`), while
    /// writing this node's own value here would count it twice.
    ///
    /// The layout is transposed relative to `ShardReduce`'s rows - rows are `[slot][dimension]` and this is
    /// `[dimension][slot]` - because the kernel indexes by dimension and runs one threadgroup per `d`.
    public func remotePartials(layer: Int,
                               experts: [Int],
                               slots: [Int],
                               activation: [Float],
                               dims: Int) throws -> [Float] {
        let rows = try contributions(layer: layer, experts: experts, slots: slots,
                                     activation: activation, dims: dims)
        var out = [Float](repeating: 0, count: dims * ShardReduce.k8)
        for slot in 0..<ShardReduce.k8 {
            guard slot < rows.count else { break }
            let row = rows[slot]
            for d in 0..<min(dims, row.count) {
                out[d * ShardReduce.k8 + slot] = row[d]
            }
        }
        return out
    }

    /// The peer contributions, ready to be added to this node's own partials.
    ///
    /// Returned as `[slot][dimension]` rows for `ShardReduce.reduceRows`, and **re-sorted by the slot each row
    /// belongs to** rather than trusted in arrival order: a reply carries its slots explicitly (`D168`), and
    /// renumbering or reordering them changes the answer (`D154`).
    public func contributions(layer: Int,
                              experts: [Int],
                              slots: [Int],
                              activation: [Float],
                              dims: Int) throws -> [[Float]] {
        var out = [[Float]](repeating: [Float](repeating: 0, count: dims), count: ShardReduce.k8)
        for reply in try replies(layer: layer, experts: experts, slots: slots, activation: activation) {
            guard reply.layer == layer else {
                throw Error.layerMismatch(expected: layer, got: reply.layer)
            }
            for (index, slot) in reply.slots.enumerated() {
                guard slot >= 0, slot < ShardReduce.k8 else { throw Error.slotOutOfRange(slot) }
                guard let row = reply.row(at: index) else {
                    // The reply does not carry a row for this slot. Reported as the shape mismatch it is, with
                    // the numbers, rather than as the trap the slice used to be.
                    throw Error.dimensionMismatch(expected: dims,
                                                  got: index * reply.dimensions > reply.values.count
                                                      ? 0 : reply.values.count - index * reply.dimensions)
                }
                guard row.count == dims else {
                    throw Error.dimensionMismatch(expected: dims, got: row.count)
                }
                // `row` is an ArraySlice, AND AN ARRAYSLICE KEEPS ITS PARENT'S INDICES. `reply.row(at: 1)` is
                // `values[4..<8]`, so `row[0]` does not exist - reading it traps with
                // `SliceBuffer.swift:317: Fatal error: Index out of bounds` and no frame naming this file. That
                // is why only the SECOND slot ever crashed, and why every test with one slot per peer passed:
                // index 0 gives a slice whose startIndex is 0, so the two spellings agree. Offsetting by
                // `startIndex` is the fix, and it is the reason this bug survived twelve rounds of otherwise
                // thorough exchange testing.
                for d in 0..<dims { out[slot][d] += row[row.startIndex + d] }
            }
        }
        return out
    }
}
