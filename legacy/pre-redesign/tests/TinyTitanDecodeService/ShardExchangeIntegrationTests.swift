import Foundation
import Testing

@testable import TinyTitanDecodeProtocol

/// The whole data path, end to end and without a model: a routed expert set is split by the plan, the remote
/// half crosses a **real socket** as exchange frames, and the results are assembled in **slot order**.
///
/// Each piece is tested alone elsewhere — the plan's ownership, the frame's round trip, the channel's framing.
/// This is the test that they **compose**, which is a different property: a plan that returns slots and a frame
/// that carries them can each be right while the assembly puts them in the wrong place, and the result would be a
/// number that is merely wrong rather than an error. `D154` is why that matters: the reference's reduce is a
/// fixed k = 8 accumulation, so slot order is the arithmetic.
@Suite("Shard exchange integration", .serialized)
struct ShardExchangeIntegrationTests {
    /// The real shape: 256 experts over 4 nodes, contiguous, so node 0 owns 0..<64 and node 3 owns 192..<256.
    private static func plan() -> ShardPlan {
        ShardPlan.generate(family: "qwen3_5_moe", experts: 256, nodes: 4, distribution: .contiguous)
    }

    /// A stand-in for running an expert: deterministic in (layer, expert, dimension), so an assembly that
    /// crosses two experts, or drops a slot, produces a detectably wrong vector rather than a plausible one.
    private static func compute(layer: Int, expert: Int, dimensions: Int) -> [Float] {
        (0..<dimensions).map { Float(layer * 1_000_000 + expert * 1_000 + $0) }
    }

    private static let dimensions = 32
    private static let port: UInt16 = 45931

    /// A routed set that deliberately straddles three owners: node 0 owns 10, 20 and 30; node 1 owns 100;
    /// node 3 owns 200; and slot 4 lands on a **replicated** expert so it must never reach the wire.
    private static let routed = [10, 100, 20, 200, 30]

    // DISABLED, with the reason, rather than deleted and rather than left to hang. It hangs, and a suite that
    // cannot run whole makes every other result in it unverifiable — which is exactly how this sat unnoticed
    // for the session.
    //
    // Two defects, both found by reading it rather than by guessing:
    //
    //   1. It never drives the NODE side. `run(node:)` is the PEER's role — read a request, compute every
    //      expert, reply in order — and the test calls it on the *client* handle with nothing sending
    //      requests, so it blocks on its first `receive()`.
    //   2. The peer loops `for _ in 0..<2`, but node 0 has ONE remote peer here: `routed` is
    //      [10, 100, 20, 200, 30], and with expert 100 replicated, `remoteSlots(amongRouted:by: 0)` is
    //      `{3: [200]}`. One request crosses, so the second `receive()` blocks forever.
    //
    // Fixing it properly means constructing a `ShardExchangeParticipant` over a `ShardTransport` backed by
    // this socket pair and asserting the assembled rows — which is the same wiring the decode path still
    // needs (DC-132), so the test belongs with that change rather than before it. `assemblyPreservesSlots`
    // below already asserts the arithmetic half without a socket, which is why disabling this loses less than
    // it looks like it does.
    @Test("a routed set is split, exchanged over a socket, and reassembled in slot order")
    func routedSetRoundTripsThroughAPeer() async throws {
        let plan = Self.plan()
        let node = 0
        // Node 1's expert 100 is replicated everywhere, which removes it from every peer's queue.
        let replicating = ShardPlan(
            family: plan.family, experts: plan.experts, nodes: plan.nodes,
            distribution: .contiguous, owners: plan.owners, replicated: [100]
        )

        // The accept MUST NOT run on a cooperative-pool thread. `listenAndAccept` blocks until a peer
        // arrives, and a `Task` that blocks holds one of the pool's threads for the whole wait — so the
        // `connect` below, which needs a thread from the same pool to make progress, never gets one. That
        // deadlock is why this suite hung for the entire session and why `swift test` could not be run whole:
        // no test reported a start, because the hang was in the test's own concurrency, not in the code under
        // test. `DispatchQueue.global()` gives the blocking call a real thread and the continuation carries
        // its result back into the async world.
        // Only node 3's expert 200 crosses. ONE peer, so ONE request — the body this replaces expected two
        // and blocked on the second `receive()` forever, which is why the suite could not run whole.
        let expectedRequests = replicating.remoteSlots(amongRouted: Self.routed, by: node).count

        // The accept MUST NOT run on a cooperative-pool thread. `listenAndAccept` blocks until a peer
        // arrives, and a `Task` that blocks holds one of the pool's threads for the whole wait — so the
        // `connect` below, which needs a thread from the same pool, never gets one. `DispatchQueue.global()`
        // gives the blocking call a real thread.
        let server = Task { () -> Void in
            let accepted: (input: FileHandle, output: FileHandle) =
                try await withCheckedThrowingContinuation { continuation in
                    DispatchQueue.global().async {
                        do {
                            continuation.resume(returning: try DecodeTCPSocket.listenAndAccept(
                                host: "127.0.0.1", port: Self.port))
                        } catch {
                            continuation.resume(throwing: error)
                        }
                    }
                }
            // The PEER role, which the previous body never ran on the peer's own handle.
            try await Self.run(node: accepted, requests: expectedRequests)
        }

        // The NODE role, which the previous body never drove at all.
        let activation = [Float](repeating: 0.5, count: Self.dimensions)
        var last: Error = POSIXError(.ECONNREFUSED)
        var rows: [[Float]] = []
        for _ in 0..<100 {
            do {
                let client = try DecodeTCPSocket.connect(host: "127.0.0.1", port: Self.port)
                let channel = ShardPeerChannel(input: client.input, output: client.output)
                let participant = ShardExchangeParticipant(
                    plan: replicating, node: node, transport: ChannelTransport(channel: channel))
                rows = try participant.contributions(
                    layer: 0, experts: Self.routed, slots: Array(0..<Self.routed.count),
                    activation: activation, dims: Self.dimensions)
                try await server.value
                break
            } catch {
                last = error
                try await Task.sleep(nanoseconds: 20_000_000)
            }
        }
        guard !rows.isEmpty else { throw last }

        // The peer's contribution lands on ITS slot and nowhere else — the whole point of the exercise.
        #expect(Array(rows[3]) == Self.compute(layer: 0, expert: 200, dimensions: Self.dimensions))
        for slot in [0, 1, 2, 4, 5, 6, 7] {
            #expect(Array(rows[slot]) == [Float](repeating: 0, count: Self.dimensions))
        }
    }

    /// The node side's transport: the exchange frames over the peer channel, which is a Unix socket or TCP
    /// without the participant knowing which.
    private struct ChannelTransport: ShardTransport {
        let channel: ShardPeerChannel
        func exchange(_ request: ShardExchange.Request, to peer: Int) throws -> ShardExchange.Reply {
            try channel.send(try ShardExchange.encode(request))
            return try ShardExchange.decodeReply(from: try channel.receive())
        }
    }

    /// The peer side: read each request, compute every requested expert, reply in the order asked.
    ///
    /// Keeping the reply in the requested order is the peer's half of the slot contract — it does not need to
    /// know the slots, only to answer in sequence, which is why `ShardExchange.Reply` carries them back anyway:
    /// the receiver must not have to trust that the order was preserved.
    private static func run(node handles: (input: FileHandle, output: FileHandle),
                            requests: Int) async throws {
        let channel = ShardPeerChannel(input: handles.input, output: handles.output)
        for _ in 0..<requests {
            let request = try ShardExchange.decodeRequest(from: try channel.receive())
            let values = request.experts.flatMap { expert in
                compute(layer: request.layer, expert: expert, dimensions: request.activation.count)
            }
            try channel.send(try ShardExchange.encode(ShardExchange.Reply(
                layer: request.layer, slots: request.slots,
                dimensions: request.activation.count, values: values
            )))
        }
    }

    /// The node side, asserted here rather than inside the socket dance so a failure names the arithmetic.
    @Test("assembly places every contribution in the slot the router gave it")
    func assemblyPreservesSlots() throws {
        let plan = Self.plan()
        let replicating = ShardPlan(
            family: plan.family, experts: plan.experts, nodes: plan.nodes,
            distribution: .contiguous, owners: plan.owners, replicated: [100]
        )

        // What this node computes itself, and what it must ask for.
        let local = replicating.ownedSlots(amongRouted: Self.routed, by: 0)
        let remote = replicating.remoteSlots(amongRouted: Self.routed, by: 0)

        // 10 at slot 0, 100 (replicated) at slot 1, 20 at slot 2, 30 at slot 4 — four of the five.
        #expect(local.map(\.slot) == [0, 1, 2, 4])
        // Only 200, which is node 3's, must cross the wire.
        #expect(remote.keys.sorted() == [3])
        #expect(remote[3]?.map(\.slot) == [3])

        // Assemble a full k = 8 contribution set: zeros everywhere, filled where a slot was computed.
        var assembled = [Float](repeating: 0, count: 8 * Self.dimensions)
        func place(slot: Int, expert: Int) {
            let row = Self.compute(layer: 0, expert: expert, dimensions: Self.dimensions)
            for dimension in 0..<Self.dimensions {
                assembled[slot * Self.dimensions + dimension] = row[dimension]
            }
        }
        for entry in local { place(slot: entry.slot, expert: entry.expert) }
        for entry in remote.values.flatMap({ $0 }) { place(slot: entry.slot, expert: entry.expert) }

        // Every routed slot is filled, in place; the three unused slots stay exactly zero.
        for (slot, expert) in Self.routed.enumerated() {
            let expected = Self.compute(layer: 0, expert: expert, dimensions: Self.dimensions)
            let actual = Array(assembled[(slot * Self.dimensions)..<((slot + 1) * Self.dimensions)])
            #expect(actual == expected, "slot \(slot) holds expert \(expert)")
        }
        for slot in [5, 6, 7] {
            let row = assembled[(slot * Self.dimensions)..<((slot + 1) * Self.dimensions)]
            #expect(row.allSatisfy { $0 == 0 }, "unused slot \(slot) must be exactly zero — the reduce sums it")
        }
        // And the owned/remote halves together are the whole routed set: nothing may be dropped silently.
        #expect(local.count + (remote.values.flatMap { $0 }.count) == Self.routed.count)
    }
}
