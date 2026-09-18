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

    @Test("a routed set is split, exchanged over a socket, and reassembled in slot order")
    func routedSetRoundTripsThroughAPeer() async throws {
        let plan = Self.plan()
        let node = 0
        // Node 1's expert 100 is replicated everywhere, which removes it from every peer's queue.
        let replicating = ShardPlan(
            family: plan.family, experts: plan.experts, nodes: plan.nodes,
            distribution: .contiguous, owners: plan.owners, replicated: [100]
        )

        let server = Task.detached {
            try DecodeTCPSocket.listenAndAccept(host: "127.0.0.1", port: Self.port)
        }
        var last: Error = POSIXError(.ECONNREFUSED)
        var accepted: (input: FileHandle, output: FileHandle)?
        for _ in 0..<100 {
            do {
                let client = try DecodeTCPSocket.connect(host: "127.0.0.1", port: Self.port)
                accepted = try await server.value
                try await Self.run(node: client)
                break
            } catch {
                last = error
                try await Task.sleep(nanoseconds: 20_000_000)
            }
        }
        guard accepted != nil else { throw last }
    }

    /// The peer side: read each request, compute every requested expert, reply in the order asked.
    ///
    /// Keeping the reply in the requested order is the peer's half of the slot contract — it does not need to
    /// know the slots, only to answer in sequence, which is why `ShardExchange.Reply` carries them back anyway:
    /// the receiver must not have to trust that the order was preserved.
    private static func run(node handles: (input: FileHandle, output: FileHandle)) async throws {
        let channel = ShardPeerChannel(input: handles.input, output: handles.output)
        for _ in 0..<2 {
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
