import Foundation
import Testing

@testable import TinyTitanDecodeProtocol

/// The whole exchange, both halves, over a real socket: a participant asks, a server answers by running the
/// named experts, and the replies are merged into the slot-ordered sum the kernel will consume.
///
/// Every piece is tested alone elsewhere. This is the test that they **compose**, and `D207` is why it matters
/// here specifically: until the server existed, every exchange test used a fake peer, so "the client and the
/// server agree" was never exercised by anything that ships.
///
/// **This test found a real bug and was withdrawn for it.** It crashed with
/// `SliceBuffer.swift:317: Fatal error: Index out of bounds`, which bisection localised to
/// `ShardExchangeParticipant.contributions` and then to an `ArraySlice` being indexed from zero when its parent's
/// indices started at four - so only the SECOND slot a peer owned ever crashed. It is restored now that the fix
/// is in, and it is the case that keeps the fix honest.
@Suite("Shard exchange end to end", .serialized)
struct ShardExchangeEndToEndTests {
  private static let port: UInt16 = 45957
  private static let dims = 4

  /// Deterministic in (layer, expert, dimension), so a contribution merged onto the wrong slot, or dropped, is
  /// detectably wrong rather than plausible.
  private static func compute(layer: Int, experts: [Int], activation: [Float]) -> [Float] {
    experts.flatMap { expert in
      (0..<activation.count).map { d in Float(layer) * 100 + Float(expert) + Float(d) / 10 }
    }
  }

  @Test func aPeerContributionMergesIntoTheSingleNodeAnswerInSlotOrder() async throws {
    // 8 experts over 4 nodes round-robin: node 0 owns 0 and 4; node 2 owns 2 and 6.
    let plan = ShardPlan.generate(family: "qwen3_5_moe", experts: 8, nodes: 4, distribution: .roundRobin)
    let dims = Self.dims
    let activation = [Float](repeating: 0.5, count: dims)

    let server = ShardExchangeServer(port: Self.port) { layer, experts, act in
      Self.compute(layer: layer, experts: experts, activation: act)
    }
    // serve is async now, so the continuation-around-a-blocking-dispatch is unnecessary: the Task awaits it
    // directly and holds no thread while it waits.
    let serving = Task { () -> Void in
      do { try await server.serve(connections: 1) } catch { }
    }

    // RETRY THE CONNECT. The server binds on another thread, so a client that connects immediately can be
    // refused - which it was, and only under the full suite, where scheduling differs from running this test
    // alone. Passing in isolation and failing in the suite is the shape of a race, not of a wrong assertion.
    var client: (input: FileHandle, output: FileHandle)?
    for _ in 0..<100 {
      client = try? DecodeTCPSocket.connect(host: "127.0.0.1", port: Self.port)
      if client != nil { break }
      try await Task.sleep(nanoseconds: 20_000_000)
    }
    let connected = try #require(client)
    let channel = ShardPeerChannel(input: connected.input, output: connected.output)
    let participant = ShardExchangeParticipant(
      plan: plan, node: 0, transport: DirectTransport(channel: channel))

    // Slots 0..3 carry experts 0 (ours), 2 (node 2's), 4 (ours), 6 (node 2's) - so ONE peer owns TWO slots,
    // which is the shape that crashed.
    let experts = [0, 2, 4, 6]
    let slots = [0, 1, 2, 3]
    let remote = try participant.remotePartials(layer: 5, experts: experts, slots: slots,
                                                activation: activation, dims: dims)
    // The server's loop is terminated by the peer closing, so close before awaiting it.
    channel.close()
    try await serving.value

    var nodeOwn = [[Float]](repeating: [Float](repeating: 0, count: dims), count: 8)
    for (index, expert) in experts.enumerated() where plan.owner(of: expert) == 0 {
      nodeOwn[slots[index]] = Self.compute(layer: 5, experts: [expert], activation: activation)
    }
    var peer = [[Float]](repeating: [Float](repeating: 0, count: dims), count: 8)
    for slot in 0..<8 where slot < remote.count / dims {
      for d in 0..<dims { peer[slot][d] = remote[d * 8 + slot] }
    }

    let merged = try ShardReduce.reduceRows([nodeOwn, peer], dimensions: dims,
                                            residuals: [Float](repeating: 0, count: dims))
    var full = [[Float]](repeating: [Float](repeating: 0, count: dims), count: 8)
    for (index, expert) in experts.enumerated() {
      full[slots[index]] = Self.compute(layer: 5, experts: [expert], activation: activation)
    }
    let single = try ShardReduce.reduceRows([full], dimensions: dims,
                                            residuals: [Float](repeating: 0, count: dims))

    for d in 0..<dims {
      #expect(merged[d].bitPattern == single[d].bitPattern,
              "dimension \(d): merged \(merged[d]) against single \(single[d])")
    }
    #expect(single.contains { $0 != 0 })
  }

  private struct DirectTransport: ShardTransport {
    let channel: ShardPeerChannel
    func exchange(_ request: ShardExchange.Request, to peer: Int) throws -> ShardExchange.Reply {
      try channel.send(try ShardExchange.encode(request))
      return try ShardExchange.decodeReply(from: try channel.receive())
    }
  }
}
