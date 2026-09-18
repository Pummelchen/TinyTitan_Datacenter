import Foundation
import Testing

@testable import TinyTitanDecodeProtocol

/// The peer set routes; it does not decide. These tests pin the routing and, more importantly, that it
/// **fails on a peer it does not have** rather than answering from another - the failure that would present as
/// a wrong token rather than as an error.
@Suite("Shard peer set", .serialized)
struct ShardPeerSetTests {
  private func plan() -> ShardPlan {
    ShardPlan.generate(family: "qwen3_5_moe", experts: 8, nodes: 4, distribution: .roundRobin)
  }

  @Test func onlyPeersThatOwnExpertsAreWorthConnecting() {
    let p = plan()
    // A node with a connection to itself, and to a peer index the plan has no owners for.
    let set = ShardPeerSet(plan: p, node: 0, peers: [
      0: (host: "127.0.0.1", port: 1),
      1: (host: "127.0.0.1", port: 2),
      9: (host: "127.0.0.1", port: 3),
    ])
    // Node 0 is dropped because it is us; 9 is dropped because the plan gives it no experts.
    #expect(set.reachablePeers() == [1])
  }

  @Test func aPeerWithNoConnectionIsRefusedByNameNotAnsweredFromAnother() {
    let set = ShardPeerSet(plan: plan(), node: 0, peers: [1: (host: "127.0.0.1", port: 2)])
    let request = ShardExchange.Request(layer: 0, slots: [0], experts: [1], activation: [1])
    // Never in the address map: a routing mistake, and it says so.
    #expect(throws: ShardPeerSet.Error.unknownPeer(7)) {
      _ = try set.exchange(request, to: 7)
    }
    // Known but not yet connected: a lifecycle mistake, and it says that instead.
    #expect(throws: ShardPeerSet.Error.notConnected(1)) {
      _ = try set.exchange(request, to: 1)
    }
  }

  @Test func aConnectedPeerAnswersOverARealSocket() async throws {
    let port: UInt16 = 45947
    let request = ShardExchange.Request(layer: 3, slots: [0, 1], experts: [4, 5], activation: [0.25, 0.5])
    let dims = 2

    // The peer: accept once, answer once, close.
    let server = Task { () -> Void in
      let accepted: (input: FileHandle, output: FileHandle) =
        try await withCheckedThrowingContinuation { continuation in
          DispatchQueue.global().async {
            do {
              continuation.resume(returning: try DecodeTCPSocket.listenAndAccept(
                host: "127.0.0.1", port: port))
            } catch {
              continuation.resume(throwing: error)
            }
          }
        }
      let channel = ShardPeerChannel(input: accepted.input, output: accepted.output)
      let received = try ShardExchange.decodeRequest(from: try channel.receive())
      // Answer in the requested order, which is the peer's half of the slot contract.
      #expect(received.experts == request.experts)
      try channel.send(try ShardExchange.encode(ShardExchange.Reply(
        layer: received.layer, slots: received.slots, dimensions: dims,
        values: [10, 11, 20, 21])))
      channel.close()
    }

    let set = ShardPeerSet(plan: plan(), node: 0, peers: [1: (host: "127.0.0.1", port: port)])
    var reply: ShardExchange.Reply?
    for _ in 0..<100 {
      do {
        try set.connect()
        reply = try set.exchange(request, to: 1)
        break
      } catch {
        try await Task.sleep(nanoseconds: 20_000_000)
      }
    }
    try await server.value

    let got = try #require(reply)
    #expect(got.layer == 3)
    #expect(got.slots == [0, 1])
    #expect(Array(try #require(got.row(at: 0))) == [10, 11])
    #expect(Array(try #require(got.row(at: 1))) == [20, 21])
    #expect(set.connectedPeers == [1])
    set.close()
    #expect(set.connectedPeers.isEmpty)
  }
}
