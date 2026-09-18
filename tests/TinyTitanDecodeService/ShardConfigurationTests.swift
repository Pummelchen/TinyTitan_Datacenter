import Testing

@testable import TinyTitanDecodeProtocol

/// A sharded run must fail at configuration time rather than compute a different answer from its peers. These
/// pin the three ways that can happen, each of which would otherwise present as a wrong token.
@Suite("Shard configuration")
struct ShardConfigurationTests {
  private func plan(experts: Int = 8, nodes: Int = 4) -> ShardPlan {
    ShardPlan.generate(family: "qwen3_5_moe", experts: experts, nodes: nodes, distribution: .roundRobin)
  }

  private func allPeers(_ n: Int, except me: Int) -> [Int: ShardConfiguration.PeerAddress] {
    var out: [Int: ShardConfiguration.PeerAddress] = [:]
    for p in 0..<n where p != me { out[p] = .init(host: "127.0.0.1", port: UInt16(9000 + p)) }
    return out
  }

  @Test func aCompleteConfigurationIsAcceptedAndDescribesItself() throws {
    let cfg = try ShardConfiguration(plan: plan(), node: 0, peers: allPeers(4, except: 0))
    #expect(cfg.peerIndices == [1, 2, 3])
    #expect(cfg.summary.contains("shard node=0/4"))
    #expect(cfg.summary.contains("peers=[1, 2, 3]"))
    // The digest is in the summary because it is the thing every node must agree on.
    #expect(cfg.summary.contains("digest="))
  }

  @Test func anOutOfRangeNodeIsRefused() {
    #expect(throws: ShardConfiguration.Error.nodeOutOfRange(node: 4, nodes: 4)) {
      _ = try ShardConfiguration(plan: plan(), node: 4, peers: allPeers(4, except: 4))
    }
  }

  @Test func aMissingPeerIsRefusedByNameBecauseItWouldChangeTheAnswer() {
    // Node 0 asking only node 1 would silently never receive node 2's and 3's contributions.
    var peers = allPeers(4, except: 0)
    peers.removeValue(forKey: 2)
    peers.removeValue(forKey: 3)
    #expect(throws: ShardConfiguration.Error.missingPeers([2, 3])) {
      _ = try ShardConfiguration(plan: plan(), node: 0, peers: peers)
    }
  }

  @Test func aNodeIsNotGivenAnAddressForItself() {
    var peers = allPeers(4, except: 0)
    peers[0] = .init(host: "127.0.0.1", port: 9999)
    #expect(throws: ShardConfiguration.Error.selfInPeers(0)) {
      _ = try ShardConfiguration(plan: plan(), node: 0, peers: peers)
    }
  }

  @Test func theTransportItBuildsReachesExactlyThePlanPeers() throws {
    let cfg = try ShardConfiguration(plan: plan(), node: 2, peers: allPeers(4, except: 2))
    let transport = cfg.makeTransport()
    #expect(transport.node == 2)
    #expect(transport.reachablePeers() == [0, 1, 3])
    // Nothing is connected until connect() is called, and the failure says which.
    #expect(transport.connectedPeers.isEmpty)
  }
}
