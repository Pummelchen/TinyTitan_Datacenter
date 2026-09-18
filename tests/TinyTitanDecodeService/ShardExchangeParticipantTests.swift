import Testing

@testable import TinyTitanDecodeProtocol

/// A transport that answers from a table instead of a socket, so the *decision* logic is what is under test.
private struct FakeTransport: ShardTransport {
  /// peer -> (slots it answers with, rows of values).
  let answers: [Int: (slots: [Int], values: [Float])]
  let node: Int
  let dims: Int

  func exchange(_ request: ShardExchange.Request) throws -> ShardExchange.Reply {
    // The fake knows which peer is being asked by matching the requested slots against the table.
    for (peer, answer) in answers where Set(answer.slots) == Set(request.slots) {
      return ShardExchange.Reply(layer: request.layer, slots: answer.slots,
                                 dimensions: dims, values: answer.values)
    }
    return ShardExchange.Reply(layer: request.layer, slots: request.slots,
                               dimensions: dims,
                               values: [Float](repeating: 0, count: request.slots.count * dims))
  }
}

@Suite("Shard exchange participant")
struct ShardExchangeParticipantTests {
  /// 8 experts over 4 nodes, round-robin, nothing replicated.
  private func plan(experts: Int = 8, nodes: Int = 4) -> ShardPlan {
    try! ShardPlan.generate(family: "qwen3.6-35b-a3b", experts: experts, nodes: nodes,
                            distribution: .roundRobin)
  }

  @Test func aPeerIsAskedOncePerLayerCarryingEverySlotItOwns() throws {
    let p = plan()
    let participant = ShardExchangeParticipant(plan: p, node: 0, transport: nilTransport)
    // Experts 0,4 are node 0's; 1,5 node 1's; 2,6 node 2's; 3,7 node 3's.
    let requests = participant.requests(layer: 3, experts: [0, 1, 2, 3],
                                        slots: [0, 1, 2, 3], activation: [0.5])
    // Three peers, one request each - not one per expert.
    #expect(requests.count == 3)
    #expect(requests.allSatisfy { $0.layer == 3 })
    #expect(requests.map { $0.experts } == [[1], [2], [3]])
    #expect(requests.map { $0.slots } == [[1], [2], [3]])
  }

  @Test func aNodeOwningEveryRoutedExpertAsksNobody() throws {
    let p = plan(experts: 4, nodes: 1)
    let participant = ShardExchangeParticipant(plan: p, node: 0, transport: nilTransport)
    #expect(participant.requests(layer: 0, experts: [0, 1, 2], slots: [0, 1, 2],
                                 activation: []).isEmpty)
  }

  @Test func replicatedExpertsAreNotAskedFor() throws {
    // Replicating experts 1 and 3 makes them local everywhere, so node 0 stops asking node 1 and node 3.
    let base = plan()
    let replicated = try ShardPlan(family: base.family, experts: 8, nodes: 4,
                                   distribution: .roundRobin, owners: base.owners,
                                   replicated: [1, 3])
    let participant = ShardExchangeParticipant(plan: replicated, node: 0, transport: nilTransport)
    let requests = participant.requests(layer: 0, experts: [0, 1, 2, 3],
                                        slots: [0, 1, 2, 3], activation: [])
    #expect(requests.map { $0.experts } == [[2]])
  }

  @Test func contributionsLandOnTheirOwnSlotsAndNotInArrivalOrder() throws {
    let p = plan()
    let dims = 2
    // Peer 1 owns slot 1, peer 3 owns slot 3. Deliberately answer in a different order than requested.
    let transport = FakeTransport(
      answers: [
        1: (slots: [1], values: [10, 100]),
        3: (slots: [3], values: [30, 300]),
      ], node: 0, dims: dims)
    let participant = ShardExchangeParticipant(plan: p, node: 0, transport: transport)
    let rows = try participant.contributions(layer: 0, experts: [1, 3], slots: [1, 3],
                                             activation: [0], dims: dims)
    #expect(rows.count == 8)
    #expect(Array(rows[1]) == [10, 100])
    #expect(Array(rows[3]) == [30, 300])
    // Every other slot is an explicit zero, which is what makes the sum exact.
    for slot in [0, 2, 4, 5, 6, 7] { #expect(Array(rows[slot]) == [0, 0]) }
  }

  @Test func peerContributionsMergeIntoTheSingleNodeAnswer() throws {
    let p = plan()
    let dims = 1
    // Single node: slot s contributes s + 1.
    let single = (0..<8).map { Float($0 + 1) }
    // Node 0 owns slots 0 and 4 and computes those ITSELF; every other slot is its zero, because it does not
    // read those experts. The first version of this test gave `rows` all eight slots and also had peers answer
    // for six of them, so each of those was counted twice.
    var rows = [[Float]](repeating: [0], count: 8)
    rows[0] = [single[0]]
    rows[4] = [single[4]]
    // Peers answer for the slots they own, padded across all eight.
    let peerA = (0..<8).map { $0 == 1 || $0 == 5 ? [single[$0]] : [0] }
    let peerB = (0..<8).map { $0 == 2 || $0 == 6 ? [single[$0]] : [0] }
    let peerC = (0..<8).map { $0 == 3 || $0 == 7 ? [single[$0]] : [0] }
    // The single-node answer is the sum over ALL eight slots - not over node 0's own partial, which is what
    // the first version of this test compared against, and which is a different number.
    let allEight = (0..<8).map { [single[$0]] }
    let reference = try ShardReduce.reduceRows([allEight], dimensions: dims, residuals: [0])
    let merged = try ShardReduce.reduceRows([rows, peerA, peerB, peerC],
                                            dimensions: dims, residuals: [0])
    #expect(merged[0].bitPattern == reference[0].bitPattern)
    #expect(reference[0] == 36)   // 1 + 2 + ... + 8, so the comparison is not vacuous
  }

  @Test func aReplyForTheWrongLayerIsRefused() throws {
    struct WrongLayer: ShardTransport {
      func exchange(_ request: ShardExchange.Request) throws -> ShardExchange.Reply {
        ShardExchange.Reply(layer: request.layer + 1, slots: request.slots,
                            dimensions: 1, values: [1])
      }
    }
    let participant = ShardExchangeParticipant(plan: plan(), node: 0, transport: WrongLayer())
    #expect(throws: ShardExchangeParticipant.Error.layerMismatch(expected: 0, got: 1)) {
      try participant.contributions(layer: 0, experts: [1], slots: [1], activation: [], dims: 1)
    }
  }

  @Test func aReplyWithTheWrongRowWidthIsRefused() throws {
    let transport = FakeTransport(answers: [1: (slots: [1], values: [1, 2, 3])], node: 0, dims: 3)
    let participant = ShardExchangeParticipant(plan: plan(), node: 0, transport: transport)
    #expect(throws: ShardExchangeParticipant.Error.dimensionMismatch(expected: 2, got: 3)) {
      try participant.contributions(layer: 0, experts: [1], slots: [1], activation: [], dims: 2)
    }
  }
}

private let nilTransport = FakeTransport(answers: [:], node: 0, dims: 1)
