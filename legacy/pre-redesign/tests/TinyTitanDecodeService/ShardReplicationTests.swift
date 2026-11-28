import Testing

@testable import TinyTitanDecodeProtocol

/// Replication is the only lever that removes bytes from the wire without touching the arithmetic — and the
/// arithmetic in this suite is what stops it being sized by wishful thinking.
@Suite("Replicated expert selection")
struct ShardReplicationTests {
  @Test func theMostRoutedExpertsAreChosen() throws {
    // Expert 3 routed most, then 1, then 4; the rest never.
    var counts = [Int](repeating: 0, count: 8)
    counts[3] = 100
    counts[1] = 50
    counts[4] = 25

    let chosen = try ShardReplication.replicatedExperts(routingCounts: counts, experts: 8, count: 2)
    #expect(chosen == [1, 3])
  }

  @Test func tiesAreBrokenByExpertIndexSoTwoNodesAgree() throws {
    // All counts equal: the choice must still be deterministic, because a replication decision that differed
    // between nodes would make `isLocal` disagree across the cluster and one node would wait for an expert
    // another had decided it already held.
    let counts = [Int](repeating: 7, count: 10)
    let first = try ShardReplication.replicatedExperts(routingCounts: counts, experts: 10, count: 4)
    for _ in 0..<20 {
      #expect(try ShardReplication.replicatedExperts(routingCounts: counts, experts: 10, count: 4) == first)
    }
    #expect(first == [0, 1, 2, 3])
  }

  @Test func theResultIsCanonicalSoTheDigestIsStable() throws {
    var counts = [Int](repeating: 0, count: 8)
    counts[7] = 9
    counts[2] = 8
    counts[5] = 7
    let chosen = try ShardReplication.replicatedExperts(routingCounts: counts, experts: 8, count: 3)
    #expect(chosen == chosen.sorted())
    #expect(chosen == [2, 5, 7])
  }

  @Test func zeroIsAllowedAndAsksForNothing() throws {
    let counts = [Int](repeating: 1, count: 4)
    #expect(try ShardReplication.replicatedExperts(routingCounts: counts, experts: 4, count: 0).isEmpty)
  }

  @Test func askingForMoreThanTheModelHasIsRefusedNotClamped() {
    // Clamping would replicate fewer experts than asked for while reporting success - which is how a wire
    // budget ends up wrong by exactly the amount it was relying on.
    let counts = [Int](repeating: 1, count: 4)
    #expect(throws: ShardReplication.Error.countExceedsModel(requested: 5, experts: 4)) {
      try ShardReplication.replicatedExperts(routingCounts: counts, experts: 4, count: 5)
    }
    #expect(throws: ShardReplication.Error.negativeCount(-1)) {
      try ShardReplication.replicatedExperts(routingCounts: counts, experts: 4, count: -1)
    }
    #expect(throws: ShardReplication.Error.countMismatch(routingCounts: 3, experts: 4)) {
      try ShardReplication.replicatedExperts(routingCounts: [1, 2, 3], experts: 4, count: 1)
    }
  }

  /// **The correction this suite exists to force.** An expert ID is replicated in **every layer** — expert 64
  /// of layer 0 is a different weight from expert 64 of layer 1 — so the resident cost of R experts is
  /// `R x layers x expertBytes`, not `R x expertBytes`.
  ///
  /// The earlier plan recorded R = 64 as **"116 MB/node"**. That figure is `64 x 1,769,472 = 113 MB`, which is
  /// **one layer**, so the whole-model cost was understated by the layer count: **40x**. On a node with a
  /// 2.83 GB measured-optimal expert cache and ~1.9 GB of dense weights in 8 GB of RAM, 4.53 GB of replicated
  /// experts does not fit at all.
  @Test func wholeModelReplicationOf64ExpertsDoesNotFitOnThisNode() {
    let r64 = ShardReplication.residentBytes(count: 64, layers: 40)
    #expect(r64 == 4_529_848_320)                        // 4.53 GB, not 116 MB
    #expect(r64 > 4_000_000_000)
    // The per-layer figure that was mistaken for the total:
    #expect(ShardReplication.expertBytes * 64 == 113_246_208)   // ~113 MB
    // And what actually fits: the cache alone is 2.83 GB of an 8 GB machine.
    let cache = 40 * ShardReplication.expertBytes * 40        // 40 slots x 40 layers
    #expect(cache == 2_831_155_200)
    #expect(r64 + cache > 7_000_000_000, "replication plus the measured cache already exceeds the machine")
  }

  @Test func aSizedSetCostsWhatItCosts() {
    // Small sets are what fit: 8 experts across 40 layers is 566 MB, which is affordable only if the cache is
    // reduced - and D178 measured that reducing the cache costs more throughput than replication returns.
    #expect(ShardReplication.residentBytes(count: 8, layers: 40) == 566_231_040)
    #expect(ShardReplication.uniformShare(count: 64, experts: 256) == 0.25)
    #expect(ShardReplication.uniformShare(count: 0, experts: 256) == 0)
  }

  @Test func theWireModelScalesWithShareAndNotWithCountDirectly() {
    // Under uniform routing R = 64 of 256 removes a quarter of the routed slots' bytes per layer per step.
    let share = ShardReplication.uniformShare(count: 64, experts: 256)
    let saving = ShardReplication.modelledWireSavingBytes(
      count: 64, experts: 256, routedPerLayer: 8, layers: 40, share: share)
    // 8 routed slots x 0.25 x 40 layers = 80 expert-reads avoided, at 1,769,472 B each.
    #expect(saving == 141_557_760)
    // No replication, no saving.
    #expect(ShardReplication.modelledWireSavingBytes(
      count: 0, experts: 256, routedPerLayer: 8, layers: 40, share: 0) == 0)
  }
}
