import Testing

@testable import TinyTitanDecodeProtocol

/// `D154`/`D168`: a sharded MoE reduce is **bit-identical** to the single-node one, by construction.
///
/// The claim under test is not "close enough". `moe_phase2_down_reduce_k8` is a fixed k = 8 fp32 sum in slot
/// order, and a node contributes **zero-padded across all eight slots** for the ones it does not own. Since
/// IEEE addition of zero is exact, accumulating padded contributions cannot perturb the value; what is left is
/// the same ordered sum of the same eight numbers. These tests assert **exact equality** on the bit pattern,
/// because that is the property the exchange depends on and a tolerance would not falsify it.
@Suite("Shard reduce is exact under any partition")
struct ShardReduceTests {
  /// Eight slot partials, deliberately awkward: negative, tiny, and a value where fp32 addition order is
  /// observable.
  private let partials: [Float] = [0.375, -1.0 / 3.0, 1e-7, 2.5, -0.125, 3.0, 0.1, -7.25]

  private func padded(_ values: [Float], owning slots: Set<Int>, k: Int = 8) -> [Float] {
    var out = [Float](repeating: 0, count: k)
    for (index, slot) in slots.sorted().enumerated() { out[slot] = values[index] }
    return out
  }

  @Test func oneNodeOwningEverySlotReproducesTheReferenceSum() throws {
    let single = ShardReduce.orderedSum(partials, start: 0)
    let reduced = try ShardReduce.reduce([partials], residual: 0)
    #expect(reduced == single)
    #expect(reduced.bitPattern == single.bitPattern)
  }

  @Test func twoNodesInAnyPartitionGiveTheIdenticalBitPattern() throws {
    let single = ShardReduce.orderedSum(partials, start: 0)

    // Two different partitions of the same eight slots, plus the degenerate ones.
    let partitions: [Set<Int>] = [
      [0, 2, 5], [1, 3, 4, 6, 7],       // uneven
      [0, 1, 2, 3], [4, 5, 6, 7],       // contiguous halves
      [0, 3, 6], [1, 4, 7], [2, 5],     // three nodes
      [], [0, 1, 2, 3, 4, 5, 6, 7],     // one node owns nothing at all
    ]

    for partition in partitions {
      let owned = partition.sorted()
      let mine = padded(owned.map { partials[$0] }, owning: partition)
      let theirs = padded(
        (0..<8).filter { !partition.contains($0) }.map { partials[$0] },
        owning: Set((0..<8).filter { !partition.contains($0) }))

      let sharded = try ShardReduce.reduce([mine, theirs], residual: 0)
      #expect(sharded.bitPattern == single.bitPattern,
              "partition \(owned) changed the answer: \(sharded) vs \(single)")
    }
  }

  @Test func theOrderOfRepliesDoesNotChangeTheResult() throws {
    let a = padded([partials[0], partials[2]], owning: [0, 2])
    let b = padded([partials[1], partials[3]], owning: [1, 3])
    let c = padded([partials[4], partials[5]], owning: [4, 5])

    let abc = try ShardReduce.reduce([a, b, c], residual: 0.25)
    let cba = try ShardReduce.reduce([c, b, a], residual: 0.25)
    let bca = try ShardReduce.reduce([b, c, a], residual: 0.25)
    #expect(abc.bitPattern == cba.bitPattern)
    #expect(abc.bitPattern == bca.bitPattern)
  }

  @Test func renumberingTheSlotsChangesTheAnswerSoSlotsMustTravel() {
    // Why `D154` insists slots are carried and never renumbered: the same eight values summed in a different
    // order can be a different fp32 number.
    //
    // The first version of this test used the suite's `partials` vector and **failed**, because that vector
    // happens to be order-insensitive — so it was asserting a property its own data could not demonstrate.
    // This is the classic observable case instead: 1.0 is absorbed by 1e8 when it is added second, and
    // survives when the two large terms cancel first.
    let big: Float = 1e8
    let ordered: [Float] = [big, 1.0, -big, 0, 0, 0, 0, 0]
    let renumbered: [Float] = [big, -big, 1.0, 0, 0, 0, 0, 0]

    let a = ShardReduce.orderedSum(ordered, start: 0)
    let b = ShardReduce.orderedSum(renumbered, start: 0)
    #expect(a != b, "the vector no longer demonstrates order sensitivity, so this test guards nothing")
    #expect(a == 0)
    #expect(b == 1.0)
  }

  @Test func aNodeOwningNothingContributesExactlyZero() throws {
    let zeros = [Float](repeating: 0, count: 8)
    let full = try ShardReduce.reduce([partials], residual: 0)
    let withIdlePeer = try ShardReduce.reduce([partials, zeros], residual: 0)
    #expect(withIdlePeer.bitPattern == full.bitPattern)
  }

  @Test func anUnpaddedContributionIsRefusedRatherThanAssumed() {
    // A short contribution is indistinguishable from one whose zeros were never sent, so it is an error and
    // not a shortfall to be padded here.
    #expect(throws: ShardReduce.Error.contributionNotPadded(expected: 8, got: 3)) {
      try ShardReduce.accumulate([[1, 2, 3]])
    }
    #expect(throws: ShardReduce.Error.contributionNotPadded(expected: 8, got: 2)) {
      try ShardReduce.accumulateRows([[[1], [2]]], dimensions: 1)
    }
  }

  @Test func rowsReduceLikeTheSingleNodePath() throws {
    let dimensions = 3
    // rows[slot][dimension] for the full single-node set.
    let full: [[Float]] = (0..<8).map { slot in
      (0..<dimensions).map { d in Float(slot) * 0.25 - Float(d) * 1e-6 }
    }
    let residuals: [Float] = [0.5, -0.25, 2.0]

    let single = try ShardReduce.reduceRows([full], dimensions: dimensions, residuals: residuals)
    // Split by slot: node A takes the even slots, node B the odd.
    let a = (0..<8).map { $0.isMultiple(of: 2) ? full[$0] : [Float](repeating: 0, count: dimensions) }
    let b = (0..<8).map { $0.isMultiple(of: 2) ? [Float](repeating: 0, count: dimensions) : full[$0] }
    let sharded = try ShardReduce.reduceRows([a, b], dimensions: dimensions, residuals: residuals)

    #expect(sharded.count == single.count)
    for d in 0..<dimensions {
      #expect(sharded[d].bitPattern == single[d].bitPattern, "dimension \(d): \(sharded[d]) vs \(single[d])")
    }
  }
}
