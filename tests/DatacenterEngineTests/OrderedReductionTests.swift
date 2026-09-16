import XCTest

@testable import DatacenterEngine

/// The reduction contract (`D17`, I2) at fixture scale.
///
/// M2's gate is "2 nodes bit-identical to the 1-node baseline", and the only reason that can be true
/// is that the N-node reduction performs **the same sequence of additions** as the single-node path.
/// These tests hold the contract rather than the arithmetic: a partition may change who computes a
/// term, never the order the terms are summed in.
final class OrderedReductionTests: XCTestCase {
    /// A deterministic generator, so a failure is reproducible and no test depends on a seed.
    private struct Source {
        var state: UInt64 = 0x2545_F491_4F6C_DD1D
        mutating func next() -> Float {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Float(Int64(bitPattern: state >> 11) % 1_000) / 128
        }
    }

    private func makeContributions(tokens: Int, experts: Int, width: Int, active: Int) -> [ExpertContribution] {
        var source = Source()
        var contributions: [ExpertContribution] = []
        for token in 0..<tokens {
            for step in 0..<active {
                let expert = (token * 3 + step * 5) % experts
                contributions.append(
                    ExpertContribution(
                        token: token, expert: expert,
                        values: (0..<width).map { _ in source.next() },
                        scale: 0.125 + Float(step) / 8
                    )
                )
            }
        }
        return contributions
    }

    /// The single-node path, written out independently: a token's experts are accumulated in ascending
    /// expert id, with the weight applied before each addition. This is what the engine does today
    /// (`MixtureOfExperts.experts`), so it is the thing an N-node run has to reproduce.
    private func singleNode(_ contributions: [ExpertContribution], tokens: Int, width: Int) -> [Float] {
        var output = [Float](repeating: 0, count: tokens * width)
        for contribution in contributions.sorted(by: { $0.expert < $1.expert }) {
            let base = contribution.token * width
            for index in 0..<width {
                output[base + index] =
                    output[base + index] + contribution.values[index] * contribution.scale
            }
        }
        return output
    }

    func testTheReductionReproducesTheSingleNodeSequenceBitForBit() {
        let contributions = makeContributions(tokens: 5, experts: 8, width: 32, active: 4)
        let reference = singleNode(contributions, tokens: 5, width: 32)
        let reduced = OrderedReduction.accumulate(contributions, tokens: 5, hiddenSize: 32)

        for index in 0..<reference.count {
            XCTAssertEqual(
                reference[index].bitPattern, reduced[index].bitPattern,
                "element \(index): the contract must be the single-node sequence, bit for bit"
            )
        }
    }

    /// The invariant M2's gate rests on: a partition decides *who computes* a term and nothing else.
    /// Each partition's terms arrive in a different internal order, and the partitions are concatenated
    /// in a different order, exactly as a network would deliver them.
    func testPartitioningAndArrivalOrderDoNotMoveTheBits() {
        let contributions = makeContributions(tokens: 4, experts: 8, width: 16, active: 6)
        let reference = OrderedReduction.accumulate(contributions, tokens: 4, hiddenSize: 16)

        for partitions in [1, 2, 4, 8] {
            var delivered: [ExpertContribution] = []
            for node in (0..<partitions).reversed() {
                var owned = contributions.filter { $0.expert % partitions == node }
                owned.reverse()  // arrival order within a node is not the computation order
                delivered.append(contentsOf: owned)
            }
            let reduced = OrderedReduction.accumulate(delivered, tokens: 4, hiddenSize: 16)
            for index in 0..<reference.count {
                XCTAssertEqual(
                    reference[index].bitPattern, reduced[index].bitPattern,
                    "\(partitions) partition(s), element \(index): sharding is not allowed to move a bit"
                )
            }
        }
    }

    /// The trap the contract exists for, and the reason a node may not pre-sum the experts it owns.
    ///
    /// In fp32 at this magnitude the spacing is 2, so adding two terms to a third gives a different
    /// result depending on the grouping. A per-node partial (`b + c` first) lands on `20000006`; the
    /// canonical left-to-right sequence lands on `20000008`. Both are "the sum of the same three
    /// numbers" and they are not the same float — which is precisely the difference between passing
    /// M2's gate and failing it in a way that looks like a conversion bug.
    func testPreSummedPartialsWouldNotBeBitIdenticalAndTheContractIs() {
        let a: Float = 2e7, b: Float = 3, c: Float = 3
        let canonical = OrderedReduction.accumulate(
            [
                ExpertContribution(token: 0, expert: 0, values: [a], scale: 1),
                ExpertContribution(token: 0, expert: 1, values: [b], scale: 1),
                ExpertContribution(token: 0, expert: 2, values: [c], scale: 1),
            ],
            tokens: 1, hiddenSize: 1
        )[0]
        let preSummedPartial = a + (b + c)

        XCTAssertEqual(canonical, (a + b) + c, "the contract is the left-to-right sequence")
        XCTAssertNotEqual(
            canonical.bitPattern, preSummedPartial.bitPattern,
            "if these two agreed, this design would be unnecessary — the test is worthless if the trap closes"
        )
    }

    func testTheOrderKeyIsTokenThenExpert() {
        let terms = [
            ExpertContribution(token: 1, expert: 0, values: [0], scale: 1),
            ExpertContribution(token: 0, expert: 3, values: [0], scale: 1),
            ExpertContribution(token: 0, expert: 1, values: [0], scale: 1),
        ]
        let sorted = terms.sorted(by: OrderedReduction.precedes)
        XCTAssertEqual(sorted.map { "\($0.token):\($0.expert)" }, ["0:1", "0:3", "1:0"])
    }
}
