import XCTest

@testable import DatacenterEngine
@testable import DatacenterIR

/// The expert slot bank's footprint, against the node's memory — the arithmetic `D12` needed.
///
/// `expertSlotsPerLayer` used to be the literal `16`, which was never derived from anything: on the
/// real geometry that is **8.05 GB across 40 layers** against the brief's ~4.5 GB usable, and this file
/// reported it as an **expected failure** with a note saying the marker should be deleted deliberately
/// the day the bank was sized from a budget. That day is `D31`: the capacity now comes from
/// `expertBankBudgetBytes` and the layer's own geometry, and the marker is gone because the constraint
/// is met rather than recorded.
///
/// The real geometry matters here. A fixture's experts are kilobytes, so no fixture-scale test can see
/// a gigabyte — which is how a 14.5 GB change once shipped past 98 green tests. Geometry from
/// [`Qwen/Qwen3.6-35B-A3B`'s `config.json`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/raw/main/config.json),
/// confirmed present in the real install's `spec.config` (`DC-094`): `hidden_size` 2048,
/// `moe_intermediate_size` 512, `num_hidden_layers` 40, 256 routed experts per layer, top-8.
final class SlotBudgetTests: XCTestCase {
    struct RealModel {
        var hiddenSize = 2048
        var moeIntermediateSize = 512
        var layers = 40
        var experts = 256
        var topK = 8
    }

    /// The brief's limit: "Usable RAM per node after macOS: assume ~4.5 GB."
    let usableBytes = 4_500_000_000

    private func shape(_ model: RealModel) -> MixtureShape {
        MixtureShape(
            hiddenSize: model.hiddenSize, experts: model.experts, topK: model.topK,
            intermediate: model.moeIntermediateSize, sharedIntermediate: 512
        )
    }

    private func bytesPerExpert(_ model: RealModel) -> Int {
        // gate and up are fused into one `[2 * inter, hidden]` stack, down is `[hidden, inter]`, and
        // the cache holds fp32 reals.
        let gateUp = 2 * model.moeIntermediateSize * model.hiddenSize * 4
        let down = model.moeIntermediateSize * model.hiddenSize * 4
        return gateUp + down
    }

    private func totalBytes(_ model: RealModel, slots: Int) -> Int {
        // Per layer the bank holds up to `slots` experts' gate/up **and** up to `slots` experts' down
        // weights — two maps, two capacities, which is why the per-layer figure is `slots` times one
        // whole expert rather than half of one.
        model.layers * slots * bytesPerExpert(model)
    }

    /// The arithmetic itself, which is not in doubt and is the part the tracker quotes.
    func testTheArithmeticMatchesTheTrackersNumbers() {
        let model = RealModel()
        XCTAssertEqual(bytesPerExpert(model), 12_582_912, "one expert should be 12.58 MB")
        XCTAssertEqual(totalBytes(model, slots: 16), 8_053_063_680, "16 slots x 40 layers should be 8.05 GB")
        XCTAssertGreaterThan(totalBytes(model, slots: 16), usableBytes, "which is why 16 cannot be kept")
    }

    /// The constraint, now met. The default budget's bank fits, and the arithmetic that says so is the
    /// same arithmetic the reader above uses.
    func testTheDefaultBankFitsInTheNodesUsableMemory() {
        let model = RealModel()
        let slots = Qwen3_5Forward.expertSlots(
            budgetBytes: Qwen3_5Forward.expertBankBudgetBytes, layers: model.layers, shape: shape(model)
        )
        let total = totalBytes(model, slots: slots)
        XCTAssertGreaterThanOrEqual(slots, 1, "a bank of zero slots would silently disable the cache")
        // The number the sweep chose. `D31` measured the real install at 1, 2, 4, 8 and 16 slots:
        // **2218 expert requests, 0 hits, and byte-for-byte the same 3,060,562,432 bytes read at every
        // size** — routing does not repeat an expert inside a layer, so a slot never serves a hit and
        // paying memory for more of them buys nothing. The default is therefore the smallest bank, and
        // this pins it so a future change has to argue with the measurement rather than drift past it.
        XCTAssertEqual(
            slots, 1,
            "the default should be the smallest bank: the measured hit rate is 0 at every size, so more "
                + "slots cost memory for nothing (D31, DC-092)"
        )
        XCTAssertLessThanOrEqual(
            total, usableBytes,
            "the default bank is \(slots) slots = \(total / 1_000_000) MB against "
                + "\(usableBytes / 1_000_000) MB usable"
        )
    }

    /// The property, not just the answer: whatever the budget, the bank it buys fits inside it.
    func testTheFootprintNeverExceedsTheBudgetItWasDerivedFrom() {
        let model = RealModel()
        var budgets = [0, 1, 1_000_000, 512 * 1_048_576, 1_500_000_000, 8_000_000_000]
        budgets += (1...40).map { $0 * 100_000_000 }
        for budget in budgets {
            let slots = Qwen3_5Forward.expertSlots(
                budgetBytes: budget, layers: model.layers, shape: shape(model)
            )
            XCTAssertGreaterThanOrEqual(slots, 1, "budget \(budget)")
            XCTAssertLessThanOrEqual(slots, model.experts, "budget \(budget): no bank needs more experts than exist")
            if budget >= bytesPerExpert(model) * model.layers {
                XCTAssertLessThanOrEqual(
                    totalBytes(model, slots: slots), budget,
                    "a budget of \(budget) bought \(slots) slots, which cost more than it"
                )
            }
        }
    }

    /// The sweep instrument still works, and a bad value falls back to the budget rather than disabling
    /// the cache this exists to measure.
    func testTheSweepOverrideWinsAndBadValuesFallBackToTheBudget() {
        let model = RealModel()
        let fromBudget = Qwen3_5Forward.slotCapacity(
            environment: [:], shape: shape(model), layers: model.layers
        )
        XCTAssertEqual(
            Qwen3_5Forward.slotCapacity(
                environment: ["SHARD_EXPERT_SLOTS": "2"], shape: shape(model), layers: model.layers
            ), 2
        )
        XCTAssertEqual(
            Qwen3_5Forward.slotCapacity(
                environment: ["SHARD_EXPERT_SLOTS": "64"], shape: shape(model), layers: model.layers
            ), 64
        )
        for bad in ["0", "-3", "many", ""] {
            XCTAssertEqual(
                Qwen3_5Forward.slotCapacity(
                    environment: ["SHARD_EXPERT_SLOTS": bad], shape: shape(model), layers: model.layers
                ), fromBudget,
                "\(bad) is not a slot count; refusing it is the point, because a bank of zero would "
                    + "silently disable the cache this exists to measure"
            )
        }
    }

    /// A budget nobody could hold is refused rather than clamped quietly: a bank sized from `Int.max`
    /// is a node that swaps, which is exactly what `DC-091` reverted.
    func testAnAbsurdBudgetFallsBackInsteadOfBeingTrusted() {
        let sane = Qwen3_5Forward.bankBudgetBytes(environment: [:])
        for bad in ["-1", "many", "", "99999999"] {
            XCTAssertEqual(
                Qwen3_5Forward.bankBudgetBytes(environment: ["SHARD_EXPERT_BANK_MB": bad]), sane,
                "\(bad) is not a budget"
            )
        }
        XCTAssertEqual(
            Qwen3_5Forward.bankBudgetBytes(environment: ["SHARD_EXPERT_BANK_MB": "1024"]),
            1024 * 1_048_576
        )
    }

    /// What a given budget allows, so the relationship stays legible next to the number chosen.
    func testWhatATotalBudgetWouldAllow() {
        let model = RealModel()
        let budget = 1_500_000_000
        let slots = budget / (model.layers * bytesPerExpert(model))
        XCTAssertEqual(slots, 2, "1.5 GB across 40 layers is 2 whole slots per layer")
    }
}
