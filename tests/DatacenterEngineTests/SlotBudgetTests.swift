import XCTest

@testable import DatacenterEngine
@testable import DatacenterIR

/// The expert slot bank's footprint, against the node's memory, as an **expected failure**.
///
/// `D12` is open: `expertSlotsPerLayer` is the literal `16`, the brief asks for "per-layer LRU slot
/// banks" without saying how large, and nothing in the repository checks what that number costs. The
/// audit that found this is the same one that caught a 14.5 GB change being shipped with 98 green
/// tests — because a fixture's experts are kilobytes, no fixture-scale test can see it.
///
/// So this test does the arithmetic the fixture cannot, on the **checkpoint's** geometry, and asserts
/// the result against the brief's own hardware limit (`~4.5 GB` usable per node). It fails today, and
/// it is marked as an expected failure so that:
///
/// - the number is visible to anybody who runs the suite, without reading the wiki;
/// - the suite stays green and honest, because a known-unmet constraint recorded as met would be a lie;
/// - the day the bank is sized from a budget, this test reports an **unexpected pass** and has to be
///   deleted deliberately rather than quietly left behind.
///
/// Geometry, from [`Qwen/Qwen3.6-35B-A3B`'s `config.json`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/raw/main/config.json)
/// and confirmed present in the real install's `spec.config` (`DC-094`): `hidden_size` 2048,
/// `moe_intermediate_size` 512, `num_hidden_layers` 40, every layer a mixture of 256 routed experts.
final class SlotBudgetTests: XCTestCase {
    /// The checkpoint's geometry. Not the fixture's: the fixture is deliberately tiny and its numbers
    /// say nothing about the machine this has to run on.
    struct RealModel {
        var hiddenSize = 2048
        var moeIntermediateSize = 512
        var layers = 40
    }

    /// The brief's limit: "Usable RAM per node after macOS: assume ~4.5 GB."
    let usableBytes = 4_500_000_000

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
        XCTAssertGreaterThan(totalBytes(model, slots: 16), usableBytes, "and it does not fit")
    }

    /// The constraint, which is **not** met today. See `D12`.
    func testTheBankFitsInTheNodesUsableMemory() {
        XCTExpectFailure("""
            D12 is open: expertSlotsPerLayer = 16 costs 8.05 GB against ~4.5 GB usable, so the bank as \
            sized cannot be kept resident. If this test passes, the bank has been sized from a budget \
            and this marker should be removed rather than left here.
            """)
        let model = RealModel()
        let total = totalBytes(model, slots: Qwen3_5Forward.expertSlotsPerLayer)
        XCTAssertLessThanOrEqual(
            total, usableBytes,
            "the slot bank would need \(total / 1_000_000) MB against \(usableBytes / 1_000_000) MB usable"
        )
    }

    /// What a budget would actually allow, so the decision has its answer attached.
    func testWhatATotalBudgetWouldAllow() {
        let model = RealModel()
        let budget = 1_500_000_000
        let perLayerBudget = budget / model.layers
        let slots = perLayerBudget / bytesPerExpert(model)
        XCTAssertEqual(slots, 2, "1.5 GB across 40 layers is 2 whole slots per layer")
    }
}
