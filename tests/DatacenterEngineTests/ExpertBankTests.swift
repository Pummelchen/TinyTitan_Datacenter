import XCTest

@testable import DatacenterEngine

/// The generation-scoped expert bank (`DC-119`).
///
/// The point of the change is a **lifetime**, and a lifetime is invisible to a test that asks one question
/// once. So every test here asks twice: once to populate and once to see whether the second ask went to the
/// loader. The other half is the safety property — a slice must never cross a layer or a projection, because
/// the only thing that would notice is a wrong number in a trace.
final class ExpertBankTests: XCTestCase {
    /// A loader that counts and can be told to fail, so "it did not load" is distinguishable from "it loaded
    /// and returned something".
    private final class Loader: @unchecked Sendable {
        private let lock = NSLock()
        private var calls: [String] = []
        var count: Int { lock.lock(); defer { lock.unlock() }; return calls.count }
        var asked: [String] { lock.lock(); defer { lock.unlock() }; return calls }

        func load(_ what: String, values: [Float]) -> [Float] {
            lock.lock()
            calls.append(what)
            lock.unlock()
            return values
        }
    }

    private func shape(intermediate: Int = 32, hiddenSize: Int = 64) -> MixtureShape {
        MixtureShape(
            hiddenSize: hiddenSize, experts: 8, topK: 2, intermediate: intermediate,
            sharedIntermediate: intermediate
        )
    }

    func testASliceSurvivesTheLoadThatFetchedIt() {
        // The whole change in one test: the second ask is the *next token's* layer load, and it must not go
        // back to the source.
        let loader = Loader()
        let bank = ExpertBank(budgetBytes: 1 << 20)
        let first = bank.gateUp(layer: 3, expert: 5) { loader.load("gateUp", values: [1.5, -0.25]) }.values
        let second = bank.gateUp(layer: 3, expert: 5) { loader.load("gateUp", values: [9.0]) }.values
        XCTAssertEqual(loader.count, 1, "the second ask must be served from the bank")
        XCTAssertEqual(first.map(\.bitPattern), second.map(\.bitPattern), "a hit returns the loaded values")
        XCTAssertEqual(bank.metrics.hits, 1)
        XCTAssertEqual(bank.metrics.misses, 1)
        XCTAssertEqual(bank.metrics.hitRate, 0.5, accuracy: 1e-12)
    }

    func testASliceNeverCrossesALayer() {
        // The key includes the layer. If it did not, layer 4 would be answered with layer 3's weights and the
        // only symptom would be a wrong trace.
        let loader = Loader()
        let bank = ExpertBank(budgetBytes: 1 << 20)
        let atThree = bank.down(layer: 3, expert: 1) { loader.load("layer3", values: [3]) }.values
        let atFour = bank.down(layer: 4, expert: 1) { loader.load("layer4", values: [4]) }.values
        XCTAssertEqual(loader.count, 2)
        XCTAssertEqual(atThree, [3])
        XCTAssertEqual(atFour, [4])
    }

    func testASliceNeverCrossesAProjection() {
        // A fused gate/up and a down are different arrays of different widths; sharing one entry would return
        // the wrong one.
        let loader = Loader()
        let bank = ExpertBank(budgetBytes: 1 << 20)
        let gateUp = bank.gateUp(layer: 0, expert: 0) { loader.load("gateUp", values: [1, 2, 3, 4]) }.values
        let down = bank.down(layer: 0, expert: 0) { loader.load("down", values: [5, 6]) }.values
        XCTAssertEqual(loader.asked, ["gateUp", "down"])
        XCTAssertEqual(gateUp.count, 4)
        XCTAssertEqual(down.count, 2)
    }

    func testTheBudgetIsACeilingAndTheEvictionIsLeastRecentlyUsed() {
        // Four floats per slice, so a 16-byte budget holds four slices and the fifth must evict the oldest.
        let loader = Loader()
        let bank = ExpertBank(budgetBytes: 16)
        for expert in 0..<4 {
            _ = bank.gateUp(layer: 0, expert: expert) { loader.load("e\(expert)", values: [Float(expert)]) }
        }
        XCTAssertEqual(bank.metrics.misses, 4)
        // Touch expert 0 so expert 1 becomes the least recently used, then add a fifth.
        _ = bank.gateUp(layer: 0, expert: 0) { loader.load("e0-again", values: [0]) }
        _ = bank.gateUp(layer: 0, expert: 4) { loader.load("e4", values: [4]) }
        XCTAssertLessThanOrEqual(bank.residentBytes, 16, "the budget is a ceiling, not a target")
        XCTAssertEqual(bank.residentSlices, 4)
        // Expert 0 was touched, so it survived; expert 1 was the oldest and did not.
        let before = loader.count
        _ = bank.gateUp(layer: 0, expert: 0) { loader.load("e0-reload", values: [0]) }
        XCTAssertEqual(loader.count, before, "the most recently used slice is still resident")
        _ = bank.gateUp(layer: 0, expert: 1) { loader.load("e1-reload", values: [1]) }
        XCTAssertEqual(loader.count, before + 1, "the least recently used one was evicted")
    }

    func testAZeroBudgetNeverHoldsAnything() {
        // The uncached path, expressed as a budget rather than as a second code path.
        let loader = Loader()
        let bank = ExpertBank(budgetBytes: 0)
        for _ in 0..<3 {
            _ = bank.gateUp(layer: 0, expert: 2) { loader.load("gateUp", values: [1]) }
        }
        XCTAssertEqual(loader.count, 3)
        XCTAssertEqual(bank.residentBytes, 0)
        XCTAssertEqual(bank.metrics.hits, 0)
        XCTAssertEqual(bank.metrics.misses, 3)
    }

    func testThePerLayerAdapterRoutesThroughTheBankAndReturnsItsValues() throws {
        // The provider protocol is unchanged, so the mixture cannot tell the difference — but two things must
        // hold: the values are the upstream's, and the counters a caller reads are the bank's, not a per-layer
        // count that would answer a question nobody asked.
        // Three experts, 2·inter rows of `hidden` for gate/up and `hidden` rows of `inter` for down — built
        // rather than written out, because the stride arithmetic is the thing being exercised.
        let small = shape(intermediate: 2, hiddenSize: 3)
        let gateUpSlice = 2 * small.intermediate * small.hiddenSize
        let downSlice = small.hiddenSize * small.intermediate
        let gateUpStack = (0..<(3 * gateUpSlice)).map { Float($0) }
        let downStack = (0..<(3 * downSlice)).map { Float($0) }
        let bank = ExpertBank(budgetBytes: 1 << 20)
        let provider = ExpertSlotCache(
            upstream: ArrayExpertProvider(gateUp: gateUpStack, down: downStack), bank: bank, layer: 2
        )
        let expected = Array(gateUpStack[gateUpSlice..<(2 * gateUpSlice)])
        let first = try provider.gateUp(expert: 1, shape: small)
        let second = try provider.gateUp(expert: 1, shape: small)
        XCTAssertEqual(first.count, gateUpSlice)
        XCTAssertEqual(first.map(\.bitPattern), expected.map(\.bitPattern), "expert 1's own rows")
        XCTAssertEqual(
            second.map(\.bitPattern), expected.map(\.bitPattern),
            "a hit must return the same weights, not merely the same shape"
        )
        XCTAssertEqual(bank.metrics.requests, 2, "the adapter must route every request through the bank")
        XCTAssertEqual(provider.metrics.hits, 1, "and report the bank's counters")
        XCTAssertEqual(bank.residentSlices, 1)
    }

    func testTheExplicitSlotCapMeansTheSameThingItAlwaysDid() {
        // `SHARD_EXPERT_SLOTS` was experts per layer per projection, so for a generation-wide bank it is that
        // many slices times the layers times two projections.
        let environment = ["SHARD_EXPERT_SLOTS": "4"]
        XCTAssertEqual(
            Qwen3_5Forward.sliceCap(environment: environment, shape: shape(), layers: 3), 4 * 3 * 2
        )
        // Without it, the cap is whatever the byte budget works out to, so the two cannot disagree.
        let fromBudget = Qwen3_5Forward.sliceCap(
            environment: ["SHARD_EXPERT_BANK_MB": "1"], shape: shape(), layers: 3
        )
        XCTAssertGreaterThan(fromBudget, 0)
    }
}
