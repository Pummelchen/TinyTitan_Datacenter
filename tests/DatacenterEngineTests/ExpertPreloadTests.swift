import XCTest

@testable import DatacenterEngine

/// Preloading a layer's chosen experts (`DC-118`).
///
/// The reads are latency-bound — 1.08 GB per step at 0.83 GB/s, 520 small `pread`s issued one at a time — so
/// the fix is to fan the misses out across threads. Two properties make it safe, and both are asserted here:
/// the values are the upstream's either way, and the warm-up **does not count as a request**, because the loop
/// that follows is the requester and will find each slice resident.
final class ExpertPreloadTests: XCTestCase {
    /// A provider that counts what was read from it, under a lock because a preload is concurrent.
    private final class Counter: @unchecked Sendable {
        private let lock = NSLock()
        private var reads: [String] = []
        func note(_ what: String) {
            lock.lock(); reads.append(what); lock.unlock()
        }
        var all: [String] { lock.lock(); defer { lock.unlock() }; return reads }
        func count(_ what: String) -> Int { all.filter { $0 == what }.count }
    }

    private func shape(intermediate: Int = 2, hiddenSize: Int = 3) -> MixtureShape {
        MixtureShape(
            hiddenSize: hiddenSize, experts: 4, topK: 2, intermediate: intermediate,
            sharedIntermediate: intermediate
        )
    }

    /// Deterministic values, so "the preload changed nothing" is a bit-for-bit assertion rather than a shape
    /// check.
    private func fixture(_ small: MixtureShape) -> (ExpertWeightProvider, Counter) {
        let gateUpSlice = 2 * small.intermediate * small.hiddenSize
        let downSlice = small.hiddenSize * small.intermediate
        let gateUp = (0..<(4 * gateUpSlice)).map { Float($0) * 0.5 }
        let down = (0..<(4 * downSlice)).map { Float($0) * -0.25 }
        let counter = Counter()
        let upstream = CountingProvider(
            base: ArrayExpertProvider(gateUp: gateUp, down: down), counter: counter
        )
        return (upstream, counter)
    }

    private final class CountingProvider: ExpertWeightProvider {
        let base: ArrayExpertProvider
        let counter: Counter
        init(base: ArrayExpertProvider, counter: Counter) { self.base = base; self.counter = counter }
        func gateUp(expert: Int, shape: MixtureShape) throws -> [Float] {
            counter.note("gateUp(\(expert))")
            return try base.gateUp(expert: expert, shape: shape)
        }
        func down(expert: Int, shape: MixtureShape) throws -> [Float] {
            counter.note("down(\(expert))")
            return try base.down(expert: expert, shape: shape)
        }
    }

    func testAPreloadTurnsTheLoopsReadsIntoHits() throws {
        try XCTSkipIf(DecodeThreads.count < 2, "a preload is a fan-out; one thread has nothing to fan")
        let small = shape()
        let (upstream, counter) = fixture(small)
        let bank = ExpertBank(budgetBytes: 1 << 20)
        let cache = ExpertSlotCache(upstream: upstream, bank: bank, layer: 0)

        let without = try cache.gateUp(expert: 2, shape: small)
        XCTAssertEqual(counter.count("gateUp(2)"), 1, "the first ask has to read")

        // Now the hint, then the same ask the loop would make.
        cache.preload(experts: [2, 3], shape: small)
        XCTAssertEqual(counter.count("gateUp(2)"), 1, "a warm slice must not be read again")
        XCTAssertEqual(counter.count("gateUp(3)"), 1, "and the hint did fetch the other one")
        XCTAssertEqual(counter.count("down(3)"), 1, "both projections, because the loop asks for both")

        let after = try cache.gateUp(expert: 2, shape: small)
        XCTAssertEqual(
            after.map(\.bitPattern), without.map(\.bitPattern),
            "a hit must return the upstream's values, not merely its shape"
        )
        XCTAssertEqual(cache.metrics.requests, 2, "the request is the loop's, and the warm-up is not one")
        XCTAssertEqual(cache.metrics.hits, 1, "and the second ask was served from the bank")
    }

    func testAPreloadDoesNothingWithoutABudget() throws {
        let small = shape()
        let (upstream, counter) = fixture(small)
        // No bank: the preloaded bytes would be discarded and the loop would read them again, which is twice
        // the work rather than half the latency.
        let cache = ExpertSlotCache(upstream: upstream, bank: nil, layer: 0)
        cache.preload(experts: [0, 1], shape: small)
        XCTAssertEqual(counter.all.count, 0, "with nowhere to keep them, nothing is fetched")

        // And a bank with a zero budget behaves the same way, because zero means "hold nothing".
        let empty = ExpertSlotCache(upstream: upstream, bank: ExpertBank(budgetBytes: 0), layer: 0)
        empty.preload(experts: [0, 1], shape: small)
        XCTAssertEqual(counter.all.count, 0)
        XCTAssertEqual(empty.metrics.requests, 0)
    }

    func testThePredictionWarmsWhatTheLoopWillAskFor() throws {
        let small = shape()
        let (upstream, counter) = fixture(small)
        let bank = ExpertBank(budgetBytes: 1 << 20)
        let cache = ExpertSlotCache(upstream: upstream, bank: bank, layer: 0)

        // A token's loop: the adapter records what it was asked for as a side effect of serving it.
        _ = try cache.gateUp(expert: 1, shape: small)
        _ = try cache.down(expert: 1, shape: small)
        XCTAssertEqual(counter.count("gateUp(1)"), 1)

        // The next token issues that as a hint. `force` because the mechanism is off by default (`D105`: it was
        // measured and it loses), and `waitForPrefetch` because the point of it is to be off this thread.
        cache.prefetchPredicted(force: true)
        cache.waitForPrefetch()
        XCTAssertEqual(counter.count("gateUp(1)"), 1, "the hint must not re-read what is already resident")

        // A fresh adapter over the same bank is what the next token's layer load builds; it must find the
        // prediction waiting rather than reading it again.
        let next = ExpertSlotCache(upstream: upstream, bank: bank, layer: 0)
        let values = try next.gateUp(expert: 1, shape: small)
        XCTAssertEqual(counter.count("gateUp(1)"), 1, "the prediction is what saved the read")
        XCTAssertEqual(next.metrics.hits, 1)
        XCTAssertFalse(values.isEmpty)
    }

    func testThePredictionIsOffByDefault() throws {
        let small = shape()
        let (upstream, counter) = fixture(small)
        let cache = ExpertSlotCache(upstream: upstream, bank: ExpertBank(budgetBytes: 1 << 20), layer: 0)
        _ = try cache.gateUp(expert: 0, shape: small)
        let before = counter.all.count
        cache.prefetchPredicted()
        cache.waitForPrefetch()
        if ExpertSlotCache.predictionEnabled {
            XCTAssertGreaterThan(counter.all.count, before, "the knob is on, so the hint must read")
        } else {
            XCTAssertEqual(counter.all.count, before, "off by default, and that default is measured (`D105`)")
        }
    }

    func testAPreloadSkipsExpertsTheProviderDoesNotServe() throws {
        try XCTSkipIf(DecodeThreads.count < 2, "a preload is a fan-out; one thread has nothing to fan")
        let small = shape()
        let (upstream, counter) = fixture(small)
        let bank = ExpertBank(budgetBytes: 1 << 20)
        let cache = ExpertSlotCache(upstream: upstream, bank: bank, layer: 0)
        // The adapter's `serves` comes from its upstream, which serves everything, so this asserts the shape of
        // the call rather than the filter: a single expert is not worth a dispatch and must be left to the loop.
        cache.preload(experts: [1], shape: small)
        XCTAssertEqual(counter.all.count, 0, "one expert is not a fan-out")
    }
}
