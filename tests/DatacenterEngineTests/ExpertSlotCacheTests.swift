import XCTest

@testable import DatacenterEngine

/// The slot bank's contract, which M1's gate depends on and nothing has ever exercised.
///
/// The gate asks for "a measured cache hit rate"; `D12` is open because the bank's size was never
/// derived from a budget. Before a real-model run sweeps that size, the *mechanism* has to be pinned,
/// because an instrument that has never been run is not an instrument:
///
/// - the brief asks for **LRU** slot banks, so eviction order is part of the contract, not an
///   implementation detail;
/// - the bank must be **bounded**, and `peakResidentExperts` is the metric that says so — the figure
///   the budget arithmetic in `SlotBudgetTests` multiplies by a per-expert byte count;
/// - and the hit rate must be **non-decreasing** in the bank size, or a sweep would produce a curve
///   that cannot be read.
///
/// The provider is a stub that counts reads, so every number here is deterministic and none of it
/// depends on a checkpoint.
final class ExpertSlotCacheTests: XCTestCase {
    /// Counts what actually reached the source, which is the SSD traffic the cache exists to avoid.
    final class CountingProvider: ExpertWeightProvider {
        let shape: MixtureShape
        private(set) var gateUpReads: [Int] = []
        private(set) var downReads: [Int] = []

        init(shape: MixtureShape) { self.shape = shape }

        func gateUp(expert: Int, shape: MixtureShape) throws -> [Float] {
            gateUpReads.append(expert)
            return [Float(expert), 0, 0, 0, 0, 0]
        }

        func down(expert: Int, shape: MixtureShape) throws -> [Float] {
            downReads.append(expert)
            return [Float(expert), 0, 0]
        }
    }

    private let shape = MixtureShape(hiddenSize: 3, experts: 8, topK: 2, intermediate: 2, sharedIntermediate: 2)

    private func bank(capacity: Int) -> (ExpertSlotCache, CountingProvider) {
        let provider = CountingProvider(shape: shape)
        // A generous byte budget and an explicit per-projection cap, so this exercises the *capacity* rule
        // rather than the byte rule.
        return (ExpertSlotCache(
            upstream: provider,
            bank: ExpertBank(budgetBytes: 1 << 20, sliceCap: capacity), layer: 0
        ), provider)
    }

    private func touch(_ cache: ExpertSlotCache, _ expert: Int) throws {
        _ = try cache.gateUp(expert: expert, shape: shape)
        _ = try cache.down(expert: expert, shape: shape)
    }

    func testARepeatedExpertComesFromTheBankTheSecondTime() throws {
        let (cache, provider) = bank(capacity: 1)
        try touch(cache, 3)
        try touch(cache, 3)
        XCTAssertEqual(cache.metrics.misses, 2, "one expert is two requests: gateUp and down")
        XCTAssertEqual(cache.metrics.hits, 2, "the second touch must not reach the source")
        XCTAssertEqual(provider.gateUpReads, [3], "the source saw the expert once")
        XCTAssertEqual(provider.downReads, [3])
    }

    func testTheBankIsBoundedByItsCapacity() throws {
        let (cache, provider) = bank(capacity: 2)
        for expert in 0..<10 { try touch(cache, expert) }
        XCTAssertLessThanOrEqual(
            cache.metrics.peakResidentExperts, 2,
            "peakResidentExperts is what the byte budget is computed from, so it must respect capacity"
        )
        XCTAssertEqual(provider.gateUpReads.count, 10, "every distinct expert had to be read once")
        XCTAssertEqual(cache.metrics.hits, 0, "ten distinct experts in a bank of two cannot hit")
    }

    /// The brief says **LRU**, so the eviction order is a promise rather than a detail.
    func testEvictionIsLeastRecentlyUsed() throws {
        let (cache, _) = bank(capacity: 2)
        try touch(cache, 0)                    // bank: 0
        try touch(cache, 1)                    // bank: 0, 1
        try touch(cache, 0)                    // 0 is now the most recent; bank: 1, 0
        try touch(cache, 2)                    // evicts 1, not 0
        let hitsBefore = cache.metrics.hits
        try touch(cache, 0)
        XCTAssertEqual(cache.metrics.hits, hitsBefore + 2, "0 was kept because it was used recently")
        try touch(cache, 1)
        XCTAssertEqual(cache.metrics.hits, hitsBefore + 2, "1 was evicted, so it must be read again")
    }

    /// A routing sequence with the skew real top-k shows, replayed at several bank sizes.
    private let routing = [0, 0, 1, 0, 2, 0, 1, 3, 0, 4, 0, 1, 5, 0, 2, 0, 6, 0, 1, 0, 7, 0, 3, 0]

    func testTheHitRateRisesWithTheBankSizeAndNeverFalls() throws {
        var previousHits = -1
        var curve: [(capacity: Int, hits: Int, requests: Int)] = []
        for capacity in [1, 2, 4, 8] {
            let (cache, _) = bank(capacity: capacity)
            for expert in routing { try touch(cache, expert) }
            // `requests` counts both projections, so the ratio is over the same denominator everywhere.
            XCTAssertGreaterThanOrEqual(
                cache.metrics.hits, previousHits,
                "a bigger bank cannot hit less: capacity \(capacity) hit fewer than the size below it"
            )
            previousHits = cache.metrics.hits
            curve.append((capacity, cache.metrics.hits, cache.metrics.requests))
        }
        let smallest = curve.first!, largest = curve.last!
        XCTAssertGreaterThan(smallest.hits, 0, "a skewed sequence must hit even in a bank of one")
        XCTAssertEqual(
            largest.hits, largest.requests - 16,
            "eight experts x two projections is sixteen compulsory misses, and the other thirty-two "
                + "must hit: 32/48 is the arithmetic, not 48/48, which is what I first asserted"
        )
        for point in curve {
            XCTAssertEqual(point.requests, routing.count * 2, "both projections are requested every time")
        }
        // A curve that cannot be printed is a curve nobody can decide from, so it is printed.
        print("hit rate by capacity: " + curve.map { "\($0.capacity)->\($0.hits)/\($0.requests)" }.joined(separator: " "))
    }
}
