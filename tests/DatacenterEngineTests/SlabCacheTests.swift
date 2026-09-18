import XCTest

@testable import DatacenterEngine

/// The packed slab cache (`DC-120`).
///
/// The cache's *correctness* end to end is asserted by the trace digest — every A/B in `D106` came out
/// `89d654ff54b0fd03` whether the cache was on, off, or half-filling — so what is asserted here is the cache's
/// own behaviour: what it holds, what it drops, and that a budget of zero means exactly that.
final class SlabCacheTests: XCTestCase {
    private func payload(_ size: Int, seed: UInt8) -> Data {
        Data((0..<size).map { UInt8((Int($0) &+ Int(seed)) % 251) })
    }

    func testItHoldsWhatItWasGivenAndCountsTheHit() {
        let cache = InstallFile.SlabCache(budget: 1 << 20)
        XCTAssertNil(cache.value(for: "a"), "nothing is resident before anything is stored")
        cache.store(payload(100, seed: 1), named: "a")
        XCTAssertEqual(cache.value(for: "a")?.count, 100)
        XCTAssertEqual(cache.metrics.slabHits, 1)
        XCTAssertEqual(cache.metrics.slabMisses, 1, "the first lookup was a miss and must be counted as one")
        XCTAssertEqual(cache.metrics.slabBytesHeld, 100)
    }

    func testABudgetOfZeroHoldsNothingAndCountsNothing() {
        // The default, and the point of the row: `D106` measured the cache as a loss on this node, so a budget of
        // zero has to mean *no residency at all* rather than a cache that quietly fills.
        let cache = InstallFile.SlabCache(budget: 0)
        cache.store(payload(100, seed: 2), named: "a")
        XCTAssertNil(cache.value(for: "a"))
        XCTAssertEqual(cache.metrics.slabBytesHeld, 0)
        XCTAssertEqual(cache.metrics.slabHits, 0)
        XCTAssertEqual(cache.metrics.slabMisses, 0, "a disabled cache is not asked, so it counts nothing")
    }

    func testTheLeastRecentlyUsedSlabIsEvictedAndTheBudgetHolds() {
        let cache = InstallFile.SlabCache(budget: 300)
        cache.store(payload(100, seed: 3), named: "a")
        cache.store(payload(100, seed: 4), named: "b")
        cache.store(payload(100, seed: 5), named: "c")
        XCTAssertEqual(cache.metrics.slabBytesHeld, 300)
        // Touching `a` makes `b` the least recently used, so `d` must evict `b` and not `a`.
        XCTAssertNotNil(cache.value(for: "a"))
        cache.store(payload(100, seed: 6), named: "d")
        XCTAssertEqual(cache.metrics.slabBytesHeld, 300, "the budget is a ceiling, not a target")
        XCTAssertNotNil(cache.value(for: "a"), "the recently used slab survives")
        XCTAssertNil(cache.value(for: "b"), "the least recently used one is the victim")
    }

    func testASlabLargerThanTheWholeBudgetIsNotStored() {
        let cache = InstallFile.SlabCache(budget: 64)
        cache.store(payload(128, seed: 7), named: "big")
        XCTAssertNil(cache.value(for: "big"))
        XCTAssertEqual(cache.metrics.slabBytesHeld, 0, "a slab that cannot fit must not evict everything to try")
    }

    func testTheBudgetComesFromTheEnvironmentAndRefusesNonsense() {
        // **256 MiB, and measured** (`D110`): the packed expert path is what uses this cache, and a sweep of
        // 256/512/768/1024 MiB found the step worse above 256 (1.465, 1.467, 1.483, 1.667 s) because the
        // resident bytes cost more in memory pressure than the reads they save. The unit is bytes, which the
        // first version of this default got wrong by returning the bare literal `256`.
        // **128 MiB, re-measured after the buffer cache was turned on** (`D112`): three alternated pairs put
        // 128 at 0.930 s against 256 at 0.957, because the kernel's own cache holds the same slabs in clean,
        // evictable pages. Zero is worse than any non-zero size — `preloadPacked` declines with no cache, so
        // the fan-out disappears.
        XCTAssertEqual(InstallFile.slabCacheBudget(environment: [:]), 128 << 20, "the default is 128 MiB")
        XCTAssertEqual(InstallFile.slabCacheBudget(environment: ["SHARD_SLAB_CACHE_MB": "512"]), 512 << 20)
        XCTAssertEqual(InstallFile.slabCacheBudget(environment: ["SHARD_SLAB_CACHE_MB": "0"]), 0)
        // Refused rather than clamped, like every other budget in this engine: a budget nobody could hold is a node
        // that swaps, and a silent clamp hides the typo that caused it.
        XCTAssertEqual(InstallFile.slabCacheBudget(environment: ["SHARD_SLAB_CACHE_MB": "-1"]), 0)
        XCTAssertEqual(InstallFile.slabCacheBudget(environment: ["SHARD_SLAB_CACHE_MB": "9999999"]), 0)
        XCTAssertEqual(InstallFile.slabCacheBudget(environment: ["SHARD_SLAB_CACHE_MB": "lots"]), 0)
    }
}
