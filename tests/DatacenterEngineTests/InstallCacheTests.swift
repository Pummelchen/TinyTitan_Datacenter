import XCTest

@testable import DatacenterEngine

/// `DC-106`: the dense backbone stays resident across forwards.
///
/// The forward walks every layer for every token, so the same norms, attention, router and shared-expert
/// weights were read and dequantised again per token — measured on the real install at **2.95 s of a
/// 19.8 s forward**. The payload cache holds those whole-tensor payloads, and the acceptance is a
/// **count**, not a timing: a second forward must read no whole-tensor payload at all.
///
/// It holds the **packed** bytes. 0.74 GB of the real install's 0.796 GB dense payload is int4, which is
/// eight times larger decoded, so caching the decoded form would need ~6 GB against this node's ~4.5 GB
/// budget — the `DC-091` mistake a second time.
final class InstallCacheTests: XCTestCase {
    private func installURL() throws -> URL {
        try XCTUnwrap(
            Bundle.module.url(forResource: "tiny-qwen36", withExtension: nil, subdirectory: "Fixtures")
        ).appendingPathComponent("install")
    }

    /// The acceptance, verbatim: the second forward reads **zero** whole-tensor payload bytes.
    func testASecondForwardReadsNoDensePayload() throws {
        let forward = try Qwen3_5Forward(install: try installURL())
        let tokens = [1, 2, 3]

        _ = try forward.forwardWithDecisions(tokens: tokens)
        let firstRead = forward.sourceBytesRead
        let firstPayload = forward.payloadCacheMetrics.bytesRead
        XCTAssertGreaterThan(firstPayload, 0, "the first forward has to read something")
        XCTAssertEqual(
            forward.payloadCacheMetrics.hits, 0, "nothing can be cached before the first read"
        )

        _ = try forward.forwardWithDecisions(tokens: tokens)
        let secondPayload = forward.payloadCacheMetrics.bytesRead - firstPayload

        XCTAssertEqual(
            secondPayload, 0,
            "a second forward must read no whole-tensor payload: it read \(secondPayload) bytes while "
                + "the first read \(firstPayload)"
        )
        XCTAssertGreaterThan(
            forward.payloadCacheMetrics.hits, 0, "and the cache is what did it, not a shorter forward"
        )
        XCTAssertGreaterThan(
            forward.sourceBytesRead - firstRead, 0,
            "row-range reads still happen — the embedding's rows and the expert banks are streamed, which "
                + "is the point: a cache of whole tensors cannot hold 18 GB of stacked experts"
        )
    }

    /// The claim that made the cache safe to put on the payload funnel: a stacked expert bank is read a
    /// row range at a time, so it never lands in a whole-tensor cache.
    func testTheStackedExpertBanksNeverEnterTheCache() throws {
        let forward = try Qwen3_5Forward(install: try installURL())
        _ = try forward.forwardWithDecisions(tokens: [1, 2, 3])
        // The source is where the cache lives, so the claim is checked there rather than inferred.
        let held = try XCTUnwrap(forward.source as? InstallFile).payloadCacheNames
        XCTAssertFalse(held.isEmpty, "the dense payload should be held")
        for name in held {
            XCTAssertFalse(
                name.contains("experts"),
                "\(name) is a stacked expert bank and must be streamed, not cached"
            )
        }
        XCTAssertLessThan(
            forward.payloadCacheMetrics.bytesHeld, 64 * 1_048_576,
            "the fixture's dense payload is small; a cache holding megabytes would be holding experts"
        )
    }

    /// The cache type on its own: least-recently-used, bounded in bytes, and it refuses what cannot fit
    /// rather than evicting everything to make room for it.
    func testTheCacheEvictsLeastRecentlyUsedAndRespectsItsBudget() throws {
        let cache = InstallFile.PayloadCache(budget: 250)
        func payload(_ count: Int, _ fill: UInt8) -> Data { Data(repeating: fill, count: count) }

        cache.store(payload(100, 1), named: "a")
        cache.store(payload(100, 2), named: "b")
        XCTAssertEqual(cache.bytes, 200)
        XCTAssertEqual(cache.value(for: "a"), payload(100, 1), "a is held")
        _ = cache.value(for: "a")  // a is now the most recently used

        cache.store(payload(100, 3), named: "c")
        XCTAssertEqual(cache.bytes, 200, "c evicted one of a and b, and it must not be a")
        XCTAssertNotNil(cache.value(for: "a"))
        XCTAssertNil(cache.value(for: "b"), "the least recently used should have gone")

        cache.store(payload(1_000, 4), named: "huge")
        XCTAssertNil(cache.value(for: "huge"), "a payload bigger than the budget is not cached")
        XCTAssertEqual(cache.bytes, 200, "and it does not evict the cache to make room for it")

        let disabled = InstallFile.PayloadCache(budget: 0)
        disabled.store(payload(10, 5), named: "x")
        XCTAssertNil(disabled.value(for: "x"))
        XCTAssertEqual(disabled.bytes, 0)
        XCTAssertEqual(disabled.hits, 0)
    }

    /// A budget nobody could hold is refused rather than clamped quietly — the same rule the expert bank
    /// learned, for the same reason.
    func testAnAbsurdBudgetFallsBackInsteadOfBeingTrusted() {
        let sane = InstallFile.payloadCacheBudget(environment: [:])
        for bad in ["-1", "many", "", "99999999"] {
            XCTAssertEqual(InstallFile.payloadCacheBudget(environment: ["SHARD_DENSE_CACHE_MB": bad]), sane)
        }
        XCTAssertEqual(
            InstallFile.payloadCacheBudget(environment: ["SHARD_DENSE_CACHE_MB": "2048"]), 2048 * 1_048_576
        )
        XCTAssertEqual(
            InstallFile.payloadCacheBudget(environment: ["SHARD_DENSE_CACHE_MB": "0"]), 0,
            "zero is a real setting: it is how the measurement turns the cache off for a comparison"
        )
    }
}
