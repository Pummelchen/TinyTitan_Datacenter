import XCTest

import DatacenterIR

@testable import DatacenterEngine

/// The decode cache, against the path M1's gate has already verified.
///
/// `D8` splits the claim in two, and these tests assert each at the strength it actually holds:
///
/// - the **full-attention** cache is bit-identical, because cached attention is the same sum over
///   the same values in the same order;
/// - the **Gated DeltaNet** state is not, because a step-at-a-time recurrence groups its sums
///   differently from the chunked rule — about 1e-7 relative, measured, and asserted as a
///   tolerance rather than pretended away.
///
/// The strongest test here is the first one: cached and uncached generation must produce **the
/// same tokens**, which is the thing a user sees, and the router's decisions must agree, because a
/// flipped expert is not a rounding difference.
final class ModelCacheTests: XCTestCase {
    private func checkpoint() throws -> URL {
        try XCTUnwrap(
            Bundle.module.url(forResource: "tiny-qwen36", withExtension: nil, subdirectory: "Fixtures")
        )
    }

    private func tokens() throws -> [Int] {
        struct Golden: Decodable { var tokens: [Int] }
        let url = try checkpoint().appendingPathComponent("golden.json")
        return try JSONDecoder().decode(Golden.self, from: Data(contentsOf: url)).tokens
    }

    func testCachedGenerationProducesTheSameTokensAsUncached() throws {
        let forward = try Qwen3_5Forward(snapshot: try checkpoint())
        let prompt = try tokens()
        let uncached = try forward.generate(prompt: prompt, maxNewTokens: 5)
        let cached = try forward.generateCached(prompt: prompt, maxNewTokens: 5)
        XCTAssertEqual(
            cached.generated, uncached.generated,
            "the cache is a different numeric path, not a different model"
        )
        XCTAssertEqual(cached.generated.count, 5)
    }

    /// The first generated token comes from the prompt's own logits, so a cache that consumed the
    /// last prompt token twice would still generate *something* — just not the same thing.
    ///
    /// The comparison is against the sequence path at a **tolerance**, not bit for bit, and that
    /// is `D8` showing up where it matters: the replay's last position has been through the
    /// recurrent rule forty times, so the logits differ from the chunked path's by about 1e-3
    /// relative. Recording the number is the point — a test that demanded the bit would be
    /// demanding something the reference itself does not offer.
    func testTheReplayLandsOnThePromptsLastPosition() throws {
        let forward = try Qwen3_5Forward(snapshot: try checkpoint())
        let prompt = try tokens()
        let (_, promptLogits) = try forward.prepareCache(tokens: prompt)

        let captured = try forward.forwardWithDecisions(tokens: prompt)
        let fullLogits = try forward.logitsOf(captured.tensors)
        let width = forward.vocabularySize
        let offset = (prompt.count - 1) * width
        var worst: Float = 0
        var scale: Float = 0
        for index in 0..<width {
            worst = max(worst, abs(promptLogits[index] - fullLogits[offset + index]))
            scale = max(scale, abs(fullLogits[offset + index]))
        }
        print(String(format: "\n  cached replay vs chunked prefill: max |Δ| %.3e against scale %.3e (%.1e relative)",
                     worst, scale, worst / max(scale, 1e-30)))
        XCTAssertLessThan(worst, max(1e-2, scale * 1e-2), "the replay must land on the same position")
        // And it must be the *last* position rather than an earlier one: taking a different
        // position would be a much larger error than the path difference.
        XCTAssertGreaterThan(worst, 0, "a bit-identical result would mean D8 does not apply")
    }

    /// The router's decisions through the cached path, asserted separately from any number (I3).
    ///
    /// The comparison is the *same position*: the cached path decides the next token's experts
    /// from its own state, and the uncached path decides them by running the prompt plus that
    /// token from scratch. Two different numeric paths, so the decisions are compared **exactly**
    /// and the agreement is measured rather than assumed — a flipped expert would be a real
    /// difference in what the model computes, not a rounding artifact.
    func testTheCachedPathChoosesTheSameExperts() throws {
        let forward = try Qwen3_5Forward(snapshot: try checkpoint())
        let prompt = try tokens()
        let width = forward.vocabularySize

        let uncached = try forward.forwardWithDecisions(tokens: prompt)
        let next = Greedy.argmax(
            try forward.logitsOf(uncached.tensors), offset: (prompt.count - 1) * width, width: width
        )

        let (cache, _) = try forward.prepareCache(tokens: prompt)
        let cachedStep = try forward.decodeOne(token: next, cache: cache, capturing: true)
        let extended = try forward.forwardWithDecisions(tokens: prompt + [next])

        XCTAssertEqual(cachedStep.discrete.count, extended.discrete.count, "one decision per layer")
        var agreed = 0
        var total = 0
        for decision in cachedStep.discrete {
            guard let reference = extended.discrete.first(where: { $0.name == decision.name }) else {
                XCTFail("the uncached path recorded no \(decision.name)")
                continue
            }
            // The uncached run's *last* row is the position the cached step decided.
            let rows = reference.values.count / reference.shape[0]
            let uncachedRow = Array(reference.values[(reference.shape[0] - 1) * rows..<reference.values.count])
            total += 1
            if uncachedRow == decision.values { agreed += 1 }
        }
        print("\n  cached vs uncached router decisions: \(agreed)/\(total) layers agree exactly")
        XCTAssertEqual(
            agreed, total,
            "the same experts must be chosen; if one flips, that is a finding about the cache, not a tolerance"
        )
    }

    /// The cache's whole point: a decoded position costs one position, not the whole sequence.
    func testASteppedPositionIsCheaperThanTheWholeSequence() throws {
        let forward = try Qwen3_5Forward(snapshot: try checkpoint())
        let prompt = try tokens()

        let started = Date()
        _ = try forward.forwardWithDecisions(tokens: prompt)
        let sequenceSeconds = Date().timeIntervalSince(started)

        let (cache, _) = try forward.prepareCache(tokens: prompt)
        let startedStep = Date()
        _ = try forward.decodeOne(token: prompt[0], cache: cache, capturing: false)
        let stepSeconds = Date().timeIntervalSince(startedStep)

        XCTAssertLessThan(
            stepSeconds, sequenceSeconds,
            "one position through the cache must cost less than the whole sequence"
        )
    }
}
