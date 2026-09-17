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

    /// The cached decode marks the same phases the sequence path does, and it is the step the throughput
    /// gate measures — the one step that had no breakdown until `D88`.
    func testTheCachedDecodeReportsItsPhasesWhenAskedFor() throws {
        let forward = try Qwen3_5Forward(snapshot: try checkpoint())
        let generation = try forward.generateCached(prompt: try tokens(), maxNewTokens: 2, profiling: true)

        let profile = try XCTUnwrap(generation.profile, "profiling was asked for")
        XCTAssertGreaterThan(profile.layers, 0)
        XCTAssertGreaterThan(profile.seconds.values.reduce(0, +), 0)
        for phase in ["embed", "load", "attn.norm", "attn.core", "attn.add", "ff.norm", "head"] {
            XCTAssertNotNil(profile.seconds[phase], "the cached decode marks \(phase)")
        }
        // The fixture is a mixture (`qwen3_5_moe`, 8 experts, top-2), and the mixture is most of a step, so
        // its own phases have to be here — they are the reason this wiring was needed.
        XCTAssertTrue(
            profile.seconds.keys.contains { $0.hasPrefix("mix.") },
            "the mixture's phases are missing: \(profile.seconds.keys.sorted())"
        )
    }

    /// Nil is not zero: with the instrument off the generation carries no profile rather than zeroes.
    func testWithoutProfilingTheGenerationCarriesNoProfile() throws {
        let forward = try Qwen3_5Forward(snapshot: try checkpoint())
        let generation = try forward.generateCached(prompt: try tokens(), maxNewTokens: 1, profiling: false)
        XCTAssertNil(generation.profile)
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
        // **Exactly zero**, and that is not an accident: this prompt is nine positions and the
        // chunked rule's chunk is sixty-four, so the whole prompt is one chunk and the chunked
        // rule *is* the recurrence. `D8`'s divergence needs a second chunk — the Python contract
        // measures 3.2e-07 relative for the recurrent rule against the chunked one over seventy
        // positions, where the chunk boundary finally matters. So this test pins the boundary of
        // the claim as well as the claim: inside one chunk, cached and uncached are the same
        // bytes; across chunks they are the same recurrence to rounding.
        XCTAssertEqual(worst, 0, "a prompt inside one chunk must replay bit for bit")
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

    /// The cached attention, against the sequence attention, bit for bit.
    ///
    /// This is the half of `D8` that is *not* allowed to differ: cached attention computes the
    /// same sum over the same values in the same order, so any difference at all is a defect
    /// rather than a numeric path. Written because the Gated DeltaNet decode step turned out to
    /// be correct to 1.6e-06 relative on the same fixtures — far too small to explain the real
    /// model's order-one logit divergence — which leaves this as the suspect.
    func testCachedAttentionIsBitIdenticalToTheSequenceAttention() throws {
        let forward = try Qwen3_5Forward(snapshot: try checkpoint())
        let golden = try tokens()
        let hiddenSize = forward.config.hiddenSize

        let attentionBlocks = forward.spec.tensors.filter { $0.role == .attnQ }.map(\.block).sorted()
        let block = try XCTUnwrap(attentionBlocks.first, "the fixture must have a full-attention layer")
        let index = Int(block.split(separator: ".")[1])!
        let layer = try forward.loadLayer(index)

        // A short synthetic sequence, so the comparison does not depend on the prompt.
        var hidden = [Float](repeating: 0, count: 5 * hiddenSize)
        for position in 0..<hidden.count { hidden[position] = Float((position * 37) % 101) * 0.01 - 0.5 }
        _ = golden

        let length = 5
        let tables = forward.ropeTables(positions: (0..<length).map(Double.init))
        var mask = [Float](repeating: 0, count: length * length)
        for row in 0..<length {
            for column in 0..<length { mask[row * length + column] = column > row ? -Float.infinity : 0 }
        }
        let sequence = try forward.attention(
            hidden, weights: layer.weights, length: length, tables: tables, mask: mask
        )

        var keys: [Float] = []
        var values: [Float] = []
        var stepped = [Float](repeating: 0, count: length * hiddenSize)
        for position in 0..<length {
            let one = Array(hidden[(position * hiddenSize)..<((position + 1) * hiddenSize)])
            let stepTables = forward.ropeTables(positions: [Double(position)])
            let step = try forward.attentionStep(
                one, weights: layer.weights, tables: stepTables, keys: &keys, values: &values,
                cachedLength: position
            )
            for offset in 0..<hiddenSize { stepped[position * hiddenSize + offset] = step[offset] }
        }

        var worst: Float = 0
        for offset in 0..<sequence.count { worst = max(worst, abs(stepped[offset] - sequence[offset])) }
        print(String(format: "\n  cached attention vs sequence attention: worst |Δ| %.3e", worst))
        XCTAssertEqual(
            worst, 0,
            "cached attention is the same sum in the same order, so it must be bit-identical"
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
