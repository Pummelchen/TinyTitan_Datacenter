import XCTest

@testable import DatacenterEngine

/// The Gated DeltaNet against the contract's golden bit patterns.
///
/// Two cases: a single chunk, and one long enough to need two. The sequential scan over
/// chunks only runs in the second, and that is where a padding or state-threading error
/// hides — a single-chunk case would never execute it.
final class GatedDeltaNetTests: XCTestCase {
    struct Vector: Decodable {
        var shape: [Int]
        var bits: [UInt32]
        var floats: [Float] { bits.map { Float(bitPattern: $0) } }
    }

    struct Config: Decodable {
        var hidden_size: Int
        var num_key_heads: Int
        var num_value_heads: Int
        var key_head_dim: Int
        var value_head_dim: Int
        var conv_kernel: Int
        var eps: Double
        var positions: Int
    }

    struct Case: Decodable {
        var config: Config
        var hidden: Vector
        var out: Vector
        var weights: [String: Vector]
    }

    private func fixture() throws -> (single: Case, multi: Case) {
        let url = try XCTUnwrap(
            Bundle.module.url(forResource: "contract-vectors", withExtension: "json", subdirectory: "Fixtures")
        )
        struct Fixture: Decodable { var gdn: Case; var gdn_multichunk: Case }
        let fixture = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: url))
        return (fixture.gdn, fixture.gdn_multichunk)
    }

    private func shape(_ config: Config) -> GatedDeltaNetShape {
        GatedDeltaNetShape(
            hiddenSize: config.hidden_size, keyHeads: config.num_key_heads, valueHeads: config.num_value_heads,
            keyHeadDim: config.key_head_dim, valueHeadDim: config.value_head_dim,
            convKernel: config.conv_kernel, eps: Float(config.eps)
        )
    }

    private func weights(_ vectors: [String: Vector]) -> GatedDeltaNetWeights {
        GatedDeltaNetWeights(
            inQKV: vectors["in_proj_qkv"]!.floats, inZ: vectors["in_proj_z"]!.floats,
            inB: vectors["in_proj_b"]!.floats, inA: vectors["in_proj_a"]!.floats,
            conv: vectors["conv1d"]!.floats, aLog: vectors["A_log"]!.floats,
            dtBias: vectors["dt_bias"]!.floats, norm: vectors["norm"]!.floats,
            outProj: vectors["out_proj"]!.floats
        )
    }

    private func assertSameBits(
        _ actual: [Float], _ expected: Vector, _ label: String, file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(actual.count, expected.bits.count, "\(label): count", file: file, line: line)
        for index in 0..<min(actual.count, expected.bits.count) where actual[index].bitPattern != expected.bits[index] {
            let want = Float(bitPattern: expected.bits[index])
            XCTFail(
                "\(label)[\(index)]: got \(actual[index]) (0x\(String(actual[index].bitPattern, radix: 16))) "
                    + "want \(want) (0x\(String(expected.bits[index], radix: 16)))",
                file: file, line: line
            )
            return
        }
    }

    private func run(_ testCase: Case) -> [Float] {
        GatedDeltaNet.layer(
            hidden: testCase.hidden.floats, weights: weights(testCase.weights), shape: shape(testCase.config),
            batch: 1, length: testCase.config.positions
        )
    }

    func testSingleChunkMatchesTheContract() throws {
        let (single, _) = try fixture()
        assertSameBits(run(single), single.out, "gdn")
    }

    /// **More value heads than key heads** — grouped-query style, and the case the 2 B model
    /// never exercised because it has sixteen of each. `Qwen3_5MoeGatedDeltaNet.forward:645`
    /// repeat-interleaves the query and key heads up to the value head count before the rule
    /// runs; without that step the per-head slice indexes past the end of the key and pairs
    /// the wrong heads, which produces plausible numbers and a wrong model.
    ///
    /// This vector pins the *order* of the expansion as well as its existence: the contract
    /// expands consecutively (`[k0, k0, k1, k1]`), and an interleaved reading (`[k0, k1, k0,
    /// k1]`) pairs different heads and gives different bits, so a hand-written ordering test
    /// would say no more than this comparison already does.
    func testAsymmetricHeadsMatchTheContract() throws {
        let url = try XCTUnwrap(
            Bundle.module.url(forResource: "contract-vectors", withExtension: "json", subdirectory: "Fixtures")
        )
        struct Fixture: Decodable { var gdn_asymmetric: Case }
        let asymmetric = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: url)).gdn_asymmetric
        XCTAssertGreaterThan(
            asymmetric.config.num_value_heads, asymmetric.config.num_key_heads,
            "the case is pointless unless the head counts differ"
        )
        XCTAssertEqual(
            asymmetric.config.num_value_heads % asymmetric.config.num_key_heads, 0,
            "the reference repeats an integer number of times"
        )
        assertSameBits(run(asymmetric), asymmetric.out, "gdn_asymmetric")
    }

    func testMultipleChunksMatchTheContract() throws {
        let (_, multi) = try fixture()
        XCTAssertGreaterThan(multi.config.positions, GatedDeltaNet.chunkSize, "the case must need a second chunk")
        assertSameBits(run(multi), multi.out, "gdn_multichunk")
    }

    /// The conv is the one place where a plausible reading (centred, or right-padded) gives
    /// the right shape and the wrong answer.
    func testTheConvIsCausalAndLeftPadded() {
        // One channel, kernel 4, the last tap reading the current position.
        let weight: [Float] = [1, 0, 0, 0]
        var input = [Float](repeating: 0, count: 6)
        input[0] = 1
        let out = GatedDeltaNet.depthwiseCausalConv(input, channels: 1, length: 6, weight: weight, kernel: 4, activate: false)
        XCTAssertEqual(out[3], 1, "tap 0 is the oldest of four, so input 0 lands at output 3")
        XCTAssertEqual(out[2], 0)
        XCTAssertEqual(out[4], 0)
    }

    func testSoftplusUsesTheThreshold() {
        XCTAssertEqual(GatedDeltaNet.softplus(25), 25, "above the threshold the reference returns the input")
        let small = GatedDeltaNet.softplus(0)
        XCTAssertEqual(small, Float(log1p(exp(0.0))))
        XCTAssertGreaterThan(GatedDeltaNet.softplus(-5), 0)
    }

    /// A unit lower triangular solve, checked against an explicit substitution.
    func testTriangularSolveMatchesForwardSubstitution() {
        let lower: [Float] = [1, 0, 0, 2, 1, 0, -1, 3, 1]
        let rhs: [Float] = [1, 2, 3]
        let solution = GatedDeltaNet.solveUnitLower(lower: lower, rhs: rhs, rows: 3, columns: 1)
        // Row 0: x0 = 1. Row 1: x1 = 2 - 2*x0 = 0. Row 2: x2 = 3 - (-1*x0 + 3*x1) = 4.
        XCTAssertEqual(solution, [1, 0, 4])
    }
}
