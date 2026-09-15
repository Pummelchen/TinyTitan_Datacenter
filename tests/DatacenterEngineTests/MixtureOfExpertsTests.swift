import XCTest

@testable import DatacenterEngine

/// The mixture of experts against the contract's golden bit patterns.
///
/// The top-k index set is asserted **separately** from the numbers, because I3 says so and
/// because it is the assertion quantisation breaks first: a 1-ULP difference can change which
/// experts are chosen, and from there the continuations are unrelated while every per-tensor
/// comparison still looks green.
final class MixtureOfExpertsTests: XCTestCase {
    struct Vector: Decodable {
        var shape: [Int]
        var bits: [UInt32]
        var floats: [Float] { bits.map { Float(bitPattern: $0) } }
    }

    struct Config: Decodable {
        var tokens: Int
        var hidden_size: Int
        var experts: Int
        var top_k: Int
        var intermediate: Int
    }

    struct Indices: Decodable {
        var shape: [Int]
        var values: [Int]
    }

    struct Case: Decodable {
        var config: Config
        var hidden: Vector
        var out: Vector
        var indices: Indices
        var weights: [String: Vector]
        var top_k_weights: Vector
    }

    private func fixture() throws -> Case {
        let url = try XCTUnwrap(
            Bundle.module.url(forResource: "contract-vectors", withExtension: "json", subdirectory: "Fixtures")
        )
        struct Fixture: Decodable { var moe: Case }
        return try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: url)).moe
    }

    private func shape(_ config: Config) -> MixtureShape {
        MixtureShape(
            hiddenSize: config.hidden_size, experts: config.experts, topK: config.top_k,
            intermediate: config.intermediate, sharedIntermediate: config.intermediate
        )
    }

    private func weights(_ vectors: [String: Vector]) -> MixtureWeights {
        MixtureWeights(
            router: vectors["router_weight"]!.floats, gateUp: vectors["gate_up"]!.floats,
            down: vectors["down"]!.floats, sharedGate: vectors["shared_gate"]!.floats,
            sharedUp: vectors["shared_up"]!.floats, sharedDown: vectors["shared_down"]!.floats,
            sharedScalarGate: vectors["shared_scalar_gate"]!.floats
        )
    }

    private func run(_ testCase: Case) -> (output: [Float], indices: [[Int]], weights: [[Float]]) {
        MixtureOfExperts.block(
            hidden: testCase.hidden.floats, tokens: testCase.config.tokens,
            weights: weights(testCase.weights), shape: shape(testCase.config)
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

    func testTheBlockMatchesTheContractBitForBit() throws {
        let testCase = try fixture()
        let result = run(testCase)
        assertSameBits(result.output, testCase.out, "moe.out")
    }

    func testTheTopKWeightsMatchTheContract() throws {
        let testCase = try fixture()
        let result = run(testCase)
        let flattened = result.weights.flatMap { $0 }
        assertSameBits(flattened, testCase.top_k_weights, "moe.top_k_weights")
    }

    /// I3, as its own assertion: the same experts, in the same order.
    func testTheTopKIndexSetMatchesTheContractExactly() throws {
        let testCase = try fixture()
        let result = run(testCase)
        let flattened = result.indices.flatMap { $0 }
        XCTAssertEqual(flattened, testCase.indices.values, "the chosen experts must be the same set, in the same order")
    }

    /// The tie-break is ours — the reference leaves it open — so it is tested directly rather
    /// than only through a vector that happens not to contain a tie.
    func testTiesBreakToTheLowestExpertIndex() {
        // Four experts with identical logits for one token: the top-2 must be experts 0 and 1,
        // in that order. `sorted(by:)` is not a stable sort, so this is the whole reason the
        // comparator carries the index.
        let hidden: [Float] = [1, 0, 0, 0]
        let router: [Float] = [
            1, 0, 0, 0,
            1, 0, 0, 0,
            1, 0, 0, 0,
            1, 0, 0, 0,
        ]
        let (_, indices, _) = MixtureOfExperts.router(
            hidden: hidden, tokens: 1, weights: router, experts: 4, topK: 2
        )
        XCTAssertEqual(indices[0], [0, 1])
    }

    /// A tie above the cut: one expert clearly best, then a tie for the remaining place.
    func testATieAtTheCutKeepsTheHighestAndTakesTheLowestIndexOfTheRest() {
        let hidden: [Float] = [1, 0, 0, 0]
        let router: [Float] = [
            2, 0, 0, 0,  // expert 0, clearly the largest
            1, 0, 0, 0,  // expert 1, tied
            1, 0, 0, 0,  // expert 2, tied
            0, 0, 0, 0,
        ]
        let (_, indices, _) = MixtureOfExperts.router(
            hidden: hidden, tokens: 1, weights: router, experts: 4, topK: 2
        )
        XCTAssertEqual(indices[0], [0, 1])
    }

    /// The routed weights are renormalised over the chosen experts, unconditionally.
    func testTheTopKWeightsSumToOne() throws {
        let testCase = try fixture()
        let result = run(testCase)
        for row in result.weights {
            XCTAssertEqual(row.reduce(0, +), 1.0, accuracy: 1e-5)
        }
    }

    /// The shared expert is added rather than ranked: with a router that sends every token to
    /// no expert at all, the block's output is still the gated shared expert's.
    func testTheSharedExpertIsAddedNotRanked() {
        let tokens = 2
        let shape = MixtureShape(hiddenSize: 2, experts: 2, topK: 1, intermediate: 2, sharedIntermediate: 2)
        let zeros = [Float](repeating: 0, count: 2 * 2)
        let weights = MixtureWeights(
            router: zeros, gateUp: [Float](repeating: 0, count: 2 * 4 * 2),
            down: [Float](repeating: 0, count: 2 * 2 * 2),
            sharedGate: [Float](repeating: 0.5, count: 4), sharedUp: [Float](repeating: 0.5, count: 4),
            sharedDown: [1, 0, 0, 1], sharedScalarGate: [0, 0]
        )
        let hidden: [Float] = [1, 1, 2, 2]
        let (output, _, _) = MixtureOfExperts.block(hidden: hidden, tokens: tokens, weights: weights, shape: shape)
        // sigmoid(0) = 0.5 on the shared expert's output; the routed half is zero because the
        // experts are zero.
        XCTAssertEqual(output.count, tokens * shape.hiddenSize)
        XCTAssertTrue(output.allSatisfy { $0 != 0 }, "the shared expert must contribute")
    }
}
