import XCTest

import DatacenterIR

@testable import DatacenterEngine

/// The whole `qwen3_5_moe` tower in Swift — M1's model — against the contract's golden bits.
///
/// The checkpoint is tiny (236 KB rather than 67 GB) but it is a *real* one: the real naming
/// under `model.language_model.`, an untied `lm_head`, the real nesting in its configuration
/// file, one Gated DeltaNet layer and one full-attention layer, a partial RoPE, eight experts
/// with a top-2, a shared expert with its scalar gate — and **two key heads to four value
/// heads**, the asymmetry the 2 B model could not exercise.
///
/// So this is M1's arithmetic end to end: the reader, the importer, the layer, the mixture, the
/// head. The router's decisions are asserted **separately** from the tensors, because I3 says
/// they are a different kind of claim from "the numbers are close".
final class Qwen3_5MoEForwardTests: XCTestCase {
    struct Vector: Decodable {
        var shape: [Int]
        var bits: [UInt32]
        var floats: [Float] { bits.map { Float(bitPattern: $0) } }
    }

    struct Decisions: Decodable {
        var shape: [Int]
        var values: [Int]
    }

    struct Golden: Decodable {
        var tokens: [Int]
        // The checkpoint's golden carries these and the install's does not; neither is used
        // here, so they are optional rather than duplicated into a second type.
        var layer_types: [String]?
        var tensors: [String: Vector]
        var discrete: [String: Decisions]
        var argmax: [Int]?
    }

    private func checkpoint() throws -> URL {
        try XCTUnwrap(
            Bundle.module.url(forResource: "tiny-qwen36", withExtension: nil, subdirectory: "Fixtures")
        )
    }

    private func golden() throws -> Golden {
        try JSONDecoder().decode(Golden.self, from: Data(contentsOf: try checkpoint().appendingPathComponent("golden.json")))
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

    /// The slot banks are per **layer**, not per **call**.
    ///
    /// M1's gate measured a cache hit rate of **0.0000 over 2240 requests**, and the cause was
    /// structural: `loadLayer` built a fresh `ExpertSlotCache` every forward, so no token could ever
    /// hit what an earlier one had read. A bank that is dropped with the layer is not a bank. This is
    /// the fixture-scale proof that it now survives — the same token routes to the same experts, so
    /// the second pass must hit what the first one filled.
    func testTheExpertBanksSurviveATokenSoTheSecondOneHits() throws {
        let golden = try golden()
        let forward = try Qwen3_5Forward(snapshot: try checkpoint())
        let first = try forward.forwardWithDecisions(tokens: golden.tokens)
        let second = try forward.forwardWithDecisions(tokens: golden.tokens)
        let firstHits = first.expertMetrics.reduce(0) { $0 + $1.hits }
        let secondHits = second.expertMetrics.reduce(0) { $0 + $1.hits }
        let secondRequests = second.expertMetrics.reduce(0) { $0 + $1.requests }
        XCTAssertEqual(firstHits, 0, "a cold bank cannot hit; if this fails the test proves nothing")
        XCTAssertGreaterThan(secondRequests, 0, "the fixture must route experts at all")
        XCTAssertGreaterThan(
            secondHits, 0,
            "the same token routes to the same experts, so the second pass must hit the bank the first one filled"
        )
    }

    func testTheWholeTowerMatchesTheContractBitForBit() throws {
        let golden = try golden()
        let forward = try Qwen3_5Forward(snapshot: try checkpoint())
        let result = try forward.forwardWithDecisions(tokens: golden.tokens)

        var byName: [String: [Float]] = [:]
        for tensor in result.tensors { byName[tensor.name] = tensor.values }

        for (name, expected) in golden.tensors.sorted(by: { $0.key < $1.key }) {
            guard let actual = byName[name] else {
                XCTFail("the engine did not capture \(name)")
                continue
            }
            assertSameBits(actual, expected, name)
        }
        XCTAssertEqual(result.tensors.count, golden.tensors.count, "same set of captured tensors")
    }

    /// I3, as its own assertion: the same experts, in the same order, at every layer.
    func testTheRouterDecisionsMatchTheContractExactly() throws {
        let golden = try golden()
        let forward = try Qwen3_5Forward(snapshot: try checkpoint())
        let result = try forward.forwardWithDecisions(tokens: golden.tokens)

        XCTAssertEqual(result.discrete.count, golden.discrete.count, "one decision per layer, and no extras")
        for decision in result.discrete {
            guard let expected = golden.discrete[decision.name] else {
                XCTFail("the contract recorded no decision named \(decision.name)")
                continue
            }
            XCTAssertEqual(decision.shape, expected.shape, "\(decision.name): shape")
            XCTAssertEqual(decision.values, expected.values, "\(decision.name): the chosen experts")
        }
    }

    /// The decisions are not tensors: a trace that folded them into a float tensor would make
    /// them comparable by tolerance, and a tolerance cannot express "the same experts".
    func testTheDecisionsAreCarriedApartFromTheTensors() throws {
        let forward = try Qwen3_5Forward(snapshot: try checkpoint())
        let result = try forward.forwardWithDecisions(tokens: [3, 1, 4])
        XCTAssertFalse(result.discrete.isEmpty, "a mixture must record its decisions")
        XCTAssertFalse(
            result.tensors.contains { $0.name.contains("router") },
            "no router value belongs in the tensor list"
        )
    }

    private func goldenInstall() throws -> Golden {
        try JSONDecoder().decode(
            Golden.self, from: Data(contentsOf: try checkpoint().appendingPathComponent("golden-install.json"))
        )
    }

    /// The int4 path, from the Swift side: the install reader, the streaming provider over a
    /// **stacked** expert payload, and the mixture.
    ///
    /// This is the only test that reads a rank-3 quantized tensor in Swift. The Python reader
    /// sized that payload's code block from `shape[0]` — `experts` rows where there are
    /// `experts × rows` — and a reader that does the same still reconstructs *plausible*
    /// weights, so nothing but a bit-for-bit comparison against the other implementation
    /// catches it.
    func testTheInstallPathMatchesTheContractOnTheInstall() throws {
        let golden = try goldenInstall()
        let forward = try Qwen3_5Forward(install: try checkpoint().appendingPathComponent("install"))
        let result = try forward.forwardWithDecisions(tokens: golden.tokens)

        var byName: [String: [Float]] = [:]
        for tensor in result.tensors { byName[tensor.name] = tensor.values }
        for (name, expected) in golden.tensors.sorted(by: { $0.key < $1.key }) {
            guard let actual = byName[name] else {
                XCTFail("the engine did not capture \(name) from the install")
                continue
            }
            assertSameBits(actual, expected, "install:\(name)")
        }
        XCTAssertEqual(result.tensors.count, golden.tensors.count, "same set of captured tensors")

        // The decisions must survive 4-bit exactly: the policy keeps the router at bf16 for
        // this reason, and a tolerance cannot express the claim.
        XCTAssertEqual(result.discrete.count, golden.discrete.count)
        for decision in result.discrete {
            guard let expected = golden.discrete[decision.name] else {
                XCTFail("the contract recorded no decision named \(decision.name)")
                continue
            }
            XCTAssertEqual(decision.values, expected.values, "\(decision.name) from the install")
        }
    }

    /// The engine's install path reads the family from the **spec inside the artifact**, so the
    /// loader does not need the caller to know what it was handed.
    func testTheInstallCarriesItsOwnFamily() throws {
        let install = try checkpoint().appendingPathComponent("install")
        let file = try InstallFile(url: install)
        XCTAssertEqual(file.manifest.spec.family, "qwen3_5_moe")
        let forward = try ModelLoader.open(snapshot: install)
        XCTAssertEqual(forward.spec.family, "qwen3_5_moe")
        XCTAssertFalse(try forward.forwardWithDecisions(tokens: [3, 1, 4]).discrete.isEmpty)
    }

    func testTheGreedyContinuationMatchesTheContract() throws {
        let golden = try golden()
        let forward = try Qwen3_5Forward(snapshot: try checkpoint())
        let logits = try XCTUnwrap(
            try forward.forwardWithDecisions(tokens: golden.tokens).tensors.first { $0.name == "logits" }
        ).values
        var argmax: [Int] = []
        for position in 0..<golden.tokens.count {
            let offset = position * forward.vocabularySize
            argmax.append(Greedy.argmax(logits, offset: offset, width: forward.vocabularySize))
        }
        XCTAssertEqual(argmax, try XCTUnwrap(golden.argmax))
    }

    /// The mixture is a fact about the layer's roles, not about its family name: the same
    /// forward pass serves both, which is what `compare_reference_modules.py` licensed.
    func testTheSpecNamesThisFamilyAndCarriesTheMixtureRoles() throws {
        let forward = try Qwen3_5Forward(snapshot: try checkpoint())
        XCTAssertEqual(forward.spec.family, "qwen3_5_moe")
        let roles = Set(forward.spec.tensors.map(\.role))
        for role in [TensorRole.routerLogits, .expertGateUpStack, .expertDownStack, .sharedExpertGateScalar] {
            XCTAssertTrue(roles.contains(role), "the spec must carry \(role.rawValue)")
        }
        // Stacked, not per expert: the checkpoint's own layout.
        let stacked = try XCTUnwrap(forward.spec.tensors.first { $0.role == .expertGateUpStack })
        XCTAssertEqual(stacked.shape, [8, 32, 32], "eight experts, gate and up fused")
    }
}
