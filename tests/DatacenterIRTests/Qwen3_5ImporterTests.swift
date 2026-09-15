import XCTest

@testable import DatacenterIR

/// The `qwen3_5` importer against the **real** inventory of `Qwen/Qwen3.5-2B`: 632 tensors
/// dumped from the safetensors header, vision tower and MTP head included.
///
/// The family is where guessing would be costly. Its text tower alternates three Gated
/// DeltaNet layers with one full-attention layer, its query projection carries an output
/// gate, and most of the checkpoint's tensors are not text at all. Every one of those facts
/// is checked here against the checkpoint rather than against a reading of it.
final class Qwen3_5ImporterTests: XCTestCase {
    private struct Source: Decodable {
        var repo: String
        var revision: String
        var weights: [String]
    }

    private struct Fixture: Decodable {
        var source: Source
        var config: Qwen3_5Importer.HuggingFaceConfig
        var tensors: [String: [Int]]
    }

    private func loadFixture() throws -> Fixture {
        let url = try XCTUnwrap(
            Bundle.module.url(
                forResource: "qwen35-2b-tensors", withExtension: "json", subdirectory: "Fixtures"
            )
        )
        return try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: url))
    }

    private func makeSpec(_ fixture: Fixture) throws -> IRSpec {
        try Qwen3_5Importer.makeSpec(
            source: Provenance(
                repo: fixture.source.repo, revision: fixture.source.revision, files: ["model.safetensors": "…"]
            ),
            config: fixture.config.modelConfig(),
            inventory: fixture.tensors.map { (name: $0.key, shape: $0.value) }.sorted { $0.name < $1.name }
        )
    }

    // MARK: - The configuration, as the checkpoint states it

    func testReadsTheNestedTextConfiguration() throws {
        let config = try loadFixture().config.modelConfig()
        XCTAssertEqual(config.hiddenSize, 2048)
        XCTAssertEqual(config.numLayers, 24)
        XCTAssertEqual(config.numAttentionHeads, 8)
        XCTAssertEqual(config.numKeyValueHeads, 2)
        XCTAssertEqual(config.headDim, 256)
        XCTAssertEqual(config.intermediateSize, 6144)
        XCTAssertEqual(config.vocabSize, 248320)
        XCTAssertEqual(config.rmsNormEps, 1e-6)
        XCTAssertEqual(config.fullAttentionInterval, 4)
        XCTAssertTrue(config.attnOutputGate, "q_proj is 4096 wide for 8 heads of 256: the gate is in the checkpoint")
        XCTAssertTrue(config.tieWordEmbeddings)
        XCTAssertEqual(config.mtpNumHiddenLayers, 1)
        // The linear-attention geometry: 16 key heads and 16 value heads of 128.
        XCTAssertEqual(config.linearKeyDim, 2048)
        XCTAssertEqual(config.linearValueDim, 2048)
        XCTAssertEqual(config.linearValueHeads, 16)
        XCTAssertEqual(config.linearValueHeadDim, 128)
        XCTAssertEqual(config.linearConvKernelDim, 4)
    }

    // MARK: - The mapping, over the whole checkpoint

    func testEveryTextTensorMapsAndNothingElseIsDropped() throws {
        let fixture = try loadFixture()
        XCTAssertEqual(fixture.tensors.count, 632, "the fixture is the whole checkpoint")

        let spec = try makeSpec(fixture)
        let mapped = Set(spec.tensors.map(\.name))
        let excluded = fixture.tensors.keys.filter(Qwen3_5Importer.isExcluded)

        // Every tensor is either mapped or excluded by a declared prefix — there is no
        // third category, which is the property that makes an exclusion a decision.
        XCTAssertEqual(mapped.count + excluded.count, 632)
        XCTAssertEqual(spec.diagnostics(), [])
        XCTAssertEqual(spec.family, "qwen3_5")
        XCTAssertEqual(spec.tensors.count, 320, "1 embedding + 18 GDN layers + 6 attention layers + 1 final norm")
    }

    func testTheDeclaredExclusionsAreExactlyTheVisionTowerAndTheMtpHead() throws {
        let fixture = try loadFixture()
        let excluded = fixture.tensors.keys.filter(Qwen3_5Importer.isExcluded).sorted()
        XCTAssertEqual(excluded.count, 312, "297 vision + 15 MTP")
        XCTAssertTrue(excluded.allSatisfy { $0.hasPrefix("model.visual.") || $0.hasPrefix("mtp.") })
        XCTAssertFalse(excluded.contains { $0.hasPrefix("model.language_model.") })

        // Named, with reasons, so the next reader does not have to reconstruct the decision.
        XCTAssertEqual(Qwen3_5Importer.excludedPrefixes.map(\.prefix), ["model.visual.", "mtp."])
        XCTAssertTrue(Qwen3_5Importer.excludedPrefixes.allSatisfy { !$0.reason.isEmpty })
    }

    func testLayerKindFollowsTheIntervalAndTheCheckpointAgrees() throws {
        let fixture = try loadFixture()
        let config = fixture.config.modelConfig()
        let spec = try makeSpec(fixture)

        let byLayer = Dictionary(grouping: spec.tensors.filter { $0.block.hasPrefix("layer.") }, by: \.block)
        for index in 0..<config.numLayers {
            let block = String(format: "layer.%02d", index)
            let roles = Set(byLayer[block, default: []].map(\.role))
            let full = Qwen3_5Importer.isFullAttention(layerIndex: index, config: config)
            if full {
                XCTAssertTrue(roles.contains(.attnQ), "\(block) should be full attention")
                XCTAssertFalse(roles.contains(.linearInQKV), "\(block) should not be linear attention")
            } else {
                XCTAssertTrue(roles.contains(.linearInQKV), "\(block) should be linear attention")
                XCTAssertFalse(roles.contains(.attnQ), "\(block) should not be full attention")
            }
        }
        // The rule and the checkpoint agree: indices 3, 7, 11, 15, 19, 23 are full attention.
        let fullLayers = (0..<config.numLayers).filter { Qwen3_5Importer.isFullAttention(layerIndex: $0, config: config) }
        XCTAssertEqual(fullLayers, [3, 7, 11, 15, 19, 23])
        XCTAssertEqual(fullLayers.count, 6)
    }

    func testShapesThatTheDenseContractWouldHaveRejected() throws {
        let spec = try makeSpec(try loadFixture())
        let named = Dictionary(uniqueKeysWithValues: spec.tensors.map { ($0.name, $0) })
        let prefix = "model.language_model.layers"

        // The query projection carries the output gate: 2 × 8 heads × 256.
        XCTAssertEqual(named["\(prefix).3.self_attn.q_proj.weight"]?.shape, [4096, 2048])
        XCTAssertEqual(named["\(prefix).3.self_attn.k_proj.weight"]?.shape, [512, 2048])

        // Gated DeltaNet: Q and K are 16 heads of 128, V is 16 heads of 128, concatenated.
        XCTAssertEqual(named["\(prefix).0.linear_attn.in_proj_qkv.weight"]?.shape, [6144, 2048])
        XCTAssertEqual(named["\(prefix).0.linear_attn.in_proj_z.weight"]?.shape, [2048, 2048])
        XCTAssertEqual(named["\(prefix).0.linear_attn.in_proj_a.weight"]?.shape, [16, 2048])
        XCTAssertEqual(named["\(prefix).0.linear_attn.conv1d.weight"]?.shape, [6144, 1, 4])
        XCTAssertEqual(named["\(prefix).0.linear_attn.norm.weight"]?.shape, [128])
        XCTAssertEqual(named["\(prefix).0.linear_attn.A_log"]?.shape, [16])
        XCTAssertEqual(named["\(prefix).0.linear_attn.dt_bias"]?.shape, [16])

        // Tied embeddings: the checkpoint ships no `lm_head` at all.
        XCTAssertEqual(named["model.language_model.embed_tokens.weight"]?.role, .tokenEmbedding)
        XCTAssertNil(named["lm_head.weight"])
        XCTAssertFalse(spec.tensors.contains { $0.role == .outputHead })
    }

    func testTheSpecRecordsNoHeadAndStillValidates() throws {
        let spec = try makeSpec(try loadFixture())
        // `tieWordEmbeddings` is true, so the missing head is legal — and the test states
        // that rather than relying on the validator's silence.
        XCTAssertTrue(spec.config.tieWordEmbeddings)
        XCTAssertEqual(spec.diagnostics(), [])
    }

    func testAnUnknownTextTensorIsStillAHardError() throws {
        let fixture = try loadFixture()
        var inventory = fixture.tensors.map { (name: $0.key, shape: $0.value) }
        inventory.append((name: "model.language_model.layers.0.linear_attn.mystery.weight", shape: [8, 8]))

        XCTAssertThrowsError(
            try Qwen3_5Importer.makeSpec(
                source: Provenance(repo: "x", revision: "y"),
                config: fixture.config.modelConfig(),
                inventory: inventory
            )
        ) { error in
            guard case Qwen3_5Importer.ImportError.unmappedTensors(let names) = error else {
                return XCTFail("expected unmappedTensors, got \(error)")
            }
            XCTAssertEqual(names, ["model.language_model.layers.0.linear_attn.mystery.weight"])
        }
    }

    /// A DenseNet-shaped check that the gate flag really changes the contract: with the flag
    /// off, the same q_proj is a shape mismatch, which is what would have happened had the
    /// flag been guessed rather than read from the config.
    func testTheGateFlagIsWhatMakesTheQueryShapeLegal() throws {
        let fixture = try loadFixture()
        var config = fixture.config
        config.text_config.attn_output_gate = false
        XCTAssertThrowsError(
            try Qwen3_5Importer.makeSpec(
                source: Provenance(repo: "x", revision: "y"),
                config: config.modelConfig(),
                inventory: fixture.tensors.map { (name: $0.key, shape: $0.value) }
            )
        ) { error in
            guard case Qwen3_5Importer.ImportError.invalidSpec(let diagnostics) = error else {
                return XCTFail("expected invalidSpec, got \(error)")
            }
            XCTAssertTrue(diagnostics.contains { $0.kind == .shapeMismatch && $0.subject.hasSuffix("q_proj.weight") })
        }
    }
}
