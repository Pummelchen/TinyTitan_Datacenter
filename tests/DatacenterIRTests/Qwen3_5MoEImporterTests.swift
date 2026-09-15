import XCTest

@testable import DatacenterIR

/// The `qwen3_5_moe` importer against the **real** inventory of `Qwen/Qwen3.6-35B-A3B`:
/// 1045 tensors read from the headers of 26 shards, vision tower and MTP head included.
///
/// The family is where the mixture of experts first appears, and where the checkpoint's
/// layout differs from the layout an install carries: the routed experts are **stacked**,
/// one tensor per projection holding all 256, not one tensor per expert.
final class Qwen3_5MoEImporterTests: XCTestCase {
    private struct Source: Decodable {
        var repo: String
        var revision: String
        var shards: Int
    }

    private struct Tensor: Decodable {
        var shape: [Int]
        var dtype: String
    }

    private struct Fixture: Decodable {
        var source: Source
        var config: Qwen3_5MoEImporter.HuggingFaceConfig
        var tensors: [String: Tensor]
    }

    private func loadFixture() throws -> Fixture {
        let url = try XCTUnwrap(
            Bundle.module.url(
                forResource: "qwen36-35b-a3b-tensors", withExtension: "json", subdirectory: "Fixtures"
            )
        )
        return try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: url))
    }

    private func makeSpec(_ fixture: Fixture) throws -> IRSpec {
        try Qwen3_5MoEImporter.makeSpec(
            source: Provenance(repo: fixture.source.repo, revision: fixture.source.revision),
            config: fixture.config.modelConfig(),
            inventory: fixture.tensors.map { (name: $0.key, shape: $0.value.shape) }.sorted { $0.name < $1.name }
        )
    }

    // MARK: - The configuration, as the checkpoint states it

    func testReadsTheMoEConfiguration() throws {
        let config = try loadFixture().config.modelConfig()
        XCTAssertEqual(config.numLayers, 40)
        XCTAssertEqual(config.hiddenSize, 2048)
        XCTAssertEqual(config.numAttentionHeads, 16)
        XCTAssertEqual(config.numKeyValueHeads, 2)
        XCTAssertEqual(config.headDim, 256)
        XCTAssertEqual(config.vocabSize, 248320)
        XCTAssertEqual(config.numExperts, 256)
        XCTAssertEqual(config.numExpertsPerToken, 8)
        XCTAssertEqual(config.moeIntermediateSize, 512)
        XCTAssertEqual(config.fullAttentionInterval, 4)
        XCTAssertTrue(config.attnOutputGate)
        // This family is NOT tied: it ships a separate head.
        XCTAssertFalse(config.tieWordEmbeddings)
        // 16 key heads and 32 value heads of 128: the value dimension is twice the key's.
        XCTAssertEqual(config.linearKeyDim, 2048)
        XCTAssertEqual(config.linearValueDim, 4096)
        XCTAssertEqual(config.linearValueHeads, 32)
    }

    // MARK: - The mapping, over the whole checkpoint

    func testEveryTextTensorMapsAndNothingElseIsDropped() throws {
        let fixture = try loadFixture()
        XCTAssertEqual(fixture.tensors.count, 1045)
        let spec = try makeSpec(fixture)
        let mapped = Set(spec.tensors.map(\.name))
        let excluded = fixture.tensors.keys.filter(Qwen3_5MoEImporter.isExcluded)
        XCTAssertEqual(mapped.count + excluded.count, 1045, "no third category")
        XCTAssertEqual(spec.diagnostics(), [])
        XCTAssertEqual(spec.family, "qwen3_5_moe")
        // 30 Gated DeltaNet layers of 18 (9 linear attention + 7 mixture + 2 norms), 10
        // full-attention layers of 15 (6 + 7 + 2), plus the embedding, the final norm and
        // the head. The mixture is seven tensors and not three, because the experts are
        // stacked: a router, a fused gate/up stack, a down stack, and the shared expert's
        // three projections plus its scalar gate.
        XCTAssertEqual(spec.tensors.count, 30 * 18 + 10 * 15 + 3)
    }

    func testTheLayerKindFollowsTheIntervalAndTheCheckpointAgrees() throws {
        let fixture = try loadFixture()
        let config = fixture.config.modelConfig()
        let spec = try makeSpec(fixture)
        let byLayer = Dictionary(grouping: spec.tensors.filter { $0.block.hasPrefix("layer.") }, by: \.block)
        let fullLayers = (0..<config.numLayers).filter {
            Qwen3_5MoEImporter.isFullAttention(layerIndex: $0, config: config)
        }
        XCTAssertEqual(fullLayers, [3, 7, 11, 15, 19, 23, 27, 31, 35, 39])
        for index in 0..<config.numLayers {
            let roles = Set(byLayer[String(format: "layer.%02d", index), default: []].map(\.role))
            if fullLayers.contains(index) {
                XCTAssertTrue(roles.contains(.attnQ))
                XCTAssertFalse(roles.contains(.linearInQKV))
            } else {
                XCTAssertTrue(roles.contains(.linearInQKV))
                XCTAssertFalse(roles.contains(.attnQ))
            }
        }
    }

    func testTheExpertsAreStackedAndTheShapesSaySo() throws {
        let spec = try makeSpec(try loadFixture())
        let named = Dictionary(uniqueKeysWithValues: spec.tensors.map { ($0.name, $0) })
        let prefix = "model.language_model.layers.0."
        // One tensor per projection holding all 256 experts, with gate and up fused.
        XCTAssertEqual(named[prefix + "mlp.experts.gate_up_proj"]?.role, .expertGateUpStack)
        XCTAssertEqual(named[prefix + "mlp.experts.gate_up_proj"]?.shape, [256, 1024, 2048])
        XCTAssertEqual(named[prefix + "mlp.experts.down_proj"]?.role, .expertDownStack)
        XCTAssertEqual(named[prefix + "mlp.experts.down_proj"]?.shape, [256, 2048, 512])
        // The router, the shared expert and its scalar gate.
        XCTAssertEqual(named[prefix + "mlp.gate.weight"]?.role, .routerLogits)
        XCTAssertEqual(named[prefix + "mlp.gate.weight"]?.shape, [256, 2048])
        XCTAssertEqual(named[prefix + "mlp.shared_expert_gate.weight"]?.role, .sharedExpertGateScalar)
        XCTAssertEqual(named[prefix + "mlp.shared_expert_gate.weight"]?.shape, [1, 2048])
        // The head exists and is not the embedding.
        XCTAssertEqual(named["lm_head.weight"]?.role, .outputHead)
        XCTAssertTrue(spec.tensors.contains { $0.role == .outputHead })
    }

    func testAGateUpStackThatIsNotFusedIsRefused() throws {
        let fixture = try loadFixture()
        var inventory = fixture.tensors.map { (name: $0.key, shape: $0.value.shape) }
        // A stack that held only the gate would be the wrong reading of the same name.
        inventory = inventory.map {
            $0.name.hasSuffix("layers.0.mlp.experts.gate_up_proj") ? (name: $0.name, shape: [256, 512, 2048]) : $0
        }
        XCTAssertThrowsError(
            try Qwen3_5MoEImporter.makeSpec(
                source: Provenance(repo: "x", revision: "y"),
                config: fixture.config.modelConfig(),
                inventory: inventory
            )
        ) { error in
            guard case Qwen3_5MoEImporter.ImportError.invalidSpec(let diagnostics) = error else {
                return XCTFail("expected invalidSpec, got \(error)")
            }
            XCTAssertEqual(diagnostics.first?.kind, .shapeMismatch)
        }
    }

    func testAnUnknownTextTensorIsStillAHardError() throws {
        let fixture = try loadFixture()
        var inventory = fixture.tensors.map { (name: $0.key, shape: $0.value.shape) }
        inventory.append((name: "model.language_model.layers.0.mlp.experts.mystery", shape: [1]))
        XCTAssertThrowsError(
            try Qwen3_5MoEImporter.makeSpec(
                source: Provenance(repo: "x", revision: "y"),
                config: fixture.config.modelConfig(),
                inventory: inventory
            )
        ) { error in
            guard case Qwen3_5MoEImporter.ImportError.unmappedTensors(let names) = error else {
                return XCTFail("expected unmappedTensors, got \(error)")
            }
            XCTAssertEqual(names, ["model.language_model.layers.0.mlp.experts.mystery"])
        }
    }
}
