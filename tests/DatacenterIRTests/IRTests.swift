import XCTest

@testable import DatacenterIR

/// Tests for the IR and the `qwen3` importer.
///
/// The inventory is a real checkpoint's: `Qwen/Qwen3-0.6B` at revision
/// `c1899de2…`, 311 tensors with their shapes, dumped from the safetensors header.
/// Testing a mapping against the names a vendor actually ships is the difference
/// between a mapping and a guess.
final class IRTests: XCTestCase {
    // MARK: - Fixture

    private struct FixtureSource: Decodable {
        var repo: String
        var revision: String
        var sha256: String
    }

    private struct Fixture: Decodable {
        var source: FixtureSource
        var config: Qwen3Importer.HuggingFaceConfig
        var tensors: [String: [Int]]
    }

    private func loadFixture() throws -> Fixture {
        // The fixture is a declared resource, so it travels with the test bundle
        // rather than being read from a path that only exists in a checkout.
        let url = try XCTUnwrap(
            Bundle.module.url(
                forResource: "qwen3-0.6b-tensors", withExtension: "json", subdirectory: "Fixtures"
            )
        )
        return try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: url))
    }

    private func makeSpec(_ fixture: Fixture) throws -> IRSpec {
        try Qwen3Importer.makeSpec(
            source: Provenance(
                repo: fixture.source.repo,
                revision: fixture.source.revision,
                files: ["model.safetensors": fixture.source.sha256]
            ),
            config: fixture.config.modelConfig(),
            inventory: fixture.tensors.map { (name: $0.key, shape: $0.value) }.sorted { $0.name < $1.name }
        )
    }

    // MARK: - Importing a real checkpoint

    func testImportsEveryTensorOfTheRealCheckpoint() throws {
        let fixture = try loadFixture()
        XCTAssertEqual(fixture.tensors.count, 311)

        let spec = try makeSpec(fixture)
        XCTAssertEqual(spec.family, "qwen3")
        XCTAssertEqual(spec.tensors.count, 311, "every tensor in the checkpoint must map to a role")
        XCTAssertEqual(spec.config.numLayers, 28)
        XCTAssertEqual(spec.blocks.filter { $0.kind == .decoder }.count, 28)
        XCTAssertEqual(spec.diagnostics(), [], "a spec built from a real checkpoint must validate cleanly")
    }

    func testMapsTheDetailsThatAreEasyToGetWrong() throws {
        let spec = try makeSpec(try loadFixture())
        let byName = Dictionary(uniqueKeysWithValues: spec.tensors.map { ($0.name, $0) })

        // QK-norm is per head, over head_dim — not over hidden_size.
        XCTAssertEqual(byName["model.layers.0.self_attn.q_norm.weight"]?.role, .attnQNorm)
        XCTAssertEqual(byName["model.layers.0.self_attn.q_norm.weight"]?.shape, [128])

        // 16 query heads, 8 KV heads, head_dim 128.
        XCTAssertEqual(byName["model.layers.0.self_attn.q_proj.weight"]?.shape, [2048, 1024])
        XCTAssertEqual(byName["model.layers.0.self_attn.k_proj.weight"]?.shape, [1024, 1024])
        XCTAssertEqual(byName["model.layers.0.self_attn.o_proj.weight"]?.shape, [1024, 2048])

        // SwiGLU order in the checkpoint is gate/up/down, each with its own role.
        XCTAssertEqual(byName["model.layers.27.mlp.gate_proj.weight"]?.role, .mlpGate)
        XCTAssertEqual(byName["model.layers.27.mlp.down_proj.weight"]?.shape, [1024, 3072])

        // The head is a separate tensor even though the config says tied.
        XCTAssertEqual(byName["lm_head.weight"]?.role, .outputHead)
        XCTAssertEqual(byName["model.embed_tokens.weight"]?.role, .tokenEmbedding)
    }

    func testBlocksFollowTheExecutionOrder() throws {
        let spec = try makeSpec(try loadFixture())
        XCTAssertEqual(spec.blocks.first?.kind, .embedding)
        XCTAssertEqual(spec.blocks.last?.kind, .outputHead)
        XCTAssertEqual(spec.blocks.dropFirst().dropLast(2).map(\.id).first, "layer.00")
        for tensor in spec.tensors where tensor.name.hasPrefix("model.layers.") {
            let index = Int(tensor.name.split(separator: ".")[2])!
            XCTAssertEqual(tensor.block, String(format: "layer.%02d", index))
        }
    }

    func testPolicyIsPopulatedForEveryRoleInUse() throws {
        let spec = try makeSpec(try loadFixture())
        for role in Set(spec.tensors.map(\.role)) {
            XCTAssertEqual(spec.policy.quant(for: role), .bf16, "role \(role.rawValue) has no quantization policy")
            XCTAssertEqual(spec.policy.shard(for: role), .replicate, "role \(role.rawValue) has no sharding policy")
        }
    }

    // MARK: - Refusing what it does not understand

    func testUnmappedTensorIsAHardError() throws {
        let fixture = try loadFixture()
        var inventory = fixture.tensors.map { (name: $0.key, shape: $0.value) }
        inventory.append((name: "model.vision.patch_embed.weight", shape: [16, 16, 3, 768]))

        XCTAssertThrowsError(
            try Qwen3Importer.makeSpec(
                source: Provenance(repo: "x", revision: "y"),
                config: fixture.config.modelConfig(),
                inventory: inventory
            )
        ) { error in
            guard case Qwen3Importer.ImportError.unmappedTensors(let names) = error else {
                return XCTFail("expected unmappedTensors, got \(error)")
            }
            XCTAssertEqual(names, ["model.vision.patch_embed.weight"])
        }
    }

    func testUnknownRoleInASpecFileIsRefusedByName() throws {
        let spec = try makeSpec(try loadFixture())
        var json = try JSONSerialization.jsonObject(with: spec.encodeJSON()) as! [String: Any]
        var tensors = json["tensors"] as! [[String: Any]]
        tensors[0]["role"] = "attn.mystery_projection"
        json["tensors"] = tensors
        let data = try JSONSerialization.data(withJSONObject: json)

        XCTAssertThrowsError(try IRSpec.decodeJSON(data)) { error in
            let text = String(describing: error)
            XCTAssertTrue(text.contains("attn.mystery_projection"), text)
            XCTAssertTrue(text.contains("head.lm"), "the refusal should list what is known: \(text)")
        }
    }

    func testShapeMismatchIsCaughtAgainstTheRoleContract() throws {
        let fixture = try loadFixture()
        var inventory = fixture.tensors.map { (name: $0.key, shape: $0.value) }
        // A plausible transcription error: q_proj shaped as if q_norm's head_dim were the hidden size.
        inventory = inventory.map {
            $0.name == "model.layers.3.self_attn.q_proj.weight" ? (name: $0.name, shape: [128, 1024]) : $0
        }
        XCTAssertThrowsError(
            try Qwen3Importer.makeSpec(
                source: Provenance(repo: "x", revision: "y"),
                config: fixture.config.modelConfig(),
                inventory: inventory
            )
        ) { error in
            guard case Qwen3Importer.ImportError.invalidSpec(let diagnostics) = error else {
                return XCTFail("expected invalidSpec, got \(error)")
            }
            XCTAssertEqual(diagnostics.first?.kind, .shapeMismatch)
            XCTAssertEqual(diagnostics.first?.subject, "model.layers.3.self_attn.q_proj.weight")
        }
    }

    func testMissingPolicyIsADiagnostic() throws {
        var spec = try makeSpec(try loadFixture())
        spec.policy.quant.removeValue(forKey: TensorRole.attnQ.rawValue)
        let diagnostics = spec.diagnostics()
        XCTAssertTrue(
            diagnostics.contains { $0.kind == .missingPolicy && $0.subject == TensorRole.attnQ.rawValue },
            "\(diagnostics)"
        )
    }

    func testDuplicateTensorIsADiagnostic() throws {
        var spec = try makeSpec(try loadFixture())
        spec.tensors.append(spec.tensors[0])
        XCTAssertTrue(spec.diagnostics().contains { $0.kind == .duplicateTensor })
    }

    func testUntiedModelWithoutAHeadIsRefused() throws {
        let fixture = try loadFixture()
        var config = fixture.config
        config.tie_word_embeddings = false
        let inventory = fixture.tensors
            .filter { $0.key != "lm_head.weight" }
            .map { (name: $0.key, shape: $0.value) }

        XCTAssertThrowsError(
            try Qwen3Importer.makeSpec(
                source: Provenance(repo: "x", revision: "y"),
                config: config.modelConfig(),
                inventory: inventory
            )
        ) { error in
            guard case Qwen3Importer.ImportError.invalidSpec(let diagnostics) = error else {
                return XCTFail("expected invalidSpec, got \(error)")
            }
            XCTAssertTrue(diagnostics.contains { $0.kind == .missingOutputHead }, "\(diagnostics)")
        }
    }

    // MARK: - The spec is data

    func testSpecRoundTripsThroughJSON() throws {
        let spec = try makeSpec(try loadFixture())
        let decoded = try IRSpec.decodeJSON(spec.encodeJSON())
        XCTAssertEqual(decoded, spec)
    }

    func testUnsupportedIRVersionIsRefused() throws {
        var spec = try makeSpec(try loadFixture())
        spec.irVersion = 99
        XCTAssertTrue(spec.diagnostics().contains { $0.kind == .unsupportedIRVersion })
        XCTAssertThrowsError(try spec.validate()) { error in
            guard case IRValidationError.invalid(let diagnostics) = error else {
                return XCTFail("expected IRValidationError, got \(error)")
            }
            XCTAssertEqual(diagnostics.map(\.kind), [.unsupportedIRVersion])
        }
    }

    func testProvenanceCarriesTheRevisionAndThePassList() throws {
        let spec = try makeSpec(try loadFixture())
        XCTAssertEqual(spec.source.repo, "Qwen/Qwen3-0.6B")
        XCTAssertEqual(spec.source.revision, "c1899de289a04d12100db370d81485cdf75e47ca")
        XCTAssertEqual(spec.source.files.keys.sorted(), ["model.safetensors"])
        XCTAssertEqual(spec.source.passes, [], "nothing has been transformed yet, and the spec says so")
    }
}
