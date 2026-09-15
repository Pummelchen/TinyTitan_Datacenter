import Foundation

/// The `qwen3` family importer: a pure name-to-role mapping, and nothing else.
///
/// No math, no quantization, no reshaping decisions beyond what the role's shape
/// contract states. When Qwen ships a point release with unchanged ops, this is the
/// only file that changes (L2) — and if a tensor appears that this table does not
/// know, the import fails loudly rather than dropping it.
///
/// Source of truth for the mapping: the checkpoint's own tensor names, and the
/// reference implementation's module structure
/// (`transformers` v5.17.0, `models/qwen3/modeling_qwen3.py`, `Qwen3DecoderLayer:283`).
/// See `docs/reference-qwen3-dense.md`.
public struct Qwen3Importer: Sendable {
    public static let family = "qwen3"

    /// Qwen3's own configuration keys, mapped once, here. Nothing downstream reads
    /// a vendor config file.
    public struct HuggingFaceConfig: Decodable, Sendable {
        public var hidden_size: Int
        public var num_hidden_layers: Int
        public var num_attention_heads: Int
        public var num_key_value_heads: Int
        public var head_dim: Int?
        public var intermediate_size: Int
        public var vocab_size: Int
        public var rms_norm_eps: Double
        public var rope_theta: Double?
        public var tie_word_embeddings: Bool?

        public func modelConfig() -> ModelConfig {
            ModelConfig(
                hiddenSize: hidden_size,
                numLayers: num_hidden_layers,
                numAttentionHeads: num_attention_heads,
                numKeyValueHeads: num_key_value_heads,
                // The checkpoint states head_dim explicitly in every Qwen3 release
                // checked; the derivation is the reference's own fallback.
                headDim: head_dim ?? (hidden_size / num_attention_heads),
                intermediateSize: intermediate_size,
                vocabSize: vocab_size,
                rmsNormEps: rms_norm_eps,
                ropeTheta: rope_theta ?? 1_000_000,
                tieWordEmbeddings: tie_word_embeddings ?? false
            )
        }
    }

    public enum ImportError: Error, Equatable, Sendable {
        case unmappedTensors([String])
        case invalidSpec([Diagnostic])
    }

    /// The mapping. Everything the family has, and nothing it does not.
    public static func role(forTensorNamed name: String) -> TensorRole? {
        switch name {
        case "model.embed_tokens.weight": return .tokenEmbedding
        case "lm_head.weight": return .outputHead
        case "model.norm.weight": return .finalNorm
        default: break
        }

        let parts = name.split(separator: ".")
        guard parts.count >= 4, parts[0] == "model", parts[1] == "layers" else { return nil }
        let layerSuffix = parts.dropFirst(3).joined(separator: ".")
        switch layerSuffix {
        case "input_layernorm.weight": return .attnNorm
        case "post_attention_layernorm.weight": return .mlpNorm
        case "self_attn.q_proj.weight": return .attnQ
        case "self_attn.k_proj.weight": return .attnK
        case "self_attn.v_proj.weight": return .attnV
        case "self_attn.o_proj.weight": return .attnO
        case "self_attn.q_norm.weight": return .attnQNorm
        case "self_attn.k_norm.weight": return .attnKNorm
        case "mlp.gate_proj.weight": return .mlpGate
        case "mlp.up_proj.weight": return .mlpUp
        case "mlp.down_proj.weight": return .mlpDown
        default: return nil
        }
    }

    /// The block a tensor belongs to, by name. Derived, not decided: the shape of the
    /// name already says which layer it is in.
    public static func block(forTensorNamed name: String) -> String? {
        switch name {
        case "model.embed_tokens.weight": return "embed"
        case "model.norm.weight": return "final"
        case "lm_head.weight": return "head"
        default: break
        }
        let parts = name.split(separator: ".")
        guard parts.count >= 3, parts[0] == "model", parts[1] == "layers",
              let index = Int(parts[2])
        else { return nil }
        return String(format: "layer.%02d", index)
    }

    /// Build a spec from a checkpoint's tensor inventory (`name -> shape`).
    public static func makeSpec(
        source: Provenance,
        config: ModelConfig,
        inventory: [(name: String, shape: [Int])]
    ) throws(ImportError) -> IRSpec {
        var tensors: [TensorEntry] = []
        var unmapped: [String] = []

        for item in inventory {
            guard let role = role(forTensorNamed: item.name),
                  let block = block(forTensorNamed: item.name)
            else {
                unmapped.append(item.name)
                continue
            }
            tensors.append(TensorEntry(name: item.name, role: role, shape: item.shape, block: block))
        }

        // An unmapped tensor is a hard error: it is a weight the engine would
        // silently not use, which is the worst of the available failures.
        guard unmapped.isEmpty else { throw .unmappedTensors(unmapped.sorted()) }

        var blocks: [BlockEntry] = [BlockEntry(id: "embed", kind: .embedding)]
        for index in 0..<config.numLayers {
            blocks.append(BlockEntry(id: String(format: "layer.%02d", index), kind: .decoder, index: index))
        }
        blocks.append(BlockEntry(id: "final", kind: .finalNorm))
        blocks.append(BlockEntry(id: "head", kind: .outputHead))

        // M0 is bf16 with nothing sharded: the policy exists, so that changing it is
        // a data change rather than a code change (I4).
        var policy = ModelPolicy()
        for role in Set(tensors.map(\.role)) {
            policy.setQuant(.bf16, for: role)
            policy.setShard(.replicate, for: role)
        }

        let spec = IRSpec(
            family: family, source: source, config: config, blocks: blocks,
            tensors: tensors.sorted { $0.name < $1.name }, policy: policy
        )
        let diagnostics = spec.diagnostics()
        guard diagnostics.isEmpty else { throw .invalidSpec(diagnostics) }
        return spec
    }
}
