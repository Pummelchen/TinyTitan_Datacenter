import Foundation

/// The `qwen3_5_moe` family importer: Qwen3.6-35B-A3B, the validation model (M1).
///
/// A pure name-to-role map, like the other two importers. What is new here is the mixture of
/// experts: a checkpoint stores the routed experts **stacked** — one tensor per projection
/// holding every expert — so the roles for that layout are `expert.stack_gate_up` and
/// `expert.stack_down`, distinct from the per-expert roles an *install* carries after L3's
/// repack pass. An importer maps names to roles; it does not reshape, so the two layouts are
/// two roles rather than one role and a convention.
///
/// The architecture, read from the checkpoint's own configuration and headers rather than
/// from the model card:
///
/// | | |
/// | --- | --- |
/// | layers | 40: 30 Gated DeltaNet + 10 full attention (`full_attention_interval: 4`) |
/// | attention | 16 heads, 2 KV heads, head_dim 256, and the query carries its output gate |
/// | linear attention | 16 key heads and **32 value heads**, head dims 128, conv kernel 4 |
/// | experts | 256 routed, top-8, intermediate 512, plus a shared expert of 512 |
/// | head | a separate `lm_head.weight` — this family is **not** tied |
///
/// Excluded by a declared prefix, as in `qwen3_5`: the vision tower (`model.visual.`) and
/// the multi-token-prediction head (`mtp.`), which is M5's feature, not M1's.
public struct Qwen3_5MoEImporter: Sendable {
    public static let family = "qwen3_5_moe"
    private static let textPrefix = "model.language_model."
    private static let layerPrefix = "model.language_model.layers."

    public static let excludedPrefixes: [(prefix: String, reason: String)] = [
        ("model.visual.", "vision tower; M1 is text only"),
        ("mtp.", "multi-token-prediction head; not part of M1"),
    ]

    public struct HuggingFaceConfig: Decodable, Sendable {
        public struct Text: Decodable, Sendable {
            public var hidden_size: Int
            public var num_hidden_layers: Int
            public var num_attention_heads: Int
            public var num_key_value_heads: Int
            public var head_dim: Int?
            public var intermediate_size: Int?
            public var vocab_size: Int
            public var rms_norm_eps: Double
            public var tie_word_embeddings: Bool?
            public var full_attention_interval: Int?
            public var attn_output_gate: Bool?
            public var num_experts: Int?
            public var num_experts_per_tok: Int?
            public var moe_intermediate_size: Int?
            public var shared_expert_intermediate_size: Int?
            public var linear_num_key_heads: Int?
            public var linear_num_value_heads: Int?
            public var linear_key_head_dim: Int?
            public var linear_value_head_dim: Int?
            public var linear_conv_kernel_dim: Int?

            public struct RopeParameters: Decodable, Sendable {
                public var rope_theta: Double?
                public var partial_rotary_factor: Double?
                public var rope_type: String?
            }
            public var rope_parameters: RopeParameters?
        }

        public var text_config: Text
        public var tie_word_embeddings: Bool?

        public func modelConfig() -> ModelConfig {
            let text = text_config
            let rope: Text.RopeParameters? = text.rope_parameters
            let valueHeads: Int? = text.linear_num_value_heads
            let keyDim: Int? = {
                guard let heads = text.linear_num_key_heads, let width = text.linear_key_head_dim else { return nil }
                return heads * width
            }()
            let valueDim: Int? = {
                guard let heads = valueHeads, let width = text.linear_value_head_dim else { return nil }
                return heads * width
            }()
            return ModelConfig(
                hiddenSize: text.hidden_size,
                numLayers: text.num_hidden_layers,
                numAttentionHeads: text.num_attention_heads,
                numKeyValueHeads: text.num_key_value_heads,
                headDim: text.head_dim ?? (text.hidden_size / text.num_attention_heads),
                // This family's dense intermediate size is absent: every layer is a mixture
                // of experts, so the per-expert width is what the shapes use.
                intermediateSize: text.intermediate_size ?? text.moe_intermediate_size ?? 0,
                vocabSize: text.vocab_size,
                rmsNormEps: text.rms_norm_eps,
                ropeTheta: rope?.rope_theta ?? 10_000,
                partialRotaryFactor: rope?.partial_rotary_factor,
                tieWordEmbeddings: text.tie_word_embeddings ?? tie_word_embeddings ?? false,
                attnOutputGate: text.attn_output_gate ?? false,
                fullAttentionInterval: text.full_attention_interval,
                numExperts: text.num_experts,
                numExpertsPerToken: text.num_experts_per_tok,
                moeIntermediateSize: text.moe_intermediate_size,
                linearKeyDim: keyDim,
                linearValueDim: valueDim,
                linearValueHeads: valueHeads,
                linearValueHeadDim: text.linear_value_head_dim,
                linearConvKernelDim: text.linear_conv_kernel_dim
            )
        }
    }

    public enum ImportError: Error, Equatable, Sendable {
        case unmappedTensors([String])
        case invalidSpec([Diagnostic])
    }

    public static func isExcluded(_ name: String) -> Bool {
        excludedPrefixes.contains { name.hasPrefix($0.prefix) }
    }

    public static func isFullAttention(layerIndex: Int, config: ModelConfig) -> Bool {
        guard let interval = config.fullAttentionInterval, interval > 0 else { return true }
        return layerIndex % interval == interval - 1
    }

    /// The mapping. Everything this family has, and nothing it does not.
    public static func role(forTensorNamed name: String) -> TensorRole? {
        switch name {
        case "\(textPrefix)embed_tokens.weight": return .tokenEmbedding
        case "\(textPrefix)norm.weight": return .finalNorm
        case "lm_head.weight": return .outputHead
        default: break
        }

        let parts = name.split(separator: ".")
        guard parts.count >= 5, parts[0] == "model", parts[1] == "language_model", parts[2] == "layers",
              Int(parts[3]) != nil
        else { return nil }
        let suffix = parts.dropFirst(4).joined(separator: ".")
        switch suffix {
        case "input_layernorm.weight": return .attnNorm
        case "post_attention_layernorm.weight": return .mlpNorm
        // Full-attention layers
        case "self_attn.q_proj.weight": return .attnQ
        case "self_attn.k_proj.weight": return .attnK
        case "self_attn.v_proj.weight": return .attnV
        case "self_attn.o_proj.weight": return .attnO
        case "self_attn.q_norm.weight": return .attnQNorm
        case "self_attn.k_norm.weight": return .attnKNorm
        // Gated DeltaNet layers
        case "linear_attn.in_proj_qkv.weight": return .linearInQKV
        case "linear_attn.in_proj_z.weight": return .linearInZ
        case "linear_attn.in_proj_a.weight": return .linearInA
        case "linear_attn.in_proj_b.weight": return .linearInB
        case "linear_attn.out_proj.weight": return .linearOut
        case "linear_attn.norm.weight": return .linearNorm
        case "linear_attn.conv1d.weight": return .linearConv
        case "linear_attn.A_log": return .linearALog
        case "linear_attn.dt_bias": return .linearDTBias
        // The mixture of experts, in the checkpoint's stacked layout
        case "mlp.gate.weight": return .routerLogits
        case "mlp.experts.gate_up_proj": return .expertGateUpStack
        case "mlp.experts.down_proj": return .expertDownStack
        case "mlp.shared_expert.gate_proj.weight": return .sharedExpertGate
        case "mlp.shared_expert.up_proj.weight": return .sharedExpertUp
        case "mlp.shared_expert.down_proj.weight": return .sharedExpertDown
        case "mlp.shared_expert_gate.weight": return .sharedExpertGateScalar
        default: return nil
        }
    }

    public static func block(forTensorNamed name: String) -> String? {
        switch name {
        case "\(textPrefix)embed_tokens.weight": return "embed"
        case "\(textPrefix)norm.weight": return "final"
        case "lm_head.weight": return "head"
        default: break
        }
        guard name.hasPrefix(layerPrefix) else { return nil }
        let parts = name.split(separator: ".")
        guard parts.count >= 4, let index = Int(parts[3]) else { return nil }
        return String(format: "layer.%02d", index)
    }

    public static func makeSpec(
        source: Provenance,
        config: ModelConfig,
        inventory: [(name: String, shape: [Int])]
    ) throws(ImportError) -> IRSpec {
        var tensors: [TensorEntry] = []
        var unmapped: [String] = []
        for item in inventory {
            if isExcluded(item.name) { continue }
            guard let role = role(forTensorNamed: item.name),
                  let block = block(forTensorNamed: item.name)
            else {
                unmapped.append(item.name)
                continue
            }
            tensors.append(TensorEntry(name: item.name, role: role, shape: item.shape, block: block))
        }
        guard unmapped.isEmpty else { throw .unmappedTensors(unmapped.sorted()) }

        var blocks: [BlockEntry] = [BlockEntry(id: "embed", kind: .embedding)]
        for index in 0..<config.numLayers {
            blocks.append(BlockEntry(id: String(format: "layer.%02d", index), kind: .decoder, index: index))
        }
        blocks.append(BlockEntry(id: "final", kind: .finalNorm))
        blocks.append(BlockEntry(id: "head", kind: .outputHead))

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
