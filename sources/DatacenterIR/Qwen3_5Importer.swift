import Foundation

/// The `qwen3_5` family importer: a pure name-to-role mapping, and nothing else.
///
/// The family is Qwen3.5, whose text tower alternates three Gated DeltaNet layers with one
/// full-attention layer (`full_attention_interval: 4`, so indices 3, 7, 11 … are full
/// attention). The mapping is derived from the checkpoint's own tensor names and shapes —
/// `model.language_model.layers.N.…` — and cross-checked against the layer-kind rule, so a
/// checkpoint whose layout disagreed with the rule would fail the import rather than be
/// silently mis-mapped.
///
/// Two groups of tensors are **not** in the text tower and are excluded by a declared
/// prefix rather than by an accident of the mapping:
///
/// | Prefix | What it is | Why excluded |
/// | --- | --- | --- |
/// | `model.visual.` | the vision tower (24 blocks, merger, patch embed) | M0 is text only, per the plan |
/// | `mtp.` | the multi-token-prediction head | an M5 feature of the other family; `docs/reference-qwen35-2b.md` records that M0 does not include it |
///
/// Excluding by a *declared* prefix is the point: anything outside the text tower is either
/// in this table or the import fails. A weight that quietly does not get used is the worst
/// available failure, and "it was a tensor we did not recognise" is not a defence.
public struct Qwen3_5Importer: Sendable {
    public static let family = "qwen3_5"
    private static let textPrefix = "model.language_model."
    private static let layerPrefix = "model.language_model.layers."

    /// Tensors outside the text tower, with the reason, so an exclusion is a decision
    /// rather than an omission.
    public static let excludedPrefixes: [(prefix: String, reason: String)] = [
        ("model.visual.", "vision tower; M0 is text only"),
        ("mtp.", "multi-token-prediction head; not part of M0"),
    ]

    public struct HuggingFaceConfig: Decodable, Sendable {
        public struct Text: Decodable, Sendable {
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
            public var full_attention_interval: Int?
            public var attn_output_gate: Bool?
            public var linear_num_key_heads: Int?
            public var linear_num_value_heads: Int?
            public var linear_key_head_dim: Int?
            public var linear_value_head_dim: Int?
            public var linear_conv_kernel_dim: Int?
            public var mtp_num_hidden_layers: Int?
        }

        public var text_config: Text
        /// The top-level key, which the multimodal config also carries; the text config's
        /// own value wins when both are present.
        public var tie_word_embeddings: Bool?

        public func modelConfig() -> ModelConfig {
            let text = text_config
            // Broken into named locals on purpose: as one expression this is more than the
            // Swift type checker will accept in reasonable time, and a chain of fourteen
            // `??` operators is not more readable for being on one line.
            let headDim: Int = text.head_dim ?? (text.hidden_size / text.num_attention_heads)
            let ropeTheta: Double = text.rope_theta ?? 10_000
            let tied: Bool = text.tie_word_embeddings ?? tie_word_embeddings ?? false
            let gate: Bool = text.attn_output_gate ?? false
            let interval: Int? = text.full_attention_interval
            let mtpLayers: Int? = text.mtp_num_hidden_layers
            let valueHeads: Int? = text.linear_num_value_heads
            let valueHeadDim: Int? = text.linear_value_head_dim
            let convKernel: Int? = text.linear_conv_kernel_dim

            let keyDim: Int? = {
                guard let heads = text.linear_num_key_heads, let width = text.linear_key_head_dim else { return nil }
                return heads * width
            }()
            let valueDim: Int? = {
                guard let heads = valueHeads, let width = valueHeadDim else { return nil }
                return heads * width
            }()

            return ModelConfig(
                hiddenSize: text.hidden_size,
                numLayers: text.num_hidden_layers,
                numAttentionHeads: text.num_attention_heads,
                numKeyValueHeads: text.num_key_value_heads,
                headDim: headDim,
                intermediateSize: text.intermediate_size,
                vocabSize: text.vocab_size,
                rmsNormEps: text.rms_norm_eps,
                ropeTheta: ropeTheta,
                tieWordEmbeddings: tied,
                attnOutputGate: gate,
                fullAttentionInterval: interval,
                mtpNumHiddenLayers: mtpLayers,
                linearKeyDim: keyDim,
                linearValueDim: valueDim,
                linearValueHeads: valueHeads,
                linearValueHeadDim: valueHeadDim,
                linearConvKernelDim: convKernel
            )
        }
    }

    public enum ImportError: Error, Equatable, Sendable {
        case unmappedTensors([String])
        case invalidSpec([Diagnostic])
    }

    /// Whether a layer index is full attention rather than linear attention.
    ///
    /// The rule comes from the configuration (`full_attention_interval`) and is checked
    /// against the checkpoint's own tensors by the tests: the two must agree, because a
    /// layer mapped to the wrong kind would produce a plausible, wrong forward pass.
    public static func isFullAttention(layerIndex: Int, config: ModelConfig) -> Bool {
        guard let interval = config.fullAttentionInterval, interval > 0 else { return true }
        return layerIndex % interval == interval - 1
    }

    public static func isExcluded(_ name: String) -> Bool {
        excludedPrefixes.contains { name.hasPrefix($0.prefix) }
    }

    /// The mapping, for a text-tower tensor in a known layer.
    public static func role(forTensorNamed name: String) -> TensorRole? {
        switch name {
        case "\(textPrefix)embed_tokens.weight": return .tokenEmbedding
        case "\(textPrefix)norm.weight": return .finalNorm
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
        // Shared by both kinds
        case "mlp.gate_proj.weight": return .mlpGate
        case "mlp.up_proj.weight": return .mlpUp
        case "mlp.down_proj.weight": return .mlpDown
        default: return nil
        }
    }

    public static func block(forTensorNamed name: String) -> String? {
        switch name {
        case "\(textPrefix)embed_tokens.weight": return "embed"
        case "\(textPrefix)norm.weight": return "final"
        default: break
        }
        guard name.hasPrefix(layerPrefix) else { return nil }
        let parts = name.split(separator: ".")
        guard parts.count >= 4, let index = Int(parts[3]) else { return nil }
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

        var policy = ModelPolicy()
        for role in Set(tensors.map(\.role)) {
            policy.setQuant(.bf16, for: role)
            policy.setShard(.replicate, for: role)
        }

        // The text tower is tied: Qwen3.5 ships no `lm_head`, so the embedding is the head
        // and the spec records that rather than leaving it to be inferred at run time.
        let spec = IRSpec(
            family: family, source: source, config: config, blocks: blocks,
            tensors: tensors.sorted { $0.name < $1.name }, policy: policy
        )
        let diagnostics = spec.diagnostics()
        guard diagnostics.isEmpty else { throw .invalidSpec(diagnostics) }
        return spec
    }
}
