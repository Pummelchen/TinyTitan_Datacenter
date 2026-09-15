import Foundation

/// The tensor roles the IR dispatches on.
///
/// The pipeline dispatches on **role**, never on tensor name (L1), so this
/// vocabulary is part of the on-disk contract and is versioned with the IR. A role
/// that is not in this list is not importable: refusing an unknown role is how a
/// vendor's new head becomes a conversation instead of a silent omission.
///
/// The list carries roles for both target families even though M0 implements only
/// the dense ones, because I4 requires one policy format to be expressible for
/// every role — a policy file that cannot name `expert.gate` could not describe
/// the model M1 runs.
public enum TensorRole: String, Codable, Sendable, CaseIterable {
    // Token level
    case tokenEmbedding = "token.embedding"
    case outputHead = "head.lm"

    // Norms
    case attnNorm = "norm.attn"
    case mlpNorm = "norm.mlp"
    case finalNorm = "norm.final"

    // Full attention (qwen3, and the full-attention layers of qwen3_5)
    case attnQ = "attn.q"
    case attnK = "attn.k"
    case attnV = "attn.v"
    case attnO = "attn.o"
    case attnQNorm = "attn.q_norm"
    case attnKNorm = "attn.k_norm"
    /// Compressed KV projection (DeepSeek's latent attention, M4).
    case attnKVCompress = "attn.kv_compress"

    // Dense feed-forward
    case mlpGate = "mlp.gate"
    case mlpUp = "mlp.up"
    case mlpDown = "mlp.down"

    // Mixture of experts (M1+).
    //
    // Two layouts, and the difference matters: a *checkpoint* stores the experts stacked
    // into one tensor per projection (`experts.gate_up_proj` is `[experts, 2·inter, hidden]`
    // with the gate and up halves fused), while the *install* L3 produces stores them
    // per role. Both are described here rather than one being reshaped into the other by an
    // importer, because an importer is a name-to-role map and reshaping is a transform pass.
    case routerLogits = "router.logits"
    /// `[experts, 2·moe_intermediate, hidden]`: gate and up fused, stacked.
    case expertGateUpStack = "expert.stack_gate_up"
    /// `[experts, hidden, moe_intermediate]`.
    case expertDownStack = "expert.stack_down"
    /// `[1, hidden]`: the scalar gate on the shared expert's output.
    case sharedExpertGateScalar = "expert.shared.scalar"
    case expertGate = "expert.gate"
    case expertUp = "expert.up"
    case expertDown = "expert.down"
    case sharedExpertGate = "expert.shared.gate"
    case sharedExpertUp = "expert.shared.up"
    case sharedExpertDown = "expert.shared.down"

    // Linear attention / Gated DeltaNet (M0b, M5)
    case linearInQKV = "linear.in_qkv"
    case linearInZ = "linear.in_z"
    case linearInB = "linear.in_b"
    case linearInA = "linear.in_a"
    case linearOut = "linear.out"
    case linearNorm = "linear.norm"
    case linearConv = "linear.conv"
    case linearALog = "linear.a_log"
    case linearDTBias = "linear.dt_bias"

    // Whole-table roles
    case ngramTable = "ngram.table"
    case mtpHead = "mtp.head"
}

/// The shape a role must have for a given configuration, in row-major order.
///
/// This is a *contract*, not a description: an importer that maps a tensor to a
/// role whose shape does not match is wrong, and the importer's output is checked
/// against this before anything downstream trusts it.
public extension TensorRole {
    func expectedShape(_ config: ModelConfig) -> [Int]? {
        switch self {
        case .tokenEmbedding, .outputHead:
            return [config.vocabSize, config.hiddenSize]

        case .attnNorm, .mlpNorm, .finalNorm:
            return [config.hiddenSize]

        case .attnQ:
            // Qwen3.5's query projection carries the output gate as well, so the contract
            // is twice as wide and the layout is `[query | gate]`. A role whose shape
            // depended on nothing would have rejected every qwen3_5 checkpoint.
            let width = config.numAttentionHeads * config.headDim * (config.attnOutputGate ? 2 : 1)
            return [width, config.hiddenSize]
        case .attnK, .attnV:
            return [config.numKeyValueHeads * config.headDim, config.hiddenSize]
        case .attnO:
            return [config.hiddenSize, config.numAttentionHeads * config.headDim]
        case .attnQNorm, .attnKNorm:
            // QK-norm is per head, over head_dim — the detail the reference
            // implementation comments on ("unlike olmo, only on the head dim!").
            return [config.headDim]

        case .mlpGate, .mlpUp:
            return [config.intermediateSize, config.hiddenSize]
        case .mlpDown:
            return [config.hiddenSize, config.intermediateSize]

        case .expertGateUpStack:
            guard let experts = config.numExperts, let width = config.moeIntermediateSize else { return nil }
            return [experts, 2 * width, config.hiddenSize]
        case .expertDownStack:
            guard let experts = config.numExperts, let width = config.moeIntermediateSize else { return nil }
            return [experts, config.hiddenSize, width]
        case .sharedExpertGateScalar:
            return [1, config.hiddenSize]

        case .expertGate, .expertUp:
            guard let width = config.moeIntermediateSize else { return nil }
            return [width, config.hiddenSize]
        case .expertDown:
            guard let width = config.moeIntermediateSize else { return nil }
            return [config.hiddenSize, width]
        // The shared expert has its own width in the configuration.
        case .sharedExpertGate, .sharedExpertUp:
            guard let width = config.sharedExpertIntermediateSize ?? config.moeIntermediateSize else { return nil }
            return [width, config.hiddenSize]
        case .sharedExpertDown:
            guard let width = config.sharedExpertIntermediateSize ?? config.moeIntermediateSize else { return nil }
            return [config.hiddenSize, width]
        case .routerLogits:
            guard let experts = config.numExperts else { return nil }
            return [experts, config.hiddenSize]

        case .linearInQKV:
            guard let keyDim = config.linearKeyDim, let valueDim = config.linearValueDim else { return nil }
            return [keyDim * 2 + valueDim, config.hiddenSize]
        case .linearInZ:
            guard let valueDim = config.linearValueDim else { return nil }
            return [valueDim, config.hiddenSize]
        case .linearInB, .linearInA:
            guard let heads = config.linearValueHeads else { return nil }
            return [heads, config.hiddenSize]
        case .linearOut:
            guard let valueDim = config.linearValueDim else { return nil }
            return [config.hiddenSize, valueDim]
        case .linearNorm:
            guard let headDim = config.linearValueHeadDim else { return nil }
            return [headDim]
        case .linearConv:
            guard let keyDim = config.linearKeyDim,
                  let valueDim = config.linearValueDim,
                  let kernel = config.linearConvKernelDim
            else { return nil }
            return [keyDim * 2 + valueDim, 1, kernel]
        case .linearALog, .linearDTBias:
            guard let heads = config.linearValueHeads else { return nil }
            return [heads]

        case .attnKVCompress, .ngramTable, .mtpHead:
            // Whole-table or family-specific: no shape contract is expressible from
            // the dense configuration, and inventing one would be worse than saying so.
            return nil
        }
    }
}
