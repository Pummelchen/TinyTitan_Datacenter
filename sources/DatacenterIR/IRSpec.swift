import Foundation

/// The model's hyperparameters, as the IR understands them.
///
/// Every field the pipeline needs is here explicitly; nothing downstream reads the
/// vendor's config file. The importer is the only place that knows what a vendor
/// calls these (L2).
public struct ModelConfig: Codable, Sendable, Equatable {
    public var hiddenSize: Int
    public var numLayers: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var headDim: Int
    public var intermediateSize: Int
    public var vocabSize: Int
    public var rmsNormEps: Double
    public var ropeTheta: Double
    public var tieWordEmbeddings: Bool

    // Present only in the families that have them; nil means "this model has none",
    // which is different from zero and is why these are optional.
    public var numExperts: Int?
    public var numExpertsPerToken: Int?
    public var moeIntermediateSize: Int?
    public var linearKeyDim: Int?
    public var linearValueDim: Int?
    public var linearValueHeads: Int?
    public var linearValueHeadDim: Int?
    public var linearConvKernelDim: Int?

    public init(
        hiddenSize: Int, numLayers: Int, numAttentionHeads: Int, numKeyValueHeads: Int,
        headDim: Int, intermediateSize: Int, vocabSize: Int, rmsNormEps: Double,
        ropeTheta: Double, tieWordEmbeddings: Bool, numExperts: Int? = nil,
        numExpertsPerToken: Int? = nil, moeIntermediateSize: Int? = nil,
        linearKeyDim: Int? = nil, linearValueDim: Int? = nil, linearValueHeads: Int? = nil,
        linearValueHeadDim: Int? = nil, linearConvKernelDim: Int? = nil
    ) {
        self.hiddenSize = hiddenSize
        self.numLayers = numLayers
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.headDim = headDim
        self.intermediateSize = intermediateSize
        self.vocabSize = vocabSize
        self.rmsNormEps = rmsNormEps
        self.ropeTheta = ropeTheta
        self.tieWordEmbeddings = tieWordEmbeddings
        self.numExperts = numExperts
        self.numExpertsPerToken = numExpertsPerToken
        self.moeIntermediateSize = moeIntermediateSize
        self.linearKeyDim = linearKeyDim
        self.linearValueDim = linearValueDim
        self.linearValueHeads = linearValueHeads
        self.linearValueHeadDim = linearValueHeadDim
        self.linearConvKernelDim = linearConvKernelDim
    }
}

/// How a tensor is stored (I4). The policy is data: changing it changes no code.
public enum QuantPolicy: String, Codable, Sendable, CaseIterable {
    case fp32
    case bf16
    case fp16
    /// Vendor-native FP8 with block scaling; the transcoder preserves it (I5).
    case fp8Block = "fp8-block"
    /// Vendor-native 4-bit block format for routed experts (I5).
    case fp4Block = "fp4-block"
    /// Our own affine int4, group 64 — used only where a vendor format is not reusable.
    case int4Affine = "int4-affine"
}

/// Where a tensor lives when the model is split across nodes (I4).
public enum ShardPolicy: String, Codable, Sendable, CaseIterable {
    /// Every node holds a copy: the dense backbone and anything every token needs.
    case replicate
    case shardByExpert = "shard-by-expert"
    case shardByHead = "shard-by-head"
    case shardByRow = "shard-by-row"
}

/// A tensor's role, shape and where it sits.
public struct TensorEntry: Codable, Sendable, Equatable {
    public var name: String
    public var role: TensorRole
    public var shape: [Int]
    /// The block this tensor belongs to; must name a block in `blocks`.
    public var block: String

    public init(name: String, role: TensorRole, shape: [Int], block: String) {
        self.name = name
        self.role = role
        self.shape = shape
        self.block = block
    }
}

/// One block of the model's execution sequence.
public struct BlockEntry: Codable, Sendable, Equatable {
    public enum Kind: String, Codable, Sendable, CaseIterable {
        case embedding
        case decoder
        case finalNorm = "final-norm"
        case outputHead = "output-head"
    }

    public var id: String
    public var kind: Kind
    /// Which decoder layer this is, for `decoder` blocks.
    public var index: Int?

    public init(id: String, kind: Kind, index: Int? = nil) {
        self.id = id
        self.kind = kind
        self.index = index
    }
}

/// Where the model came from (I6). Every converted artifact carries this.
public struct Provenance: Codable, Sendable, Equatable {
    public var repo: String
    public var revision: String
    /// sha256 of each source weight file, by file name.
    public var files: [String: String]
    /// The ordered list of transform passes applied, empty before conversion.
    public var passes: [String]
    /// The policy files used, by name.
    public var policies: [String]

    public init(repo: String, revision: String, files: [String: String] = [:],
                passes: [String] = [], policies: [String] = []) {
        self.repo = repo
        self.revision = revision
        self.files = files
        self.passes = passes
        self.policies = policies
    }
}

/// The per-role quantization and sharding tables (I4).
public struct ModelPolicy: Codable, Sendable, Equatable {
    public var quant: [String: QuantPolicy]
    public var shard: [String: ShardPolicy]

    public init(quant: [String: QuantPolicy] = [:], shard: [String: ShardPolicy] = [:]) {
        self.quant = quant
        self.shard = shard
    }

    public func quant(for role: TensorRole) -> QuantPolicy? { quant[role.rawValue] }
    public func shard(for role: TensorRole) -> ShardPolicy? { shard[role.rawValue] }

    public mutating func setQuant(_ policy: QuantPolicy, for role: TensorRole) {
        quant[role.rawValue] = policy
    }

    public mutating func setShard(_ policy: ShardPolicy, for role: TensorRole) {
        shard[role.rawValue] = policy
    }
}

/// The IR spec: a declarative description of a model, not code.
///
/// It is data on disk because the pipeline dispatches on it — a new family is a new
/// spec plus whatever kernels the spec's ops need, and a point release with unchanged
/// ops is a new spec with the same ops and a new revision.
public struct IRSpec: Codable, Sendable, Equatable {
    /// Bumped whenever the meaning of a field changes. A reader that does not know
    /// this number refuses the file rather than guessing.
    public static let currentVersion = 1

    public var irVersion: Int
    public var family: String
    public var source: Provenance
    public var config: ModelConfig
    public var blocks: [BlockEntry]
    public var tensors: [TensorEntry]
    public var policy: ModelPolicy

    public init(irVersion: Int = IRSpec.currentVersion, family: String, source: Provenance,
                config: ModelConfig, blocks: [BlockEntry], tensors: [TensorEntry],
                policy: ModelPolicy) {
        self.irVersion = irVersion
        self.family = family
        self.source = source
        self.config = config
        self.blocks = blocks
        self.tensors = tensors
        self.policy = policy
    }

    public func encodeJSON(pretty: Bool = true) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = pretty ? [.prettyPrinted, .sortedKeys] : [.sortedKeys]
        return try encoder.encode(self)
    }

    public static func decodeJSON(_ data: Data) throws -> IRSpec {
        try JSONDecoder().decode(IRSpec.self, from: data)
    }
}

extension TensorRole {
    /// Refuse an unknown role by name, listing what is known. A role this version
    /// does not understand is a conversation, not a silent default.
    public init(from decoder: any Decoder) throws {
        let raw = try decoder.singleValueContainer().decode(String.self)
        guard let role = TensorRole(rawValue: raw) else {
            throw DecodingError.dataCorrupted(
                .init(
                    codingPath: decoder.codingPath,
                    debugDescription: "unknown tensor role '\(raw)'; this IR version knows: "
                        + TensorRole.allCases.map(\.rawValue).sorted().joined(separator: ", ")
                )
            )
        }
        self = role
    }
}
