import DatacenterIR
import Foundation

/// The `qwen3_5` text tower, driven by the IR, with weights loaded **one layer at a time**.
///
/// The structural problem this solves: 2 B parameters are 8 GB in fp32, and the nodes have
/// 8 GB total with about 4.5 GB usable. So nothing is loaded whole. The embedding is read
/// a row at a time — a token needs one row of a `[248320, 2048]` matrix — the output head is
/// the same matrix read in blocks, and each decoder layer's tensors are read, used and
/// released before the next layer starts. Peak residency is one layer (~330 MB in fp32 for
/// this model) plus its activations, which leaves room for the trace on a node that is also
/// running macOS.
///
/// Everything numeric lives in `Ops` and `GatedDeltaNet`, which are bit-identical to
/// `tools/ordered_qwen35.py`; this file is the wiring, and the wiring is what the tests
/// against a real checkpoint check.
public struct Qwen3_5Forward: ForwardPass {
    /// How many vocabulary rows the head is computed in. 8192 rows is 64 MB of fp32, which
    /// is small next to a layer and large enough that the block count stays modest.
    public static let headBlockRows = 8192

    public let spec: IRSpec
    public let config: ModelConfig
    private let source: any WeightSource
    private let namesByBlock: [String: [TensorRole: String]]
    private let embeddingName: String
    private let finalNormName: String
    private let headName: String

    public enum Error: Swift.Error, CustomStringConvertible {
        case missingTensor(block: String, role: TensorRole)
        case noEmbedding
        case emptyPrompt
        case notAMixture

        public var description: String {
            switch self {
            case .missingTensor(let block, let role):
                return "no tensor with role \(role.rawValue) in block \(block)"
            case .noEmbedding: return "the checkpoint has no embedding tensor"
            case .emptyPrompt: return "the prompt must contain at least one token"
            case .notAMixture: return "the configuration describes no mixture of experts"
            }
        }
    }

    /// Open a checkpoint: the spec is built from its inventory by the importer, which is the
    /// only place that knows what a tensor name means.
    public init(snapshot: URL) throws {
        // One factory for both layouts: a 26-shard checkpoint and a single file present the
        // same inventory to the importer, which is what keeps the file format out of L2.
        let opened = try SnapshotWeights.open(snapshot)
        let file = opened.source

        let configData = try Data(contentsOf: snapshot.appendingPathComponent("config.json"))
        let inventory = opened.inventory.inventory
        let source = Provenance(repo: snapshot.lastPathComponent, revision: "local")

        // The family's **importer** is chosen by the checkpoint's own `model_type`, which is
        // what the loader above already does. Two importers share one forward pass because the
        // arithmetic they describe is shared; they differ in the names they map.
        let declared = (try? JSONSerialization.jsonObject(with: configData)) as? [String: Any]
        let modelType = (declared?["model_type"] as? String) ?? "qwen3_5"
        let config: ModelConfig
        let spec: IRSpec
        if modelType == Qwen3_5MoEImporter.family {
            let hfConfig = try JSONDecoder().decode(Qwen3_5MoEImporter.HuggingFaceConfig.self, from: configData)
            config = hfConfig.modelConfig()
            spec = try Qwen3_5MoEImporter.makeSpec(source: source, config: config, inventory: inventory)
        } else {
            let hfConfig = try JSONDecoder().decode(Qwen3_5Importer.HuggingFaceConfig.self, from: configData)
            config = hfConfig.modelConfig()
            spec = try Qwen3_5Importer.makeSpec(source: source, config: config, inventory: inventory)
        }
        try self.init(source: file, config: config, spec: spec)
    }

    /// Open a quantized install. The spec travels inside the artifact, so nothing else is
    /// needed — and the config is reconstructed from it, which is the check that the spec is
    /// genuinely sufficient to run the model (L1).
    public init(install: URL) throws {
        let file = try InstallFile(url: install)
        let config = try Qwen3_5Forward.config(from: file.manifest.spec)
        try self.init(source: file, config: config, spec: file.manifest.spec)
    }

    /// The IR's configuration, in the form the kernels take.
    public static func config(from spec: IRSpec) throws -> ModelConfig { spec.config }

    private init(source: any WeightSource, config: ModelConfig, spec: IRSpec) throws {
        self.source = source
        self.config = config
        self.spec = spec

        var namesByBlock: [String: [TensorRole: String]] = [:]
        for tensor in spec.tensors { namesByBlock[tensor.block, default: [:]][tensor.role] = tensor.name }
        self.namesByBlock = namesByBlock
        guard let embedding = namesByBlock["embed"]?[.tokenEmbedding] else { throw Error.noEmbedding }
        self.embeddingName = embedding
        guard let norm = namesByBlock["final"]?[.finalNorm] else {
            throw Error.missingTensor(block: "final", role: .finalNorm)
        }
        self.finalNormName = norm
        // The family ties its embeddings and ships no head, so the embedding is the head.
        self.headName = namesByBlock["head"]?[.outputHead] ?? embedding
    }

    /// How many expert slices a layer's cache may hold.
    ///
    /// A token asks for its `topK` experts and several of a prompt's positions usually ask for
    /// the same ones, so a slot bank somewhat larger than `topK` serves those repeats without
    /// touching the source. This is a **starting** value: the hit rate it produces is what
    /// `DC-032` measures and what the M1 gate reports, and it should be chosen from that
    /// measurement rather than from this comment. It is a `let` because Swift 6's strict
    /// concurrency rejects mutable global state, and it should become a field of the IR's
    /// policy when there is a measurement to put in it.
    public static let expertSlotsPerLayer = 16

    /// The feed-forward half of a decoder layer. The reference branches inside its decoder
    /// layer between `Qwen3_5MLP` and `Qwen3_5SparseMoeBlock`, and so does this.
    private enum FeedForward {
        case dense(gate: [Float], up: [Float], down: [Float])
        case mixture(MixtureWeights, provider: ExpertSlotCache)
    }

    /// The mixture's geometry, from the IR.
    public func mixtureShape() throws -> MixtureShape {
        guard let experts = config.numExperts, let topK = config.numExpertsPerToken,
              let intermediate = config.moeIntermediateSize
        else { throw Error.notAMixture }
        return MixtureShape(
            hiddenSize: config.hiddenSize, experts: experts, topK: topK, intermediate: intermediate,
            sharedIntermediate: config.sharedExpertIntermediateSize ?? intermediate
        )
    }

    /// The weights a decoder layer needs, loaded and then released.
    private func loadLayer(_ index: Int) throws -> (weights: [TensorRole: [Float]], gdn: GatedDeltaNetWeights?, feedForward: FeedForward) {
        let block = String(format: "layer.%02d", index)
        guard let byRole = namesByBlock[block] else { throw Error.missingTensor(block: block, role: .attnNorm) }
        func load(_ role: TensorRole) throws -> [Float] {
            guard let name = byRole[role] else { throw Error.missingTensor(block: block, role: role) }
            return try source.tensor(named: name)
        }
        var weights: [TensorRole: [Float]] = [.attnNorm: try load(.attnNorm), .mlpNorm: try load(.mlpNorm)]

        // Which feed-forward this layer has is a fact about its roles, not about its family:
        // a block carrying a router is a mixture.
        let feedForward: FeedForward
        if byRole[.routerLogits] != nil {
            guard let gateUpName = byRole[.expertGateUpStack], let downName = byRole[.expertDownStack] else {
                throw Error.missingTensor(block: block, role: .expertGateUpStack)
            }
            // The experts are **not** loaded here. A stacked `[experts, 2·inter, hidden]` for
            // this model is 3.2 GB in fp32, so the layer keeps a provider that reads one
            // expert's row range on demand and a bounded cache in front of it.
            let stacked = StackedExpertProvider(source: source, gateUpName: gateUpName, downName: downName)
            let cache = ExpertSlotCache(upstream: stacked, capacity: Self.expertSlotsPerLayer)
            feedForward = .mixture(
                MixtureWeights(
                    router: try load(.routerLogits),
                    sharedGate: try load(.sharedExpertGate),
                    sharedUp: try load(.sharedExpertUp),
                    sharedDown: try load(.sharedExpertDown),
                    sharedScalarGate: try load(.sharedExpertGateScalar),
                    experts: cache
                ),
                provider: cache
            )
        } else {
            let gate = try load(.mlpGate)
            let up = try load(.mlpUp)
            let down = try load(.mlpDown)
            weights[.mlpGate] = gate
            weights[.mlpUp] = up
            weights[.mlpDown] = down
            feedForward = .dense(gate: gate, up: up, down: down)
        }

        let isFullAttention = byRole[.attnQ] != nil
        if isFullAttention {
            for role in [TensorRole.attnQ, .attnK, .attnV, .attnO, .attnQNorm, .attnKNorm] {
                weights[role] = try load(role)
            }
            return (weights: weights, gdn: nil, feedForward: feedForward)
        }
        var gdnWeights: [TensorRole: [Float]] = [:]
        for role in [TensorRole.linearInQKV, .linearInZ, .linearInA, .linearInB, .linearConv, .linearALog, .linearDTBias, .linearNorm, .linearOut] {
            gdnWeights[role] = try load(role)
        }
        weights.merge(gdnWeights) { current, _ in current }
        let gdn = GatedDeltaNetWeights(
            inQKV: gdnWeights[.linearInQKV]!, inZ: gdnWeights[.linearInZ]!,
            inB: gdnWeights[.linearInB]!, inA: gdnWeights[.linearInA]!,
            conv: gdnWeights[.linearConv]!, aLog: gdnWeights[.linearALog]!,
            dtBias: gdnWeights[.linearDTBias]!, norm: gdnWeights[.linearNorm]!,
            outProj: gdnWeights[.linearOut]!
        )
        return (weights: weights, gdn: gdn, feedForward: feedForward)
    }

    /// The Gated DeltaNet's geometry, assembled from the IR's configuration.
    public func gatedShape() throws -> GatedDeltaNetShape {
        guard let keyDim = config.linearKeyDim,
              let valueHeads = config.linearValueHeads, let valueHeadDim = config.linearValueHeadDim,
              let convKernel = config.linearConvKernelDim
        else { throw Error.missingTensor(block: "config", role: .linearInQKV) }
        // The key head width is the key dimension over the value head count: this family has
        // one key head per value head, and the IR stores the totals.
        // Sixteen keys to thirty-two values in the MoE family: deriving one count from
        // the other was a silent bug for it.
        let keyHeads = config.linearKeyHeads ?? valueHeads
        let keyHeadDim = keyDim / max(keyHeads, 1)
        return GatedDeltaNetShape(
            hiddenSize: config.hiddenSize, keyHeads: keyHeads, valueHeads: valueHeads,
            keyHeadDim: keyHeadDim, valueHeadDim: valueHeadDim, convKernel: convKernel,
            eps: Float(config.rmsNormEps)
        )
    }

    /// The text RoPE tables — `partial_rotary_factor` of `head_dim`, computed in double and
    /// rounded, as the contract states.
    public func ropeTables(positions: [Double]) -> (cos: [Float], sin: [Float]) {
        // Read from the IR, which read it from the checkpoint: 0.25 here, so 64 of 256
        // channels rotate.
        let partial = config.partialRotaryFactor ?? 1.0
        let dim = Int(Double(config.headDim) * partial)
        let base = Float(config.ropeTheta)
        var frequencies: [Float] = []
        frequencies.reserveCapacity(dim / 2)
        for index in stride(from: 0, to: dim, by: 2) {
            frequencies.append(Float(1) / powf(base, Float(index) / Float(dim)))
        }
        var cosines = [Float](repeating: 0, count: positions.count * dim)
        var sines = [Float](repeating: 0, count: positions.count * dim)
        for (row, position) in positions.enumerated() {
            for index in 0..<dim {
                let angle = position * Double(frequencies[index % frequencies.count])
                cosines[row * dim + index] = Float(cos(angle))
                sines[row * dim + index] = Float(sin(angle))
            }
        }
        return (cosines, sines)
    }

    public var vocabularySize: Int { config.vocabSize }

    /// The whole tower, capturing the same tensors the Python contract does.
    public func forward(tokens: [Int]) throws -> [TraceWriter.Tensor] {
        try forwardWithDecisions(tokens: tokens).tensors
    }

    public func forwardWithDecisions(tokens: [Int]) throws -> ForwardResult {
        guard !tokens.isEmpty else { throw Error.emptyPrompt }
        let length = tokens.count
        let hiddenSize = config.hiddenSize
        var captured: [TraceWriter.Tensor] = []
        var discrete: [TraceWriter.Discrete] = []
        var expertMetrics: [ExpertProviderMetrics] = []

        var hidden = [Float](repeating: 0, count: length * hiddenSize)
        for (row, token) in tokens.enumerated() {
            precondition(token >= 0 && token < config.vocabSize, "token \(token) outside the vocabulary")
            let values = try source.rows(named: embeddingName, range: token..<(token + 1))
            for index in 0..<hiddenSize { hidden[row * hiddenSize + index] = values[index] }
        }
        captured.append(TraceWriter.Tensor(name: "embed.out", shape: [length, hiddenSize], values: hidden))

        let tables = ropeTables(positions: (0..<length).map(Double.init))
        var mask = [Float](repeating: 0, count: length * length)
        for row in 0..<length {
            for column in 0..<length {
                mask[row * length + column] = column > row ? -Float.infinity : 0
            }
        }

        for index in 0..<config.numLayers {
            let tag = String(format: "layer.%02d", index)
            captured.append(TraceWriter.Tensor(name: "\(tag).hidden_in", shape: [length, hiddenSize], values: hidden))

            let layer = try loadLayer(index)
            let normed = rmsNorm(
                hidden, weight: layer.weights[.attnNorm]!, rows: length, width: hiddenSize,
                eps: Float(config.rmsNormEps)
            )
            let mixed: [Float]
            if let gdn = layer.gdn {
                mixed = GatedDeltaNet.layer(
                    hidden: normed, weights: gdn, shape: try gatedShape(), batch: 1, length: length
                )
            } else {
                mixed = try attention(
                    normed, weights: layer.weights, length: length, tables: tables, mask: mask
                )
            }
            hidden = add(hidden, mixed)

            let residual = hidden
            let postNormed = rmsNorm(
                hidden, weight: layer.weights[.mlpNorm]!, rows: length, width: hiddenSize,
                eps: Float(config.rmsNormEps)
            )
            switch layer.feedForward {
            case .dense(let gate, let up, let down):
                let projectedGate = Ops.orderedMatmul(
                    x: postNormed, w: gate, rows: length, k: hiddenSize, out: config.intermediateSize
                )
                let projectedUp = Ops.orderedMatmul(
                    x: postNormed, w: up, rows: length, k: hiddenSize, out: config.intermediateSize
                )
                var activated = [Float](repeating: 0, count: length * config.intermediateSize)
                for position in 0..<activated.count {
                    activated[position] = Ops.silu(projectedGate[position]) * projectedUp[position]
                }
                let downProjected = Ops.orderedMatmul(
                    x: activated, w: down, rows: length, k: config.intermediateSize, out: hiddenSize
                )
                hidden = add(residual, downProjected)

            case .mixture(let weights, let provider):
                let shape = try mixtureShape()
                let (routed, indices, _) = try MixtureOfExperts.block(
                    hidden: postNormed, tokens: length, weights: weights, shape: shape
                )
                // Kept so the caller can report a measured hit rate rather than an assurance.
                // M1's gate asks for the number, and the number is not visible from outside.
                expertMetrics.append(provider.metrics)
                hidden = add(residual, routed)
                // The router's decision, recorded as its own kind of thing rather than as a
                // tensor: I3 asserts it apart from any tolerance, and the trace's digest
                // covers it (I1).
                discrete.append(
                    TraceWriter.Discrete(
                        name: "\(tag).router.topk", shape: [length, shape.topK],
                        values: indices.flatMap { $0 }
                    )
                )
            }
            captured.append(TraceWriter.Tensor(name: "\(tag).hidden_out", shape: [length, hiddenSize], values: hidden))
        }

        let finalWeight = try source.tensor(named: finalNormName)
        hidden = rmsNorm(hidden, weight: finalWeight, rows: length, width: hiddenSize, eps: Float(config.rmsNormEps))
        captured.append(TraceWriter.Tensor(name: "final_norm.out", shape: [length, hiddenSize], values: hidden))

        // The head, one block of vocabulary rows at a time: the same matrix as the embedding,
        // and materialising it whole is the thing this design exists to avoid.
        var logits = [Float](repeating: 0, count: length * config.vocabSize)
        var row = 0
        while row < config.vocabSize {
            let upper = min(row + Self.headBlockRows, config.vocabSize)
            let block = try source.rows(named: headName, range: row..<upper)
            let product = Ops.orderedMatmul(
                x: hidden, w: block, rows: length, k: hiddenSize, out: upper - row
            )
            for position in 0..<length {
                for column in 0..<(upper - row) {
                    logits[position * config.vocabSize + row + column] = product[position * (upper - row) + column]
                }
            }
            row = upper
        }
        captured.append(
            TraceWriter.Tensor(name: "logits", shape: [length, config.vocabSize], values: logits)
        )
        return ForwardResult(tensors: captured, discrete: discrete, expertMetrics: expertMetrics)
    }

    private func attention(
        _ hidden: [Float], weights: [TensorRole: [Float]], length: Int,
        tables: (cos: [Float], sin: [Float]), mask: [Float]
    ) throws -> [Float] {
        let heads = config.numAttentionHeads
        let kvHeads = config.numKeyValueHeads
        let headDim = config.headDim
        let eps = Float(config.rmsNormEps)
        let scaling = Float(1) / Float(headDim).squareRoot()
        let rotary = tables.cos.count / max(length, 1)

        let projected = Ops.orderedMatmul(
            x: hidden, w: weights[.attnQ]!, rows: length, k: config.hiddenSize, out: heads * headDim * 2
        )
        var query = [Float](repeating: 0, count: length * heads * headDim)
        var gate = [Float](repeating: 0, count: length * heads * headDim)
        for position in 0..<length {
            for head in 0..<heads {
                for index in 0..<headDim {
                    let source = (position * heads + head) * headDim * 2
                    query[(position * heads + head) * headDim + index] = projected[source + index]
                    gate[(position * heads + head) * headDim + index] = projected[source + headDim + index]
                }
            }
        }
        var key = Ops.orderedMatmul(
            x: hidden, w: weights[.attnK]!, rows: length, k: config.hiddenSize, out: kvHeads * headDim
        )
        let value = Ops.orderedMatmul(
            x: hidden, w: weights[.attnV]!, rows: length, k: config.hiddenSize, out: kvHeads * headDim
        )

        query = rmsNorm(query, weight: weights[.attnQNorm]!, rows: length * heads, width: headDim, eps: eps)
        key = rmsNorm(key, weight: weights[.attnKNorm]!, rows: length * kvHeads, width: headDim, eps: eps)

        // Partial rotation: the first `rotary` channels rotate with rotate-half inside
        // themselves and the rest pass through.
        query = applyPartialRope(query, tables: tables, length: length, heads: heads, headDim: headDim, rotary: rotary)
        key = applyPartialRope(key, tables: tables, length: length, heads: kvHeads, headDim: headDim, rotary: rotary)

        let groups = heads / kvHeads
        var mixer = [Float](repeating: 0, count: length * heads * headDim)
        for head in 0..<heads {
            let kvHead = head / groups
            let queryHead = sliceHead(query, length: length, heads: heads, headDim: headDim, head: head)
            let keyHead = sliceHead(key, length: length, heads: kvHeads, headDim: headDim, head: kvHead)
            var scores = Ops.orderedMatmul(x: queryHead, w: keyHead, rows: length, k: headDim, out: length)
            for index in 0..<scores.count { scores[index] = scores[index] * scaling + mask[index] }
            let attention = Ops.softmax(x: scores, rows: length, width: length)
            let valueHead = sliceHead(value, length: length, heads: kvHeads, headDim: headDim, head: kvHead)
            var transposed = [Float](repeating: 0, count: headDim * length)
            for position in 0..<length {
                for index in 0..<headDim { transposed[index * length + position] = valueHead[position * headDim + index] }
            }
            let mixed = Ops.orderedMatmul(x: attention, w: transposed, rows: length, k: length, out: headDim)
            for position in 0..<length {
                for index in 0..<headDim {
                    mixer[(position * heads + head) * headDim + index] = mixed[position * headDim + index]
                }
            }
        }

        var gated = [Float](repeating: 0, count: mixer.count)
        for index in 0..<mixer.count { gated[index] = mixer[index] * Ops.sigmoid(gate[index]) }
        return Ops.orderedMatmul(
            x: gated, w: weights[.attnO]!, rows: length, k: heads * headDim, out: config.hiddenSize
        )
    }

    private func applyPartialRope(
        _ x: [Float], tables: (cos: [Float], sin: [Float]), length: Int, heads: Int, headDim: Int, rotary: Int
    ) -> [Float] {
        var result = x
        let half = rotary / 2
        for position in 0..<length {
            for head in 0..<heads {
                let base = (position * heads + head) * headDim
                let table = position * rotary
                for index in 0..<rotary {
                    let value = x[base + index]
                    let paired = index < half ? x[base + index + half] : x[base + index - half]
                    let rotated = index < half ? -paired : paired
                    result[base + index] = value * tables.cos[table + index] + rotated * tables.sin[table + index]
                }
            }
        }
        return result
    }

    private func sliceHead(_ values: [Float], length: Int, heads: Int, headDim: Int, head: Int) -> [Float] {
        var slice = [Float](repeating: 0, count: length * headDim)
        for row in 0..<length {
            let source = (row * heads + head) * headDim
            for index in 0..<headDim { slice[row * headDim + index] = values[source + index] }
        }
        return slice
    }

    private func rmsNorm(_ x: [Float], weight: [Float], rows: Int, width: Int, eps: Float) -> [Float] {
        var result = [Float](repeating: 0, count: rows * width)
        for row in 0..<rows {
            let base = row * width
            var squares: [Float] = []
            squares.reserveCapacity(width)
            for index in 0..<width {
                let value = x[base + index]
                squares.append(value * value)
            }
            let variance = Ops.orderedSum(squares) / Float(width)
            let inverse = Float(1) / (variance + eps).squareRoot()
            // This family's backbone norm is weight-OFFSET: (1 + weight), not weight.
            for index in 0..<width {
                result[base + index] = (Float(1) + weight[index]) * (x[base + index] * inverse)
            }
        }
        return result
    }

    private func add(_ left: [Float], _ right: [Float]) -> [Float] {
        var result = [Float](repeating: 0, count: left.count)
        for index in 0..<left.count { result[index] = left[index] + right[index] }
        return result
    }
}
