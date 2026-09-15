import DatacenterIR
import Foundation

/// The `qwen3` dense forward, driven by the IR.
///
/// Dispatch is on **role**, never on tensor name (L1): the layer's weights are looked up
/// by `(block, role)`, so the same code runs any checkpoint whose importer maps its names
/// onto the same roles. The arithmetic is `Ops`, which is bit-identical to
/// `tools/ordered_reference.py` — that is the whole point of this file being so plain.
public struct Qwen3Forward: ForwardPass {
    public struct LayerWeights {
        let attnNorm: [Float]
        let q: [Float]
        let k: [Float]
        let v: [Float]
        let o: [Float]
        let qNorm: [Float]
        let kNorm: [Float]
        let mlpNorm: [Float]
        let gate: [Float]
        let up: [Float]
        let down: [Float]
    }

    public let spec: IRSpec
    public let config: ModelConfig
    private let layers: [LayerWeights]
    private let embedding: [Float]
    private let finalNorm: [Float]
    private let head: [Float]

    public enum Error: Swift.Error, CustomStringConvertible {
        case missingTensor(block: String, role: TensorRole)
        case unexpectedShape(String, [Int], [Int])
        case emptyPrompt

        public var description: String {
            switch self {
            case .missingTensor(let block, let role):
                return "no tensor with role \(role.rawValue) in block \(block)"
            case .unexpectedShape(let name, let got, let want):
                return "\(name): shape \(got) does not match the IR contract \(want)"
            case .emptyPrompt: return "the prompt must contain at least one token"
            }
        }
    }

    /// Build from a checkpoint, using the IR importer to decide what every tensor is.
    public init(snapshot: URL) throws {
        let weightsURL = snapshot.appendingPathComponent("model.safetensors")
        let file = try SafetensorsFile(url: weightsURL)
        let configData = try Data(contentsOf: snapshot.appendingPathComponent("config.json"))
        let hfConfig = try JSONDecoder().decode(Qwen3Importer.HuggingFaceConfig.self, from: configData)
        let config = hfConfig.modelConfig()

        let source = Provenance(repo: snapshot.lastPathComponent, revision: "local")
        let inventory = file.names.map { (name: $0, shape: file.tensors[$0]!.shape) }
        let spec = try Qwen3Importer.makeSpec(source: source, config: config, inventory: inventory)

        // block id → role → tensor name, built once from the spec.
        var byBlock: [String: [TensorRole: String]] = [:]
        for tensor in spec.tensors {
            byBlock[tensor.block, default: [:]][tensor.role] = tensor.name
        }
        func load(_ block: String, _ role: TensorRole) throws -> [Float] {
            guard let name = byBlock[block]?[role] else { throw Error.missingTensor(block: block, role: role) }
            return try file.float32(name)
        }

        var layers: [LayerWeights] = []
        layers.reserveCapacity(config.numLayers)
        for index in 0..<config.numLayers {
            let block = String(format: "layer.%02d", index)
            layers.append(
                LayerWeights(
                    attnNorm: try load(block, .attnNorm), q: try load(block, .attnQ), k: try load(block, .attnK),
                    v: try load(block, .attnV), o: try load(block, .attnO), qNorm: try load(block, .attnQNorm),
                    kNorm: try load(block, .attnKNorm), mlpNorm: try load(block, .mlpNorm),
                    gate: try load(block, .mlpGate), up: try load(block, .mlpUp), down: try load(block, .mlpDown)
                )
            )
        }

        let embedding = try load("embed", .tokenEmbedding)
        let finalNorm = try load("final", .finalNorm)
        // A separate head is used when the checkpoint ships one, even if the config says
        // the weights are tied: Qwen3-0.6B does exactly that, and the file wins.
        let head = (try? load("head", .outputHead)) ?? embedding

        self.spec = spec
        self.config = config
        self.layers = layers
        self.embedding = embedding
        self.finalNorm = finalNorm
        self.head = head
    }

    public var vocabularySize: Int { config.vocabSize }

    /// Run the forward, capturing the same tensors the reference does, in the same order.
    public func forward(tokens: [Int]) throws -> [TraceWriter.Tensor] {
        guard !tokens.isEmpty else { throw Error.emptyPrompt }
        let tokenCount = tokens.count
        let hidden = config.hiddenSize
        let heads = config.numAttentionHeads
        let kvHeads = config.numKeyValueHeads
        let headDim = config.headDim
        let groups = heads / kvHeads
        let scale = Float(pow(Double(headDim), -0.5))

        var captured: [TraceWriter.Tensor] = []
        func capture(_ name: String, _ shape: [Int], _ values: [Float]) {
            captured.append(TraceWriter.Tensor(name: name, shape: shape, values: values))
        }

        // Embedding rows, gathered in prompt order.
        var state = [Float](repeating: 0, count: tokenCount * hidden)
        for (row, token) in tokens.enumerated() {
            precondition(token >= 0 && token < config.vocabSize, "token \(token) outside the vocabulary")
            let base = token * hidden
            for index in 0..<hidden { state[row * hidden + index] = embedding[base + index] }
        }
        capture("embed.out", [tokenCount, hidden], state)

        let tables = Ops.ropeTables(
            headDim: headDim, positions: (0..<tokenCount).map(Double.init), theta: config.ropeTheta
        )

        for layerIndex in 0..<config.numLayers {
            let layer = layers[layerIndex]
            let tag = String(format: "layer.%02d", layerIndex)
            let hiddenIn = state
            capture("\(tag).hidden_in", [tokenCount, hidden], hiddenIn)

            let normed = Ops.rmsNorm(
                x: state, weight: layer.attnNorm, rows: tokenCount, width: hidden, eps: Float(config.rmsNormEps)
            )

            var query = Ops.orderedMatmul(
                x: normed, w: layer.q, rows: tokenCount, k: hidden, out: heads * headDim
            )
            let key = Ops.orderedMatmul(
                x: normed, w: layer.k, rows: tokenCount, k: hidden, out: kvHeads * headDim
            )
            let value = Ops.orderedMatmul(
                x: normed, w: layer.v, rows: tokenCount, k: hidden, out: kvHeads * headDim
            )

            // QK-norm is per head, over head_dim, before RoPE — S4, not an implementation
            // choice. Values are not normalised.
            query = Ops.rmsNorm(
                x: query, weight: layer.qNorm, rows: tokenCount * heads, width: headDim,
                eps: Float(config.rmsNormEps)
            )
            var normalizedKey = Ops.rmsNorm(
                x: key, weight: layer.kNorm, rows: tokenCount * kvHeads, width: headDim,
                eps: Float(config.rmsNormEps)
            )

            query = Ops.applyRope(
                x: query, cos: tables.cos, sin: tables.sin, tokens: tokenCount, heads: heads, headDim: headDim
            )
            normalizedKey = Ops.applyRope(
                x: normalizedKey, cos: tables.cos, sin: tables.sin, tokens: tokenCount, heads: kvHeads,
                headDim: headDim
            )

            var mixer = [Float](repeating: 0, count: tokenCount * heads * headDim)
            for head in 0..<heads {
                let kvHead = head / groups
                var scores = Ops.orderedMatmul(
                    x: Self.sliceHeads(query, tokenCount: tokenCount, heads: heads, headDim: headDim, head: head),
                    w: Self.sliceHeads(
                        normalizedKey, tokenCount: tokenCount, heads: kvHeads, headDim: headDim, head: kvHead
                    ),
                    rows: tokenCount, k: headDim, out: tokenCount
                )
                for row in 0..<tokenCount {
                    for column in 0..<tokenCount {
                        // Causal mask, then the scale — the contract's order, applied to
                        // the same values in the same sequence.
                        scores[row * tokenCount + column] =
                            (column <= row) ? scores[row * tokenCount + column] * scale : -Float.infinity
                    }
                }
                let weights = Ops.softmax(x: scores, rows: tokenCount, width: tokenCount)
                // weights[t, s] · v[s, d]: the value head is transposed, as in the contract.
                let values = Self.sliceHeads(
                    value, tokenCount: tokenCount, heads: kvHeads, headDim: headDim, head: kvHead
                )
                var transposed = [Float](repeating: 0, count: headDim * tokenCount)
                for row in 0..<tokenCount {
                    for index in 0..<headDim { transposed[index * tokenCount + row] = values[row * headDim + index] }
                }
                let mixed = Ops.orderedMatmul(
                    x: weights, w: transposed, rows: tokenCount, k: tokenCount, out: headDim
                )
                for row in 0..<tokenCount {
                    for index in 0..<headDim { mixer[(row * heads + head) * headDim + index] = mixed[row * headDim + index] }
                }
            }

            let attention = Ops.orderedMatmul(
                x: mixer, w: layer.o, rows: tokenCount, k: heads * headDim, out: hidden
            )
            capture("\(tag).mixer_out", [tokenCount, hidden], attention)
            state = Self.add(hiddenIn, attention)

            let postNormed = Ops.rmsNorm(
                x: state, weight: layer.mlpNorm, rows: tokenCount, width: hidden, eps: Float(config.rmsNormEps)
            )
            let gated = Ops.orderedMatmul(
                x: postNormed, w: layer.gate, rows: tokenCount, k: hidden, out: config.intermediateSize
            )
            let up = Ops.orderedMatmul(
                x: postNormed, w: layer.up, rows: tokenCount, k: hidden, out: config.intermediateSize
            )
            var activated = [Float](repeating: 0, count: tokenCount * config.intermediateSize)
            for index in 0..<activated.count { activated[index] = Ops.silu(gated[index]) * up[index] }
            let mlp = Ops.orderedMatmul(
                x: activated, w: layer.down, rows: tokenCount, k: config.intermediateSize, out: hidden
            )
            capture("\(tag).mlp_out", [tokenCount, hidden], mlp)
            state = Self.add(state, mlp)
        }

        let final = Ops.rmsNorm(
            x: state, weight: finalNorm, rows: tokenCount, width: hidden, eps: Float(config.rmsNormEps)
        )
        capture("final_norm.out", [tokenCount, hidden], final)
        let logits = Ops.orderedMatmul(
            x: final, w: head, rows: tokenCount, k: hidden, out: config.vocabSize
        )
        capture("logits", [tokenCount, config.vocabSize], logits)
        return captured
    }

    /// Rows `row * heads + head` for every row, flattened back to `[tokens, headDim]`.
    private static func sliceHeads(
        _ values: [Float], tokenCount: Int, heads: Int, headDim: Int, head: Int
    ) -> [Float] {
        var slice = [Float](repeating: 0, count: tokenCount * headDim)
        for row in 0..<tokenCount {
            let source = (row * heads + head) * headDim
            let destination = row * headDim
            for index in 0..<headDim { slice[destination + index] = values[source + index] }
        }
        return slice
    }

    private static func add(_ left: [Float], _ right: [Float]) -> [Float] {
        var result = [Float](repeating: 0, count: left.count)
        for index in 0..<left.count { result[index] = left[index] + right[index] }
        return result
    }
}
