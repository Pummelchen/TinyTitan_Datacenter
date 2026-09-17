import Foundation

import DatacenterIR

/// The per-layer state a decoder keeps between tokens.
///
/// Two kinds, because the model has two kinds of token mixer, and they cache different things:
///
/// - a **Gated DeltaNet** layer keeps a `GatedDeltaNet.State` — the convolution's window and the
///   recurrent state. What it does *not* keep is the past's token mixers, because the recurrence
///   has already absorbed them.
/// - a **full-attention** layer keeps its keys and values, because attention needs them all.
///
/// `D8` applies to the first and not the second: the recurrence evaluated a step at a time agrees
/// with the chunked rule to about 1e-7 rather than to the bit, while cached attention over an
/// append-only key list is the *same* sum over the same values in the same order, so it is
/// bit-identical. The tests assert each at the strength it holds.
public enum LayerCache {
    case gdn(GatedDeltaNet.State)
    case attention(keys: [Float], values: [Float], length: Int)

    /// How many positions this cache holds, for the attention path's softmax width.
    public var cachedLength: Int {
        switch self {
        case .gdn: return 0
        case .attention(_, _, let length): return length
        }
    }
}

/// A whole model's decode state, one entry per layer.
public final class ModelCache {
    public var layers: [LayerCache]

    public init(layers: [LayerCache]) {
        self.layers = layers
    }
}

extension Qwen3_5Forward {
    /// Build the decode state for a prompt.
    ///
    /// The prompt's **outputs** come from the verified sequence path — that is what M1's gate
    /// compares to the contract, and re-deriving them here would be a second implementation of
    /// the thing the gate checks. The **state** is built by replaying the same tokens through the
    /// decode path, which is what the cache will continue from; for the Gated DeltaNet that replay
    /// differs from the chunked path by about 1e-7, by `D8`, and for attention it does not differ
    /// at all.
    public func prepareCache(tokens: [Int]) throws -> (cache: ModelCache, logits: [Float]) {
        var layers: [LayerCache] = []
        for index in 0..<config.numLayers {
            let block = String(format: "layer.%02d", index)
            guard let byRole = namesByBlock[block] else { throw Error.missingTensor(block: block, role: .attnNorm) }
            if byRole[.attnQ] != nil {
                layers.append(.attention(keys: [], values: [], length: 0))
            } else {
                let shape = try gatedShape()
                layers.append(
                    .gdn(
                        GatedDeltaNet.State(
                            batch: 1, convDim: shape.convDim, kernel: shape.convKernel,
                            valueHeads: shape.valueHeads, keyHeadDim: shape.keyHeadDim,
                            valueHeadDim: shape.valueHeadDim
                        )
                    )
                )
            }
        }
        let cache = ModelCache(layers: layers)
        // Every prompt token is consumed exactly once, and the logits of the **last** one are
        // returned: they are what chooses the first generated token. Returning the cache without
        // them invites the caller to decode the last token again, which advances the state twice
        // and produces plausible text from a state that has seen one token too many.
        var logits: [Float] = []
        for token in tokens {
            // `profiler: nil` on purpose: the prompt's forward is not one of the measured steps, and
            // `Generation.secondsPerStep` does not include it either, so the profile must not either.
            let result = try decodeOne(token: token, cache: cache, capturing: false, profiler: nil)
            logits = try logitsOf(result.tensors)
        }
        return (cache, logits)
    }

    /// One token through the whole model against `cache`, which it updates in place.
    ///
    /// `profiler` is the same instrument the sequence path uses, with the **same phase names**, so a cached
    /// step and a full-sequence forward can be read side by side. It was missing here until `D88`: the marks
    /// inside `MixtureOfExperts` existed, but this path never passed a profiler down to them, so the step the
    /// throughput gate actually measures was the one step with no breakdown — and the mixture is most of it.
    public func decodeOne(
        token: Int, cache: ModelCache, capturing: Bool,
        profiler: Profiler? = Qwen3_5Forward.requestedProfiler,
        layerCache: LayerWeightCache? = nil
    ) throws -> (tensors: [TraceWriter.Tensor], discrete: [TraceWriter.Discrete]) {
        let hiddenSize = config.hiddenSize
        precondition(token >= 0 && token < config.vocabSize, "token \(token) outside the vocabulary")
        var captured: [TraceWriter.Tensor] = []
        var discrete: [TraceWriter.Discrete] = []

        var hidden = try source.rows(named: embeddingName, range: token..<(token + 1))
        hidden = Array(hidden[0..<hiddenSize])
        profiler?.mark("embed")

        // The position this token occupies, which is what RoPE needs. With a cache the position is
        // whatever has been seen already; without one it is the token's index in the sequence.
        for index in 0..<config.numLayers {
            let tag = String(format: "layer.%02d", index)
            let layer = try loadLayer(index, cache: layerCache)
            profiler?.mark("load")
            let tables: (cos: [Float], sin: [Float])
            switch cache.layers[index] {
            case .attention(_, _, let length):
                tables = ropeTables(positions: [Double(length)])
            case .gdn:
                tables = ropeTables(positions: [0])
            }

            var normed = rmsNorm(
                hidden, weight: layer.weights[.attnNorm]!, rows: 1, width: hiddenSize,
                eps: Float(config.rmsNormEps)
            )
            profiler?.mark("attn.norm")
            var mixed: [Float]
            switch cache.layers[index] {
            case .gdn(let state):
                mixed = GatedDeltaNet.decodeStep(
                    hidden: normed, weights: layer.gdn!, shape: try gatedShape(), state: state
                )
            case .attention(var keys, var values, let length):
                mixed = try attentionStep(
                    normed, weights: layer.weights, tables: tables,
                    keys: &keys, values: &values, cachedLength: length
                )
                cache.layers[index] = .attention(keys: keys, values: values, length: length + 1)
            }
            profiler?.mark("attn.core")
            hidden = add(hidden, mixed)
            profiler?.mark("attn.add")

            let residual = hidden
            normed = rmsNorm(
                hidden, weight: layer.weights[.mlpNorm]!, rows: 1, width: hiddenSize,
                eps: Float(config.rmsNormEps)
            )
            profiler?.mark("ff.norm")
            switch layer.feedForward {
            case .dense(gate: let gate, up: let up, down: let down):
                let projectedGate = Ops.orderedMatmul(
                    x: normed, w: gate, rows: 1, k: hiddenSize, out: config.intermediateSize
                )
                let projectedUp = Ops.orderedMatmul(
                    x: normed, w: up, rows: 1, k: hiddenSize, out: config.intermediateSize
                )
                var activated = [Float](repeating: 0, count: config.intermediateSize)
                for position in 0..<activated.count {
                    activated[position] = Ops.silu(projectedGate[position]) * projectedUp[position]
                }
                hidden = add(
                    residual,
                    Ops.orderedMatmul(
                        x: activated, w: down, rows: 1, k: config.intermediateSize, out: hiddenSize
                    )
                )
            case .mixture(let weights, _):
                let shape = try mixtureShape()
                // The same entry point the sequence path uses, so a sharded decode all-reduces here too.
                let (routed, indices) = try mixtureOutput(
                    hidden: normed, tokens: 1, weights: weights, shape: shape, profiler: profiler
                )
                hidden = add(residual, routed)
                if capturing {
                    discrete.append(
                        TraceWriter.Discrete(
                            name: "\(tag).router.topk", shape: [1, shape.topK], values: indices[0]
                        )
                    )
                }
            }
            profiler?.mark("ff")
            if capturing {
                captured.append(
                    TraceWriter.Tensor(name: "\(tag).hidden_out", shape: [1, hiddenSize], values: hidden)
                )
            }
            profiler?.mark("trace.copy")
        }

        let finalWeight = try source.tensor(named: finalNormName)
        hidden = rmsNorm(hidden, weight: finalWeight, rows: 1, width: hiddenSize, eps: Float(config.rmsNormEps))
        profiler?.mark("final_norm")

        // Only the last row's logits are needed to choose the next token, so the head is read and
        // multiplied one vocabulary block at a time for one row rather than for the sequence.
        //
        // In a sharded run each node computes only its **vocabulary slice** and the slices are then exchanged,
        // so every node ends with the same array: the head is 1.05 s/step of replicated work (`D88`), and
        // `D93` splits it without moving a single value — the dot product for a row does not depend on which
        // other rows the same node computed.
        var logits = [Float](repeating: 0, count: config.vocabSize)
        let slice = shard.map {
            VocabSlice(node: $0.node, nodes: $0.ownership.nodes, vocabSize: config.vocabSize)
        }
        let rows = slice?.range ?? 0..<config.vocabSize
        var row = rows.lowerBound
        while row < rows.upperBound {
            let upper = min(row + Self.headBlockRows, rows.upperBound)
            let block = try source.rows(named: headName, range: row..<upper)
            let product = Ops.orderedMatmul(x: hidden, w: block, rows: 1, k: hiddenSize, out: upper - row)
            for column in 0..<(upper - row) { logits[row + column] = product[column] }
            row = upper
        }
        profiler?.mark("head")
        if let shard, let slice {
            try shard.gatherHeadSlice(into: &logits, slice: slice)
        }
        captured.append(TraceWriter.Tensor(name: "logits", shape: [1, config.vocabSize], values: logits))
        return (captured, discrete)
    }

    /// Greedy generation with a cache: the prompt once, then one position per token.
    ///
    /// `profiling` defaults to what `SHARD_PROFILE=1` asked for, and is a parameter so a test can turn it on
    /// without the environment variable, which is read once at load time.
    public func generateCached(
        prompt: [Int], maxNewTokens: Int, profiling: Bool = Qwen3_5Forward.profilingEnabled,
        layerCacheBudgetBytes: Int = Qwen3_5Forward.layerCacheBudget
    ) throws -> Generation {
        let (cache, promptLogits) = try prepareCache(tokens: prompt)
        // One cache for the whole generation: a decode revisits every layer on every token, so holding a layer
        // pays on every step that follows (`D88`).
        let layerWeights = LayerWeightCache(
            budgetBytes: layerCacheBudgetBytes, layerCount: config.numLayers
        )
        var generated: [Int] = []
        var seconds: [Double] = []
        var margins: [Float] = []
        var reports: [ProfileReport] = []
        var logits = promptLogits
        var result: (tensors: [TraceWriter.Tensor], discrete: [TraceWriter.Discrete]) = ([], [])

        for _ in 0..<maxNewTokens {
            let next = Greedy.argmax(logits, offset: 0, width: vocabularySize)
            generated.append(next)
            margins.append(Greedy.margin(logits, offset: 0, width: vocabularySize))
            // A fresh profiler per step, so a phase's seconds are that step's rather than a running total
            // with the gap between steps folded into whichever phase happened to come first; the reports are
            // added afterwards.
            let profiler = profiling ? Profiler() : nil
            let started = Date()
            result = try decodeOne(
                token: next, cache: cache, capturing: false, profiler: profiler, layerCache: layerWeights
            )
            seconds.append(Date().timeIntervalSince(started))
            if let report = profiler?.report(layers: config.numLayers) { reports.append(report) }
            logits = try logitsOf(result.tensors)
        }
        return Generation(
            prompt: prompt, generated: generated, secondsPerStep: seconds, captured: result.tensors,
            margins: margins, profile: ProfileReport.combined(reports),
            layerCache: layerCacheBudgetBytes > 0 ? layerWeights.metrics : nil
        )
    }

    func logitsOf(_ tensors: [TraceWriter.Tensor]) throws -> [Float] {
        guard let logits = tensors.first(where: { $0.name == "logits" }) else { throw Error.emptyPrompt }
        return logits.values
    }
}
