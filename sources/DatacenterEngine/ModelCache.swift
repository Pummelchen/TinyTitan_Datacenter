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
    /// The first error any head block raised, so a fan-out cannot swallow a failure.
    private final class HeadBlockFailure: @unchecked Sendable {
        private let lock = NSLock()
        // `Swift.Error`, spelled out: inside `extension Qwen3_5Forward` the bare name resolves to the model's
        // own nested `Error` type, which is a different thing and does not accept the reader's failures.
        private var stored: (any Swift.Error)?
        func record(_ error: any Swift.Error) {
            lock.lock(); if stored == nil { stored = error }; lock.unlock()
        }
        var first: (any Swift.Error)? { lock.lock(); defer { lock.unlock() }; return stored }
    }

    /// The LM head's logits for one row, a vocabulary block at a time — the **safe half of `DC-122`**.
    ///
    /// **Why this may be threaded where the general matmul could not be.** `DC-122` threaded
    /// `Ops.orderedMatmulVectorized` and it **moved bits**: bounding each thread's work by its own `last`
    /// regrouped the four-wide columns and moved the scalar tail, and a comparison that had held since `D9`
    /// caught it — so that function is the original single-threaded body and the attempt is on the record
    /// rather than shipped. The unit here is a **whole vocabulary block of rows**. Every block runs the *same*
    /// `Ops.orderedMatmul` call, over the same ascending `k`, with the same internal grouping; the only thing
    /// the decomposition changes is **which thread** runs it, which is exactly what `D63` allows. Nothing is
    /// regrouped because no block is ever split, and that is a structural argument rather than a measurement —
    /// which is why `HeadLogitsTests` walks a grid of block sizes and compares bit patterns.
    ///
    /// The reads happen inside the blocks and are therefore concurrent. That is safe for the same reason the
    /// expert fan-out of `D101` is: the reader's counters are write-locked and `pread` carries its own offset.
    ///
    /// `rows` is the vocabulary slice this node owns — the whole vocabulary when there is no shard — and the
    /// returned array is always `vocabulary` wide, so a sharded gather still receives the full logits array.
    static func headLogits(
        x: [Float], source: any WeightSource, name: String, rows: Range<Int>, vocabulary: Int,
        hiddenSize: Int, blockRows: Int, threads: Int
    ) throws -> [Float] {
        var logits = [Float](repeating: 0, count: vocabulary)
        let blocks = rows.isEmpty ? 0 : (rows.count + blockRows - 1) / blockRows
        guard blocks > 0 else { return logits }
        let failure = HeadBlockFailure()
        nonisolated(unsafe) let upstream = source
        // **The stored form is fetched exactly once, before the fan-out** (`D109`). Asking for it per block
        // looked harmless — `storedRows` slices a cached payload — but the cache is only populated by the
        // *first* read, so 31 concurrent blocks each missed and each read the whole 1.017 GB head: 31 GB of
        // transient allocations, which the node answered with six gigabytes of swap and the disk watchdog.
        // One fetch, then cheap slices. `storedRows(named:range:)` over the whole slice is the same call a
        // block would make, so a source without a stored form simply falls through to the old path.
        // **The device copy is taken once** (`D115`). `resident` is `(key, stored)` when the head is mapped on
        // the device; `stored` is only non-nil on the call that mapped it, so the fallback below stays
        // available for a source with no stored form.
        // **The key names the window, not the tensor** (`D115`). A sharded node holds only its own slice of
        // the vocabulary, so keying on the name alone let node 2 reuse node 1's mapping and produce node 1's
        // tokens — which is exactly what `ShardedGenerateTests` failed on the first time this was written. A
        // mapped weight is a (tensor, row window) pair and the key has to say so.
        let headKey0 = "\(name)#\(rows.lowerBound)..<\(rows.upperBound)"
        let resident: (key: String, stored: StoredRows?)? = try {
            guard MetalBf16Matmul.enabled, MetalBf16Matmul.residentEnabled, MetalBf16Matmul.isAvailable
            else { return nil }
            if MetalBf16Matmul.isResident(key: headKey0) { return (headKey0, nil) }
            guard let whole = try upstream.storedRows(named: name, range: rows),
                whole.dtype == "bf16", whole.width == hiddenSize
            else { return nil }
            try MetalBf16Matmul.upload(key: headKey0, w: whole.data)
            return (headKey0, whole)
        }()
        let storedHead = resident?.stored
        let headKey = resident?.key
        let stride = hiddenSize * 2
        logits.withUnsafeMutableBufferPointer { buffer in
            nonisolated(unsafe) let target = buffer.baseAddress!
            let body: @Sendable (Int) -> Void = { block in
                let lower = rows.lowerBound + block * blockRows
                let upper = min(lower + blockRows, rows.upperBound)
                do {
                    if let headKey {
                        // The offset is into the mapped weight, so the slice arithmetic — `Data` slices keep
                        // the parent's indices — never happens on this path.
                        let product = try MetalBf16Matmul.matmulResident(
                            x: x, key: headKey, offset: block * blockRows * stride,
                            rows: 1, k: hiddenSize, out: upper - lower
                        )
                        for column in 0..<(upper - lower) { target[lower + column] = product[column] }
                        return
                    }
                    if let storedHead {
                        // The slice is a view of the cached payload: no copy, no decode. **`Data` slices keep
                        // the parent's indices**, so the window is offset from `startIndex` rather than from
                        // zero — indexing a sharded vocabulary window from zero is out of bounds, which is
                        // exactly how `ShardedGenerateTests` found this.
                        let base = storedHead.data.startIndex
                        let start = base + block * blockRows * stride
                        let end = min(start + (upper - lower) * stride, storedHead.data.endIndex)
                        let product = try MetalBf16Matmul.matmul(
                            x: x, w: storedHead.data[start..<end], rows: 1, k: hiddenSize, out: upper - lower
                        )
                        for column in 0..<(upper - lower) { target[lower + column] = product[column] }
                        return
                    }
                    let weights = try upstream.rows(named: name, range: lower..<upper)
                    // **The serial body, not the chooser** (`D109`). `Ops.orderedMatmul` fans out across
                    // output columns when there is enough work, and this loop is *already* a fan-out over
                    // blocks — so the public entry point nested one `concurrentPerform` inside another and
                    // put 31 × 8 tasks on 8 cores. `Ops`' own comment says a caller that is already parallel
                    // at a higher level calls the serial body; this one did not. The two are bit-identical by
                    // `D104`'s aligned-chunk rule, which is why the swap cannot move a digest.
                    let product = Ops.orderedMatmulVectorized(
                        x: x, w: weights, rows: 1, k: hiddenSize, out: upper - lower
                    )
                    for column in 0..<(upper - lower) { target[lower + column] = product[column] }
                } catch {
                    failure.record(error)
                }
            }
            if threads > 1 && blocks > 1 {
                DispatchQueue.concurrentPerform(iterations: blocks, execute: body)
            } else {
                for block in 0..<blocks { body(block) }
            }
        }
        if let error = failure.first { throw error }
        return logits
    }

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
            // `DC-121`: the reads this layer is about to want are the ones it wanted on the **last token**, and
            // the routing is strongly correlated between consecutive tokens — so they are issued here, on a
            // background thread, and overlap the attention and the router below. A wrong guess costs exactly the
            // read the loop would have made anyway; a right one takes the read off the critical path, which is
            // where `D101` left it. It has to be issued *before* the attention: placed after it, the background
            // work has nothing left to hide behind.
            if case .mixture(_, let provider) = layer.feedForward { provider.prefetchPredicted() }
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
                    normed, weights: layer.weights, packed: layer.packedWeights, tables: tables,
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
        let slice = shard.map {
            VocabSlice(node: $0.node, nodes: $0.ownership.nodes, vocabSize: config.vocabSize)
        }
        let rows = slice?.range ?? 0..<config.vocabSize
        var logits = try Self.headLogits(
            x: hidden, source: source, name: headName, rows: rows, vocabulary: config.vocabSize,
            hiddenSize: hiddenSize, blockRows: Self.headBlockRows, threads: DecodeThreads.count
        )
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
            layerCache: layerCacheBudgetBytes > 0 ? layerWeights.metrics : nil,
            experts: expertBankMetrics,
            expertBudgetBytes: expertBankBudgetBytes
        )
    }

    func logitsOf(_ tensors: [TraceWriter.Tensor]) throws -> [Float] {
        guard let logits = tensors.first(where: { $0.name == "logits" }) else { throw Error.emptyPrompt }
        return logits.values
    }
}
