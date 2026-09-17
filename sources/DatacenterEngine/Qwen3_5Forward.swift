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
    /// Set when this node is one member of a sharded cluster. Nil is a single-node run, and it is the
    /// only difference between the two: the arithmetic below is the same code either way.
    let shard: ShardExecution?
    let source: any WeightSource
    let namesByBlock: [String: [TensorRole: String]]
    let embeddingName: String
    let finalNormName: String
    let headName: String

    /// `ForwardPass`: what this model's install has actually read, counted by the reader — the
    /// figure M1's gate needs, rather than one derived from element counts.
    public var sourceBytesRead: Int { source.bytesReadFromSource }

    /// `ForwardPass`: what the reading, verifying and unpacking cost, in seconds.
    public var sourceTiming: SourceTiming { source.sourceTiming }
    public var payloadCacheMetrics: PayloadCacheMetrics { source.payloadCacheMetrics }
    public var payloadRequestCounts: [(name: String, count: Int)] { source.payloadRequestCounts }

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
    public init(install: URL, shard: ShardExecution? = nil) throws {
        let file = try InstallFile(url: install)
        let config = try Qwen3_5Forward.config(from: file.manifest.spec)
        try self.init(source: file, config: config, spec: file.manifest.spec, shard: shard)
    }

    /// The IR's configuration, in the form the kernels take.
    public static func config(from spec: IRSpec) throws -> ModelConfig { spec.config }

    private init(
        source: any WeightSource, config: ModelConfig, spec: IRSpec, shard: ShardExecution? = nil
    ) throws {
        self.source = source
        self.config = config
        self.spec = spec
        self.shard = shard

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
    /// How many experts one layer's slot bank may hold. Overridable for **measurement**.
    ///
    /// The expert slot bank's budget in bytes, for the whole model (`D12`, `DC-091`).
    ///
    /// The brief asks for per-layer LRU slot banks and does not say how large. It was the literal `16`,
    /// never derived from anything: 16 slots across this chain's 40 layers is **8.05 GB** against roughly
    /// 4.5 GB usable, which `SlotBudgetTests` reported as an expected failure until the number came from
    /// a budget. The default below is the measured answer (`D31`), not a guess: it is the smallest bank
    /// at which the read savings stop improving, so paying more memory buys nothing.
    public static let expertBankBudgetBytes = bankBudgetBytes(
        environment: ProcessInfo.processInfo.environment
    )

    /// How many bytes of **decoded layer weights** one generation may hold, from `SHARD_LAYER_CACHE_MB`.
    ///
    /// `D88` measured the decode's `load` phase at **30.5% of a step**, all of it the same constants being
    /// dequantised again on every token — 160 times per generation — while the payload it comes from is already
    /// resident. A layer held here is a hit on **every** step by construction, unlike the expert slot bank whose
    /// hit rate `D31` measured at zero at every size, so this is where a node's spare bytes belong.
    ///
    /// **The default is zero, and that is a measurement rather than caution** (`D89`). On this 8 GB node, with
    /// budgets of 256 MB, 1 GB and 2 GB alternated against no cache on one binary, the cache did exactly what it
    /// was built to do — holding 2, 8 and 16 layers of 40, `load` falling by about 0.035 s per layer per step,
    /// which across all 40 layers would be 1.4 s — and **the step still got slower as it held more**, because the
    /// resident fp32 arrays cost more elsewhere than they saved: at 2 GB, `head` was +0.20 s/step, `attn.core`
    /// +0.27 and `mix.read` +0.17, on a machine already swapping. At 256 MB it was neutral; at 1 GB it was
    /// 5.33 → 5.46 s/step.
    ///
    /// So the knob is for a node with headroom and the default is for this one. A 24 GB or 64 GB machine should
    /// set `SHARD_LAYER_CACHE_MB` and re-measure; the metrics below record what each node actually held, which is
    /// `DC-052`'s done-when either way.
    static func layerCacheBudgetBytes(environment: [String: String]) -> Int {
        let defaultMegabytes = 0
        guard let raw = environment["SHARD_LAYER_CACHE_MB"], let megabytes = Int(raw) else {
            return defaultMegabytes * 1_048_576
        }
        // Refused rather than clamped, for the reason `bankBudgetBytes` gives: a budget nobody could hold is a
        // node that swaps, and a silent clamp hides the typo that caused it.
        guard megabytes >= 0, megabytes <= 1 << 20 else { return defaultMegabytes * 1_048_576 }
        return megabytes * 1_048_576
    }

    /// The budget asked for on this node.
    public static let layerCacheBudget = layerCacheBudgetBytes(
        environment: ProcessInfo.processInfo.environment
    )

    /// The budget, from `SHARD_EXPERT_BANK_MB` when it is sane.
    static func bankBudgetBytes(environment: [String: String]) -> Int {
        let defaultMegabytes = 512
        guard let raw = environment["SHARD_EXPERT_BANK_MB"], let megabytes = Int(raw) else {
            return defaultMegabytes * 1_048_576
        }
        // A budget nobody could hold is refused rather than clamped quietly: `Int.max` here is a typo,
        // and a bank sized from it is a node that swaps — which is what `DC-091` reverted.
        guard megabytes >= 0, megabytes <= 1 << 20 else { return defaultMegabytes * 1_048_576 }
        return megabytes * 1_048_576
    }

    /// How many experts one layer may keep, given a budget for the whole bank.
    ///
    /// `ExpertSlotCache` holds one expert's fused gate/up (`2·inter·hidden`) and its down
    /// (`inter·hidden`) as fp32 reals, so one expert costs `3·inter·hidden·4` bytes and the whole bank
    /// costs that times the slots times the layers. The arithmetic is asserted in `SlotBudgetTests`
    /// against the **real** geometry, because a fixture's experts are kilobytes and cannot see a
    /// gigabyte — which is how a 14.5 GB change once shipped past 98 green tests.
    public static func expertSlots(budgetBytes: Int, layers: Int, shape: MixtureShape) -> Int {
        let (perRow, rowOverflow) = shape.intermediate.multipliedReportingOverflow(by: shape.hiddenSize)
        let (perExpert, expertOverflow) = perRow.multipliedReportingOverflow(by: 3 * 4)
        let (whole, wholeOverflow) = perExpert.multipliedReportingOverflow(by: max(1, layers))
        guard !rowOverflow, !expertOverflow, !wholeOverflow, whole > 0, budgetBytes > 0 else { return 1 }
        return max(1, min(budgetBytes / whole, shape.experts))
    }

    /// The capacity one layer's bank gets: an explicit sweep wins, otherwise the budget decides.
    ///
    /// `SHARD_EXPERT_SLOTS=<n>` sets it for one process, which is how a measurement is a series of runs
    /// rather than a rebuild. Swift 6 forbids a mutable global and is right to: a value that can change
    /// under a running forward is a race.
    static func slotCapacity(environment: [String: String], shape: MixtureShape, layers: Int) -> Int {
        if let raw = environment["SHARD_EXPERT_SLOTS"], let slots = Int(raw), slots >= 1 { return slots }
        return expertSlots(
            budgetBytes: bankBudgetBytes(environment: environment), layers: layers, shape: shape
        )
    }

    /// Whether to time the forward's phases. Immutable, so Swift 6's concurrency checking is satisfied
    /// and a profile cannot change under a running forward. `SHARD_PROFILE=1` turns it on.
    public static let profilingEnabled = ProcessInfo.processInfo.environment["SHARD_PROFILE"] == "1"

    /// A profiler when `SHARD_PROFILE=1` asked for one, and nothing otherwise.
    ///
    /// Used as a default argument so every entry point — the sequence forward, the cached decode, the
    /// generation loop — answers "was profiling asked for?" in one place, and so a test can hand one in
    /// directly rather than needing the environment variable, which is read once at load time.
    public static var requestedProfiler: Profiler? { profilingEnabled ? Profiler() : nil }

    /// Whether the trace records what is *inside* a decoder layer as well as at its boundaries.
    ///
    /// Off by default, and it has to be: the trace's digest covers the tensor list, so a trace with extra
    /// tensors is a different artifact and every recorded digest would move. `SHARD_TRACE_INTERNALS=1` turns
    /// it on. It exists because a divergence localised to "inside layer 0" cannot be bisected from
    /// boundaries alone, and the first attempt to bisect it without these points produced a number that
    /// disagreed with the reference's own record — a broken instrument rather than a finding.
    public static let tracingInternals = ProcessInfo.processInfo.environment["SHARD_TRACE_INTERNALS"] == "1"

    /// The feed-forward half of a decoder layer. The reference branches inside its decoder
    /// layer between `Qwen3_5MLP` and `Qwen3_5SparseMoeBlock`, and so does this.
    enum FeedForward {
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

    /// The weights a decoder layer needs, through `cache` when there is one.
    ///
    /// Without a cache this is what it always was: decode, use, release. With one, a layer that fits is decoded
    /// once per generation instead of once per token — which is the whole of `D88`'s 30.5%.
    func loadLayer(_ index: Int, cache: LayerWeightCache? = nil) throws -> DecodedLayer {
        if let cache { return try cache.layer(index) { try decodeLayer(index) } }
        return try decodeLayer(index)
    }

    /// Decode one layer from the source, uncached.
    func decodeLayer(_ index: Int) throws -> DecodedLayer {
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
            let cache = ExpertSlotCache(
                upstream: stacked,
                capacity: Self.slotCapacity(
                    environment: ProcessInfo.processInfo.environment, shape: try mixtureShape(),
                    layers: config.numLayers
                )
            )
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
            return DecodedLayer(weights: weights, gdn: nil, feedForward: feedForward)
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
        return DecodedLayer(weights: weights, gdn: gdn, feedForward: feedForward)
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

    /// A mixture layer's output, sharded or not — **one** implementation, because a second call site is
    /// a second chance to forget the reduce, and that failure mode is a plausible token sequence computed
    /// from a fraction of the experts. The cached decode path calls this too (`DC-109`); before it did,
    /// a sharded `--cached` run would have ignored the shard entirely and looked fine.
    func mixtureOutput(
        hidden: [Float], tokens: Int, weights: MixtureWeights, shape: MixtureShape,
        profiler: Profiler? = nil
    ) throws -> (output: [Float], indices: [[Int]]) {
        guard let shard else {
            let whole = try MixtureOfExperts.block(
                hidden: hidden, tokens: tokens, weights: weights, shape: shape, profiler: profiler
            )
            return (whole.output, whole.indices)
        }
        let (routed, indices) = try shardedRouted(
            hidden: hidden, tokens: tokens, weights: weights, shape: shape, shard: shard,
            profiler: profiler
        )
        return (routed, indices)
    }

    /// One node's half of a mixture layer: its own experts' terms, the all-reduce with its peers, and the
    /// shared expert added back by the same `combine` the single-node path uses.
    ///
    /// The router runs here on every node rather than being shipped, because the dense backbone is
    /// replicated — every node already has what it needs to decide, and a router decision that travelled
    /// would be a second source of truth for it.
    private func shardedRouted(
        hidden: [Float], tokens: Int, weights: MixtureWeights, shape: MixtureShape,
        shard: ShardExecution, profiler: Profiler?
    ) throws -> (routed: [Float], indices: [[Int]]) {
        let (_, indices, chosen) = MixtureOfExperts.router(
            hidden: hidden, tokens: tokens, weights: weights.router, experts: shape.experts, topK: shape.topK
        )
        let owned = OwnedExpertProvider(
            base: weights.experts, ownership: shard.ownership, node: shard.node
        )
        let terms = try MixtureOfExperts.expertContributions(
            hidden: hidden, tokens: tokens, provider: owned, indices: indices,
            weights: chosen, shape: shape, profiler: profiler
        )
        // One all-reduce per mixture layer, carrying terms rather than partial sums (`D17`).
        let reduced = try ShardExchange.allReduce(
            own: terms, peers: shard.peers, indices: indices, tokens: tokens,
            hiddenSize: shape.hiddenSize, policy: shard.policy, ledger: shard.ledger
        )
        let parts = MixtureOfExperts.sharedPart(
            hidden: hidden, tokens: tokens, weights: weights, shape: shape
        )
        return (
            MixtureOfExperts.combine(
                routed: reduced, shared: parts.shared, scalar: parts.scalar,
                tokens: tokens, hiddenSize: shape.hiddenSize
            ),
            indices
        )
    }

    public func forwardWithDecisions(tokens: [Int]) throws -> ForwardResult {
        guard !tokens.isEmpty else { throw Error.emptyPrompt }
        let length = tokens.count
        let hiddenSize = config.hiddenSize
        var captured: [TraceWriter.Tensor] = []
        var discrete: [TraceWriter.Discrete] = []
        var expertMetrics: [ExpertProviderMetrics] = []
        // A profile of the real path, off unless asked for. The marks sit *after* the work they name,
        // so a phase's seconds are the time since the previous mark and the phases sum to the run.
        let profiler = Self.profilingEnabled ? Profiler() : nil

        var hidden = [Float](repeating: 0, count: length * hiddenSize)
        for (row, token) in tokens.enumerated() {
            precondition(token >= 0 && token < config.vocabSize, "token \(token) outside the vocabulary")
            let values = try source.rows(named: embeddingName, range: token..<(token + 1))
            for index in 0..<hiddenSize { hidden[row * hiddenSize + index] = values[index] }
        }
        captured.append(TraceWriter.Tensor(name: "embed.out", shape: [length, hiddenSize], values: hidden))
        profiler?.mark("embed")

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
            profiler?.mark("load")
            let normed = rmsNorm(
                hidden, weight: layer.weights[.attnNorm]!, rows: length, width: hiddenSize,
                eps: Float(config.rmsNormEps)
            )
            profiler?.mark("attn.norm")
            let mixed: [Float]
            if let gdn = layer.gdn {
                mixed = GatedDeltaNet.layer(
                    hidden: normed, weights: gdn, shape: try gatedShape(), batch: 1, length: length,
                    record: Self.tracingInternals ? { name, values, shape in
                        captured.append(
                            TraceWriter.Tensor(name: "\(tag).\(name)", shape: shape, values: values)
                        )
                    } : nil
                )
            } else {
                mixed = try attention(
                    normed, weights: layer.weights, length: length, tables: tables, mask: mask
                )
            }
            profiler?.mark("attn.core")
            hidden = add(hidden, mixed)
            // The mixture's input, which is also the router's input after its norm: the quantity whose
            // divergence would explain everything downstream of it.
            if Self.tracingInternals {
                captured.append(
                    TraceWriter.Tensor(name: "\(tag).attn_out", shape: [length, hiddenSize], values: hidden)
                )
            }
            profiler?.mark("attn.add")

            let residual = hidden
            let postNormed = rmsNorm(
                hidden, weight: layer.weights[.mlpNorm]!, rows: length, width: hiddenSize,
                eps: Float(config.rmsNormEps)
            )
            profiler?.mark("ff.norm")
            switch layer.feedForward {
            case .dense(let gate, let up, let down):
                let projectedGate = MetalMatmul.ordered(
                    x: postNormed, w: gate, rows: length, k: hiddenSize, out: config.intermediateSize
                )
                let projectedUp = MetalMatmul.ordered(
                    x: postNormed, w: up, rows: length, k: hiddenSize, out: config.intermediateSize
                )
                var activated = [Float](repeating: 0, count: length * config.intermediateSize)
                for position in 0..<activated.count {
                    activated[position] = Ops.silu(projectedGate[position]) * projectedUp[position]
                }
                let downProjected = MetalMatmul.ordered(
                    x: activated, w: down, rows: length, k: config.intermediateSize, out: hiddenSize
                )
                if Self.tracingInternals {
                    captured.append(
                        TraceWriter.Tensor(
                            name: "\(tag).ff_out", shape: [length, hiddenSize], values: downProjected
                        )
                    )
                }
                hidden = add(residual, downProjected)

            case .mixture(let weights, let provider):
                let shape = try mixtureShape()
                let (routed, indices) = try mixtureOutput(
                    hidden: postNormed, tokens: length, weights: weights, shape: shape, profiler: profiler
                )
                // Kept so the caller can report a measured hit rate rather than an assurance.
                // M1's gate asks for the number, and the number is not visible from outside.
                expertMetrics.append(provider.metrics)
                if Self.tracingInternals {
                    captured.append(
                        TraceWriter.Tensor(name: "\(tag).ff_out", shape: [length, hiddenSize], values: routed)
                    )
                }
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
            profiler?.mark("ff")
            captured.append(TraceWriter.Tensor(name: "\(tag).hidden_out", shape: [length, hiddenSize], values: hidden))
            profiler?.mark("trace.copy")
        }

        let finalWeight = try source.tensor(named: finalNormName)
        hidden = rmsNorm(hidden, weight: finalWeight, rows: length, width: hiddenSize, eps: Float(config.rmsNormEps))
        captured.append(TraceWriter.Tensor(name: "final_norm.out", shape: [length, hiddenSize], values: hidden))
        profiler?.mark("final_norm")

        // The head, one block of vocabulary rows at a time: the same matrix as the embedding,
        // and materialising it whole is the thing this design exists to avoid.
        var logits = [Float](repeating: 0, count: length * config.vocabSize)
        var row = 0
        while row < config.vocabSize {
            let upper = min(row + Self.headBlockRows, config.vocabSize)
            let block = try source.rows(named: headName, range: row..<upper)
            let product = MetalMatmul.ordered(
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
        profiler?.mark("head")
        return ForwardResult(
            tensors: captured, discrete: discrete, expertMetrics: expertMetrics,
            profile: profiler?.report(layers: config.numLayers)
        )
    }

    func attention(
        _ hidden: [Float], weights: [TensorRole: [Float]], length: Int,
        tables: (cos: [Float], sin: [Float]), mask: [Float]
    ) throws -> [Float] {
        let heads = config.numAttentionHeads
        let kvHeads = config.numKeyValueHeads
        let headDim = config.headDim
        let eps = Float(config.rmsNormEps)
        let scaling = Float(1) / Float(headDim).squareRoot()
        let rotary = tables.cos.count / max(length, 1)

        let projected = MetalMatmul.ordered(
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
        var key = MetalMatmul.ordered(
            x: hidden, w: weights[.attnK]!, rows: length, k: config.hiddenSize, out: kvHeads * headDim
        )
        let value = MetalMatmul.ordered(
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
            var scores = MetalMatmul.ordered(x: queryHead, w: keyHead, rows: length, k: headDim, out: length)
            for index in 0..<scores.count { scores[index] = scores[index] * scaling + mask[index] }
            let attention = Ops.softmax(x: scores, rows: length, width: length)
            let valueHead = sliceHead(value, length: length, heads: kvHeads, headDim: headDim, head: kvHead)
            var transposed = [Float](repeating: 0, count: headDim * length)
            for position in 0..<length {
                for index in 0..<headDim { transposed[index * length + position] = valueHead[position * headDim + index] }
            }
            let mixed = MetalMatmul.ordered(x: attention, w: transposed, rows: length, k: length, out: headDim)
            for position in 0..<length {
                for index in 0..<headDim {
                    mixer[(position * heads + head) * headDim + index] = mixed[position * headDim + index]
                }
            }
        }

        var gated = [Float](repeating: 0, count: mixer.count)
        for index in 0..<mixer.count { gated[index] = mixer[index] * Ops.sigmoid(gate[index]) }
        return MetalMatmul.ordered(
            x: gated, w: weights[.attnO]!, rows: length, k: heads * headDim, out: config.hiddenSize
        )
    }

    /// One position of full attention against an append-only key/value cache.
    ///
    /// Deliberately the *same* arithmetic as the sequence path: the scores are one ordered dot
    /// product per cached position, the softmax runs over the same width, and the output is the
    /// same ordered product. So `D8` does not apply here — cached attention is **bit-identical**
    /// to attention over the whole sequence, and the test says so at that strength rather than
    /// at a tolerance. No mask is needed: every cached position precedes the new one.
    ///
    /// `keys` and `values` are `[position, kvHead, headDim]` and are extended in place.
    func attentionStep(
        _ hidden: [Float], weights: [TensorRole: [Float]], tables: (cos: [Float], sin: [Float]),
        keys: inout [Float], values: inout [Float], cachedLength: Int
    ) throws -> [Float] {
        let heads = config.numAttentionHeads
        let kvHeads = config.numKeyValueHeads
        let headDim = config.headDim
        let eps = Float(config.rmsNormEps)
        let scaling = Float(1) / Float(headDim).squareRoot()
        let rotary = tables.cos.count

        let projected = MetalMatmul.ordered(
            x: hidden, w: weights[.attnQ]!, rows: 1, k: config.hiddenSize, out: heads * headDim * 2
        )
        var query = [Float](repeating: 0, count: heads * headDim)
        var gate = [Float](repeating: 0, count: heads * headDim)
        for head in 0..<heads {
            for index in 0..<headDim {
                query[head * headDim + index] = projected[(head * headDim) * 2 + index]
                gate[head * headDim + index] = projected[(head * headDim) * 2 + headDim + index]
            }
        }
        var key = MetalMatmul.ordered(
            x: hidden, w: weights[.attnK]!, rows: 1, k: config.hiddenSize, out: kvHeads * headDim
        )
        let value = MetalMatmul.ordered(
            x: hidden, w: weights[.attnV]!, rows: 1, k: config.hiddenSize, out: kvHeads * headDim
        )

        query = rmsNorm(query, weight: weights[.attnQNorm]!, rows: heads, width: headDim, eps: eps)
        key = rmsNorm(key, weight: weights[.attnKNorm]!, rows: kvHeads, width: headDim, eps: eps)
        query = applyPartialRope(query, tables: tables, length: 1, heads: heads, headDim: headDim, rotary: rotary)
        key = applyPartialRope(key, tables: tables, length: 1, heads: kvHeads, headDim: headDim, rotary: rotary)

        keys.append(contentsOf: key)
        values.append(contentsOf: value)
        let width = cachedLength + 1
        let groups = heads / kvHeads

        var mixer = [Float](repeating: 0, count: heads * headDim)
        for head in 0..<heads {
            let kvHead = head / groups
            var queryHead = [Float](repeating: 0, count: headDim)
            for index in 0..<headDim { queryHead[index] = query[head * headDim + index] }

            // `Ops.orderedMatmul` takes `w` as `[out, k]` — the layout of a `Linear.weight` — so
            // the scores' weight is `[cached positions, headDim]`, which is the key head *as
            // stored*. Building it the other way round multiplies the wrong pairs and still
            // returns numbers of the right shape; this is the bug the bit-identity test caught.
            var keyRows = [Float](repeating: 0, count: width * headDim)
            for position in 0..<width {
                for index in 0..<headDim {
                    keyRows[position * headDim + index] = keys[(position * kvHeads + kvHead) * headDim + index]
                }
            }
            var scores = MetalMatmul.ordered(x: queryHead, w: keyRows, rows: 1, k: headDim, out: width)
            for index in 0..<scores.count { scores[index] = scores[index] * scaling }
            let attention = Ops.softmax(x: scores, rows: 1, width: width)

            // And here `out` is the head dimension and `k` is the cached positions, so the weight
            // is `[headDim, width]` — the transpose, exactly as the sequence path builds it.
            var valueRows = [Float](repeating: 0, count: headDim * width)
            for index in 0..<headDim {
                for position in 0..<width {
                    valueRows[index * width + position] =
                        values[(position * kvHeads + kvHead) * headDim + index]
                }
            }
            let mixed = MetalMatmul.ordered(x: attention, w: valueRows, rows: 1, k: width, out: headDim)
            for index in 0..<headDim { mixer[head * headDim + index] = mixed[index] }
        }

        var gated = [Float](repeating: 0, count: mixer.count)
        for index in 0..<mixer.count { gated[index] = mixer[index] * Ops.sigmoid(gate[index]) }
        return MetalMatmul.ordered(
            x: gated, w: weights[.attnO]!, rows: 1, k: heads * headDim, out: config.hiddenSize
        )
    }

    func applyPartialRope(
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

    func sliceHead(_ values: [Float], length: Int, heads: Int, headDim: Int, head: Int) -> [Float] {
        var slice = [Float](repeating: 0, count: length * headDim)
        for row in 0..<length {
            let source = (row * heads + head) * headDim
            for index in 0..<headDim { slice[row * headDim + index] = values[source + index] }
        }
        return slice
    }

    func rmsNorm(_ x: [Float], weight: [Float], rows: Int, width: Int, eps: Float) -> [Float] {
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

    func add(_ left: [Float], _ right: [Float]) -> [Float] {
        var result = [Float](repeating: 0, count: left.count)
        for index in 0..<left.count { result[index] = left[index] + right[index] }
        return result
    }
}
