import TinyTitan

public enum PrefillChunkChoice: Equatable, Sendable {
    case fixed(Int)
    case auto
}

public struct Args: Equatable, Sendable {
    public var model: String
    public var prompt: String?
    public var messagesFile: String?
    public var maxNew: Int
    public var maxContext: Int
    public var temperature: Float
    public var topK: Int?
    public var topP: Float?
    /// Whether the caller named these, as opposed to inheriting the house
    /// values. The family's own defaults are applied in `Run` once the
    /// manifest is read; an explicit flag always wins over them.
    public var temperatureWasSet: Bool = false
    public var topKWasSet: Bool = false
    public var topPWasSet: Bool = false
    public var repetitionPenalty: Float
    public var seed: UInt64?
    /// Sharded-run configuration, all three or none. Absent on every single-node run, so the engine's default
    /// behaviour is untouched and the ownership filter stays unset rather than set to an identity (`D164`).
    public var shardPlanPath: String?
    public var shardNode: Int?
    public var shardPeersSpec: String?
    /// Port to serve peer expert requests on. Without it a node generates but does not answer.
    public var shardServePort: Int?
    /// Serve peers and nothing else. D278 measured a peer request at 2.95 ms of compute against about 0.5 ms for the
    /// same work single-node, and named contention between a node's own forward pass and the requests it answers.
    /// A node cannot test that today: it always generates, and a Task does not keep the process alive once main
    /// returns, so the server dies with the generation. With this the node serves on the main path instead (D279).
    /// The layers this node owns, as `start:end` - the whole of Design A's partition. `nil` means all of them,
    /// which is the single-node configuration and the one A1 proved bit-identical.
    public var layerRange: String?

    public var shardServeOnly: Bool
    public var stops: [String]
    public var quiet: Bool
    public var concise: Bool
    public var thinkingMode: ModelThinkingMode
    public var reasoningEffort: ModelReasoningEffort?
    /// Routed-expert slots, or nil to derive from the family's measured
    /// tuning and the payload's expert stride. A flat default cannot be
    /// right for both families: 8 GiB is 128 slots at the 35B stride and 96
    /// at Qwen3.8-Flash-Next's.
    public var expertCacheSlots: Int?
    public var rdadvise: String
    public var prefillChunk: PrefillChunkChoice?
    public var kvCachePrecision: KVCachePrecision
    public var ropeScalingMode: RuntimeRoPEScalingMode

    public init(model: String,
                prompt: String? = nil,
                messagesFile: String? = nil,
                maxNew: Int = 1_024,
                maxContext: Int = 4096,
                temperature: Float = GenerationDefaults.temperature,
                topK: Int? = GenerationDefaults.topK,
                topP: Float? = GenerationDefaults.topP,
                temperatureWasSet: Bool = false,
                topKWasSet: Bool = false,
                topPWasSet: Bool = false,
                repetitionPenalty: Float = 1.0,
                seed: UInt64? = nil,
                shardPlanPath: String? = nil,
                shardNode: Int? = nil,
                shardPeersSpec: String? = nil,
                shardServePort: Int? = nil,
                shardServeOnly: Bool = false,
                layerRange: String? = nil,
                stops: [String] = [],
                quiet: Bool = false,
                concise: Bool = false,
                thinkingMode: ModelThinkingMode = .off,
                reasoningEffort: ModelReasoningEffort? = nil,
                expertCacheSlots: Int? = nil,
                rdadvise: String = "default",
                prefillChunk: PrefillChunkChoice? = nil,
                kvCachePrecision: KVCachePrecision = .int8,
                ropeScalingMode: RuntimeRoPEScalingMode = .none) {
        self.model = model
        self.prompt = prompt
        self.messagesFile = messagesFile
        self.maxNew = maxNew
        self.maxContext = maxContext
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.temperatureWasSet = temperatureWasSet
        self.topKWasSet = topKWasSet
        self.topPWasSet = topPWasSet
        self.repetitionPenalty = repetitionPenalty
        self.expertCacheSlots = expertCacheSlots
        self.rdadvise = rdadvise
        self.prefillChunk = prefillChunk
        self.kvCachePrecision = kvCachePrecision
        self.ropeScalingMode = ropeScalingMode
        self.seed = seed
        self.shardPlanPath = shardPlanPath
        self.shardNode = shardNode
        self.shardPeersSpec = shardPeersSpec
        self.shardServePort = shardServePort
        self.shardServeOnly = shardServeOnly
        self.layerRange = layerRange
        self.stops = stops
        self.quiet = quiet
        self.concise = concise
        self.thinkingMode = thinkingMode
        self.reasoningEffort = reasoningEffort
    }
}

public enum ArgsError: Error, Equatable, CustomStringConvertible {
    case helpRequested
    case unknownFlag(String)
    case missingValue(flag: String)
    case invalidValue(flag: String, value: String)
    case requiredMissing(String)
    case mutuallyExclusive(String, String)
    case modeMissing

    public var description: String {
        switch self {
        case .helpRequested: return "help requested"
        case .unknownFlag(let flag): return "unknown flag: \(flag)"
        case .missingValue(let flag): return "missing value for \(flag)"
        case .invalidValue(let flag, let value): return "invalid value for \(flag): \(value)"
        case .requiredMissing(let flag): return "required flag missing: \(flag)"
        case .mutuallyExclusive(let a, let b): return "\(a) and \(b) are mutually exclusive"
        case .modeMissing: return "one of --prompt or --messages-file is required"
        }
    }
}

extension Args {
    /// The accepted slot counts, spelled from the validator that enforces them.
    ///
    /// The help text used to list them by hand and had already drifted: it
    /// stopped at 128 while `RuntimeConfiguration.allowedExpertCacheSlots`
    /// accepts 40, 48, 112, 160, 192 and 256 as well, so a caller reading
    /// `--help` would not know 160 was legal. Spelled from the source of truth
    /// so it cannot drift again.
    static var expertCacheSlotsHelp: String {
        RuntimeConfiguration.allowedExpertCacheSlots
            .map(String.init).joined(separator: ", ")
    }

    public static let usage = """
    TinyTitanCLI — Qwen3.5-MoE 35B-A3B text generation

    usage: TinyTitanCLI --model <dir> (--prompt <string> | --messages-file <path>) [options]

    required:
      --model <dir>             Path to a .gturbo model directory.
      --prompt <string>         Raw-completion prompt.
      --messages-file <path>    JSON chat messages with role and content fields.

    options:
      --max-new <int>           Generated-token limit (default 1024).
      --max-context <int>       Native context limit, 1...262144 (default 262144).
                                With YaRN: 524288 or 1048576 (default 1048576).
      --rope-scaling <mode>     Context scaling: none or yarn (default none).
      --temperature <float>     Sampling temperature (0 = greedy). Default is
                                the family's: 1.0 for Qwen3.8-Flash-Next,
                                0.6 elsewhere.
      --top-k <int>             Top-k truncation, 1...256 (default 20; 0 = off).
      --top-p <float>           Nucleus truncation (default 0.95).
      --repetition-penalty <f>  Repetition penalty (default 1.0).
      --seed <uint64>           Deterministic sampling seed (default off).
      --stop <string>           Stop substring (repeatable).
      --rdadvise <mode>         Expert read-ahead advice: off, default,
                                bounded, or adaptive. The default is
                                `default`, which leaves advice ON; pass
                                off to disable it.
      --expert-cache-slots <n>  Routed-expert cache slots per layer:
                                \(Self.expertCacheSlotsHelp). The default
                                is derived from the model profile's tuned
                                budget, not fixed; 64 is only the
                                fallback when the manifest cannot be read.
                                More slots raise the hit rate but use more
                                memory.
      --prefill-chunk <n|auto>  Prefill chunk tokens. Larger chunks reduce
                                routed-expert file sweeps but use more GPU
                                scratch. Allowed: 32, 64, 128, 256, 512,
                                1024, 2048, 4096; auto covers the prompt with
                                the smallest allowed chunk.
      --kv-bits <4|8|16>        KV-cache storage precision (default 8).
      --concise                 Inject the per-quantization concise-mode
                                system prompt (answers without preamble,
                                filler, or closing codas).
      --thinking <off|on>       Ornith/Qwen reasoning mode (default off).
                                These models do not define effort levels.
      --reasoning-effort <lvl>  Reasoning-effort level: low, medium, or
                                xhigh. Requires --thinking on and a model
                                family whose chat template defines effort
                                levels (Qwen3.8-Flash-Next); Ornith 1.5 and
                                Qwen 3.6 reject it.
      --quiet                   Suppress the timing footer.
      --help                    Show this message.
    """

    /// lint:allow-long same shape as ServerArguments.parse: a flag table
    /// where the exhaustive switch is the point.
    public static func parse(_ argv: [String]) throws -> Args {
        var model: String?
        var prompt: String?
        var messagesFile: String?
        var maxNew = 1_024
        // Matches the server and the published benchmark settings. The
        // allocation is lazy -- a server started at 262,144 loads with the
        // same resident footprint as one started at 4,096 -- so defaulting
        // low only surprised people whose prompt was longer than 4k.
        var maxContext = RuntimeConfiguration.nativeMaximumContextTokens
        var maxContextWasSet = false
        var temperatureWasSet = false
        var topKWasSet = false
        var topPWasSet = false
        var temperature: Float = GenerationDefaults.temperature
        var topK: Int? = GenerationDefaults.topK
        var topP: Float? = GenerationDefaults.topP
        var repetitionPenalty: Float = 1.0
        var seed: UInt64?
        var shardPlanPath: String?
        var shardNode: Int?
        var shardPeersSpec: String?
        var shardServePort: Int?
        var shardServeOnly = false
        var layerRange: String?
        var stops: [String] = []
        var quiet = false
        var concise = false
        var thinkingMode: ModelThinkingMode = .off
        var reasoningEffort: ModelReasoningEffort?
        var expertCacheSlots: Int?
        var rdadvise = "default"
        var prefillChunk: PrefillChunkChoice?
        var kvCachePrecision: KVCachePrecision = .int8
        var ropeScalingMode: RuntimeRoPEScalingMode = .none

        var index = 0
        while index < argv.count {
            let flag = argv[index]
            switch flag {
            case "--help":
                throw ArgsError.helpRequested
            case "--quiet":
                quiet = true
                index += 1
            case "--concise":
                concise = true
                index += 1
            case "--thinking":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = ModelThinkingMode(rawValue: value) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                thinkingMode = parsed
            case "--reasoning-effort":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = ModelReasoningEffort(rawValue: value) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                reasoningEffort = parsed
            case "--shard-plan":
                shardPlanPath = try takeValue(argv, &index, flag: flag)
            case "--shard-node":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = Int(value), parsed >= 0 else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                shardNode = parsed
            case "--shard-peers":
                shardPeersSpec = try takeValue(argv, &index, flag: flag)
            case "--shard-serve":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = Int(value), (1...65535).contains(parsed) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                shardServePort = parsed
            case "--shard-serve-only":
                shardServeOnly = true
            case "--layer-range":
                layerRange = try takeValue(argv, &index, flag: flag)
            case "--model":
                model = try takeValue(argv, &index, flag: flag)
            case "--prompt":
                prompt = try takeValue(argv, &index, flag: flag)
            case "--messages-file":
                messagesFile = try takeValue(argv, &index, flag: flag)
            case "--max-new":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = Int(value),
                      (1...RuntimeConfiguration.maximumContextTokens).contains(parsed) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                maxNew = parsed
            case "--max-context":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = Int(value),
                      (1...RuntimeConfiguration.maximumContextTokens).contains(parsed) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                maxContext = parsed
                maxContextWasSet = true
            case "--rope-scaling":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = RuntimeRoPEScalingMode(rawValue: value) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                ropeScalingMode = parsed
            case "--temperature":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = Float(value), parsed >= 0, parsed <= 2 else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                temperatureWasSet = true
                temperature = parsed
            case "--top-k":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = Int(value), (0...256).contains(parsed) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                topK = parsed == 0 ? nil : parsed
                topKWasSet = true
            case "--top-p":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = Float(value), parsed > 0, parsed <= 1 else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                topP = parsed
                topPWasSet = true
            case "--repetition-penalty":
                let value = try takeValue(argv, &index, flag: flag)
                // At least 1, matching AppGenerationRequest: a penalty below
                // 1 multiplies the repeated logit instead of dividing it, so
                // it rewards repetition -- the opposite of the flag. This CLI
                // is the scripted verification path, so it must not accept a
                // sampling configuration every other front end rejects.
                guard let parsed = Float(value), parsed.isFinite, parsed >= 1 else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                repetitionPenalty = parsed
            case "--seed":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = UInt64(value) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                seed = parsed
            case "--expert-cache-slots":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = Int(value),
                      RuntimeConfiguration.allowedExpertCacheSlots.contains(parsed) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                expertCacheSlots = parsed
            case "--rdadvise":
                let value = try takeValue(argv, &index, flag: flag)
                guard ["off", "default", "bounded", "adaptive"].contains(value) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                rdadvise = value
            case "--prefill-chunk":
                let value = try takeValue(argv, &index, flag: flag)
                if value == "auto" {
                    prefillChunk = .auto
                } else if let parsed = Int(value),
                          RuntimeConfiguration.allowedPrefillChunkTokens.contains(parsed) {
                    prefillChunk = .fixed(parsed)
                } else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
            case "--kv-bits":
                let value = try takeValue(argv, &index, flag: flag)
                guard let bits = Int(value),
                      let parsed = KVCachePrecision(rawValue: bits) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                kvCachePrecision = parsed
            case "--stop":
                stops.append(try takeValue(argv, &index, flag: flag))
            default:
                throw ArgsError.unknownFlag(flag)
            }
        }

        guard let model else { throw ArgsError.requiredMissing("--model") }
        if prompt != nil && messagesFile != nil {
            throw ArgsError.mutuallyExclusive("--prompt", "--messages-file")
        }
        if prompt == nil && messagesFile == nil { throw ArgsError.modeMissing }
        if temperature > 0, topK == nil, let topP, topP < 1 {
            throw ArgsError.invalidValue(
                flag: "--top-p",
                value: "\(topP) requires --top-k between 1 and 256")
        }
        if ropeScalingMode == .yarn {
            if !maxContextWasSet {
                maxContext = RuntimeConfiguration.defaultYaRNContextTokens
            }
            guard RuntimeConfiguration.supportedYaRNContextTokens.contains(maxContext) else {
                throw ArgsError.invalidValue(flag: "--max-context", value: String(maxContext))
            }
        } else if maxContext > RuntimeConfiguration.nativeMaximumContextTokens {
            throw ArgsError.invalidValue(flag: "--max-context", value: String(maxContext))
        }
        // Family support is checked in Run against the installed manifest;
        // the effort/thinking combination is a pure argument error here.
        if let effort = reasoningEffort, !thinkingMode.isEnabled {
            throw ArgsError.invalidValue(
                flag: "--reasoning-effort",
                value: "\(effort.rawValue) requires --thinking on")
        }
        return Args(model: model,
                    prompt: prompt,
                    messagesFile: messagesFile,
                    maxNew: maxNew,
                    maxContext: maxContext,
                    temperature: temperature,
                    topK: topK,
                    topP: topP,
                    temperatureWasSet: temperatureWasSet,
                    topKWasSet: topKWasSet,
                    topPWasSet: topPWasSet,
                    repetitionPenalty: repetitionPenalty,
                    seed: seed,
                    shardPlanPath: shardPlanPath,
                    shardNode: shardNode,
                    shardPeersSpec: shardPeersSpec,
                    shardServePort: shardServePort,
                    shardServeOnly: shardServeOnly,
                    layerRange: layerRange,
                    stops: stops,
                    quiet: quiet,
                    concise: concise,
                    thinkingMode: thinkingMode,
                    reasoningEffort: reasoningEffort,
                    expertCacheSlots: expertCacheSlots,
                    rdadvise: rdadvise,
                    prefillChunk: prefillChunk,
                    kvCachePrecision: kvCachePrecision,
                    ropeScalingMode: ropeScalingMode)
    }

    private static func takeValue(_ argv: [String],
                                  _ index: inout Int,
                                  flag: String) throws -> String {
        guard index + 1 < argv.count else { throw ArgsError.missingValue(flag: flag) }
        let value = argv[index + 1]
        index += 2
        return value
    }
}
