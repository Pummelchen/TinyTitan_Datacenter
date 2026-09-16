import DatacenterIR
import Foundation

/// What a forward produces: the captured tensors, and — kept apart from them — the
/// **discrete decisions** the invariants treat separately (I3). A trace that folded a router's
/// top-k into a float tensor would make it comparable by tolerance, and a tolerance is exactly
/// what cannot express "the same experts".
public struct ForwardResult {
    public let tensors: [TraceWriter.Tensor]
    public let discrete: [TraceWriter.Discrete]
    /// One entry per mixture layer: what its expert reads actually cost. M1's gate asks for a
    /// measured cache hit rate, and a number that is not returned by the forward pass is a
    /// number nobody can reproduce.
    public let expertMetrics: [ExpertProviderMetrics]
    /// Phase timings, present only when `SHARD_PROFILE=1` asked for them. Nil means *not measured*,
    /// which is not the same as zero.
    public let profile: ProfileReport?

    public init(
        tensors: [TraceWriter.Tensor], discrete: [TraceWriter.Discrete] = [],
        expertMetrics: [ExpertProviderMetrics] = [], profile: ProfileReport? = nil
    ) {
        self.tensors = tensors
        self.discrete = discrete
        self.expertMetrics = expertMetrics
        self.profile = profile
    }

    /// Expert-slice reads served from memory over all mixture layers, or zero when there were
    /// no requests — a `0/0` rate would be a claim, not a measurement.
    public var expertHitRate: Double {
        let requests = expertMetrics.reduce(0) { $0 + $1.requests }
        let hits = expertMetrics.reduce(0) { $0 + $1.hits }
        return requests == 0 ? 0 : Double(hits) / Double(requests)
    }

    public var expertElementsRead: Int { expertMetrics.reduce(0) { $0 + $1.elementsRead } }
}

/// Phase timings for one forward, in seconds, when `SHARD_PROFILE=1`.
///
/// A profile of the real code path rather than arithmetic about it. This session produced three
/// wrong readings of where a token's time goes — an estimate of SSD bytes, a division by that
/// estimate, and a read benchmark that was answered by cache — and each was replaced by a
/// measurement. This is the instrument that was missing.
public struct ProfileReport: Sendable {
    /// Seconds per phase, accumulated over every layer, so a phase's share is a division by the total.
    public let seconds: [String: Double]
    /// How many layers the accumulation covers, so a per-layer average is derived rather than guessed.
    public let layers: Int
}

/// Accumulates phase timings between `mark` calls. Allocated only when profiling is on, so the
/// instrument costs nothing when it is off.
public final class Profiler {
    private var last: UInt64
    private var seconds: [String: Double] = [:]

    public init() { last = DispatchTime.now().uptimeNanoseconds }

    /// Close the phase that just ended. The name describes the work between the previous mark and
    /// this one, which is why the marks sit *after* the work rather than around it.
    public func mark(_ phase: String) {
        let now = DispatchTime.now().uptimeNanoseconds
        seconds[phase, default: 0] += Double(now &- last) / 1e9
        last = now
    }

    public func report(layers: Int) -> ProfileReport {
        ProfileReport(seconds: seconds, layers: layers)
    }
}

/// A model that can be run for a token sequence and asked to generate.
///
/// Two families implement it today, which is what makes it an abstraction rather than a
/// guess about the future. Everything above it — the trace writer, the differ, the
/// command-line tools — works against this and never against a family.
public protocol ForwardPass {
    /// The model's IR spec — roles, shapes, policies — so a tool can consume a model
    /// without knowing its family, and so the *importer* stays the only place that knows
    /// tensor names.
    var spec: IRSpec { get }
    /// The size of the vocabulary, so a caller can slice the logits without knowing the
    /// model.
    var vocabularySize: Int { get }
    /// The captured tensors for a token sequence, in the reference's order.
    func forward(tokens: [Int]) throws -> [TraceWriter.Tensor]
    /// The same, plus the discrete decisions. A family with no such decisions — every family
    /// before the mixture — gets the default, so this costs the earlier ones nothing.
    func forwardWithDecisions(tokens: [Int]) throws -> ForwardResult
    /// Payload bytes the family's weight source has read since it was opened, when it counts.
    ///
    /// This is a protocol **requirement** rather than an extension member on purpose: an extension
    /// member is dispatched statically, so a caller holding `any ForwardPass` would silently get
    /// the default and the figure would look measured while being a constant. Zero means *not
    /// counted* — the dense family holds its weights from open time.
    var sourceBytesRead: Int { get }
    /// The payload cache's counters, on the same footing as `sourceBytesRead` (`DC-106`).
    var payloadCacheMetrics: PayloadCacheMetrics { get }
    /// Whole-tensor requests per tensor, most-requested first — the `DC-106` repeat-read audit.
    var payloadRequestCounts: [(name: String, count: Int)] { get }
    /// Where the source's time went — reading, verifying, unpacking — when it counts it.
    var sourceTiming: SourceTiming { get }
}

extension ForwardPass {
    public var sourceBytesRead: Int { 0 }
    public var payloadCacheMetrics: PayloadCacheMetrics { PayloadCacheMetrics() }
    public var payloadRequestCounts: [(name: String, count: Int)] { [] }
    public var sourceTiming: SourceTiming { SourceTiming() }
}

extension ForwardPass {
    public func forwardWithDecisions(tokens: [Int]) throws -> ForwardResult {
        ForwardResult(tensors: try forward(tokens: tokens))
    }

    /// Greedy generation. M0 has no KV cache: every step re-runs the whole sequence,
    /// because a cache is a second numeric path through attention and M0's job is to
    /// establish one correct path before there are two.
    ///
    /// The loop is shaped so the last forward is over the full sequence, which is what a
    /// trace of a generation records; the Python contract uses the identical loop.
    public func generate(prompt: [Int], maxNewTokens: Int) throws -> Generation {
        var tokens = prompt
        var generated: [Int] = []
        var seconds: [Double] = []

        var captured = try forward(tokens: tokens)
        var margins: [Float] = []
        for _ in 0..<maxNewTokens {
            guard let logits = captured.last, logits.name == "logits" else {
                throw GenerationError.noLogits
            }
            let width = vocabularySize
            let offset = (tokens.count - 1) * width
            let next = Greedy.argmax(logits.values, offset: offset, width: width)
            generated.append(next)
            margins.append(Greedy.margin(logits.values, offset: offset, width: width))
            tokens.append(next)

            let started = Date()
            captured = try forward(tokens: tokens)
            seconds.append(Date().timeIntervalSince(started))
        }
        return Generation(
            prompt: prompt, generated: generated, secondsPerStep: seconds, captured: captured,
            margins: margins
        )
    }
}

public enum GenerationError: Error, CustomStringConvertible {
    case noLogits
    case unknownFamily(String)

    public var description: String {
        switch self {
        case .noLogits: return "the forward pass captured no logits tensor"
        case .unknownFamily(let type): return "no importer for model_type '\(type)'"
        }
    }
}

/// Opens a checkpoint and returns the right family's forward pass.
///
/// The dispatch reads the checkpoint's own `model_type`, so adding a family is adding an
/// importer and a case here — not a flag the operator has to remember.
public enum ModelLoader {
    public static func open(snapshot: URL) throws -> any ForwardPass {
        // An install is self-describing: if its header is there, it is what we were handed.
        if FileManager.default.fileExists(atPath: snapshot.appendingPathComponent("install.json").path) {
            // An install carries its own spec, and the spec names its family, so the artifact
            // says which forward pass reads it rather than the caller having to know.
            let file = try InstallFile(url: snapshot)
            switch file.manifest.spec.family {
            // `qwen3` has no install reader of its own: the int4 path was built for the
            // family M0c measured, and a `qwen3` install would be a new claim about a family
            // nobody has quantized yet. Refusing is the honest answer.
            case "qwen3": throw GenerationError.unknownFamily("qwen3 (no install reader)")
            default: return try Qwen3_5Forward(install: snapshot)
            }
        }
        let configData = try Data(contentsOf: snapshot.appendingPathComponent("config.json"))
        guard let object = try JSONSerialization.jsonObject(with: configData) as? [String: Any],
              let modelType = object["model_type"] as? String
        else {
            throw GenerationError.unknownFamily("(no model_type in config.json)")
        }
        switch modelType {
        case "qwen3":
            return try Qwen3Forward(snapshot: snapshot)
        // One implementation serves both `qwen3_5` families: the reference branches *inside*
        // its decoder layer between a dense feed-forward and a mixture, and
        // `tools/compare_reference_modules.py` proved the attention, the Gated DeltaNet, the
        // RoPE and the conv identical between them. A second copy would be a second thing to
        // keep in step for no gain.
        case "qwen3_5", "qwen3_5_moe":
            return try Qwen3_5Forward(snapshot: snapshot)
        default:
            throw GenerationError.unknownFamily(modelType)
        }
    }
}
