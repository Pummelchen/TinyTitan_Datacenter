import DatacenterIR
import Foundation

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
}

extension ForwardPass {
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
        for _ in 0..<maxNewTokens {
            guard let logits = captured.last, logits.name == "logits" else {
                throw GenerationError.noLogits
            }
            let width = vocabularySize
            let offset = (tokens.count - 1) * width
            let next = Greedy.argmax(logits.values, offset: offset, width: width)
            generated.append(next)
            tokens.append(next)

            let started = Date()
            captured = try forward(tokens: tokens)
            seconds.append(Date().timeIntervalSince(started))
        }
        return Generation(prompt: prompt, generated: generated, secondsPerStep: seconds, captured: captured)
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
            return try Qwen3_5Forward(install: snapshot)
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
        case "qwen3_5":
            return try Qwen3_5Forward(snapshot: snapshot)
        default:
            throw GenerationError.unknownFamily(modelType)
        }
    }
}
