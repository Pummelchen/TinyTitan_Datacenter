import Foundation

/// Something wrong with a spec, named precisely enough to fix without guessing.
public struct Diagnostic: Sendable, Equatable, CustomStringConvertible {
    public enum Kind: String, Sendable {
        case unsupportedIRVersion = "unsupported-ir-version"
        case duplicateTensor = "duplicate-tensor"
        case unknownBlock = "unknown-block"
        case shapeMismatch = "shape-mismatch"
        case missingPolicy = "missing-policy"
        case layerCountMismatch = "layer-count-mismatch"
        case missingOutputHead = "missing-output-head"
    }

    public let kind: Kind
    /// What the diagnostic is about: a tensor name, a block id, a role.
    public let subject: String
    public let detail: String

    public init(kind: Kind, subject: String, detail: String) {
        self.kind = kind
        self.subject = subject
        self.detail = detail
    }

    public var description: String { "\(kind.rawValue): \(subject): \(detail)" }
}

public enum IRValidationError: Error, Equatable, Sendable {
    case invalid([Diagnostic])

    public var diagnostics: [Diagnostic] {
        switch self {
        case .invalid(let diagnostics): return diagnostics
        }
    }
}

public extension IRSpec {
    /// Every problem with this spec, not just the first — an importer that reports
    /// one error at a time turns a ten-second fix into ten runs.
    func diagnostics() -> [Diagnostic] {
        var found: [Diagnostic] = []

        if irVersion != Self.currentVersion {
            found.append(
                Diagnostic(
                    kind: .unsupportedIRVersion,
                    subject: "irVersion",
                    detail: "spec says \(irVersion), this reader implements \(Self.currentVersion)"
                )
            )
        }

        let blockIDs = Set(blocks.map(\.id))
        var seenNames = Set<String>()
        var rolesInUse = Set<TensorRole>()

        for tensor in tensors {
            if !seenNames.insert(tensor.name).inserted {
                found.append(
                    Diagnostic(kind: .duplicateTensor, subject: tensor.name,
                               detail: "listed more than once")
                )
            }
            if !blockIDs.contains(tensor.block) {
                found.append(
                    Diagnostic(kind: .unknownBlock, subject: tensor.name,
                               detail: "names block '\(tensor.block)', which the block list does not define")
                )
            }
            rolesInUse.insert(tensor.role)

            if let expected = tensor.role.expectedShape(config), expected != tensor.shape {
                found.append(
                    Diagnostic(
                        kind: .shapeMismatch,
                        subject: tensor.name,
                        detail: "role \(tensor.role.rawValue) expects \(expected), spec has \(tensor.shape)"
                    )
                )
            }

            if policy.quant(for: tensor.role) == nil {
                found.append(
                    Diagnostic(kind: .missingPolicy, subject: tensor.role.rawValue,
                               detail: "no quantization policy; I4 requires the policy to be data")
                )
            }
            if policy.shard(for: tensor.role) == nil {
                found.append(
                    Diagnostic(kind: .missingPolicy, subject: tensor.role.rawValue,
                               detail: "no sharding policy; I4 requires the policy to be data")
                )
            }
        }

        let decoderIndices = blocks.compactMap { $0.kind == .decoder ? $0.index : nil }
        if decoderIndices.count != config.numLayers {
            found.append(
                Diagnostic(
                    kind: .layerCountMismatch,
                    subject: "blocks",
                    detail: "\(decoderIndices.count) decoder block(s) for a \(config.numLayers)-layer configuration"
                )
            )
        }
        if Set(decoderIndices) != Set(0..<config.numLayers) {
            found.append(
                Diagnostic(kind: .layerCountMismatch, subject: "blocks",
                           detail: "decoder indices are not exactly 0..<\(config.numLayers)")
            )
        }

        // A separate head is required unless the weights are tied; when they are tied
        // the checkpoint may still ship one (Qwen3-0.6B does), which is legal.
        if !config.tieWordEmbeddings, !rolesInUse.contains(.outputHead) {
            found.append(
                Diagnostic(kind: .missingOutputHead, subject: "tensors",
                           detail: "tieWordEmbeddings is false but no tensor has role head.lm")
            )
        }

        return found
    }

    /// Throwing form, for callers that want to stop at the first problem.
    /// Typed throws rather than a generic `Error`, so a caller cannot forget which
    /// failure it is handling.
    func validate() throws(IRValidationError) {
        let found = diagnostics()
        if !found.isEmpty { throw .invalid(found) }
    }
}
