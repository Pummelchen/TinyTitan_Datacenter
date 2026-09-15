import CryptoKit
import Foundation

/// Writes a trace in the project's container format: a 64-byte-aligned `data.bin` plus a
/// `manifest.json` whose digest covers every tensor's identity and hash.
///
/// This is the Swift half of `tools/trace_format.py`, and it has to agree with it
/// byte-for-byte — including the canonical digest, which is computed over a
/// key-sorted, whitespace-free JSON encoding of the index. If the two disagreed, the
/// differ's digest shortcut would compare two encodings instead of two computations, and
/// a matching trace would look like a mismatch.
public struct TraceWriter {
    public static let alignment = 64
    public static let schema = 1

    public struct Tensor: Sendable {
        public let name: String
        public let shape: [Int]
        public let values: [Float]

        public init(name: String, shape: [Int], values: [Float]) {
            self.name = name
            self.shape = shape
            self.values = values
        }
    }

    public struct Entry: Codable, Equatable, Sendable {
        public var name: String
        public var file: String
        public var offset: Int
        public var nbytes: Int
        public var shape: [Int]
        public var dtype: String
        public var sha256: String
    }

    public struct Discrete: Codable, Equatable, Sendable {
        public var name: String
        public var shape: [Int]
        public var values: [Int]
    }

    public struct Manifest: Codable, Sendable {
        public var schema: Int
        public var producer: String
        public var model: [String: String]
        public var reference: [String: String]
        public var prompt: [String: String]
        public var tensors: [Entry]
        public var discrete: [Discrete]
        public var digest: String
    }

    public let producer: String
    public var model: [String: String]
    /// The discrete decisions — router top-k, block selection — kept apart from the
    /// numbers, because I3 asserts them separately from any tolerance.
    public var discrete: [Discrete]

    public init(producer: String, model: [String: String] = [:]) {
        self.producer = producer
        self.model = model
        self.discrete = []
    }

    /// Write a trace directory. Returns the manifest, so a caller can record the digest.
    @discardableResult
    public func write(to root: URL, tensors: [Tensor], prompt: [String: String] = [:]) throws -> Manifest {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)

        var blob = Data()
        var index: [Entry] = []
        index.reserveCapacity(tensors.count)

        for tensor in tensors {
            let payload = Self.payload(of: tensor.values)
            let padding = (Self.alignment - blob.count % Self.alignment) % Self.alignment
            if padding > 0 { blob.append(contentsOf: [UInt8](repeating: 0, count: padding)) }
            let offset = blob.count
            blob.append(payload)
            index.append(
                Entry(
                    name: tensor.name, file: "data.bin", offset: offset, nbytes: payload.count,
                    shape: tensor.shape, dtype: "f32", sha256: Self.sha256(payload)
                )
            )
        }

        var manifest = Manifest(
            schema: Self.schema, producer: producer, model: model, reference: [:], prompt: prompt,
            tensors: index, discrete: discrete, digest: ""
        )
        manifest.digest = Self.canonicalDigest(manifest)

        try blob.write(to: root.appendingPathComponent("data.bin"))

        let encoder = JSONEncoder()
        // Sorted keys and indentation match the Python writer's manifest exactly. The
        // digest does not depend on this formatting, but a manifest that looks different
        // for no reason is a trap for the next reader.
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        var text = String(decoding: try encoder.encode(manifest), as: UTF8.self)
        if !text.hasSuffix("\n") { text += "\n" }
        try text.write(to: root.appendingPathComponent("manifest.json"), atomically: true, encoding: .utf8)
        return manifest
    }

    static func payload(of values: [Float]) -> Data {
        var data = Data(capacity: values.count * 4)
        for value in values { withUnsafeBytes(of: value.bitPattern.littleEndian) { data.append(contentsOf: $0) } }
        return data
    }

    static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    /// The digest covers the schema, each tensor's identity and hash, and the discrete
    /// decisions — deliberately not the producer string or a timestamp, so two runs that
    /// computed the same numbers digest identically.
    static func canonicalDigest(_ manifest: Manifest) -> String {
        struct CanonicalTensor: Encodable {
            let name: String
            let shape: [Int]
            let dtype: String
            let sha256: String
        }
        struct Canonical: Encodable {
            let schema: Int
            let tensors: [CanonicalTensor]
            let discrete: [Discrete]
        }
        let canonical = Canonical(
            schema: manifest.schema,
            tensors: manifest.tensors.map {
                CanonicalTensor(name: $0.name, shape: $0.shape, dtype: $0.dtype, sha256: $0.sha256)
            },
            discrete: manifest.discrete
        )
        // `JSONEncoder` emits no whitespace and, with `.sortedKeys`, in the same order as
        // Python's `json.dumps(sort_keys=True, separators=(",", ":"))`.
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        let data = (try? encoder.encode(canonical)) ?? Data()
        return sha256(data)
    }
}
