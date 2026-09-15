import Foundation

/// A reader for the `safetensors` container the checkpoints ship.
///
/// The format is a little-endian `UInt64` header length, that many bytes of JSON mapping
/// each tensor name to `{dtype, shape, data_offsets}`, then the raw tensor bytes. The file
/// is memory-mapped rather than read: a 2 B-parameter checkpoint is 5 GB and the engine
/// must be able to touch a few tensors of it without committing the rest.
public struct SafetensorsFile {
    public struct Info: Sendable, Equatable {
        public let dtype: String
        public let shape: [Int]
        /// Byte range inside the data section (which begins after the header).
        public let range: Range<Int>

        public var elementCount: Int { shape.reduce(1, *) }
    }

    public enum Error: Swift.Error, CustomStringConvertible {
        case truncatedFile(Int)
        case malformedHeader(String)
        case unknownDtype(String)
        case unknownTensor(String)

        public var description: String {
            switch self {
            case .truncatedFile(let count): return "safetensors file is truncated (\(count) bytes)"
            case .malformedHeader(let detail): return "safetensors header is malformed: \(detail)"
            case .unknownDtype(let dtype): return "unsupported dtype '\(dtype)'"
            case .unknownTensor(let name): return "no tensor named '\(name)'"
            }
        }
    }

    private let mapped: Data
    private let dataStart: Int
    public let tensors: [String: Info]
    public let headerBytes: Int
    /// The header's optional `__metadata__` block. It is not a tensor, and the format
    /// reserves the name for it with a different value shape — a reader that treats it as
    /// a tensor fails on every real checkpoint.
    public let metadata: [String: String]

    public init(url: URL) throws {
        // `.mappedIfSafe` keeps the checkpoint out of the resident set: the reader pages in
        // only what it decodes, which is what makes a 5 GB file usable on an 8 GB node.
        let mapped = try Data(contentsOf: url, options: [.mappedIfSafe])
        guard mapped.count >= 8 else { throw Error.truncatedFile(mapped.count) }

        let headerLength = Int(mapped.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 0, as: UInt64.self) })
        guard headerLength > 0, 8 + headerLength <= mapped.count else {
            throw Error.malformedHeader("header length \(headerLength) does not fit the file")
        }
        let headerBytes = 8 + headerLength
        self.headerBytes = headerBytes
        self.dataStart = headerBytes
        self.mapped = mapped

        let header = mapped.subdata(in: 8..<headerBytes)
        guard let parsed = try JSONSerialization.jsonObject(with: header) as? [String: Any] else {
            throw Error.malformedHeader("expected an object of tensor descriptions")
        }

        var tensors: [String: Info] = [:]
        tensors.reserveCapacity(parsed.count)
        var metadata: [String: String] = [:]
        for (name, value) in parsed {
            if name == "__metadata__" {
                metadata = (value as? [String: String]) ?? [:]
                continue
            }
            guard let description = value as? [String: Any],
                  let dtype = description["dtype"] as? String,
                  let shape = description["shape"] as? [Int],
                  let offsets = description["data_offsets"] as? [Int], offsets.count == 2
            else {
                throw Error.malformedHeader("tensor '\(name)' is missing dtype, shape or data_offsets")
            }
            tensors[name] = Info(dtype: dtype, shape: shape, range: offsets[0]..<offsets[1])
        }
        self.tensors = tensors
        self.metadata = metadata
    }

    public func info(_ name: String) throws -> Info {
        guard let info = tensors[name] else { throw Error.unknownTensor(name) }
        return info
    }

    public var names: [String] { tensors.keys.sorted() }

    /// A tensor's values as `Float`, converting from whatever the checkpoint stores.
    ///
    /// bf16 → fp32 is exact (the widening is a shift), so this conversion cannot introduce
    /// the kind of error a resize would; that is why M0 can load bf16 weights and still
    /// compute in fp32.
    public func float32(_ name: String) throws -> [Float] {
        let info = try info(name)
        let start = dataStart + info.range.lowerBound
        let byteCount = info.range.count
        guard start + byteCount <= mapped.count else { throw Error.truncatedFile(mapped.count) }

        return try mapped.withUnsafeBytes { raw -> [Float] in
            let base = raw.baseAddress! + start
            // Checkpoints use upper case (`BF16`) even though the format's own examples use
            // lower case, so the comparison normalises rather than assuming.
            switch info.dtype.lowercased() {
            case "f32":
                let count = byteCount / 4
                var values = [Float](repeating: 0, count: count)
                for index in 0..<count {
                    let word = base.loadUnaligned(fromByteOffset: index * 4, as: UInt32.self)
                    values[index] = Float(bitPattern: UInt32(littleEndian: word))
                }
                return values
            case "bf16":
                let count = byteCount / 2
                var values = [Float](repeating: 0, count: count)
                for index in 0..<count {
                    let word = base.loadUnaligned(fromByteOffset: index * 2, as: UInt16.self)
                    values[index] = Float(bitPattern: UInt32(UInt16(littleEndian: word)) << 16)
                }
                return values
            case "f16":
                let count = byteCount / 2
                var values = [Float](repeating: 0, count: count)
                for index in 0..<count {
                    let word = base.loadUnaligned(fromByteOffset: index * 2, as: UInt16.self)
                    values[index] = Float(Float16(bitPattern: UInt16(littleEndian: word)))
                }
                return values
            default:
                throw Error.unknownDtype(info.dtype)
            }
        }
    }
}
