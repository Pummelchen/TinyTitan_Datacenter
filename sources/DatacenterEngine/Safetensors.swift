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
    private let url: URL
    private let dataStart: Int

    /// The uncached handle for streaming reads, opened on first use.
    ///
    /// A reference box rather than a `lazy var`, because `WeightSource` conformance is
    /// non-mutating and a struct's lazy properties can only be touched from a mutating context.
    private final class StreamHandle {
        var file: UncachedFile?
        var failed = false
    }

    private let stream = StreamHandle()
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
        self.url = url

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

    /// A tensor's values as `Float`, restricted to a row range.
    ///
    /// This is what makes an 8 GB node able to run a 2 B model: the embedding is
    /// `[248320, 2048]` — 2 GB in fp32 — and a single token needs one row of it. The
    /// output head is the same matrix, so the logits are computed a block of rows at a
    /// time rather than by materialising the whole matrix to multiply once.
    public func float32(_ name: String, rows: Range<Int>) throws -> [Float] {
        let info = try info(name)
        guard info.shape.count >= 2 else { throw Error.malformedHeader("tensor '\(name)' is not row-addressable") }
        let width = info.shape.dropFirst().reduce(1, *)
        let total = info.shape[0]
        guard rows.lowerBound >= 0, rows.upperBound <= total else {
            throw Error.malformedHeader("row range \(rows) is outside '\(name)' (\(total) rows)")
        }
        let elementSize = try Self.elementSize(info.dtype)
        let rowBytes = width * elementSize
        let start = dataStart + info.range.lowerBound + rows.lowerBound * rowBytes
        let byteCount = rows.count * rowBytes
        guard start + byteCount <= mapped.count else { throw Error.truncatedFile(mapped.count) }

        return try mapped.withUnsafeBytes { raw in
            try Self.decodeRaw(
                UnsafeRawBufferPointer(rebasing: raw[start...]),
                dtype: info.dtype,
                elementCount: rows.count * width
            )
        }
    }

    /// A row range read that **bypasses the buffer cache**.
    ///
    /// The same values as `float32(_:rows:)`, read through `UncachedFile` instead of the mapping.
    /// The distinction is which consumer is asking, not which tensor it is: a routed expert slab
    /// is read once per token and would evict everything useful from the page cache, while the
    /// embedding and the head are read every token and are exactly what the cache is for. So the
    /// *streaming* consumer asks for this and the dense one keeps the mapping.
    ///
    /// Falls back to the mapped read if the second descriptor cannot be opened, because a
    /// fallback that is slower is better than a read that fails.
    public func rowsStreaming(named name: String, range: Range<Int>) throws -> [Float] {
        let info = try info(name)
        guard info.shape.count >= 2 else { return try float32(name) }
        let width = info.shape.dropFirst().reduce(1, *)
        let total = info.shape[0]
        guard range.lowerBound >= 0, range.upperBound <= total else {
            throw Error.malformedHeader("row range \(range) is outside '\(name)' (\(total) rows)")
        }
        let elementSize = try Self.elementSize(info.dtype)
        let rowBytes = width * elementSize
        let start = dataStart + info.range.lowerBound + range.lowerBound * rowBytes
        let byteCount = range.count * rowBytes

        guard let file = streamingHandle() else { return try float32(name, rows: range) }
        let data = try file.readData(offset: start, byteCount: byteCount)
        return try data.withUnsafeBytes {
            try Self.decodeRaw($0, dtype: info.dtype, elementCount: range.count * width)
        }
    }

    private func streamingHandle() -> UncachedFile? {
        if let file = stream.file { return file }
        if stream.failed { return nil }
        do {
            let file = try UncachedFile(url: url)
            stream.file = file
            return file
        } catch {
            // Remembered, so a missing or unreadable file does not retry on every row read.
            stream.failed = true
            return nil
        }
    }

    /// Decode `elementCount` elements of `dtype` from the front of `raw`. The single decoder all
    /// three read paths share, so mapped, uncached and whole-tensor reads cannot drift apart.
    static func decodeRaw(_ raw: UnsafeRawBufferPointer, dtype: String, elementCount: Int) throws -> [Float] {
        let base = raw.baseAddress!
        var values = [Float](repeating: 0, count: elementCount)
        switch dtype.lowercased() {
        case "f32":
            for index in 0..<elementCount {
                let word = base.loadUnaligned(fromByteOffset: index * 4, as: UInt32.self)
                values[index] = Float(bitPattern: UInt32(littleEndian: word))
            }
        case "bf16":
            for index in 0..<elementCount {
                let word = base.loadUnaligned(fromByteOffset: index * 2, as: UInt16.self)
                values[index] = Float(bitPattern: UInt32(UInt16(littleEndian: word)) << 16)
            }
        case "f16":
            for index in 0..<elementCount {
                let word = base.loadUnaligned(fromByteOffset: index * 2, as: UInt16.self)
                values[index] = Float(Float16(bitPattern: UInt16(littleEndian: word)))
            }
        default:
            throw Error.unknownDtype(dtype)
        }
        return values
    }

    static func elementSize(_ dtype: String) throws -> Int {
        switch dtype.lowercased() {
        case "f32", "i32": return 4
        case "bf16", "f16": return 2
        case "i64": return 8
        case "u8": return 1
        default: throw Error.unknownDtype(dtype)
        }
    }


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
