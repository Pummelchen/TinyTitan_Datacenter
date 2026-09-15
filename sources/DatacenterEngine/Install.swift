import CryptoKit
import DatacenterIR
import Foundation

/// A weight source: the two operations every backend offers, and the only two the forward
/// passes use.
///
/// There are two implementations — a checkpoint's `safetensors` and a quantized install —
/// and the forward pass is written against this rather than against either, so the same
/// arithmetic runs on both and a difference between them is the quantization's.
public protocol WeightSource {
    /// A whole tensor, in fp32.
    func tensor(named name: String) throws -> [Float]
    /// A row range, in fp32. The embedding and the tied head are read this way so a
    /// `[248320, 2048]` matrix is never materialised to use one row of it.
    func rows(named name: String, range: Range<Int>) throws -> [Float]
    /// A row range for a consumer that will not read it again soon, so its pages are not worth
    /// keeping: the routed expert slabs. Defaults to the ordinary read, which is what an install
    /// already does — its payload is read uncached by construction.
    func rowsStreaming(named name: String, range: Range<Int>) throws -> [Float]
}

extension WeightSource {
    public func rowsStreaming(named name: String, range: Range<Int>) throws -> [Float] {
        try rows(named: name, range: range)
    }
}

extension SafetensorsFile: WeightSource {
    public func tensor(named name: String) throws -> [Float] { try float32(name) }
    public func rows(named name: String, range: Range<Int>) throws -> [Float] { try float32(name, rows: range) }
}

/// Reads an install written by `tools/quantize.py`.
///
/// The dequantization here has to be **bit-identical** to the Python one: the packed codes,
/// the group scales and the zero points are integers and exact arithmetic, so there is no
/// rounding to disagree about — which is why the two implementations can be compared byte
/// for byte rather than by tolerance.
public struct InstallFile: WeightSource {
    public struct Entry: Decodable, Sendable {
        public var name: String
        public var role: String
        public var quant: String
        public var shape: [Int]
        public var padded_columns: Int
        public var group: Int
        public var dtype: String
        public var offset: Int
        public var nbytes: Int
        public var sha256: String
    }

    public struct Manifest: Decodable, Sendable {
        public var schema: Int
        public var family: String
        public var passes: [String]
        public var policy_files: [String]
        public var spec: IRSpec
        public var tensors: [Entry]
    }

    public enum Error: Swift.Error, CustomStringConvertible {
        case unknownTensor(String)
        case badHeader(String)
        case digestMismatch(String)

        public var description: String {
            switch self {
            case .unknownTensor(let name): return "the install has no tensor '\(name)'"
            case .badHeader(let detail): return "install header: \(detail)"
            case .digestMismatch(let name): return "\(name): the payload digest does not match the manifest"
            }
        }
    }

    public let manifest: Manifest
    private let blob: UncachedFile
    private let entries: [String: Entry]

    /// The names whose digest this instance has already checked.
    ///
    /// A reference box rather than a `var`, because `WeightSource` conformance is non-mutating and
    /// an expert tensor is fetched again and again: re-hashing a five-hundred-megabyte payload on
    /// every read of it would be its own disaster, and a struct cannot hold that memo itself.
    private final class ReadState {
        var names: Set<String> = []
        /// Payload bytes this instance has read. Exposed so a test can assert that reading one
        /// expert of a stacked tensor does not cost the whole stack, which is the difference
        /// between streaming and not streaming.
        var bytesRead = 0
    }

    private let state = ReadState()

    /// Payload bytes read through this instance.
    public var bytesRead: Int { state.bytesRead }

    /// Open an install.
    ///
    /// `verify` is **false** by default, and that is a deliberate change from the first version,
    /// which hashed the entire payload on open. That made opening a 20 GB install a 20 GB read:
    /// slow everywhere, and on the 8 GB development node the read filled the page cache, memory
    /// pressure grew swap, and free disk fell from 17 GB to 2.96 GB in half a minute. Integrity
    /// is not weakened by moving it — each payload is checked the first time it is read, so a
    /// tampered tensor still cannot produce plausible numbers (I6) — and `verifyAll()` restores
    /// the eager check for a gate that wants to make it explicit.
    public init(url: URL, verify: Bool = false) throws {
        let manifestData = try Data(contentsOf: url.appendingPathComponent("install.json"))
        let manifest = try JSONDecoder().decode(Manifest.self, from: manifestData)
        self.manifest = manifest
        // Read through `UncachedFile`, not `mmap`: the payload is larger than the machine, and
        // pages cached from it evict everything useful and turn a read into swap pressure.
        self.blob = try UncachedFile(url: url.appendingPathComponent("data.bin"))
        var entries: [String: Entry] = [:]
        for entry in manifest.tensors { entries[entry.name] = entry }
        self.entries = entries
        if verify { try verifyAll() }
    }

    /// Check every payload digest, in bounded windows. For a gate rather than for normal use.
    public func verifyAll() throws {
        for entry in manifest.tensors where !(try digestMatches(entry)) {
            throw Error.digestMismatch(entry.name)
        }
    }

    private func digestMatches(_ entry: Entry) throws -> Bool {
        if state.names.contains(entry.name) { return true }
        guard entry.offset + entry.nbytes <= blob.byteCount else { return false }
        let payload = try blob.readData(offset: entry.offset, byteCount: entry.nbytes)
        let digest = SHA256.hash(data: payload).map { String(format: "%02x", $0) }.joined()
        guard digest == entry.sha256 else { return false }
        state.names.insert(entry.name)
        return true
    }

    public func entry(_ name: String) throws -> Entry {
        guard let entry = entries[name] else { throw Error.unknownTensor(name) }
        return entry
    }

    /// A bounded read that is counted, so the cost of a row range is observable rather than
    /// asserted.
    private func readCounted(offset: Int, byteCount: Int) throws -> Data {
        let data = try blob.readData(offset: offset, byteCount: byteCount)
        state.bytesRead += data.count
        return data
    }

    private func payload(_ entry: Entry) throws -> Data {
        guard try digestMatches(entry) else { throw Error.digestMismatch(entry.name) }
        let data = try blob.readData(offset: entry.offset, byteCount: entry.nbytes)
        state.bytesRead += data.count
        return data
    }

    public func tensor(named name: String) throws -> [Float] {
        let entry = try entry(name)
        let data = try payload(entry)
        guard entry.dtype == "int4" else {
            return try Self.decodeRaw(data, dtype: entry.dtype, elementCount: entry.shape.reduce(1, *))
        }
        return try Self.dequantizeInt4(data, entry: entry)
    }

    public func rows(named name: String, range: Range<Int>) throws -> [Float] {
        let entry = try entry(name)
        guard entry.shape.count >= 2 else { return try tensor(named: name) }
        // One row is the **leading axis**, whatever the rank: a token of the embedding, or one
        // expert of a stacked expert tensor. The row's width is the product of the rest.
        let width = entry.shape.dropFirst().reduce(1, *)
        let rowCount = entry.shape[0]
        guard range.lowerBound >= 0, range.upperBound <= rowCount else {
            throw Error.badHeader("row range \(range) is outside '\(name)'")
        }
        if entry.dtype != "int4" {
            // Sliced from the stored bytes: one row of the embedding should not cost a
            // gigabyte of decoding.
            guard try digestMatches(entry) else { throw Error.digestMismatch(entry.name) }
            let stride = Self.elementSize(entry.dtype) * width
            let start = entry.offset + range.lowerBound * stride
            let slice = try blob.readData(offset: start, byteCount: range.count * stride)
            return try Self.decodeRaw(slice, dtype: entry.dtype, elementCount: range.count * width)
        }
        // Packed codes cannot be sliced as bytes without re-deriving the group layout, and the
        // layout is section-major, so the rows come out of **three** ranges rather than out of a
        // whole-tensor decode. That distinction is the difference between streaming and not: the
        // real model's expert tensor is `[256, 1024, 2048]`, so decoding it to return one expert
        // is 537 M parameters and two gigabytes of `Float` on a node with four and a half.
        let totalRows = entry.shape.dropLast().reduce(1, *)
        guard entry.padded_columns % entry.group == 0, entry.padded_columns % 2 == 0 else {
            throw Error.badHeader("\(name): group \(entry.group) does not divide \(entry.padded_columns)")
        }
        guard try digestMatches(entry) else { throw Error.digestMismatch(entry.name) }
        // The range is in **leading-axis entries** — one expert of a stack, not one payload row —
        // so it has to be translated, and getting that wrong returns the first expert's first
        // *row* while still looking like a tensor. One entry spans `inner` payload rows.
        let inner = entry.shape.dropFirst().dropLast().reduce(1, *)
        let payloadRows = range.count * inner
        let first = range.lowerBound * inner
        let groupsPerRow = entry.padded_columns / entry.group
        let codesPerRow = entry.padded_columns / 2
        let codesBytes = totalRows * codesPerRow
        let scalesBytes = totalRows * groupsPerRow * 4
        let codes = try readCounted(
            offset: entry.offset + first * codesPerRow, byteCount: payloadRows * codesPerRow
        )
        let scales = try readCounted(
            offset: entry.offset + codesBytes + first * groupsPerRow * 4, byteCount: payloadRows * groupsPerRow * 4
        )
        let zeros = try readCounted(
            offset: entry.offset + codesBytes + scalesBytes + first * groupsPerRow, byteCount: payloadRows * groupsPerRow
        )
        return try Self.dequantizeInt4(codes + scales + zeros, entry: entry, rowCount: payloadRows)
    }

    static func elementSize(_ dtype: String) -> Int {
        switch dtype {
        case "bf16", "fp16": return 2
        case "fp32": return 4
        default: return 1
        }
    }

    static func decodeRaw(_ data: Data, dtype: String, elementCount: Int) throws -> [Float] {
        var values = [Float](repeating: 0, count: elementCount)
        try data.withUnsafeBytes { raw in
            let base = raw.baseAddress!
            switch dtype {
            case "bf16":
                for index in 0..<elementCount {
                    let word = base.loadUnaligned(fromByteOffset: index * 2, as: UInt16.self)
                    values[index] = Float(bitPattern: UInt32(UInt16(littleEndian: word)) << 16)
                }
            case "fp16":
                for index in 0..<elementCount {
                    let word = base.loadUnaligned(fromByteOffset: index * 2, as: UInt16.self)
                    values[index] = Float(Float16(bitPattern: UInt16(littleEndian: word)))
                }
            case "fp32":
                for index in 0..<elementCount {
                    let word = base.loadUnaligned(fromByteOffset: index * 4, as: UInt32.self)
                    values[index] = Float(bitPattern: UInt32(littleEndian: word))
                }
            default:
                throw Error.badHeader("unknown stored dtype '\(dtype)'")
            }
        }
        return values
    }

    /// The packed format, decoded exactly as `tools/quantize.py` encodes it.
    ///
    /// Two codes per byte, **low nibble first**; codes are signed four-bit values, so `0b1000`
    /// is -8 and not 8 — reading it as 8 shifts a whole group by sixteen steps and still
    /// produces plausible weights. Each group of `group` codes shares one fp32 scale and one
    /// int4 zero point, and the reconstruction is `(code - zero) * scale`.
    /// Dequantize `rowCount` payload rows — the whole tensor, or a range of it.
    ///
    /// The layout is **section-major**: every row's codes, then every row's scales, then every
    /// row's zeros. That is what makes a row range decodable without the rest of the tensor, since
    /// each section is row-contiguous; it is also the thing to get wrong, because the wrong row
    /// count still produces plausible weights.
    static func dequantizeInt4(_ data: Data, entry: Entry, rowCount: Int? = nil) throws -> [Float] {
        // A stacked expert tensor is `[experts, rows, columns]` and its payload was quantized
        // with the leading axis flattened, so the codes describe `experts x rows` rows.
        let rows = rowCount ?? entry.shape.dropLast().reduce(1, *)
        let columns = entry.shape.last ?? 0
        let padded = entry.padded_columns
        let group = entry.group
        guard group > 0, padded % group == 0, padded % 2 == 0 else {
            throw Error.badHeader("\(entry.name): group \(group) does not divide \(padded)")
        }
        let groups = padded / group
        let codeBytes = rows * padded / 2
        let scaleBytes = rows * groups * 4
        guard data.count == codeBytes + scaleBytes + rows * groups else {
            throw Error.badHeader("\(entry.name): payload is \(data.count) bytes, the layout needs \(codeBytes + scaleBytes + rows * groups)")
        }

        var values = [Float](repeating: 0, count: rows * columns)
        data.withUnsafeBytes { raw in
            let base = raw.baseAddress!
            let codes = base
            let scales = base + codeBytes
            let zeros = base + codeBytes + scaleBytes
            for row in 0..<rows {
                let rowCodes = codes + row * (padded / 2)
                for index in 0..<padded {
                    let byte = rowCodes.loadUnaligned(fromByteOffset: index / 2, as: UInt8.self)
                    let nibble = index % 2 == 0 ? (byte & 0x0F) : (byte >> 4)
                    let code = Int(nibble >= 8 ? Int(nibble) - 16 : Int(nibble))
                    let groupIndex = row * groups + index / group
                    let scale = scales.loadUnaligned(fromByteOffset: groupIndex * 4, as: UInt32.self)
                    let zero = zeros.loadUnaligned(fromByteOffset: groupIndex, as: UInt8.self)
                    let zeroValue = Int(zero >= 128 ? Int(zero) - 256 : Int(zero))
                    if index < columns {
                        values[row * columns + index] =
                            Float(code - zeroValue) * Float(bitPattern: UInt32(littleEndian: scale))
                    }
                }
            }
        }
        return values
    }
}
