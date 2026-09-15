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
    private let blob: Data
    private let entries: [String: Entry]

    public init(url: URL) throws {
        let manifestData = try Data(contentsOf: url.appendingPathComponent("install.json"))
        let manifest = try JSONDecoder().decode(Manifest.self, from: manifestData)
        self.manifest = manifest
        // Memory-mapped: an install is around two gigabytes and the engine uses one layer of
        // it at a time.
        self.blob = try Data(contentsOf: url.appendingPathComponent("data.bin"), options: [.mappedIfSafe])
        var entries: [String: Entry] = [:]
        for entry in manifest.tensors { entries[entry.name] = entry }
        self.entries = entries
        for entry in manifest.tensors where !Self.digestMatches(entry, blob: blob) {
            throw Error.digestMismatch(entry.name)
        }
    }

    private static func digestMatches(_ entry: Entry, blob: Data) -> Bool {
        guard entry.offset + entry.nbytes <= blob.count else { return false }
        let payload = blob.subdata(in: entry.offset..<(entry.offset + entry.nbytes))
        return SHA256.hash(data: payload).map { String(format: "%02x", $0) }.joined() == entry.sha256
    }

    public func entry(_ name: String) throws -> Entry {
        guard let entry = entries[name] else { throw Error.unknownTensor(name) }
        return entry
    }

    private func payload(_ entry: Entry) -> Data {
        blob.subdata(in: entry.offset..<(entry.offset + entry.nbytes))
    }

    public func tensor(named name: String) throws -> [Float] {
        let entry = try entry(name)
        let data = payload(entry)
        guard entry.dtype == "int4" else {
            return try Self.decodeRaw(data, dtype: entry.dtype, elementCount: entry.shape.reduce(1, *))
        }
        return try Self.dequantizeInt4(data, entry: entry)
    }

    public func rows(named name: String, range: Range<Int>) throws -> [Float] {
        let entry = try entry(name)
        guard entry.shape.count == 2 else { return try tensor(named: name) }
        let width = entry.shape[1]
        guard range.lowerBound >= 0, range.upperBound <= entry.shape[0] else {
            throw Error.badHeader("row range \(range) is outside '\(name)'")
        }
        if entry.dtype != "int4" {
            // Sliced from the stored bytes: one row of the embedding should not cost a
            // gigabyte of decoding.
            let stride = Self.elementSize(entry.dtype) * width
            let start = entry.offset + range.lowerBound * stride
            let slice = blob.subdata(in: start..<(start + range.count * stride))
            return try Self.decodeRaw(slice, dtype: entry.dtype, elementCount: range.count * width)
        }
        // Packed codes cannot be sliced as bytes without re-deriving the group layout, so
        // the whole tensor is decoded and the rows taken. The quantized roles are never the
        // embedding or the head, which is what this shortcut exists for.
        let all = try tensor(named: name)
        return Array(all[(range.lowerBound * width)..<(range.upperBound * width)])
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
    static func dequantizeInt4(_ data: Data, entry: Entry) throws -> [Float] {
        let rows = entry.shape[0]
        let columns = entry.shape[1]
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
