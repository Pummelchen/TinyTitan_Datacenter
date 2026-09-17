import Foundation

/// The wire format for head-logit slices (`D93`).
///
/// A different message from `ContributionWire`, and a different enum, because it is a different thing: expert
/// contributions are keyed terms that get *summed*, and a head slice is a run of vocabulary rows that gets
/// *placed*. Reusing the contribution frame would have meant inventing a key and excluding it from the sum,
/// which is how a format grows a special case.
///
/// The layout is little-endian and fixed, floats travel as their IEEE-754 bit patterns exactly as they do on
/// the contribution wire (`-0.0` stays `-0.0`, a NaN keeps its payload, nothing is flushed or normalised), and
/// there is no checksum for the reason `D15` gives: TCP checksums, the trace digest catches the rest, and a
/// per-frame hash costs more than it guards.
///
/// ```
/// magic "TTDH" · version u16 · start u32 · count u32 · count × f32
/// ```
public enum HeadSliceWire {
    public static let magic: [UInt8] = Array("TTDH".utf8)
    public static let version: UInt16 = 1
    public static let headerBytes = 14

    /// Bounds so a broken frame is refused **before** anything is allocated from its length field.
    public static let maximumRows = 1 << 22

    public enum Error: Swift.Error, CustomStringConvertible, Equatable {
        case tooShort(bytes: Int)
        case badMagic
        case unsupportedVersion(UInt16)
        case absurdCount(Int)
        case wrongLength(declared: Int, found: Int)

        public var description: String {
            switch self {
            case .tooShort(let bytes):
                return "a head-slice frame of \(bytes) byte(s) cannot hold its own header"
            case .badMagic:
                return "not a head-slice frame"
            case .unsupportedVersion(let version):
                return "head-slice frame version \(version) is not one this build writes"
            case .absurdCount(let count):
                return "a head-slice frame claiming \(count) row(s) is refused rather than allocated"
            case .wrongLength(let declared, let found):
                return "the frame declares \(declared) row(s) and carries \(found)"
            }
        }
    }

    public static func encode(start: Int, values: [Float]) throws -> Data {
        guard start >= 0, values.count <= maximumRows else { throw Error.absurdCount(values.count) }
        var bytes: [UInt8] = []
        bytes.reserveCapacity(headerBytes + values.count * 4)
        bytes.append(contentsOf: magic)
        append(UInt16(version), to: &bytes)
        append(UInt32(start), to: &bytes)
        append(UInt32(values.count), to: &bytes)
        for value in values { append(value.bitPattern, to: &bytes) }
        return Data(bytes)
    }

    public static func decode(_ data: Data) throws -> (start: Int, values: [Float]) {
        guard data.count >= headerBytes else { throw Error.tooShort(bytes: data.count) }
        let bytes = [UInt8](data)
        guard Array(bytes[0..<4]) == magic else { throw Error.badMagic }
        let version = UInt16(bytes[4]) | (UInt16(bytes[5]) << 8)
        guard version == Self.version else { throw Error.unsupportedVersion(version) }
        let start = Int(readUInt32(bytes, 6))
        let count = Int(readUInt32(bytes, 10))
        guard count <= maximumRows else { throw Error.absurdCount(count) }
        guard bytes.count == headerBytes + count * 4 else {
            throw Error.wrongLength(declared: count, found: (bytes.count - headerBytes) / 4)
        }
        var values: [Float] = []
        values.reserveCapacity(count)
        for index in 0..<count {
            let offset = headerBytes + index * 4
            values.append(Float(bitPattern: readUInt32(bytes, offset)))
        }
        return (start, values)
    }

    private static func append(_ value: UInt16, to bytes: inout [UInt8]) {
        bytes.append(UInt8(value & 0xFF))
        bytes.append(UInt8((value >> 8) & 0xFF))
    }

    private static func append(_ value: UInt32, to bytes: inout [UInt8]) {
        bytes.append(UInt8(value & 0xFF))
        bytes.append(UInt8((value >> 8) & 0xFF))
        bytes.append(UInt8((value >> 16) & 0xFF))
        bytes.append(UInt8((value >> 24) & 0xFF))
    }

    private static func readUInt32(_ bytes: [UInt8], _ offset: Int) -> UInt32 {
        UInt32(bytes[offset]) | (UInt32(bytes[offset + 1]) << 8)
            | (UInt32(bytes[offset + 2]) << 16) | (UInt32(bytes[offset + 3]) << 24)
    }
}
