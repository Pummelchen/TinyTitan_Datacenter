import Foundation

/// The wire format for expert contributions (`D18`).
///
/// What crosses the network is **terms**, not per-node partial sums (`D17`), and every field is
/// encoded so that the receiving side reconstructs the *same bits*: floats travel as their IEEE-754
/// bit patterns, so `-0.0` stays `-0.0`, a NaN keeps its payload, and no value is ever normalised,
/// canonicalised or flushed on the way. The wire is the one place where "close enough" would be
/// invisible, because it would look like arithmetic.
///
/// The layout is little-endian and fixed:
///
/// ```
/// magic "TTDC" · version u16 · tokens u32 · hiddenSize u32 · count u32
/// count × ( token u32 · expert u32 · width u32 · scale f32 · width × f32 )
/// ```
///
/// **No checksum, deliberately.** `D15` removed per-read hashing from the hot path because it cost
/// more than the I/O it guarded; a per-frame hash here would be the same mistake with a smaller
/// payload. The transport is TCP, which checksums; a frame that arrives corrupt but plausible is
/// caught by the completeness guard (`OrderedReduction.isComplete`) and by the trace digest, and a
/// checksum that duplicates both is not worth a millisecond per layer.
public enum ContributionWire {
    public static let magic: [UInt8] = Array("TTDC".utf8)
    public static let version: UInt16 = 1
    public static let headerBytes = 18

    /// Bounds that exist so a hostile or broken frame is refused **before** anything is allocated
    /// from a length field. A decoder that trusts a length is a denial-of-service with extra steps.
    public static let maximumDimension = 1 << 20
    public static let maximumTerms = 1 << 20

    public static func encode(
        _ contributions: [ExpertContribution], tokens: Int, hiddenSize: Int
    ) throws -> Data {
        guard tokens >= 0, tokens <= maximumDimension else { throw ContributionWireError.absurdDimension(tokens) }
        guard hiddenSize > 0, hiddenSize <= maximumDimension else {
            throw ContributionWireError.absurdDimension(hiddenSize)
        }
        guard contributions.count <= maximumTerms else { throw ContributionWireError.absurdCount(contributions.count) }
        for contribution in contributions
        where contribution.token < 0 || contribution.token >= tokens || contribution.values.count != hiddenSize {
            throw ContributionWireError.malformedTerm(
                token: contribution.token, width: contribution.values.count, hiddenSize: hiddenSize
            )
        }

        var bytes: [UInt8] = []
        bytes.reserveCapacity(headerBytes + contributions.count * (16 + hiddenSize * 4))
        bytes.append(contentsOf: magic)
        append(UInt16(version), to: &bytes)
        append(UInt32(tokens), to: &bytes)
        append(UInt32(hiddenSize), to: &bytes)
        append(UInt32(contributions.count), to: &bytes)
        for contribution in contributions {
            append(UInt32(contribution.token), to: &bytes)
            append(UInt32(contribution.expert), to: &bytes)
            append(UInt32(contribution.values.count), to: &bytes)
            append(contribution.scale.bitPattern, to: &bytes)
            for value in contribution.values { append(value.bitPattern, to: &bytes) }
        }
        return Data(bytes)
    }

    public static func decode(_ data: Data) throws -> [ExpertContribution] {
        let bytes = [UInt8](data)
        guard bytes.count >= headerBytes else {
            throw ContributionWireError.truncated(needed: headerBytes, got: bytes.count)
        }
        guard Array(bytes[0..<4]) == magic else { throw ContributionWireError.badMagic }
        var cursor = 4
        let version = try nextUInt16(bytes, &cursor)
        guard version == Self.version else { throw ContributionWireError.unsupportedVersion(version) }
        let tokens = Int(try nextUInt32(bytes, &cursor))
        let hiddenSize = Int(try nextUInt32(bytes, &cursor))
        guard tokens <= maximumDimension, hiddenSize > 0, hiddenSize <= maximumDimension else {
            throw ContributionWireError.absurdDimension(max(tokens, hiddenSize))
        }
        let count = Int(try nextUInt32(bytes, &cursor))
        guard count <= maximumTerms else { throw ContributionWireError.absurdCount(count) }

        var contributions: [ExpertContribution] = []
        contributions.reserveCapacity(count)
        for _ in 0..<count {
            let token = Int(try nextUInt32(bytes, &cursor))
            let expert = Int(try nextUInt32(bytes, &cursor))
            let width = Int(try nextUInt32(bytes, &cursor))
            guard width == hiddenSize else {
                throw ContributionWireError.malformedTerm(token: token, width: width, hiddenSize: hiddenSize)
            }
            guard token >= 0, token < tokens else {
                throw ContributionWireError.malformedTerm(token: token, width: width, hiddenSize: hiddenSize)
            }
            let scale = Float(bitPattern: try nextUInt32(bytes, &cursor))
            var values: [Float] = []
            values.reserveCapacity(width)
            for _ in 0..<width { values.append(Float(bitPattern: try nextUInt32(bytes, &cursor))) }
            contributions.append(
                ExpertContribution(token: token, expert: expert, values: values, scale: scale)
            )
        }
        guard cursor == bytes.count else { throw ContributionWireError.trailingBytes(bytes.count - cursor) }
        return contributions
    }

    // MARK: - little-endian primitives

    private static func append(_ value: UInt16, to bytes: inout [UInt8]) {
        let little = value.littleEndian
        bytes.append(UInt8(truncatingIfNeeded: little))
        bytes.append(UInt8(truncatingIfNeeded: little >> 8))
    }

    private static func append(_ value: UInt32, to bytes: inout [UInt8]) {
        let little = value.littleEndian
        bytes.append(UInt8(truncatingIfNeeded: little))
        bytes.append(UInt8(truncatingIfNeeded: little >> 8))
        bytes.append(UInt8(truncatingIfNeeded: little >> 16))
        bytes.append(UInt8(truncatingIfNeeded: little >> 24))
    }

    private static func nextUInt16(_ bytes: [UInt8], _ cursor: inout Int) throws -> UInt16 {
        guard cursor + 2 <= bytes.count else {
            throw ContributionWireError.truncated(needed: cursor + 2, got: bytes.count)
        }
        defer { cursor += 2 }
        return UInt16(bytes[cursor]) | (UInt16(bytes[cursor + 1]) << 8)
    }

    private static func nextUInt32(_ bytes: [UInt8], _ cursor: inout Int) throws -> UInt32 {
        guard cursor + 4 <= bytes.count else {
            throw ContributionWireError.truncated(needed: cursor + 4, got: bytes.count)
        }
        defer { cursor += 4 }
        return UInt32(bytes[cursor]) | (UInt32(bytes[cursor + 1]) << 8)
            | (UInt32(bytes[cursor + 2]) << 16) | (UInt32(bytes[cursor + 3]) << 24)
    }
}

public enum ContributionWireError: Swift.Error, CustomStringConvertible, Equatable {
    case badMagic
    case unsupportedVersion(UInt16)
    case truncated(needed: Int, got: Int)
    case absurdDimension(Int)
    case absurdCount(Int)
    case malformedTerm(token: Int, width: Int, hiddenSize: Int)
    case trailingBytes(Int)

    public var description: String {
        switch self {
        case .badMagic:
            return "not a contribution frame: the magic bytes are wrong"
        case .unsupportedVersion(let version):
            return "frame version \(version) is not version \(ContributionWire.version)"
        case .truncated(let needed, let got):
            return "frame is truncated: it needs \(needed) bytes and has \(got)"
        case .absurdDimension(let value):
            return "frame declares a dimension of \(value), which no honest frame can carry"
        case .absurdCount(let count):
            return "frame declares \(count) terms, which no honest frame can carry"
        case .malformedTerm(let token, let width, let hiddenSize):
            return "term for token \(token) has width \(width) where the frame declares \(hiddenSize)"
        case .trailingBytes(let count):
            return "frame has \(count) bytes after the last term"
        }
    }
}
