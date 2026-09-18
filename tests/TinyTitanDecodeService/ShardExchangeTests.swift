import Foundation
import Testing

@testable import TinyTitanDecodeProtocol

/// The exchange frames. These are worth testing carefully because a mistake here is **silent**: a contribution
/// put in the wrong slot, or a payload read at the wrong offset, produces a number that is merely wrong rather
/// than an error — and `D168` established the reduce is a fixed-order fp32 accumulation, so "merely wrong" is
/// exactly what must not happen.
@Suite("Shard exchange")
struct ShardExchangeTests {
    private static func request() -> ShardExchange.Request {
        // The real shapes: D = 2048, and a routed set straddling two owners.
        let activation = (0..<2048).map { Float($0) * 0.001 - 1.0 }
        return ShardExchange.Request(layer: 7, slots: [1, 3], experts: [100, 200], activation: activation)
    }

    private static func reply() -> ShardExchange.Reply {
        // Two slots of 2048 floats, values that are exactly representable so a byte-level round trip is exact.
        let values = (0..<(2 * 2048)).map { Float($0) * 0.25 }
        return ShardExchange.Reply(layer: 7, slots: [1, 3], dimensions: 2048, values: values)
    }

    @Test("a request round-trips with every float bit-identical")
    func requestRoundTripsBitExactly() throws {
        let original = Self.request()
        let decoded = try ShardExchange.decodeRequest(from: try ShardExchange.encode(original))
        #expect(decoded == original)
        // Bit patterns, not equality of values: `-0.0 == 0.0` is true and they are different bytes, which is the
        // same trap D34 hit when a normalisation was folded away by the optimiser.
        #expect(decoded.activation.map(\.bitPattern) == original.activation.map(\.bitPattern))
    }

    @Test("a reply round-trips bit-identically and keeps slot order")
    func replyRoundTripsBitExactly() throws {
        let original = Self.reply()
        let decoded = try ShardExchange.decodeReply(from: try ShardExchange.encode(original))
        #expect(decoded == original)
        #expect(decoded.slots == [1, 3], "slot order is the reduce order and must survive the wire")
        #expect(Array(decoded.row(at: 1)) == Array(original.values[2048..<4096]))
    }

    /// The size claim from `D165`/`D168`, asserted rather than assumed: a reply frame must be header + 4 bytes
    /// per value, with no per-float overhead. If a future change makes this JSON, the exchange budget doubles and
    /// the throughput arithmetic is invalid.
    @Test("the payload is exactly four bytes per float")
    func payloadHasNoOverhead() throws {
        let reply = Self.reply()
        let frame = try ShardExchange.encode(reply)
        let headerLength = Int(frame.withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) }.littleEndian)
        #expect(frame.count - 4 - headerLength == reply.values.count * 4)
        #expect(headerLength < 200, "the header is small and does not scale with the payload")
    }

    @Test("a mismatched schema is refused rather than misread")
    func schemaMismatchIsRefused() throws {
        let frame = try ShardExchange.encode(Self.reply())
        let headerLength = Int(frame.withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) }.littleEndian)
        var header = try JSONSerialization.jsonObject(
            with: frame.subdata(in: 4..<(4 + headerLength))) as? [String: Any] ?? [:]
        header["schema"] = 99
        var rebuilt = Data()
        let newHeader = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        var length = UInt32(newHeader.count).littleEndian
        withUnsafeBytes(of: &length) { rebuilt.append(contentsOf: $0) }
        rebuilt.append(newHeader)
        rebuilt.append(frame.subdata(in: (4 + headerLength)..<frame.count))
        #expect(throws: ShardExchange.Error.self) { try ShardExchange.decodeReply(from: rebuilt) }
    }

    @Test("a truncated payload is an error, not a short read")
    func truncatedPayloadIsRefused() throws {
        let frame = try ShardExchange.encode(Self.reply())
        #expect(throws: ShardExchange.Error.self) {
            try ShardExchange.decodeReply(from: frame.dropLast(4))
        }
        #expect(throws: ShardExchange.Error.self) {
            try ShardExchange.decodeReply(from: frame.prefix(2))
        }
    }

    @Test("slot and expert counts must agree, and an empty request is refused")
    func inconsistentRequestsAreRefused() throws {
        let activation = [Float](repeating: 0, count: 8)
        let bad = ShardExchange.Request(layer: 0, slots: [0, 1], experts: [5], activation: activation)
        #expect(throws: ShardExchange.Error.self) { try ShardExchange.encode(bad) }
        let empty = ShardExchange.Request(layer: 0, slots: [], experts: [], activation: activation)
        #expect(throws: ShardExchange.Error.self) { try ShardExchange.encode(empty) }
    }
}
