import Foundation

/// The expert exchange: what a node asks a peer to compute, and what comes back.
///
/// **Why this is not JSON.** The frame codec the decode service uses (`DecodeFrameCodec`) is a length-prefixed
/// JSON document, which is right for commands and events and wrong for this: a contribution is `D` floats per
/// slot, and `D168` established that those floats must cross the wire at **full fp32 precision** — a narrower
/// encoding rounds them and the fixed-order fp32 sum in `moe_phase2_down_reduce_k8` stops being bit-identical to
/// a single-node run. JSON would also inflate each value to ~12 characters. So the **header is JSON** — small,
/// inspectable, and where a mistake is likely — and the **payload is raw little-endian fp32**.
///
/// **Why the slot is carried explicitly.** The reduce is a fixed-order accumulation over slots 0..7, so a
/// contribution is only correct if the receiver can put it back in the slot the router gave it. Both messages
/// carry slots, and the reply's values are row-major in the order those slots are listed. Renumbering them to a
/// compact `0..<n` would produce a different number that still looks plausible.
public enum ShardExchange {
    /// Bumped if the layout below changes. A peer that disagrees must refuse rather than misread a buffer.
    public static let schema: UInt32 = 1

    /// What the owners of `slots` must compute for one layer of one token.
    ///
    /// `activation` is the layer's hidden state — `D` floats, sent **once** for the whole request rather than
    /// once per slot, because every expert reads the same activations.
    public struct Request: Sendable, Equatable {
        public let layer: Int
        public let slots: [Int]
        public let experts: [Int]
        public let activation: [Float]

        public init(layer: Int, slots: [Int], experts: [Int], activation: [Float]) {
            self.layer = layer
            self.slots = slots
            self.experts = experts
            self.activation = activation
        }
    }

    /// The computed expert outputs, one row of `D` floats per requested slot, **in the order requested**.
    ///
    /// These are the expert outputs, not the weighted partials: the routing weight is applied by the node that
    /// owns the router's decision, so a peer never needs to know the routing weights and cannot disagree about
    /// them. That keeps the multiplication on the same side of the wire as the ordering.
    public struct Reply: Sendable, Equatable {
        public let layer: Int
        public let slots: [Int]
        public let dimensions: Int
        /// `slots.count * dimensions`, row-major.
        public let values: [Float]

        public init(layer: Int, slots: [Int], dimensions: Int, values: [Float]) {
            self.layer = layer
            self.slots = slots
            self.dimensions = dimensions
            self.values = values
        }

        /// The row for the `index`-th requested slot.
        public func row(at index: Int) -> ArraySlice<Float> {
            let start = index * dimensions
            return values[start..<(start + dimensions)]
        }
    }

    public enum Error: Swift.Error, Equatable {
        case unsupportedSchema(UInt32)
        case truncatedPayload(expected: Int, found: Int)
        case slotCountMismatch(slots: Int, experts: Int)
        case valueCountMismatch(slots: Int, dimensions: Int, found: Int)
        case emptyRequest
    }

    // MARK: - header, as JSON

    private struct RequestHeader: Codable {
        let schema: UInt32
        let layer: Int
        let slots: [Int]
        let experts: [Int]
        let dimensions: Int
    }

    private struct ReplyHeader: Codable {
        let schema: UInt32
        let layer: Int
        let slots: [Int]
        let dimensions: Int
    }

    static let headerEncoder: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return encoder
    }()

    // MARK: - framing

    /// A frame is `UInt32 headerLength | header JSON | payload`, all lengths and floats little-endian.
    ///
    /// The header is length-prefixed rather than fixed-width so a field can be added without a schema bump for
    /// readers that ignore it, and the schema is still carried so a peer that *cannot* ignore a change can say
    /// so instead of misreading the payload that follows.
    public static func encode(_ request: Request) throws -> Data {
        guard !request.slots.isEmpty else { throw Error.emptyRequest }
        guard request.slots.count == request.experts.count else {
            throw Error.slotCountMismatch(slots: request.slots.count, experts: request.experts.count)
        }
        let header = RequestHeader(
            schema: schema, layer: request.layer, slots: request.slots,
            experts: request.experts, dimensions: request.activation.count
        )
        return try frame(header: headerEncoder.encode(header), payload: floatsToData(request.activation))
    }

    public static func encode(_ reply: Reply) throws -> Data {
        guard reply.values.count == reply.slots.count * reply.dimensions else {
            throw Error.valueCountMismatch(
                slots: reply.slots.count, dimensions: reply.dimensions, found: reply.values.count)
        }
        let header = ReplyHeader(
            schema: schema, layer: reply.layer, slots: reply.slots, dimensions: reply.dimensions)
        return try frame(header: headerEncoder.encode(header), payload: floatsToData(reply.values))
    }

    public static func decodeRequest(from data: Data) throws -> Request {
        let (headerData, payload) = try split(data)
        let header = try JSONDecoder().decode(RequestHeader.self, from: headerData)
        guard header.schema == schema else { throw Error.unsupportedSchema(header.schema) }
        guard header.slots.count == header.experts.count else {
            throw Error.slotCountMismatch(slots: header.slots.count, experts: header.experts.count)
        }
        return Request(
            layer: header.layer, slots: header.slots, experts: header.experts,
            activation: try dataToFloats(payload, count: header.dimensions)
        )
    }

    public static func decodeReply(from data: Data) throws -> Reply {
        let (headerData, payload) = try split(data)
        let header = try JSONDecoder().decode(ReplyHeader.self, from: headerData)
        guard header.schema == schema else { throw Error.unsupportedSchema(header.schema) }
        let expected = header.slots.count * header.dimensions
        return Reply(
            layer: header.layer, slots: header.slots, dimensions: header.dimensions,
            values: try dataToFloats(payload, count: expected)
        )
    }

    // MARK: - bytes

    private static func frame(header: Data, payload: Data) throws -> Data {
        var out = Data(capacity: 4 + header.count + payload.count)
        var length = UInt32(header.count).littleEndian
        withUnsafeBytes(of: &length) { out.append(contentsOf: $0) }
        out.append(header)
        out.append(payload)
        return out
    }

    private static func split(_ data: Data) throws -> (header: Data, payload: Data) {
        guard data.count >= 4 else { throw Error.truncatedPayload(expected: 4, found: data.count) }
        let length = data.withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) }.littleEndian
        let start = 4 + Int(length)
        guard data.count >= start else {
            throw Error.truncatedPayload(expected: start, found: data.count)
        }
        return (data.subdata(in: 4..<start), data.subdata(in: start..<data.count))
    }

    static func floatsToData(_ values: [Float]) -> Data {
        var out = Data(capacity: values.count * 4)
        for value in values {
            var bits = value.bitPattern.littleEndian
            withUnsafeBytes(of: &bits) { out.append(contentsOf: $0) }
        }
        return out
    }

    static func dataToFloats(_ data: Data, count: Int) throws -> [Float] {
        guard data.count == count * 4 else {
            throw Error.truncatedPayload(expected: count * 4, found: data.count)
        }
        var values = [Float]()
        values.reserveCapacity(count)
        // `withUnsafeBytes` over the whole buffer, then a bounded load per element, so a short payload is a
        // thrown error rather than an over-read.
        try data.withUnsafeBytes { raw in
            for index in 0..<count {
                let bits = raw.loadUnaligned(fromByteOffset: index * 4, as: UInt32.self)
                values.append(Float(bitPattern: UInt32(littleEndian: bits)))
            }
        }
        return values
    }
}
