// PipelineFrame — the only thing that crosses the wire in Design A.
//
// A layer pipeline sends activations, never weights. A hidden state is `D` fp16 values - 4 KB for this model at
// D = 2048 - and the measured wire is 117.8 MB/s, so a frame is about 0.03 ms of link time. The same wire cannot
// carry weights at all: 566 MB of active experts a token would be 4.8 seconds (`D288`), which is why every design
// that moves them is dead and why this type carries a vector rather than a tensor.
//
// The layout is deliberately trivial: a fixed 16-byte header (token index, layer index, element count, reserved)
// followed by the fp16 payload. It reuses `DecodeTCPSocket` for the connection and nothing else from the
// expert-exchange protocol, because a pipeline has no routing table, no ownership and no reduce - a stage owns a
// contiguous layer range and forwards what it computed.

import Foundation

/// One activation handoff between two pipeline stages.
public struct PipelineFrame: Equatable, Sendable {
    /// Which token this activation belongs to. Checked on receipt, so a desynchronised pipeline fails loudly
    /// rather than computing a forward pass for the wrong position.
    public let token: Int
    /// The layer that produced it, so a stage can refuse a frame meant for someone else.
    public let layer: Int
    /// `D` hidden values, fp16 because that is what the kernels load.
    public let hidden: [Float16]

    public init(token: Int, layer: Int, hidden: [Float16]) {
        self.token = token
        self.layer = layer
        self.hidden = hidden
    }

    /// Header is fixed-width so a decoder can size the payload without a second round trip.
    public static let headerBytes = 16

    public func encode() -> Data {
        var data = Data(capacity: Self.headerBytes + hidden.count * MemoryLayout<Float16>.stride)
        var t = UInt32(truncatingIfNeeded: token)
        var l = UInt32(truncatingIfNeeded: layer)
        var n = UInt32(hidden.count)
        var reserved: UInt32 = 0
        withUnsafeBytes(of: &t) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &l) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &n) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &reserved) { data.append(contentsOf: $0) }
        hidden.withUnsafeBytes { data.append(contentsOf: $0) }
        return data
    }

    public enum DecodeError: Error, Equatable {
        case shortHeader(Int)
        case shortPayload(expected: Int, got: Int)
    }

    public static func decode(_ data: Data) throws -> PipelineFrame {
        guard data.count >= headerBytes else { throw DecodeError.shortHeader(data.count) }
        let base = data.startIndex
        func u32(_ offset: Int) -> UInt32 {
            var v: UInt32 = 0
            _ = withUnsafeMutableBytes(of: &v) { dst in
                data.copyBytes(to: dst, from: (base + offset)..<(base + offset + 4))
            }
            return v
        }
        let token = Int(u32(0)), layer = Int(u32(4)), count = Int(u32(8))
        let need = headerBytes + count * MemoryLayout<Float16>.stride
        guard data.count >= need else {
            throw DecodeError.shortPayload(expected: need, got: data.count)
        }
        var hidden = [Float16](repeating: 0, count: count)
        _ = hidden.withUnsafeMutableBytes { dst in
            data.copyBytes(to: dst, from: (base + headerBytes)..<(base + need))
        }
        return PipelineFrame(token: token, layer: layer, hidden: hidden)
    }
}
