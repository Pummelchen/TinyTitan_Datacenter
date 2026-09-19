import Foundation

/// **The pipeline's wire.** A `PipelineFrame` in each direction over a TCP pair - the whole of the transport
/// between two layer-pipeline stages.
///
/// It deliberately does **not** reuse `ShardPeerChannel`. That channel carries an expert-sharding exchange -
/// routing tables, ownership, a reduce - and a pipeline stage needs none of it: it owns a contiguous layer
/// range, receives a hidden state, computes, and forwards one. So this is its own framing over the same
/// `DecodeTCPSocket`, and `D288`'s measured 117.8 MB/s is the rate it inherits.
///
/// **The framing needs no length prefix, and that is what the header's `count` field is for.** A reader takes
/// `headerBytes`, decodes them, and now knows the payload is `count` fp16 values - so the same 16 bytes that
/// identify the frame also size it, and a stream of frames is self-delimiting. `count` is exactly the field
/// `D337` needed for a `t`-row prefill handoff, where a stage's buffer is sized from the chunk rather than from
/// one token.
public enum PipelineLink {
    public enum LinkError: Error, Equatable {
        case closed
        case shortFrame(expected: Int, got: Int)
        case badCount(Int)
    }

    /// The byte offset of `count` in the header: **after token and layer, before the reserved word.**
    ///
    /// This is written as a named constant because getting it wrong is silent in the worst way. `PipelineFrame`
    /// encodes token, layer, count, reserved as four `u32`s; reading offset 12 instead of 8 yields `reserved`,
    /// which is always zero, so a reader computes a zero-length payload, `decode` rejects it, and the sender's
    /// task dies - **presenting as `ECONNRESET` on the receiving side, which looks like a socket or port fault
    /// and is neither.** That cost a round (`D338`) before the offset was checked against `encode`.
    public static let countOffset = 8

    /// The most rows one frame may claim. A receiver sizes a buffer from `count` before reading the payload, so
    /// an unchecked count is an allocation a peer chooses - 65536 rows is 256 MB, far above any chunk this
    /// engine prefills, and refusing above it is cheaper than discovering it under memory pressure.
    public static let maxRows = 65_536

    /// Write one frame and flush. `output` is the socket's write end.
    public static func send(_ frame: PipelineFrame, to output: FileHandle) throws {
        try output.write(contentsOf: frame.encode())
    }

    /// Read exactly one frame. Blocks until it arrives or the peer closes.
    ///
    /// Both the header and the payload are read in a loop: a TCP stream has no message boundaries, so a single
    /// `read(upToCount:)` is allowed to return short - and on a LAN it will. A closed or short read is an error
    /// rather than a partial frame. `PipelineFrame.decode` does the fp16 conversion, so nothing here interprets
    /// the payload.
    public static func receive(from input: FileHandle) throws -> PipelineFrame {
        let header = try readExactly(PipelineFrame.headerBytes, from: input, what: "header")
        let count = Int(header.withUnsafeBytes {
            $0.loadUnaligned(fromByteOffset: countOffset, as: UInt32.self)
        })
        guard count <= maxRows else { throw LinkError.badCount(count) }
        let payloadBytes = count * MemoryLayout<Float16>.stride
        let payload = try readExactly(payloadBytes, from: input, what: "payload")
        return try PipelineFrame.decode(header + payload)
    }

    /// Read exactly `count` bytes, or throw. A zero-byte read is the peer closing, which is `closed` when
    /// nothing had been read yet and `shortFrame` when a frame had already started arriving.
    private static func readExactly(_ count: Int, from handle: FileHandle, what: String) throws -> Data {
        var collected = Data()
        while collected.count < count {
            guard let chunk = try handle.read(upToCount: count - collected.count), !chunk.isEmpty else {
                throw collected.isEmpty ? LinkError.closed
                                        : LinkError.shortFrame(expected: count, got: collected.count)
            }
            collected.append(chunk)
        }
        return collected
    }
}
