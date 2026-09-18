import Foundation

/// A peer link that carries `ShardExchange` frames.
///
/// It wraps an already-open `(input: FileHandle, output: FileHandle)` pair — the same pair `DecodeUnixSocket`
/// and `DecodeTCPSocket` both produce — so a peer link can be a Unix socket on one machine and TCP across the
/// LAN without this type knowing which. That is the property `D135` argued the reference's service boundary
/// already had, reused rather than reinvented.
///
/// **Framing.** `ShardExchange`'s frame is `headerLength | header | payload`, and the payload's length is only
/// derivable by parsing the header. Rather than have this layer understand the exchange's fields, every frame is
/// sent with its **total** length in front:
///
/// ```
/// UInt32 frameLength | frame
/// ```
///
/// so the channel stays a byte pipe and `ShardExchange` stays the only thing that knows the exchange's layout.
/// A layer that parsed the header to size the payload would be a second place to update when the header changes,
/// and the two would drift.
public struct ShardPeerChannel: Sendable {
    public let input: FileHandle
    public let output: FileHandle

    /// No `SO_SNDBUF`-sized frame may exceed this. A contribution for one layer of one token is
    /// `slots x 2048 x 4` bytes — about 49 KB at 6 slots — so 1 MiB leaves room for a whole layer's slots plus
    /// header while still refusing a length that a corrupt stream could use to allocate wildly.
    public static let maximumFrameBytes = 1 << 20

    public enum Error: Swift.Error, Equatable {
        case frameTooLarge(Int)
        case truncated(expected: Int, found: Int)
        case peerClosed
    }

    public init(input: FileHandle, output: FileHandle) {
        self.input = input
        self.output = output
    }

    /// Release both ends of the connection.
    ///
    /// `input` and `output` may wrap the **same** file descriptor - a socket pair handed back as two handles
    /// does - so each descriptor is closed once. Closing the same fd twice would release an unrelated descriptor
    /// that had since been allocated to something else, which is a bug that appears far from its cause.
    public func close() {
        var closed = Set<Int32>()
        for handle in [input, output] {
            let fd = handle.fileDescriptor
            guard fd >= 0, !closed.contains(fd) else { continue }
            closed.insert(fd)
            try? handle.close()
        }
    }

    public func send(_ frame: Data) throws {
        guard frame.count <= Self.maximumFrameBytes else { throw Error.frameTooLarge(frame.count) }
        var length = UInt32(frame.count).littleEndian
        var datagram = withUnsafeBytes(of: &length) { Data($0) }
        datagram.append(frame)
        try output.write(contentsOf: datagram)
    }

    public func receive() throws -> Data {
        let header = try readExactly(4)
        let length = header.withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) }.littleEndian
        guard Int(length) <= Self.maximumFrameBytes else { throw Error.frameTooLarge(Int(length)) }
        return try readExactly(Int(length))
    }

    /// Read exactly `count` bytes, or throw. A single `read` may return short — more so over a network than over
    /// loopback — so reading once would test the buffer rather than the transport, which is the mistake the
    /// decode-service socket tests already caught once.
    private func readExactly(_ count: Int) throws -> Data {
        var collected = Data()
        collected.reserveCapacity(count)
        while collected.count < count {
            guard let chunk = try input.read(upToCount: count - collected.count) else {
                throw Error.peerClosed
            }
            if chunk.isEmpty { throw Error.truncated(expected: count, found: collected.count) }
            collected.append(chunk)
        }
        return collected
    }
}
