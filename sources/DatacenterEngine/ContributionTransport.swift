import Darwin
import Foundation

/// A framed connection between nodes.
///
/// Framing is the transport's job and the codec's is pure data, so the two can be tested apart: the
/// codec can be handed a corrupted buffer, and the transport can be handed a frame that arrives in
/// pieces. A network gives you both for free, and neither is convenient to provoke on demand.
public protocol ContributionTransport {
    /// Send one frame. The transport owns the length prefix, so a reader never has to guess where a
    /// frame ends.
    func send(_ frame: Data) throws
    /// Receive exactly one frame, however the bytes arrived — one piece, or three frames coalesced.
    func receive() throws -> Data
}

public enum ContributionTransportError: Swift.Error, CustomStringConvertible, Equatable {
    case frameTooLarge(Int)
    case truncated(needed: Int, got: Int)
    case closed
    case socket(String)

    public var description: String {
        switch self {
        case .frameTooLarge(let bytes):
            return "frame of \(bytes) bytes is larger than the transport will carry"
        case .truncated(let needed, let got):
            return "connection ended mid-frame: \(got) of \(needed) bytes"
        case .closed:
            return "connection closed"
        case .socket(let message):
            return "socket error: \(message)"
        }
    }
}

/// A connection over a file descriptor, with a 32-bit length prefix in front of each frame.
///
/// `SO_NOSIGPIPE` is set on the descriptor because the alternative is that a peer that has gone away
/// kills the process with `SIGPIPE` — a failure mode that presents as "the node vanished", not as
/// "the write failed", and this project cannot afford that kind of ambiguity (`DC-043`).
public final class SocketContributionTransport: ContributionTransport {
    /// A bound so a corrupt or hostile length prefix is refused before it is allocated.
    public static let maximumFrameBytes = 1 << 26

    private let handle: FileHandle

    public init(fileDescriptor: Int32, closeOnDealloc: Bool = true) {
        var one: Int32 = 1
        setsockopt(
            fileDescriptor, SOL_SOCKET, SO_NOSIGPIPE, &one, socklen_t(MemoryLayout<Int32>.size)
        )
        self.handle = FileHandle(fileDescriptor: fileDescriptor, closeOnDealloc: closeOnDealloc)
    }

    public func send(_ frame: Data) throws {
        guard frame.count <= Self.maximumFrameBytes else {
            throw ContributionTransportError.frameTooLarge(frame.count)
        }
        var prefix: [UInt8] = []
        let length = UInt32(frame.count).littleEndian
        prefix.append(UInt8(truncatingIfNeeded: length))
        prefix.append(UInt8(truncatingIfNeeded: length >> 8))
        prefix.append(UInt8(truncatingIfNeeded: length >> 16))
        prefix.append(UInt8(truncatingIfNeeded: length >> 24))
        do {
            try handle.write(contentsOf: Data(prefix))
            try handle.write(contentsOf: frame)
        } catch {
            throw ContributionTransportError.socket("\(error)")
        }
    }

    public func receive() throws -> Data {
        let prefix = try readExactly(4)
        let length = Int(prefix[prefix.startIndex])
            | (Int(prefix[prefix.startIndex + 1]) << 8)
            | (Int(prefix[prefix.startIndex + 2]) << 16)
            | (Int(prefix[prefix.startIndex + 3]) << 24)
        guard length <= Self.maximumFrameBytes else {
            throw ContributionTransportError.frameTooLarge(length)
        }
        return try readExactly(length)
    }

    private func readExactly(_ count: Int) throws -> Data {
        var data = Data()
        while data.count < count {
            let chunk: Data?
            do {
                chunk = try handle.read(upToCount: count - data.count)
            } catch {
                throw ContributionTransportError.socket("\(error)")
            }
            guard let chunk, !chunk.isEmpty else {
                throw data.isEmpty
                    ? ContributionTransportError.closed
                    : ContributionTransportError.truncated(needed: count, got: data.count)
            }
            data.append(chunk)
        }
        return data
    }
}

public enum SocketPair {
    /// A connected pair of sockets.
    ///
    /// This is the two-node harness: the same framing, the same read loop and the same codec a real
    /// connection uses, without a network or a second process. What it cannot exercise is a peer that
    /// dies mid-frame or a link that is slow — those need real sockets and belong to `DC-043`.
    public static func make() throws -> (SocketContributionTransport, SocketContributionTransport) {
        var descriptors: [Int32] = [-1, -1]
        guard socketpair(AF_UNIX, SOCK_STREAM, 0, &descriptors) == 0 else {
            throw ContributionTransportError.socket("socketpair failed: \(String(cString: strerror(errno)))")
        }
        return (
            SocketContributionTransport(fileDescriptor: descriptors[0]),
            SocketContributionTransport(fileDescriptor: descriptors[1])
        )
    }
}
