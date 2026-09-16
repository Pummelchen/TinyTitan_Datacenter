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
    /// Apply a deadline to subsequent receives.
    ///
    /// This exists because the alternative is worse than it looks: an `ExchangePolicy` carrying a
    /// timeout that the transport never hears about is a *decorative* parameter, and the first version
    /// of this code had exactly that — a test configured 60 ms, waited 30 s per attempt, and passed
    /// anyway because it asserted behaviour and not duration. A transport with no deadline of its own
    /// ignores this, which is honest; one that has a deadline is told about it.
    ///
    /// **A transport that wraps another must forward this.** A decorator that implements `send` and
    /// `receive` and not this one silently reintroduces the same defect one layer up, which is exactly
    /// what happened the first time this was tested.
    func applyTimeout(milliseconds: Int)
}

extension ContributionTransport {
    public func applyTimeout(milliseconds: Int) {}
}

public enum ContributionTransportError: Swift.Error, CustomStringConvertible, Equatable {
    case frameTooLarge(Int)
    case truncated(needed: Int, got: Int)
    case closed
    /// A peer that did not answer inside its deadline.
    ///
    /// The flag is the diagnosis that matters: waiting for a frame that never starts is a silent or
    /// slow node, while stopping part-way through one means the stream is **desynchronised** and the
    /// connection cannot be reused, retried or reasoned about. Collapsing the two would make a broken
    /// connection look like a slow one.
    case timedOut(midFrame: Bool)
    case socket(String)

    public var description: String {
        switch self {
        case .frameTooLarge(let bytes):
            return "frame of \(bytes) bytes is larger than the transport will carry"
        case .truncated(let needed, let got):
            return "connection ended mid-frame: \(got) of \(needed) bytes"
        case .closed:
            return "connection closed"
        case .timedOut(let midFrame):
            return midFrame
                ? "peer stopped part-way through a frame, so the stream is desynchronised"
                : "peer did not answer inside its deadline"
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
    private var timeoutMilliseconds: Int32

    /// - Parameter timeoutMilliseconds: how long a `receive` waits for a peer. `poll` is used rather
    ///   than `SO_RCVTIMEO` so the deadline applies per chunk and the *reason* for a stop is visible:
    ///   nothing at all, or a frame that began and did not finish. A negative value waits forever,
    ///   which is only reasonable for a test that is certain the bytes are already in flight.
    public init(fileDescriptor: Int32, closeOnDealloc: Bool = true, timeoutMilliseconds: Int = 30_000) {
        var one: Int32 = 1
        setsockopt(
            fileDescriptor, SOL_SOCKET, SO_NOSIGPIPE, &one, socklen_t(MemoryLayout<Int32>.size)
        )
        self.handle = FileHandle(fileDescriptor: fileDescriptor, closeOnDealloc: closeOnDealloc)
        self.timeoutMilliseconds = Int32(clamping: timeoutMilliseconds)
    }

    public func applyTimeout(milliseconds: Int) {
        timeoutMilliseconds = Int32(clamping: milliseconds)
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
        let prefix = try readExactly(4, midFrame: false)
        let length = Int(prefix[prefix.startIndex])
            | (Int(prefix[prefix.startIndex + 1]) << 8)
            | (Int(prefix[prefix.startIndex + 2]) << 16)
            | (Int(prefix[prefix.startIndex + 3]) << 24)
        guard length <= Self.maximumFrameBytes else {
            throw ContributionTransportError.frameTooLarge(length)
        }
        return try readExactly(length, midFrame: true)
    }

    /// Read exactly `count` bytes, polling before every chunk so the deadline applies to the whole
    /// read rather than to a single `read` call that might return early.
    private func readExactly(_ count: Int, midFrame: Bool) throws -> Data {
        var data = Data()
        while data.count < count {
            try waitForReadable(alreadyHave: data.count, midFrame: midFrame)
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

    private func waitForReadable(alreadyHave: Int, midFrame: Bool) throws {
        guard timeoutMilliseconds >= 0 else { return }
        var descriptor = pollfd(fd: handle.fileDescriptor, events: Int16(POLLIN), revents: 0)
        let ready = poll(&descriptor, 1, timeoutMilliseconds)
        if ready == 0 {
            throw ContributionTransportError.timedOut(midFrame: midFrame || alreadyHave > 0)
        }
        if ready < 0 {
            throw ContributionTransportError.socket("poll failed: \(String(cString: strerror(errno)))")
        }
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
