import Darwin
import Foundation

/// A listening socket that hands back connected transports.
///
/// IPv4 loopback first, and that is a scope statement rather than a design one: the point of this type is
/// that a cluster node can **bind, listen and accept** at all, which the socket pair in the test harness
/// never did. Choosing and measuring the real link — Thunderbolt bridge, LAN, SFP/QSFP — is `DC-008`'s
/// other half and needs the testbed.
public final class TCPListener {
    private let descriptor: Int32
    /// The port actually bound, which matters when the caller asked for `0` and let the system choose.
    public let port: Int

    public enum Error: Swift.Error, CustomStringConvertible, Equatable {
        case cannotCreateSocket(String)
        case cannotBind(host: String, port: Int, reason: String)
        case cannotListen(String)
        case cannotAccept(String)
        case cannotResolve(host: String)
        case cannotConnect(host: String, port: Int, reason: String)

        public var description: String {
            switch self {
            case .cannotCreateSocket(let reason):
                return "could not create a socket: \(reason)"
            case .cannotBind(let host, let port, let reason):
                return "could not bind \(host):\(port): \(reason)"
            case .cannotListen(let reason):
                return "could not listen: \(reason)"
            case .cannotAccept(let reason):
                return "could not accept: \(reason)"
            case .cannotResolve(let host):
                return "could not resolve \(host)"
            case .cannotConnect(let host, let port, let reason):
                return "could not connect to \(host):\(port): \(reason)"
            }
        }
    }

    /// - Parameter port: `0` asks the system for a free port; read `port` afterwards to learn which.
    public init(host: String = "127.0.0.1", port: Int, backlog: Int32 = 8) throws {
        let fd = socket(AF_INET, SOCK_STREAM, 0)
        guard fd >= 0 else {
            throw Error.cannotCreateSocket(String(cString: strerror(errno)))
        }
        var one: Int32 = 1
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, socklen_t(MemoryLayout<Int32>.size))

        var address = sockaddr_in()
        address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        address.sin_family = sa_family_t(AF_INET)
        address.sin_port = in_port_t(UInt16(port).bigEndian)
        guard inet_pton(AF_INET, host, &address.sin_addr) == 1 else {
            close(fd)
            throw Error.cannotResolve(host: host)
        }
        let bound = withUnsafePointer(to: &address) { pointer in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) { raw in
                bind(fd, raw, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        guard bound == 0 else {
            let reason = String(cString: strerror(errno))
            close(fd)
            throw Error.cannotBind(host: host, port: port, reason: reason)
        }
        guard listen(fd, backlog) == 0 else {
            let reason = String(cString: strerror(errno))
            close(fd)
            throw Error.cannotListen(reason)
        }

        var actual = sockaddr_in()
        var length = socklen_t(MemoryLayout<sockaddr_in>.size)
        _ = withUnsafeMutablePointer(to: &actual) { pointer in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) { raw in
                getsockname(fd, raw, &length)
            }
        }
        self.descriptor = fd
        self.port = Int(UInt16(bigEndian: actual.sin_port))
    }

    deinit { close(descriptor) }

    /// Accept one connection, waiting at most `timeoutMilliseconds`.
    public func accept(timeoutMilliseconds: Int = 30_000) throws -> SocketContributionTransport {
        var ready = pollfd(fd: descriptor, events: Int16(POLLIN), revents: 0)
        let result = poll(&ready, 1, Int32(clamping: timeoutMilliseconds))
        if result == 0 {
            throw ContributionTransportError.timedOut(midFrame: false)
        }
        if result < 0 {
            throw Error.cannotAccept(String(cString: strerror(errno)))
        }
        var peer = sockaddr_in()
        var length = socklen_t(MemoryLayout<sockaddr_in>.size)
        let accepted = withUnsafeMutablePointer(to: &peer) { pointer in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) { raw in
                // Qualified: inside `accept` the method name hides the global function, and the compiler
                // says so rather than picking one silently.
                Darwin.accept(descriptor, raw, &length)
            }
        }
        guard accepted >= 0 else {
            throw Error.cannotAccept(String(cString: strerror(errno)))
        }
        return SocketContributionTransport(fileDescriptor: accepted, timeoutMilliseconds: timeoutMilliseconds)
    }
}

public enum TCPTransport {
    /// Connect to a listening node, giving up after `timeoutMilliseconds`.
    ///
    /// The socket goes non-blocking for the connect and back to blocking afterwards, because a plain
    /// `connect` to a host that is not answering blocks for the kernel's own timeout — which is minutes,
    /// and would present as a hung run rather than a refused one.
    public static func connect(
        host: String, port: Int, timeoutMilliseconds: Int = 30_000
    ) throws -> SocketContributionTransport {
        let fd = socket(AF_INET, SOCK_STREAM, 0)
        guard fd >= 0 else {
            throw TCPListener.Error.cannotCreateSocket(String(cString: strerror(errno)))
        }
        var one: Int32 = 1
        setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, socklen_t(MemoryLayout<Int32>.size))

        var address = sockaddr_in()
        address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        address.sin_family = sa_family_t(AF_INET)
        address.sin_port = in_port_t(UInt16(port).bigEndian)
        guard inet_pton(AF_INET, host, &address.sin_addr) == 1 else {
            close(fd)
            throw TCPListener.Error.cannotResolve(host: host)
        }

        let flags = fcntl(fd, F_GETFL, 0)
        _ = fcntl(fd, F_SETFL, flags | O_NONBLOCK)
        let connected = withUnsafePointer(to: &address) { pointer in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) { raw in
                Darwin.connect(fd, raw, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        if connected != 0 {
            guard errno == EINPROGRESS else {
                let reason = String(cString: strerror(errno))
                close(fd)
                throw TCPListener.Error.cannotConnect(host: host, port: port, reason: reason)
            }
            var writable = pollfd(fd: fd, events: Int16(POLLOUT), revents: 0)
            let ready = poll(&writable, 1, Int32(clamping: timeoutMilliseconds))
            if ready == 0 {
                close(fd)
                throw ContributionTransportError.timedOut(midFrame: false)
            }
            guard ready > 0 else {
                let reason = String(cString: strerror(errno))
                close(fd)
                throw TCPListener.Error.cannotConnect(host: host, port: port, reason: reason)
            }
            // `POLLOUT` can also mean the connect failed, so ask the socket itself.
            var soError: Int32 = 0
            var size = socklen_t(MemoryLayout<Int32>.size)
            getsockopt(fd, SOL_SOCKET, SO_ERROR, &soError, &size)
            guard soError == 0 else {
                close(fd)
                throw TCPListener.Error.cannotConnect(
                    host: host, port: port, reason: String(cString: strerror(soError))
                )
            }
        }
        _ = fcntl(fd, F_SETFL, flags)
        return SocketContributionTransport(fileDescriptor: fd, timeoutMilliseconds: timeoutMilliseconds)
    }
}
