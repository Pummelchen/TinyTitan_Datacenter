import Darwin
import Foundation

/// A **LAN** peer of `DecodeUnixSocket`, for running the decode service across machines.
///
/// The Unix transport answers a `(input: FileHandle, output: FileHandle)` pair, and so does this one: the
/// framing, the request and event vocabulary, the queues and the outbox are all downstream of that pair and
/// none of them change. That is the whole point — two transports, one protocol, so a caller picks a
/// transport and everything above it is identical. Nothing here invents a wire format.
///
/// The Unix path is deliberately *not* reused or generalised: its `sockaddr_un` construction, its
/// `unlink`-before-bind and its `chmod 0o600` are all about a filesystem socket in a uid-private directory,
/// and none of that has an analogue on a TCP port. Keeping them separate is what lets the secure-by-default
/// Unix behaviour stay exactly as it is.
public enum DecodeTCPSocket {
    /// Connect to a decode service listening on `host:port`.
    ///
    /// `host` is a literal IPv4 address rather than a name: a name would need `getaddrinfo`, and a decode
    /// service that resolves DNS at connect time is a service that can connect somewhere else on a later run.
    /// A LAN deployment knows its addresses.
    public static func connect(host: String, port: UInt16) throws -> (input: FileHandle, output: FileHandle) {
        let fd = socket(AF_INET, SOCK_STREAM, 0)
        guard fd >= 0 else { throw POSIXError(.init(rawValue: errno) ?? .EIO) }
        do {
            try setNoSigPipe(fd)
            var address = try makeAddress(host: host, port: port)
            let result = withUnsafePointer(to: &address) {
                $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                    Darwin.connect(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
                }
            }
            guard result == 0 else { throw POSIXError(.init(rawValue: errno) ?? .EIO) }
            return try handles(for: fd)
        } catch {
            Darwin.close(fd)
            throw error
        }
    }

    /// Bind `host:port`, wait for one peer to connect, and hand back the connected pair.
    ///
    /// `backlog` is one by default, matching the Unix path: a decode service serialises its commands anyway,
    /// and a queue of waiting clients would only hide that behind a longer wait. `SO_REUSEADDR` is set
    /// because a restarted service must be able to rebind a port whose previous connection is still in
    /// `TIME_WAIT` — on a development LAN that is the difference between restarting and waiting a minute.
    public static func listenAndAccept(
        host: String, port: UInt16, backlog: Int32 = 1
    ) throws -> (input: FileHandle, output: FileHandle) {
        let fd = socket(AF_INET, SOCK_STREAM, 0)
        guard fd >= 0 else { throw POSIXError(.init(rawValue: errno) ?? .EIO) }
        do {
            try setNoSigPipe(fd)
            var yes: Int32 = 1
            guard setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, socklen_t(MemoryLayout<Int32>.size)) == 0
            else { throw POSIXError(.init(rawValue: errno) ?? .EIO) }
            var address = try makeAddress(host: host, port: port)
            let bindResult = withUnsafePointer(to: &address) {
                $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                    Darwin.bind(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
                }
            }
            guard bindResult == 0 else { throw POSIXError(.init(rawValue: errno) ?? .EIO) }
            guard listen(fd, backlog) == 0 else { throw POSIXError(.init(rawValue: errno) ?? .EIO) }
            let accepted = Darwin.accept(fd, nil, nil)
            guard accepted >= 0 else { throw POSIXError(.init(rawValue: errno) ?? .EIO) }
            Darwin.close(fd)
            do {
                try setNoSigPipe(accepted)
            } catch {
                Darwin.close(accepted)
                throw error
            }
            return try handles(for: accepted)
        } catch {
            Darwin.close(fd)
            throw error
        }
    }

    /// The bound port, so a caller that passed `0` — "any free port" — can learn what it got.
    ///
    /// Needed for tests and for a launcher that wants the kernel to pick a port. A socket that has been
    /// `listen`ed but not yet accepted can be asked; the connected pair cannot, so this takes the listening
    /// side before `accept`.
    public static func boundPort(of fd: Int32) throws -> UInt16 {
        var address = sockaddr_in()
        var length = socklen_t(MemoryLayout<sockaddr_in>.size)
        let result = withUnsafeMutablePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                getsockname(fd, $0, &length)
            }
        }
        guard result == 0 else { throw POSIXError(.init(rawValue: errno) ?? .EIO) }
        return UInt16(bigEndian: address.sin_port)
    }

    /// A peer that vanishes mid-frame must surface as a closed handle, not as a signal that kills the
    /// process. `write` to a socket whose far end has gone raises `SIGPIPE` by default, and on Darwin
    /// `SO_NOSIGPIPE` is the per-socket way to turn that into an ordinary `EPIPE` the caller can handle.
    private static func setNoSigPipe(_ fd: Int32) throws {
        var yes: Int32 = 1
        guard setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &yes, socklen_t(MemoryLayout<Int32>.size)) == 0
        else { throw POSIXError(.init(rawValue: errno) ?? .EIO) }
    }

    private static func handles(for fd: Int32) throws
        -> (input: FileHandle, output: FileHandle) {
        let outputFD = dup(fd)
        guard outputFD >= 0 else { throw POSIXError(.init(rawValue: errno) ?? .EIO) }
        return (FileHandle(fileDescriptor: fd, closeOnDealloc: true),
                FileHandle(fileDescriptor: outputFD, closeOnDealloc: true))
    }

    private static func makeAddress(host: String, port: UInt16) throws -> sockaddr_in {
        var address = sockaddr_in()
        // The BSD convention is to carry the length in the address as well as pass it to `connect`/`bind`.
        // It was added while diagnosing an `EINVAL` and it was **not** what fixed it — the failures were a
        // test-suite port collision and an `fsync` on a socket. It is kept because it is conventional and
        // harmless, and it is labelled here so nobody later reads it as a proven requirement.
        address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        address.sin_family = sa_family_t(AF_INET)
        address.sin_port = port.bigEndian
        guard inet_pton(AF_INET, host, &address.sin_addr) == 1 else {
            throw POSIXError(.EINVAL)
        }
        return address
    }
}
