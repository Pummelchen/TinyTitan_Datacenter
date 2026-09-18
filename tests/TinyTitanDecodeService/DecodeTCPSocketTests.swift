import Foundation
import Testing

@testable import TinyTitanDecodeProtocol

/// The LAN transport, tested where it matters: that it is a **peer** of the Unix one, so the frames above it
/// are the same, and that a peer going away is an ordinary error rather than a signal.
///
/// These run on loopback with a fixed high port. The alternative — port 0 and asking the kernel what it
/// gave us — needs the listening socket before `accept` blocks, and `listenAndAccept` closes it as soon as
/// it has a connection; the accessor for that exists (`boundPort(of:)`) but exposing the listener would
/// change the shape of a function whose whole point is that it mirrors the Unix one. A loopback test with a
/// fixed port is the smaller thing to write, and `SO_REUSEADDR` is set, so a repeated run rebinds.
/// **Serialised, and the first version was not** — which failed immediately. swift-testing runs tests in
/// parallel by default, all three bound the same fixed port, and `SO_REUSEADDR` does not permit two
/// *simultaneous* listeners on one port (it permits rebinding one in `TIME_WAIT`). Two tests failed with
/// `EADDRINUSE` and the third connected to a *different test's* listener and died on `EPIPE` — a failure
/// that looks like a broken transport and was a broken fixture. The alternative is a distinct port per test,
/// or teaching `listenAndAccept` to report the port it got for port 0; serialising is the smallest change
/// that makes the port genuinely exclusive.
@Suite("Decode TCP socket", .serialized)
struct DecodeTCPSocketTests {
    /// A port in the ephemeral range, fixed so the test is deterministic. Two suites running at once would
    /// collide; `swift test` runs them in one process, so they do not.
    static let port: UInt16 = 45917

    private static func acceptInBackground(port: UInt16) -> Task<(input: FileHandle, output: FileHandle), Error> {
        Task.detached {
            try DecodeTCPSocket.listenAndAccept(host: "127.0.0.1", port: port)
        }
    }

    /// Read exactly `count` bytes, or throw if the peer closes first.
    ///
    /// A single `read(upToCount:)` is allowed to return short, and on a LAN it will — so a test that reads
    /// once is testing the loopback's buffering rather than the transport.
    private static func readExactly(_ count: Int, from handle: FileHandle) throws -> Data {
        var collected = Data()
        while collected.count < count {
            guard let chunk = try handle.read(upToCount: count - collected.count), !chunk.isEmpty else {
                throw POSIXError(.ECONNRESET)
            }
            collected.append(chunk)
        }
        return collected
    }

    @Test("a message round-trips over loopback")
    func roundTrip() async throws {
        let server = Self.acceptInBackground(port: Self.port)
        let client = try await Self.connectWhenListening(port: Self.port)
        let accepted = try await server.value

        // **No `synchronize()`**: it is `fsync`, and `fsync` on a socket returns `EINVAL` — which surfaced as
        // `NSCocoaErrorDomain 512 / POSIX 22` and looked like a broken transport for two runs. A socket write
        // is handed to the kernel immediately; there is no user-space buffer to flush.
        let payload = Data("decode-service-frame".utf8)
        try client.output.write(contentsOf: payload)

        #expect(try Self.readExactly(payload.count, from: accepted.input) == payload)

        try? client.output.close()
        try? accepted.input.close()
    }

    @Test("a partial frame arrives in pieces and still reassembles")
    func partialFrame() async throws {
        let server = Self.acceptInBackground(port: Self.port)
        let client = try await Self.connectWhenListening(port: Self.port)
        let accepted = try await server.value

        let payload = Data((0..<4096).map { UInt8($0 % 251) })
        let half = payload.count / 2
        try client.output.write(contentsOf: payload.prefix(half))
        try await Task.sleep(nanoseconds: 50_000_000)
        try client.output.write(contentsOf: payload.suffix(from: half))

        #expect(try Self.readExactly(payload.count, from: accepted.input) == payload)

        try? client.output.close()
        try? accepted.input.close()
    }

    @Test("a peer that disappears mid-frame surfaces as a closed handle, not a signal")
    func midFrameDisconnect() async throws {
        let server = Self.acceptInBackground(port: Self.port)
        let client = try await Self.connectWhenListening(port: Self.port)
        let accepted = try await server.value

        try client.output.write(contentsOf: Data(repeating: 7, count: 16))
        // Close mid-frame: the reader has been promised more than it will get.
        try client.output.close()
        try client.input.close()

        // The reader must see end-of-stream rather than hang or die. `read(upToCount:)` returning nil or an
        // empty chunk is the end; anything else means the transport is still waiting for a peer that is gone.
        var sawEnd = false
        for _ in 0..<50 {
            let chunk = try? accepted.input.read(upToCount: 32)
            if chunk == nil || chunk?.isEmpty == true {
                sawEnd = true
                break
            }
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        #expect(sawEnd, "a closed peer must end the stream")
        try? accepted.input.close()
    }

    /// The listening side is started in a detached task, so a connect can arrive before `bind`/`listen` has
    /// run. Retrying is the honest way to express that: the transport has no "ready" signal to wait on, and
    /// a fixed sleep would be a race that passes on a quiet machine and fails on a busy one.
    private static func connectWhenListening(port: UInt16) async throws
        -> (input: FileHandle, output: FileHandle)
    {
        var last: Error = POSIXError(.ECONNREFUSED)
        for _ in 0..<100 {
            do {
                return try DecodeTCPSocket.connect(host: "127.0.0.1", port: port)
            } catch {
                last = error
                try await Task.sleep(nanoseconds: 20_000_000)
            }
        }
        throw last
    }
}
