import Foundation
import Testing

@testable import TinyTitanDecodeProtocol

/// **A two-process test, with `nc` as the peer.** The in-process attempt deadlocked twice (`D338`, `D342`) because
/// a listener that blocks in `accept` inside a task the test also awaits leaves one side waiting on a socket
/// forever when the other fails. A separate process cannot do that: it either answers or it exits non-zero.
///
/// And `nc` is a *better* peer than a second copy of this code. Testing a codec against itself proves the two
/// halves agree; testing it against `nc` proves the bytes on the wire are what `encode` says they are, checked by
/// something that has never read `PipelineFrame`. That is the version of this test worth having.
@Suite("PipelineLink on a real wire", .serialized)
struct PipelineLinkWireTests {
    static let port: UInt16 = 47_690

    private func scratch(_ name: String) -> String {
        let dir = NSTemporaryDirectory() + "pipelinelink-\(UUID().uuidString)"
        try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        return dir + "/" + name
    }

    /// Retry the REAL connection until it succeeds, rather than probing for the listener first.
    ///
    /// A probe is not a harmless readiness check here: `nc -l` accepts exactly one connection and exits, so a probe
    /// **consumes** the peer and the connection that matters is then refused. That is what `ECONNREFUSED` was
    /// telling us, and it is why the repository's own helper is named `connectWhenListening` and retries the
    /// connection it actually wants instead of testing for one it does not.
    private func connectWhenListening(_ port: UInt16, retries: Int = 200) throws
        -> (input: FileHandle, output: FileHandle) {
        var last: Error = PipelineLink.LinkError.closed
        for _ in 0..<retries {
            do { return try DecodeTCPSocket.connect(host: "127.0.0.1", port: port) }
            catch { last = error; usleep(20_000) }
        }
        throw last
    }

    // THE SEND DIRECTION IS NOT TESTED HERE, and the reason is a limitation of the instrument rather than of the
    // transport. `nc -l` accepts one connection and then waits for EOF; waiting on it with `waitUntilExit()` has no
    // timeout, so a peer that does not exit hangs the suite - twice now, and a hanging test blocks everything.
    // What the send direction needs is a peer that closes on a deadline, which is a small job and not this one.
    //
    // The direction that IS tested is the one that had a real defect in it: `receive` reads `count` from the header
    // to size the payload, and reading the wrong word there (D340) silently produced an empty payload. That is now
    // checked against a frame written by `nc` - an implementation that has never read `PipelineFrame`.

    @Test("receive() decodes a frame that nc wrote")
    func receiveMatchesDecode() async throws {
        let payload = scratch("in.bin")
        let frame = PipelineFrame(token: 11, layer: 40, hidden: [0.5, 1.0, 2.0, 4.0])
        try frame.encode().write(to: URL(fileURLWithPath: payload))

        // `nc` as a *client* rather than a listener, so this half does not depend on bind timing at all.
        let listener = Task.detached { try DecodeTCPSocket.listenAndAccept(host: "127.0.0.1", port: Self.port + 1) }
        // Give the listener a moment to bind before the client process starts.
        usleep(300_000)
        let nc = Process()
        nc.executableURL = URL(fileURLWithPath: "/usr/bin/nc")
        nc.arguments = ["127.0.0.1", "\(Self.port + 1)"]
        nc.standardInput = try FileHandle(forReadingFrom: URL(fileURLWithPath: payload))
        try nc.run()

        let pair = try await listener.value
        let got = try PipelineLink.receive(from: pair.input)
        #expect(got == frame, "receive() did not reconstruct what nc wrote")
        try? pair.input.close(); try? pair.output.close()
        nc.waitUntilExit()
    }
}
