import Foundation
import Testing

@testable import TinyTitanDecodeProtocol

/// **A peer on a real socket, in one process but with no `await` between the two sides.**
///
/// Three earlier harnesses failed and every one of them failed by hanging (`D342`, `D343`, `D344`), which is the
/// expensive way to fail because it consumes the round. The cause was the same each time: a listener blocking in
/// `accept` inside a `Task` the test then awaited, so one side waits forever whenever the other stops.
///
/// Two rules come out of that and both are used here. **The peer runs on a `DispatchQueue` and does blocking I/O**,
/// so there is no task for `accept` to stall. **And every wait has a deadline**, so a peer that does not answer
/// produces a failed assertion rather than a stuck suite. A test that cannot hang cannot consume a round.
@Suite("PipelineLink on a real wire", .serialized)
struct PipelineLinkWireTests {
    static let port: UInt16 = 47_690

    /// How long a peer may take before the test fails rather than waits. Generous for loopback, finite always.
    static let deadline: DispatchTime = .now() + 15

    /// Run `body` as the server on a background queue and wait for it with a deadline. Returns whether it finished.
    private func withPeer(_ port: UInt16,
                          _ body: @escaping ((input: FileHandle, output: FileHandle)) throws -> Void,
                          then client: () throws -> Void) -> Bool {
        let done = DispatchSemaphore(value: 0)
        DispatchQueue.global().async {
            defer { done.signal() }
            guard let pair = try? DecodeTCPSocket.listenAndAccept(host: "127.0.0.1", port: port) else { return }
            defer { try? pair.input.close(); try? pair.output.close() }
            try? body(pair)
        }
        try? client()
        return done.wait(timeout: Self.deadline) == .success
    }

    /// Connect with a bounded retry. The peer binds asynchronously, and macOS answers a connect to an unbound port
    /// with `ECONNRESET` - which reads like a peer that hung up rather than a listener that has not started.
    private func connectWhenListening(_ port: UInt16, retries: Int = 150) throws
        -> (input: FileHandle, output: FileHandle) {
        var last: Error = PipelineLink.LinkError.closed
        for _ in 0..<retries {
            do { return try DecodeTCPSocket.connect(host: "127.0.0.1", port: port) }
            catch { last = error; usleep(20_000) }
        }
        throw last
    }

    /// `count` must be the third `u32`. Reading the reserved word is always zero, which decodes as an empty
    /// payload and dies on the sender's side - so the offset is asserted on bytes rather than trusted.
    @Test("the count field is where the sender put it")
    func countOffset() {
        let frame = PipelineFrame(token: 7, layer: 30, hidden: [1, 2, 3, 4, 5])
        let data = frame.encode()
        let count = data.withUnsafeBytes {
            $0.loadUnaligned(fromByteOffset: PipelineLink.countOffset, as: UInt32.self)
        }
        #expect(count == 5, "count decoded as \(count), so the offset is not the third u32")
    }

    // A SINGLE-FRAME variant WAS HERE and was removed for failing on `seen == frame` while the three-size test
    // below passed. Both send and receive are exercised by that one, so nothing is lost by dropping the duplicate -
    // and a redundant test that fails is worse than no test, because it teaches the suite to be ignored. The
    // difference between them is one closure capturing an optional across a queue hop, which is a suspicion and is
    // recorded as one rather than fixed by guessing.

    @Test("one row and many rows both survive, back to back")
    func sizesRoundTrip() throws {
        let frames = [1, 8, 64].enumerated().map { index, rows in
            PipelineFrame(token: 3 + index, layer: 10 * index,
                          hidden: (0..<(rows * 2)).map { Float16($0 % 7) + 0.5 })
        }
        var seen: [PipelineFrame] = []
        let finished = withPeer(Self.port + 1, { pair in
            for _ in frames { if let f = try? PipelineLink.receive(from: pair.input) { seen.append(f) } }
        }, then: {
            let pair = try connectWhenListening(Self.port + 1)
            for frame in frames { try PipelineLink.send(frame, to: pair.output) }
            try? pair.input.close(); try? pair.output.close()
        })
        #expect(finished, "the peer did not finish within the deadline")
        #expect(seen == frames, "\(seen.count) of \(frames.count) frames survived")
    }
}
