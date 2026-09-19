import Foundation
import Testing

@testable import TinyTitanDecodeProtocol

/// `PipelineLink` is the first code in the pipeline to touch a socket, so it is tested on one: a listener and a
/// client over loopback, frames both ways. A codec tested only against `Data` would not exercise the half that
/// matters, because a TCP stream has no message boundaries and the framing is the whole point.
///
/// The structure is deliberately the one `DecodeTCPSocketTests` arrived at - `.serialized`, a fixed port in the
/// ephemeral range, and a reader that loops - because that file records what happens without it.
@Suite("PipelineLink over a socket", .serialized)
struct PipelineLinkTests {
    /// A port in the ephemeral range, fixed so the test is deterministic and distinct from every other suite's.
    static let port: UInt16 = 47_681

    private static func acceptInBackground(port: UInt16) -> Task<(input: FileHandle, output: FileHandle), Error> {
        Task.detached { try DecodeTCPSocket.listenAndAccept(host: "127.0.0.1", port: port) }
    }

    /// Connect with a bounded retry. `listenAndAccept` binds and accepts in one call and so cannot signal that
    /// it is listening; macOS answers a connect to an unbound port with `ECONNRESET`, which reads like a peer
    /// that hung up rather than a listener that has not started.
    private static func connectWhenListening(port: UInt16, retries: Int = 100) throws
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

    // The socket round-trip is NOT here yet and its absence is deliberate (D340). A version that ran a
    // listener and a client as two detached tasks in one process hung rather than failed, and a hanging test
    // blocks the suite - which is worse than a red one. The codec test above is the one that catches the defect
    // that actually cost a round (reading the reserved word instead of `count`), and it needs no socket. The
    // round-trip belongs in a two-process test, where neither side can be starved or deallocated by the other.
}
