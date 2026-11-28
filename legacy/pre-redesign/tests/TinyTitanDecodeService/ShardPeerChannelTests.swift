import Foundation
import Testing

@testable import TinyTitanDecodeProtocol

/// The peer channel, over a real socket rather than a mock: a channel tested against a stub is a channel tested
/// against its own assumptions about short reads, and short reads are the whole reason the loop exists.
/// **Serialised, and this was learned the hard way twice.** All four tests bind the same fixed port, and
/// swift-testing runs tests in parallel by default, so without this they collide: three failed with
/// `ECONNREFUSED` and the fourth died on `EPIPE` after connecting to *another test's* listener. That is exactly
/// the failure `D143` already recorded for the socket tests, and writing a new suite I repeated it.
///
/// The alternative is a distinct port per test; serialising is the smaller change that makes the port genuinely
/// exclusive. A fixed port is used at all because `listenAndAccept` closes the listener as soon as it has a
/// connection, so there is no way to ask the kernel which port it picked.
@Suite("Shard peer channel", .serialized)
struct ShardPeerChannelTests {
    private static let port: UInt16 = 45923

    private static func pair() async throws -> (ShardPeerChannel, ShardPeerChannel) {
        let server = Task.detached { try DecodeTCPSocket.listenAndAccept(host: "127.0.0.1", port: port) }
        var last: Error = POSIXError(.ECONNREFUSED)
        for _ in 0..<100 {
            do {
                let client = try DecodeTCPSocket.connect(host: "127.0.0.1", port: port)
                let accepted = try await server.value
                return (ShardPeerChannel(input: client.input, output: client.output),
                        ShardPeerChannel(input: accepted.input, output: accepted.output))
            } catch {
                last = error
                try await Task.sleep(nanoseconds: 20_000_000)
            }
        }
        throw last
    }

    /// A real exchange frame, not a byte string: the point is that the two layers compose.
    @Test("an exchange frame crosses the channel intact")
    func exchangeFrameCrossesTheChannel() async throws {
        let (client, server) = try await Self.pair()
        let values = (0..<(2 * 2048)).map { Float($0) * 0.125 }
        let reply = ShardExchange.Reply(layer: 3, slots: [2, 5], dimensions: 2048, values: values)

        try client.send(try ShardExchange.encode(reply))
        let decoded = try ShardExchange.decodeReply(from: try server.receive())
        #expect(decoded == reply)
        #expect(decoded.values.map(\.bitPattern) == reply.values.map(\.bitPattern))
    }

    /// Several frames in a row, because a channel that works once and desynchronises on the second frame is the
    /// failure a single-frame test cannot see: the length prefix is what keeps them aligned.
    @Test("consecutive frames stay aligned")
    func consecutiveFramesDoNotDesynchronise() async throws {
        let (client, server) = try await Self.pair()
        let frames = (0..<5).map { index in
            ShardExchange.Reply(layer: index, slots: [index], dimensions: 8,
                                values: (0..<8).map { Float($0 + index) })
        }
        for frame in frames { try client.send(try ShardExchange.encode(frame)) }
        for (index, frame) in frames.enumerated() {
            let decoded = try ShardExchange.decodeReply(from: try server.receive())
            #expect(decoded == frame, "frame \(index) arrived out of step")
        }
    }

    @Test("an oversized frame is refused before it is sent")
    func oversizedFrameIsRefused() async throws {
        let (client, _) = try await Self.pair()
        let tooBig = Data(repeating: 0, count: ShardPeerChannel.maximumFrameBytes + 1)
        #expect(throws: ShardPeerChannel.Error.self) { try client.send(tooBig) }
    }

    /// A peer that disconnects between the length and the body must surface as an error rather than a hang or a
    /// short frame — the mid-frame case the transport tests already cover at the socket layer, asserted here at
    /// the channel layer where the length prefix makes it a different bug.
    @Test("a peer closing mid-frame is an error, not a short read")
    func midFrameCloseIsAnError() async throws {
        let (client, server) = try await Self.pair()
        var length = UInt32(4096).littleEndian
        var partial = withUnsafeBytes(of: &length) { Data($0) }
        partial.append(Data(repeating: 7, count: 100))
        try client.output.write(contentsOf: partial)
        try client.output.close()
        try client.input.close()
        #expect(throws: (any Swift.Error).self) { try server.receive() }
    }
}
