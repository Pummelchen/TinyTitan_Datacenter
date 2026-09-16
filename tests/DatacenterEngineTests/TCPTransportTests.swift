import Darwin
import XCTest

@testable import DatacenterEngine

/// The transport over a real socket: bind, listen, accept, connect — the part a socket pair never did.
///
/// The pair in the earlier tests is a fine harness, but it cannot fail the way a socket fails: nothing
/// binds, nothing listens, no port can be busy, and no connection can be refused. These tests use TCP on
/// loopback so those paths exist, while leaving the link itself — Thunderbolt, LAN, SFP/QSFP, measured —
/// to `DC-008`.
final class TCPTransportTests: XCTestCase {
    private func pair() throws -> (client: SocketContributionTransport, server: SocketContributionTransport, port: Int) {
        let listener = try TCPListener(port: 0)
        // Connect first: with a listener already listening the kernel completes the handshake and the
        // backlog holds it, so neither side has to be on another thread.
        let client = try TCPTransport.connect(host: "127.0.0.1", port: listener.port, timeoutMilliseconds: 5_000)
        let server = try listener.accept(timeoutMilliseconds: 5_000)
        return (client, server, listener.port)
    }

    private struct Fixture {
        let provider: any ExpertWeightProvider
        let shape: MixtureShape
        let input: [Float]
        let indices: [[Int]]
        let chosen: [[Float]]
        let tokens: Int

        func singleNode() throws -> [Float] {
            try MixtureOfExperts.experts(
                hidden: input, tokens: tokens, provider: provider,
                indices: indices, weights: chosen, shape: shape
            )
        }

        func terms(node: Int, nodes: Int) throws -> [ExpertContribution] {
            let owned = OwnedExpertProvider(
                base: provider,
                ownership: ExpertOwnership(nodes: nodes, experts: shape.experts), node: node
            )
            return try MixtureOfExperts.expertContributions(
                hidden: input, tokens: tokens, provider: owned,
                indices: indices, weights: chosen, shape: shape
            )
        }
    }

    private func fixture() throws -> Fixture {
        let root = try XCTUnwrap(
            Bundle.module.url(forResource: "tiny-qwen36", withExtension: nil, subdirectory: "Fixtures")
        )
        let forward = try Qwen3_5Forward(install: root.appendingPathComponent("install"))
        let layer = try forward.loadLayer(0)
        guard case .mixture(let weights, let provider) = layer.feedForward else {
            throw XCTSkip("the fixture's first layer is not a mixture")
        }
        let shape = try forward.mixtureShape()
        var state: UInt64 = 0xBEEF_1234_5678_9ABC
        let tokens = 2
        let input = (0..<(tokens * shape.hiddenSize)).map { _ -> Float in
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Float(Int64(bitPattern: state >> 16) % 3_500) / 512
        }
        let (_, indices, chosen) = try MixtureOfExperts.block(
            hidden: input, tokens: tokens, weights: weights, shape: shape
        )
        return Fixture(
            provider: provider, shape: shape, input: input, indices: indices, chosen: chosen, tokens: tokens
        )
    }

    private func identity(planDigest: String) -> ClusterIdentity {
        ClusterIdentity(
            family: "tiny-qwen36", revision: "abc", experts: 8, hiddenSize: 64, topK: 2,
            planDigest: planDigest
        )
    }

    func testAListenerBindsAndAConnectionCarriesAFrame() throws {
        let (client, server, port) = try pair()
        XCTAssertGreaterThan(port, 0, "a listener asked for port 0 must report the port it got")
        let frame = try ContributionWire.encode(
            [ExpertContribution(token: 0, expert: 3, values: [1, -0.0, 3], scale: 0.5)],
            tokens: 1, hiddenSize: 3
        )
        try client.send(frame)
        XCTAssertEqual(try server.receive(), frame)
        try server.send(frame)
        XCTAssertEqual(try client.receive(), frame)
    }

    func testABringUpHandshakeCompletesOverTCP() throws {
        let (client, server, _) = try pair()
        let ours = NodeDeclaration(identity: identity(planDigest: "d00d"), node: 0, nodes: 2)
        let theirs = NodeDeclaration(identity: identity(planDigest: "d00d"), node: 1, nodes: 2)
        try server.send(try ClusterHandshake.frame(theirs))
        let peers = try ClusterHandshake.perform(ours: ours, peers: [client])
        XCTAssertEqual(peers, [theirs])
        XCTAssertEqual(try ClusterHandshake.decode(try server.receive()), ours)
    }

    /// The claim the whole phase is about, over a socket rather than a socket pair.
    func testAContributionExchangeOverTCPIsBitIdenticalToTheSingleNodeForward() throws {
        let fixture = try fixture()
        let shape = fixture.shape
        let single = try fixture.singleNode()
        let (client, server, _) = try pair()

        // Node 0 is the client, node 1 the server; each ships only its own terms.
        try client.send(
            try ContributionWire.encode(
                try fixture.terms(node: 0, nodes: 2), tokens: fixture.tokens, hiddenSize: shape.hiddenSize
            )
        )
        let fromServer = try ContributionWire.decode(try server.receive()).contributions
        try server.send(
            try ContributionWire.encode(
                try fixture.terms(node: 1, nodes: 2), tokens: fixture.tokens, hiddenSize: shape.hiddenSize
            )
        )
        let fromClient = try ContributionWire.decode(try client.receive()).contributions

        let mineAtServer = try fixture.terms(node: 1, nodes: 2)
        let merged = try ShardExchange.merge(fromServer + mineAtServer, indices: fixture.indices)
        let reduced = try OrderedReduction.accumulate(
            merged, tokens: fixture.tokens, hiddenSize: shape.hiddenSize
        )
        for (index, element) in reduced.enumerated() where element.bitPattern != single[index].bitPattern {
            XCTFail("over TCP, element \(index) differs from the single-node forward by bits")
            return
        }
        XCTAssertEqual(fromClient.count, try fixture.terms(node: 0, nodes: 2).count)
    }

    // MARK: - `D19`'s failure semantics, over a real socket

    /// One end wrapped in a transport, the other held raw so the test can send half a frame and stop.
    ///
    /// The exchange tests prove the *rules* with a counting transport, and they are the right place for
    /// them — but every one of those peers is a Swift object that can be told to misbehave. These tests
    /// use a descriptor that really closes and a length prefix that really promises more than arrives, so
    /// the rules are checked against the thing that produces the bytes.
    private func socketPair(timeoutMilliseconds: Int = 5_000) throws -> (reader: SocketContributionTransport, peer: Int32) {
        var descriptors: [Int32] = [0, 0]
        guard socketpair(AF_UNIX, SOCK_STREAM, 0, &descriptors) == 0 else {
            throw ContributionTransportError.socket("socketpair failed: \(String(cString: strerror(errno)))")
        }
        let reader = SocketContributionTransport(
            fileDescriptor: descriptors[0], closeOnDealloc: true, timeoutMilliseconds: timeoutMilliseconds
        )
        return (reader, descriptors[1])
    }

    private func send(_ bytes: [UInt8], to descriptor: Int32) {
        _ = bytes.withUnsafeBufferPointer { buffer in
            write(descriptor, buffer.baseAddress, buffer.count)
        }
    }

    /// A frame is announced and the peer goes away before it arrives.
    func testAPeerThatClosesBetweenFramesReportsClosedRatherThanWaiting() throws {
        let (reader, peer) = try socketPair()
        close(peer)
        XCTAssertThrowsError(try reader.receive()) { error in
            XCTAssertEqual(
                error as? ContributionTransportError, .closed,
                "a peer that is gone must be reported as gone, not waited out until a deadline"
            )
        }
    }

    /// The distinction `D19` draws, produced by a real socket: some of the frame arrived, so the stream is
    /// desynchronised — a later read would return the *previous* frame's tail.
    func testAPeerThatClosesMidFrameReportsTruncation() throws {
        let (reader, peer) = try socketPair()
        send([16, 0, 0, 0], to: peer)          // a frame of sixteen bytes is announced…
        send([1, 2, 3, 4, 5], to: peer)        // …and five arrive
        close(peer)
        XCTAssertThrowsError(try reader.receive()) { error in
            XCTAssertEqual(
                error as? ContributionTransportError, .truncated(needed: 16, got: 5),
                "a partially delivered frame is not a closed connection, and vice versa"
            )
        }
    }

    /// A peer that stops part-way and says nothing more — the case that must **not** be retried, because
    /// the bytes after it are not a frame boundary. Asserted on the reason, so the retry rule has a fact
    /// behind it rather than a hope that the two paths look the same.
    func testAPeerThatStopsPartWayThroughAFrameReportsThatReason() throws {
        let (reader, peer) = try socketPair(timeoutMilliseconds: 60)
        send([64, 0, 0, 0], to: peer)          // sixty-four bytes announced
        send([9, 9, 9], to: peer)              // three sent, then silence
        XCTAssertThrowsError(try reader.receive()) { error in
            XCTAssertEqual(
                error as? ContributionTransportError, .timedOut(midFrame: true),
                "mid-frame is what decides whether the exchange retries, so it must be distinguishable"
            )
        }
        close(peer)
    }

    func testAcceptTimesOutWhenNobodyConnects() throws {
        let listener = try TCPListener(port: 0)
        let started = Date()
        XCTAssertThrowsError(try listener.accept(timeoutMilliseconds: 80)) { error in
            guard case ContributionTransportError.timedOut(let midFrame) = error else {
                return XCTFail("expected the accept to time out, got \(error)")
            }
            XCTAssertFalse(midFrame)
        }
        XCTAssertLessThan(Date().timeIntervalSince(started), 5, "the accept must give up on its deadline")
    }

    func testConnectingWhereNothingListensIsRefused() throws {
        // Bind and immediately release a port, so the number is almost certainly unbound.
        let port: Int
        do {
            let listener = try TCPListener(port: 0)
            port = listener.port
        }
        XCTAssertThrowsError(try TCPTransport.connect(host: "127.0.0.1", port: port, timeoutMilliseconds: 500)) {
            error in
            guard case TCPListener.Error.cannotConnect = error else {
                return XCTFail("expected a refused connection, got \(error)")
            }
        }
    }

    func testASecondListenerOnABusyPortIsRefused() throws {
        let listener = try TCPListener(port: 0)
        XCTAssertThrowsError(try TCPListener(port: listener.port)) { error in
            guard case TCPListener.Error.cannotBind = error else {
                return XCTFail("expected the bind to be refused, got \(error)")
            }
        }
    }
}
