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
