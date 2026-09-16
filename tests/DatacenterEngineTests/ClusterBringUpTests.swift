import XCTest

@testable import DatacenterEngine

/// `DC-042`: a cluster run refuses to start when its nodes disagree.
///
/// Every failure here is one that would otherwise appear as a plausible number: a different model
/// revision, a different plan, a different geometry. The handshake exists so that the disagreement is a
/// message before a token is computed rather than a wrong answer afterwards.
final class ClusterBringUpTests: XCTestCase {
    private func identity(
        family: String = "tiny-qwen36", revision: String = "abc123", experts: Int = 8,
        hiddenSize: Int = 64, topK: Int = 2, planDigest: String = "deadbeef", schema: Int = 1
    ) -> ClusterIdentity {
        ClusterIdentity(
            family: family, revision: revision, experts: experts, hiddenSize: hiddenSize,
            topK: topK, planDigest: planDigest, schema: schema
        )
    }

    private func declaration(_ identity: ClusterIdentity, node: Int, nodes: Int) -> NodeDeclaration {
        NodeDeclaration(identity: identity, node: node, nodes: nodes)
    }

    private func assertRefused(
        ours: NodeDeclaration, theirs: NodeDeclaration, _ check: (ClusterBringUpError) -> Void,
        file: StaticString = #filePath, line: UInt = #line
    ) throws {
        let (left, right) = try SocketPair.make()
        _ = left  // holds the connection open; the peer frame is pre-sent below
        try right.send(try ClusterHandshake.frame(theirs))
        XCTAssertThrowsError(
            try ClusterHandshake.perform(ours: ours, peers: [left]), file: file, line: line
        ) { error in
            guard let bringUp = error as? ClusterBringUpError else {
                return XCTFail("expected a bring-up refusal, got \(error)", file: file, line: line)
            }
            check(bringUp)
        }
    }

    /// The happy path, and both directions of it: this node learns who its peer is, and the peer
    /// received this node's declaration.
    func testAMatchingClusterCompletesBringUp() throws {
        let identity = identity()
        let ours = declaration(identity, node: 0, nodes: 2)
        let theirs = declaration(identity, node: 1, nodes: 2)
        let (left, right) = try SocketPair.make()
        try right.send(try ClusterHandshake.frame(theirs))

        let peers = try ClusterHandshake.perform(ours: ours, peers: [left])
        XCTAssertEqual(peers, [theirs], "bring-up must return what the peers declared")

        // The other direction: what the peer receives is this node's declaration, unchanged.
        let received = try ClusterHandshake.decode(try right.receive())
        XCTAssertEqual(received, ours)
        XCTAssertEqual(try received.identity.canonicalDigest(), try identity.canonicalDigest())
    }

    /// The check that matters most: two nodes with different plans must not run together.
    func testADifferentPlanDigestIsRefusedAndNamed() throws {
        try assertRefused(
            ours: declaration(identity(planDigest: "aaaa"), node: 0, nodes: 2),
            theirs: declaration(identity(planDigest: "bbbb"), node: 1, nodes: 2)
        ) { error in
            guard case ClusterBringUpError.disagreement(let node, let fields) = error else {
                return XCTFail("expected a named disagreement, got \(error)")
            }
            XCTAssertEqual(node, 1)
            XCTAssertEqual(fields, ["planDigest"], "the refusal must say which agreement failed")
        }
    }

    func testEveryComparedFieldIsCheckedAndNamed() throws {
        let cases: [(ClusterIdentity, String)] = [
            (identity(schema: 99), "schema"),
            (identity(family: "other"), "family"),
            (identity(revision: "different"), "revision"),
            (identity(experts: 16), "experts"),
            (identity(hiddenSize: 128), "hiddenSize"),
            (identity(topK: 4), "topK"),
        ]
        for (other, field) in cases {
            try assertRefused(
                ours: declaration(identity(), node: 0, nodes: 2),
                theirs: declaration(other, node: 1, nodes: 2)
            ) { error in
                guard case ClusterBringUpError.disagreement(_, let fields) = error else {
                    return XCTFail("expected a disagreement for \(field), got \(error)")
                }
                XCTAssertEqual(fields, [field], "one changed field should be reported as one field")
            }
        }
    }

    func testANodeCountDisagreementIsRefused() throws {
        try assertRefused(
            ours: declaration(identity(), node: 0, nodes: 2),
            theirs: declaration(identity(), node: 1, nodes: 3)
        ) { error in
            XCTAssertEqual(error, .nodeCountMismatch(ours: 2, theirs: 3))
        }
    }

    func testAPeerClaimingThisNodesIdentityIsRefused() throws {
        try assertRefused(
            ours: declaration(identity(), node: 0, nodes: 2),
            theirs: declaration(identity(), node: 0, nodes: 2)
        ) { error in
            guard case ClusterBringUpError.unexpectedNode(let node, let expected) = error else {
                return XCTFail("expected an unexpected node, got \(error)")
            }
            XCTAssertEqual(node, 0)
            XCTAssertEqual(expected, [1])
        }
    }

    func testAMissingNodeIsRefused() throws {
        try assertRefused(
            ours: declaration(identity(), node: 0, nodes: 3),
            theirs: declaration(identity(), node: 2, nodes: 3)
        ) { error in
            XCTAssertEqual(error, .missingNode(1))
        }
    }

    func testAFrameThatIsNotADeclarationIsRefused() throws {
        let (left, right) = try SocketPair.make()
        _ = left
        // A contribution frame travelling on the bring-up path: the two frame types must be
        // distinguishable, or a handshake could read terms as a declaration.
        let contributions = try ContributionWire.encode(
            [ExpertContribution(token: 0, expert: 0, values: [1, 2], scale: 1)], tokens: 1, hiddenSize: 2
        )
        try right.send(contributions)
        XCTAssertThrowsError(
            try ClusterHandshake.perform(
                ours: declaration(identity(), node: 0, nodes: 2), peers: [left]
            )
        ) { error in
            guard case ClusterBringUpError.notADeclaration = error else {
                return XCTFail("expected the frame to be refused, got \(error)")
            }
        }
    }

    /// And the other way round: a declaration frame must not decode as contributions.
    func testADeclarationFrameIsNotContributions() throws {
        let frame = try ClusterHandshake.frame(declaration(identity(), node: 0, nodes: 2))
        XCTAssertThrowsError(try ContributionWire.decode(frame)) { error in
            XCTAssertEqual(error as? ContributionWireError, .badMagic)
        }
    }

    func testANonPositiveTimeoutStillRefusesAHandshakeWithNoPeers() throws {
        XCTAssertThrowsError(
            try ClusterHandshake.perform(ours: declaration(identity(), node: 0, nodes: 1), peers: [])
        ) { error in
            XCTAssertEqual(error as? ClusterBringUpError, .noPeers)
        }
    }

    // MARK: - the config is data too

    func testAConfigIsValidatedAgainstThisNode() throws {
        let config = ClusterConfig(endpoints: [
            NodeEndpoint(host: "10.0.0.1", port: 9000),
            NodeEndpoint(host: "10.0.0.2", port: 9000),
        ])
        try config.validate(thisNode: 0)
        try config.validate(thisNode: 1)
        XCTAssertThrowsError(try config.validate(thisNode: 2)) { error in
            XCTAssertEqual(error as? ClusterConfig.Error, .thisNodeNotInConfig(2, nodes: 2))
        }
    }

    func testAConfigWithADuplicateOrBadEndpointIsRefused() throws {
        let duplicate = ClusterConfig(endpoints: [
            NodeEndpoint(host: "10.0.0.1", port: 9000),
            NodeEndpoint(host: "10.0.0.1", port: 9000),
        ])
        XCTAssertThrowsError(try duplicate.validate(thisNode: 0)) { error in
            guard case ClusterConfig.Error.duplicateEndpoint = error else {
                return XCTFail("expected a duplicate endpoint, got \(error)")
            }
        }

        let emptyHost = ClusterConfig(endpoints: [NodeEndpoint(host: "", port: 9000)])
        XCTAssertThrowsError(try emptyHost.validate(thisNode: 0)) { error in
            XCTAssertEqual(error as? ClusterConfig.Error, .badEndpoint(node: 0))
        }

        let badPort = ClusterConfig(endpoints: [NodeEndpoint(host: "10.0.0.1", port: 0)])
        XCTAssertThrowsError(try badPort.validate(thisNode: 0)) { error in
            XCTAssertEqual(error as? ClusterConfig.Error, .badEndpoint(node: 0))
        }
    }

    /// The declaration's plan digest must be the digest of the plan this node will actually use, and a
    /// config that disagrees with the plan's node count is a mistake worth catching here.
    func testTheDeclarationAgreesWithThePlanAndTheConfig() throws {
        let plan = ShardPlan.generate(family: "tiny-qwen36", experts: 8, nodes: 2)
        let digest = try plan.canonicalDigest()
        let declared = identity(experts: plan.experts, planDigest: digest)
        XCTAssertEqual(declared.differences(from: declared), [])

        let config = ClusterConfig(endpoints: [
            NodeEndpoint(host: "10.0.0.1", port: 9000), NodeEndpoint(host: "10.0.0.2", port: 9000),
        ])
        XCTAssertEqual(config.nodes, plan.nodes, "a config for a different node count is a different cluster")

        let stale = identity(experts: plan.experts, planDigest: "0000")
        XCTAssertEqual(stale.differences(from: declared), ["planDigest"])
    }
}
