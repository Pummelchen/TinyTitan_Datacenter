import Darwin
import XCTest

@testable import DatacenterEngine

/// `DC-043`: what a single-user cluster run does when a node goes quiet, stalls mid-frame, or dies.
///
/// The three failures must not look alike, and the retry rule is the heart of it: a clean timeout is
/// retried **because** the reduction order is canonical, so the same terms sent twice produce the same
/// bits; a stream that stopped mid-frame is not retried, because the next read would return the tail of
/// the previous frame and the failure would surface somewhere else entirely.
final class ShardExchangeTests: XCTestCase {
    /// Counts the calls that reach the transport, so "was this retried?" is an exact assertion rather
    /// than a stopwatch.
    private final class CountingTransport: ContributionTransport {
        private let inner: any ContributionTransport
        private(set) var sends = 0
        private(set) var receives = 0
        private(set) var deadlines: [Int] = []

        init(_ inner: any ContributionTransport) { self.inner = inner }

        func send(_ frame: Data) throws {
            sends += 1
            try inner.send(frame)
        }

        func receive() throws -> Data {
            receives += 1
            return try inner.receive()
        }

        /// Forwarded, and counted. Without this the wrapper takes the protocol's no-op default and the
        /// deadline never reaches the socket — which is how the first version of this test spent 90
        /// seconds proving a 60 ms policy.
        func applyTimeout(milliseconds: Int) {
            deadlines.append(milliseconds)
            inner.applyTimeout(milliseconds: milliseconds)
        }
    }

    private struct Fixture {
        let weights: MixtureWeights
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
        var state: UInt64 = 0x1234_5678_9ABC_DEF0
        let tokens = 2
        let input = (0..<(tokens * shape.hiddenSize)).map { _ -> Float in
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Float(Int64(bitPattern: state >> 14) % 3_000) / 256
        }
        let (_, indices, chosen) = try MixtureOfExperts.block(
            hidden: input, tokens: tokens, weights: weights, shape: shape
        )
        return Fixture(
            weights: weights, provider: provider, shape: shape,
            input: input, indices: indices, chosen: chosen, tokens: tokens
        )
    }

    private func assertSameBits(_ one: [Float], _ other: [Float], _ what: String) {
        XCTAssertEqual(one.count, other.count, "\(what): different lengths")
        for index in 0..<min(one.count, other.count) where one[index].bitPattern != other[index].bitPattern {
            XCTFail("\(what): element \(index) is \(one[index].bitPattern) against \(other[index].bitPattern)")
            return
        }
    }

    /// The happy path, through the exchange rather than the codec: a two-node all-reduce equals the
    /// single-node forward bit for bit.
    func testACompletedExchangeMatchesTheSingleNodeForward() throws {
        let fixture = try fixture()
        let shape = fixture.shape
        let single = try fixture.singleNode()
        let (left, right) = try SocketPair.make()

        // Each side runs its own half of the exchange concurrently, which is what the real run does;
        // the two sends happen before either receive, and the socket buffers hold these frames.
        let mine = try fixture.terms(node: 0, nodes: 2)
        let theirs = try fixture.terms(node: 1, nodes: 2)
        try left.send(try ContributionWire.encode(theirs, tokens: fixture.tokens, hiddenSize: shape.hiddenSize))
        let reduced = try ShardExchange.allReduce(
            own: mine, peers: [right], indices: fixture.indices,
            tokens: fixture.tokens, hiddenSize: shape.hiddenSize
        )
        assertSameBits(single, reduced, "two nodes over a socket")
    }

    /// A peer that says nothing is retried, and the number of attempts is the policy's.
    func testASilentPeerIsRetriedAndThenFailsTheRun() throws {
        let fixture = try fixture()
        let (left, right) = try SocketPair.make()
        _ = left  // never writes
        let counting = CountingTransport(right)
        let policy = ExchangePolicy(receiveTimeoutMilliseconds: 60, attempts: 3)
        let mine = try fixture.terms(node: 0, nodes: 2)

        // The duration is asserted, not just the behaviour. The first version of this test passed while
        // the policy's 60 ms never reached the transport and the default 30 s did the work — three
        // attempts, 90 seconds, green. A parameter that is not wired up is a lie, and only a clock
        // catches it.
        let started = Date()
        defer {
            XCTAssertLessThan(
                Date().timeIntervalSince(started), 5,
                "the policy's deadline must reach the transport: three 60 ms attempts cannot take this long"
            )
        }
        XCTAssertThrowsError(
            try ShardExchange.allReduce(
                own: mine, peers: [counting], indices: fixture.indices,
                tokens: fixture.tokens, hiddenSize: fixture.shape.hiddenSize, policy: policy
            )
        ) { error in
            guard case ContributionTransportError.timedOut(let midFrame) = error else {
                return XCTFail("expected a timeout, got \(error)")
            }
            XCTAssertFalse(midFrame, "nothing arrived at all, so this is a silent peer, not a stalled frame")
        }
        XCTAssertEqual(counting.receives, 3, "a clean timeout is retryable, and the policy says three attempts")
        XCTAssertEqual(counting.deadlines, [60, 60, 60], "each attempt must set the policy's deadline on the transport")
        XCTAssertEqual(counting.sends, 3, "each attempt resends; a peer that already had it would merge the duplicate")
    }

    /// A peer that stops part-way through a frame is **not** retried: the stream is desynchronised.
    func testAPeerThatStopsMidFrameIsNotRetried() throws {
        let fixture = try fixture()
        var descriptors: [Int32] = [-1, -1]
        guard socketpair(AF_UNIX, SOCK_STREAM, 0, &descriptors) == 0 else {
            throw XCTSkip("socketpair is unavailable")
        }
        let transport = SocketContributionTransport(fileDescriptor: descriptors[1], timeoutMilliseconds: 60)
        let writer = FileHandle(fileDescriptor: descriptors[0], closeOnDealloc: true)
        // A length prefix promising a frame that never comes.
        try writer.write(contentsOf: Data([0x00, 0x10, 0x00, 0x00]))
        let counting = CountingTransport(transport)
        let mine = try fixture.terms(node: 0, nodes: 2)

        XCTAssertThrowsError(
            try ShardExchange.allReduce(
                own: mine, peers: [counting], indices: fixture.indices,
                tokens: fixture.tokens, hiddenSize: fixture.shape.hiddenSize,
                policy: ExchangePolicy(receiveTimeoutMilliseconds: 60, attempts: 3)
            )
        ) { error in
            guard case ContributionTransportError.timedOut(let midFrame) = error else {
                return XCTFail("expected a timeout, got \(error)")
            }
            XCTAssertTrue(midFrame, "the frame had begun, so the stream cannot be reused")
        }
        XCTAssertEqual(
            counting.receives, 1,
            "a desynchronised stream must not be retried: the next read would return the previous frame's tail"
        )
    }

    /// A resend after a timeout is harmless, because the bits are identical — which is what makes
    /// retrying safe at all.
    func testAResendMergesBecauseTheBitsAreIdentical() throws {
        let fixture = try fixture()
        let shape = fixture.shape
        let single = try fixture.singleNode()
        let mine = try fixture.terms(node: 0, nodes: 2)
        let theirs = try fixture.terms(node: 1, nodes: 2)
        let frame = try ContributionWire.encode(theirs, tokens: fixture.tokens, hiddenSize: shape.hiddenSize)

        // The same peer frame twice: a retry on the other side, seen from here.
        let doubled = mine + theirs + theirs
        let merged = try ShardExchange.merge(doubled, indices: fixture.indices)
        XCTAssertEqual(merged.count, mine.count + theirs.count, "a bit-identical duplicate must collapse")
        assertSameBits(
            single,
            try OrderedReduction.accumulate(merged, tokens: fixture.tokens, hiddenSize: shape.hiddenSize),
            "a retry must not move the bits"
        )
        XCTAssertFalse(frame.isEmpty)
    }

    /// A duplicate that is *not* bit-identical means two nodes disagree about a term they both claim to
    /// have computed, and it is refused where the diagnosis is still possible.
    func testADuplicateWithDifferentBitsIsRefused() throws {
        let fixture = try fixture()
        let mine = try fixture.terms(node: 0, nodes: 2)
        let first = try XCTUnwrap(mine.first)
        var changed = first.values
        changed[0] = changed[0].nextUp
        let contradicting = ExpertContribution(
            token: first.token, expert: first.expert, values: changed, scale: first.scale
        )

        XCTAssertThrowsError(try ShardExchange.merge(mine + [contradicting], indices: fixture.indices)) { error in
            guard case ShardExchangeError.duplicateWithDifferentBits(let key) = error else {
                return XCTFail("expected a contradiction to be refused, got \(error)")
            }
            XCTAssertEqual(key, "\(first.token):\(first.expert)")
        }
    }

    /// The rule the whole design rests on: a term that never arrives fails the run. It never produces a
    /// smaller sum that looks like an answer.
    func testAMissingTermFailsTheRunInsteadOfShrinkingTheSum() throws {
        let fixture = try fixture()
        // Only one node's half arrives, so half the selected experts are missing.
        let mine = try fixture.terms(node: 0, nodes: 2)
        XCTAssertThrowsError(try ShardExchange.merge(mine, indices: fixture.indices)) { error in
            guard case ShardExchangeError.incomplete(let missing) = error else {
                return XCTFail("expected an incomplete reduction to fail, got \(error)")
            }
            XCTAssertFalse(missing.isEmpty)
            XCTAssertEqual(
                missing.count,
                fixture.indices.flatMap { $0 }.count - mine.count,
                "every selected expert with no term must be reported, not just the first"
            )
        }
    }

    /// A peer that has gone away entirely is a failure, not a timeout to wait out.
    func testAClosedPeerFailsTheRun() throws {
        var descriptors: [Int32] = [-1, -1]
        guard socketpair(AF_UNIX, SOCK_STREAM, 0, &descriptors) == 0 else {
            throw XCTSkip("socketpair is unavailable")
        }
        close(descriptors[0])
        let transport = SocketContributionTransport(fileDescriptor: descriptors[1], timeoutMilliseconds: 500)
        XCTAssertThrowsError(try transport.receive()) { error in
            guard case ContributionTransportError.closed = error else {
                return XCTFail("expected a closed connection, got \(error)")
            }
        }
    }
}
