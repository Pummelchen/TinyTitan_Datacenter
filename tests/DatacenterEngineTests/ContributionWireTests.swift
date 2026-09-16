import Darwin
import XCTest

@testable import DatacenterEngine

/// The wire protocol (`D18`): what crosses between nodes, and what must survive the crossing.
///
/// Two properties matter and they are different. The codec must be **bit-exact** — a float that goes
/// through the wire and comes back must have the same bits, including `-0.0`, subnormals and NaN
/// payloads, because `D17`'s whole argument is that the low bits are the result. And the decoder must
/// be **hostile-input safe** — a length field is not a promise, and a decoder that trusts one is a
/// denial-of-service with extra steps.
final class ContributionWireTests: XCTestCase {
    private struct Source {
        var state: UInt64 = 0xD1B5_4A32_D192_ED03
        mutating func next() -> Float {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Float(Int64(bitPattern: state >> 13) % 4_000) / 512
        }
    }

    private func frames(tokens: Int, width: Int, terms: Int) -> [ExpertContribution] {
        var source = Source()
        return (0..<terms).map { index in
            ExpertContribution(
                token: index % tokens, expert: index * 7 % 64,
                values: (0..<width).map { _ in source.next() },
                scale: Float(index) / 16
            )
        }
    }

    // MARK: - the codec

    func testRoundTripPreservesEveryBit() throws {
        // The awkward values, on purpose: the sign of zero, a subnormal, an infinity, and a NaN with a
        // payload. A codec that went through a decimal or a "cleanup" step would pass every other test
        // here and fail this one.
        let awkward: [Float] = [
            0.0, -0.0, Float.leastNonzeroMagnitude, -Float.leastNonzeroMagnitude,
            Float.infinity, -Float.infinity, Float.nan, Float(bitPattern: 0x7FC0_1234),
            1.0, -1.0, 3.4028235e38, 1.1754944e-38,
        ]
        // The second term is the same width as the frame declares: the codec refused the first version
        // of this test, which declared a 12-wide frame and passed a 1-wide term, and it was right to.
        let subnormal = [Float(bitPattern: 0x8000_0001)]
            + [Float](repeating: 0, count: awkward.count - 1)
        let terms = [
            ExpertContribution(token: 0, expert: 0, values: awkward, scale: -0.0),
            ExpertContribution(token: 1, expert: 5, values: subnormal, scale: 0.5),
        ]
        let frame = try ContributionWire.decode(
            ContributionWire.encode(terms, tokens: 2, hiddenSize: awkward.count)
        )
        XCTAssertEqual(frame.tokens, 2)
        XCTAssertEqual(frame.hiddenSize, awkward.count)
        let decoded = frame.contributions
        XCTAssertEqual(decoded.count, 2)
        for (index, value) in decoded[0].values.enumerated() {
            XCTAssertEqual(
                value.bitPattern, awkward[index].bitPattern,
                "value \(index) changed bits on the wire"
            )
        }
        XCTAssertEqual(decoded[0].scale.bitPattern, (-0.0 as Float).bitPattern, "the sign of zero must survive")
        XCTAssertEqual(decoded[1].values[0].bitPattern, 0x8000_0001, "a subnormal must survive")
        XCTAssertEqual(decoded[1].expert, 5)
        XCTAssertEqual(decoded[1].token, 1)
    }

    func testAHostileWidthIsRefusedRatherThanAllocated() throws {
        // A well-formed header followed by a term claiming a width of 2^30. Nothing may be reserved for
        // it: the refusal has to happen on the declared number, not after trying to read it.
        var bytes = Array("TTDC".utf8)
        bytes.append(contentsOf: [1, 0])  // version 1
        bytes.append(contentsOf: [1, 0, 0, 0])  // tokens 1
        bytes.append(contentsOf: [1, 0, 0, 0])  // hiddenSize 1
        bytes.append(contentsOf: [1, 0, 0, 0])  // count 1
        bytes.append(contentsOf: [0, 0, 0, 0])  // token 0
        bytes.append(contentsOf: [0, 0, 0, 0])  // expert 0
        bytes.append(contentsOf: [0, 0, 0, 0x40])  // width 2^30
        XCTAssertThrowsError(try ContributionWire.decode(Data(bytes))) { error in
            guard case ContributionWireError.malformedTerm = error else {
                return XCTFail("expected the width to be refused, got \(error)")
            }
        }
    }

    func testAnAbsurdCountIsRefusedRatherThanAllocated() throws {
        var bytes = Array("TTDC".utf8)
        bytes.append(contentsOf: [1, 0])
        bytes.append(contentsOf: [1, 0, 0, 0])
        bytes.append(contentsOf: [1, 0, 0, 0])
        bytes.append(contentsOf: [0xFF, 0xFF, 0xFF, 0x7F])  // count 2^31-1
        XCTAssertThrowsError(try ContributionWire.decode(Data(bytes))) { error in
            guard case ContributionWireError.absurdCount = error else {
                return XCTFail("expected the count to be refused, got \(error)")
            }
        }
    }

    func testBadMagicVersionsTruncationAndTrailingBytesAreRefused() throws {
        let good = try ContributionWire.encode(frames(tokens: 2, width: 4, terms: 2), tokens: 2, hiddenSize: 4)

        var wrongMagic = good; wrongMagic[0] = 0x54 &+ 1
        XCTAssertThrowsError(try ContributionWire.decode(wrongMagic)) { error in
            XCTAssertEqual(error as? ContributionWireError, .badMagic)
        }

        var wrongVersion = good; wrongVersion[4] = 99
        XCTAssertThrowsError(try ContributionWire.decode(wrongVersion)) { error in
            XCTAssertEqual(error as? ContributionWireError, .unsupportedVersion(99))
        }

        XCTAssertThrowsError(try ContributionWire.decode(good.prefix(10))) { error in
            guard case ContributionWireError.truncated = error else {
                return XCTFail("expected truncation, got \(error)")
            }
        }

        var trailing = good; trailing.append(0)
        XCTAssertThrowsError(try ContributionWire.decode(trailing)) { error in
            XCTAssertEqual(error as? ContributionWireError, .trailingBytes(1))
        }
    }

    // MARK: - the transport

    func testTwoNodesExchangeContributionsAndReduceBitIdentically() throws {
        let root = try XCTUnwrap(
            Bundle.module.url(forResource: "tiny-qwen36", withExtension: nil, subdirectory: "Fixtures")
        )
        let forward = try Qwen3_5Forward(install: root.appendingPathComponent("install"))
        let layer = try forward.loadLayer(0)
        guard case .mixture(let weights, let provider) = layer.feedForward else {
            throw XCTSkip("the fixture's first layer is not a mixture")
        }
        let shape = try forward.mixtureShape()

        var source = Source()
        let tokens = 2
        let input = (0..<(tokens * shape.hiddenSize)).map { _ in source.next() }
        let (_, indices, chosen) = try MixtureOfExperts.block(
            hidden: input, tokens: tokens, weights: weights, shape: shape
        )
        let single = try MixtureOfExperts.experts(
            hidden: input, tokens: tokens, provider: provider,
            indices: indices, weights: chosen, shape: shape
        )

        // Each node computes its own experts' terms and ships them. Nothing is pre-summed (`D17`).
        let ownership = ExpertOwnership(nodes: 2, experts: shape.experts)
        let (left, right) = try SocketPair.make()
        let transports = [left, right]
        var ownTerms: [[ExpertContribution]] = []
        for node in 0..<2 {
            let owned = OwnedExpertProvider(base: provider, ownership: ownership, node: node)
            let terms = try MixtureOfExperts.expertContributions(
                hidden: input, tokens: tokens, provider: owned,
                indices: indices, weights: chosen, shape: shape
            )
            ownTerms.append(terms)
            try transports[node].send(
                try ContributionWire.encode(terms, tokens: tokens, hiddenSize: shape.hiddenSize)
            )
        }

        // Each node now holds what it produced and what the other sent, and both reduce that same set —
        // which is what makes this an all-reduce rather than a gather at one node.
        var reduced: [[Float]] = []
        for node in 0..<2 {
            let all = ownTerms[node]
                + (try ContributionWire.decode(try transports[node].receive())).contributions
            XCTAssertTrue(
                OrderedReduction.isComplete(all, indices: indices),
                "the pair of nodes must together cover every selected expert exactly once"
            )
            reduced.append(
                try OrderedReduction.accumulate(all, tokens: tokens, hiddenSize: shape.hiddenSize)
            )
        }

        for (index, element) in reduced[0].enumerated() where element.bitPattern != single[index].bitPattern {
            XCTFail("element \(index) crossed the wire and changed: \(element.bitPattern) against \(single[index].bitPattern)")
            return
        }
        XCTAssertEqual(reduced[0].count, reduced[1].count)
        XCTAssertEqual(
            reduced[0].map(\.bitPattern), reduced[1].map(\.bitPattern),
            "both nodes must reach the same bits, or they cannot agree on the next token"
        )
    }

    func testCoalescedFramesAreReadOneAtATime() throws {
        let (left, right) = try SocketPair.make()
        let first = try ContributionWire.encode(frames(tokens: 2, width: 4, terms: 1), tokens: 2, hiddenSize: 4)
        let second = try ContributionWire.encode(frames(tokens: 2, width: 4, terms: 3), tokens: 2, hiddenSize: 4)
        // Both written before either is read: a stream does not preserve message boundaries, and a
        // reader that assumed it did would return one and a half frames here.
        try left.send(first)
        try left.send(second)
        XCTAssertEqual(try right.receive(), first)
        XCTAssertEqual(try right.receive(), second)
    }

    func testAHostileLengthPrefixIsRefusedRatherThanAllocated() throws {
        var descriptors: [Int32] = [-1, -1]
        guard socketpair(AF_UNIX, SOCK_STREAM, 0, &descriptors) == 0 else {
            throw XCTSkip("socketpair is unavailable")
        }
        let transport = SocketContributionTransport(fileDescriptor: descriptors[1])
        // A length prefix of 2^31 is a claim, not a frame. Refuse it on the number.
        let prefix: [UInt8] = [0x00, 0x00, 0x00, 0x80]
        let writeEnd = FileHandle(fileDescriptor: descriptors[0], closeOnDealloc: true)
        try writeEnd.write(contentsOf: Data(prefix))
        XCTAssertThrowsError(try transport.receive()) { error in
            guard case ContributionTransportError.frameTooLarge = error else {
                return XCTFail("expected the length to be refused, got \(error)")
            }
        }
    }
}
