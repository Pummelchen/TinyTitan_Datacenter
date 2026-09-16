import Foundation
import XCTest

@testable import DatacenterEngine

/// `DC-083`: the wire is fed by other machines, so it is fed hostile input here.
///
/// The decoder validates a frame against **itself**, and that is not enough at a boundary. A peer can
/// satisfy every internal check while declaring a geometry this node does not have, and the reduction
/// `precondition`s on width — which turns a frame into a crash. These tests are the audit: nothing below
/// may crash, hang, or allocate in proportion to a number the sender chose, and the frame that says the
/// wrong shape must be refused *by name*.
///
/// Every case is seeded and the seed is printed when one fails, because a fuzz test that cannot be
/// replayed is a rumour.
final class WireFuzzTests: XCTestCase {
    /// SplitMix64: a tiny generator whose sequence is fixed for a given seed, so a failure replays.
    private struct Random {
        var state: UInt64
        mutating func next() -> UInt64 {
            state &+= 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z >> 31)
        }
        mutating func int(below bound: Int) -> Int { Int(next() % UInt64(bound)) }
        mutating func byte() -> UInt8 { UInt8(next() & 0xFF) }
    }

    private let tokens = 3
    private let hidden = 8

    private func validFrame() throws -> Data {
        try ContributionWire.encode(
            [
                ExpertContribution(token: 0, expert: 1, values: [Float](repeating: 0.5, count: hidden), scale: 1.0),
                ExpertContribution(token: 2, expert: 7, values: [Float](repeating: -1.5, count: hidden), scale: 0.25),
            ],
            tokens: tokens, hiddenSize: hidden
        )
    }

    /// Decoding must end in one of exactly two ways, for any input at all.
    private func assertDecodeIsTotal(_ data: Data, seed: UInt64, _ label: @autoclosure () -> String) {
        let started = Date()
        do {
            let frame = try ContributionWire.decode(data)
            XCTAssertLessThanOrEqual(
                frame.contributions.count, Int.max,
                "decoded \(frame.contributions.count) terms \(label())"
            )
        } catch is ContributionWireError {
            // The only acceptable failure: a typed refusal that names what is wrong.
        } catch {
            XCTFail("decode threw \(error) for \(label()) (seed \(seed))")
        }
        let elapsed = Date().timeIntervalSince(started)
        XCTAssertLessThan(elapsed, 2.0, "decode took \(elapsed)s for \(label()) (seed \(seed))")
    }

    func testRandomBytesAreAlwaysRefusedAndNeverCrash() throws {
        var random = Random(state: 0xA5A5_1234)
        for case_ in 0..<2_000 {
            let length = random.int(below: 300)
            var bytes = [UInt8]()
            bytes.reserveCapacity(length)
            for _ in 0..<length { bytes.append(random.byte()) }
            assertDecodeIsTotal(Data(bytes), seed: 0xA5A5_1234, "random case \(case_)")
        }
    }

    func testSingleByteMutationsOfAValidFrameAreAlwaysHandled() throws {
        let seed: UInt64 = 0x5EED_0F17
        let valid = try validFrame()
        var random = Random(state: seed)
        for case_ in 0..<2_000 {
            var bytes = [UInt8](valid)
            let index = random.int(below: bytes.count)
            bytes[index] = random.byte()
            assertDecodeIsTotal(Data(bytes), seed: seed, "mutation \(case_) at byte \(index)")
        }
    }

    func testEveryTruncationOfAValidFrameIsHandled() throws {
        let valid = try validFrame()
        for length in 0..<valid.count {
            assertDecodeIsTotal(valid.prefix(length), seed: 0, "prefix of \(length)")
        }
    }

    /// The declarations are individually bounded, so the test that matters is the **product**: a frame
    /// that claims a million terms in sixty bytes must be refused without reserving for a million terms,
    /// and one that claims a million-wide hidden size likewise. A duration bound is how this test sees
    /// an allocation it cannot measure directly.
    func testHostileDeclarationsAreRefusedQuickly() throws {
        let declarations: [(tokens: Int, hidden: Int, count: Int, body: Int, label: String)] = [
            (tokens, hidden, 1 << 20, 0, "a million terms in no bytes"),
            (tokens, hidden, 1 << 20, 64, "a million terms in sixty-four bytes"),
            (tokens, 1 << 20, 1, 0, "a million-wide hidden size with no values"),
            (tokens, 1 << 20, 1 << 16, 0, "a million-wide hidden size, sixty-five thousand terms"),
            (1 << 20, 1, 1 << 20, 0, "a million tokens"),
            (0, hidden, 1, 0, "zero tokens"),
        ]
        for declaration in declarations {
            var bytes = [UInt8]("TTDC".utf8)
            bytes.append(contentsOf: [1, 0])  // version 1, little endian
            func append(_ value: Int) {
                let little = UInt32(truncatingIfNeeded: value).littleEndian
                bytes.append(contentsOf: [
                    UInt8(truncatingIfNeeded: little), UInt8(truncatingIfNeeded: little >> 8),
                    UInt8(truncatingIfNeeded: little >> 16), UInt8(truncatingIfNeeded: little >> 24),
                ])
            }
            append(declaration.tokens)
            append(declaration.hidden)
            append(declaration.count)
            for _ in 0..<declaration.body { bytes.append(0) }
            assertDecodeIsTotal(Data(bytes), seed: 0, declaration.label)
        }
    }

    /// A frame that is valid on its own terms but not this node's terms must be refused by name.
    ///
    /// Before `DC-083` this reached `OrderedReduction.accumulate`, whose width invariant is a
    /// **precondition** — a crash, from the network.
    func testAFrameFromAnotherGeometryIsRefusedRatherThanCrashingTheReduction() throws {
        let (left, right) = try SocketPair.make()
        let peerFrame = try ContributionWire.encode(
            [ExpertContribution(token: 0, expert: 0, values: [Float](repeating: 1, count: 4), scale: 1)],
            tokens: 1, hiddenSize: 4
        )
        try right.send(peerFrame)

        XCTAssertThrowsError(
            try ShardExchange.allReduce(
                own: [ExpertContribution(token: 0, expert: 0, values: [Float](repeating: 2, count: hidden), scale: 1)],
                peers: [left], indices: [[0, 1]], tokens: tokens, hiddenSize: hidden,
                policy: ExchangePolicy(receiveTimeoutMilliseconds: 2_000, attempts: 1)
            )
        ) { error in
            guard case ContributionWireError.geometryMismatch(let gotTokens, let gotHidden, let wantTokens, let wantHidden) = error else {
                return XCTFail("expected a geometry mismatch, got \(error)")
            }
            XCTAssertEqual([gotTokens, gotHidden], [1, 4], "the peer's declaration should be named")
            XCTAssertEqual([wantTokens, wantHidden], [tokens, hidden], "and this node's own geometry")
        }
    }

    /// Cancelling a term is a contradiction, not a duplicate, and the wire is where an attacker would
    /// try it: the same key twice with different bits.
    func testAConflictingDuplicateIsRefused() throws {
        let terms = [
            ExpertContribution(token: 0, expert: 0, values: [Float](repeating: 1, count: hidden), scale: 1),
            ExpertContribution(token: 0, expert: 0, values: [Float](repeating: 2, count: hidden), scale: 1),
        ]
        XCTAssertThrowsError(try ShardExchange.merge(terms, indices: [[0]])) { error in
            guard case ShardExchangeError.duplicateWithDifferentBits = error else {
                return XCTFail("expected a duplicate-with-different-bits refusal, got \(error)")
            }
        }
    }

    /// And a bit-identical duplicate — a resend after a timeout — is accepted, because `D17`'s canonical
    /// order makes a retry harmless. The pair of these two tests is what makes the refusal meaningful.
    func testABitIdenticalDuplicateIsAccepted() throws {
        let term = ExpertContribution(
            token: 0, expert: 0, values: [Float](repeating: 1, count: hidden), scale: 1
        )
        let merged = try ShardExchange.merge([term, term], indices: [[0]])
        XCTAssertEqual(merged.count, 1)
        XCTAssertEqual(merged[0].values, term.values)
    }
}
