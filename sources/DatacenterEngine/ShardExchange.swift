import Foundation

/// How long a node waits for its peers, and how many times it will try again.
public struct ExchangePolicy: Sendable, Equatable {
    /// Per-`receive` deadline, passed to the transport.
    public var receiveTimeoutMilliseconds: Int
    /// How many times a **clean** timeout may be retried. One means no retry at all.
    public var attempts: Int

    public init(receiveTimeoutMilliseconds: Int = 30_000, attempts: Int = 3) {
        self.receiveTimeoutMilliseconds = receiveTimeoutMilliseconds
        self.attempts = attempts
    }

    public static let `default` = ExchangePolicy()
}

public enum ShardExchangeError: Swift.Error, CustomStringConvertible, Equatable {
    /// The reduction did not see every term the router selected. The run fails; it never sums less.
    case incomplete(missing: [String])
    /// The same `(token, expert)` arrived twice with different bits.
    case duplicateWithDifferentBits(key: String)
    /// A head slice arrived claiming rows outside the vocabulary.
    case headSliceOutOfRange(start: Int, count: Int, vocabulary: Int)
    case noAttempts

    public var description: String {
        switch self {
        case .incomplete(let missing):
            return "the reduction is missing \(missing.count) selected term(s), starting at \(missing.first ?? "-"): "
                + "a run that summed fewer experts would be quietly wrong"
        case .duplicateWithDifferentBits(let key):
            return "term \(key) arrived twice with different bits, so one of the two is not what this node computed"
        case .headSliceOutOfRange(let start, let count, let vocabulary):
            return "a head slice claims rows \(start)..<\(start + count) of a \(vocabulary)-row head"
        case .noAttempts:
            return "an exchange policy of zero attempts cannot complete anything"
        }
    }
}

extension ExpertContribution {
    /// Bit-for-bit equality.
    ///
    /// `==` would not do: `Float` says `-0.0 == 0.0` and `NaN != NaN`, and both of those would hide
    /// exactly the divergence this comparison exists to catch. A retry is supposed to produce the same
    /// bits, and "supposed to" is what this checks.
    public func matchesExactly(_ other: ExpertContribution) -> Bool {
        guard token == other.token, expert == other.expert,
            scale.bitPattern == other.scale.bitPattern,
            values.count == other.values.count
        else { return false }
        for index in 0..<values.count where values[index].bitPattern != other.values[index].bitPattern {
            return false
        }
        return true
    }
}

/// One layer's all-reduce, and the rule that decides when it is finished.
///
/// The failure semantics are the point (`DC-043`). A single-user run over a cluster has three ways to
/// end badly and they must not look alike:
///
/// - **A peer that says nothing.** Retry it, within the policy. Retrying is safe *because* the
///   reduction order is canonical (`D17`): the same terms sent twice produce the same bits, so a
///   retry cannot change the answer, and a duplicate is recognised rather than summed twice.
/// - **A peer that stops mid-frame.** Do **not** retry. The stream is desynchronised — the next read
///   would return the tail of the previous frame — and the failure would surface much later, in a
///   place with nothing to do with its cause.
/// - **A term that never arrives.** Fail the run. `OrderedReduction.isComplete` is checked before
///   anything is summed, because a reduction over seven of a token's eight experts produces a number
///   that looks like an answer and is not one.
/// What the all-reduce cost, counted rather than inferred (`DC-081`).
///
/// A **class**, because `ShardExecution` is a struct and an exchange outlives one call — the same reason
/// `ReadFile.ReadState` and the payload cache are classes. The counters include every attempt, not only the
/// successful one: a retry that cost a round trip is part of what the run cost, and hiding it would make
/// the instrument flatter than the truth.
public final class ExchangeLedger: @unchecked Sendable {
    private let lock = NSLock()
    private var _reduces = 0
    private var _termsSent = 0
    private var _termsReceived = 0
    private var _bytesSent = 0
    private var _bytesReceived = 0
    private var _seconds = 0.0
    private var _encodeSeconds = 0.0
    private var _sendSeconds = 0.0
    private var _receiveSeconds = 0.0
    private var _reduceSeconds = 0.0

    public init() {}

    func record(
        termsSent: Int, termsReceived: Int, bytesSent: Int, bytesReceived: Int, seconds: Double,
        encodeSeconds: Double = 0, sendSeconds: Double = 0, receiveSeconds: Double = 0,
        reduceSeconds: Double = 0
    ) {
        lock.lock()
        defer { lock.unlock() }
        _reduces += 1
        _termsSent += termsSent
        _termsReceived += termsReceived
        _bytesSent += bytesSent
        _bytesReceived += bytesReceived
        _seconds += seconds
        _encodeSeconds += encodeSeconds
        _sendSeconds += sendSeconds
        _receiveSeconds += receiveSeconds
        _reduceSeconds += reduceSeconds
    }

    public var metrics: ExchangeMetrics {
        lock.lock()
        defer { lock.unlock() }
        return ExchangeMetrics(
            reduces: _reduces, termsSent: _termsSent, termsReceived: _termsReceived,
            bytesSent: _bytesSent, bytesReceived: _bytesReceived, seconds: _seconds,
            encodeSeconds: _encodeSeconds, sendSeconds: _sendSeconds,
            receiveSeconds: _receiveSeconds, reduceSeconds: _reduceSeconds
        )
    }
}

/// A snapshot of `ExchangeLedger`, on the `SourceTiming` pattern: a value the caller reports.
public struct ExchangeMetrics: Sendable, Equatable {
    public var reduces = 0
    public var termsSent = 0
    public var termsReceived = 0
    public var bytesSent = 0
    public var bytesReceived = 0
    /// Wall seconds spent inside every all-reduce, **including retries and the time spent waiting for a
    /// peer**. It is an observation, not a throughput claim: on a shared network it measures the farm as
    /// much as the engine.
    public var seconds = 0.0
    /// Where those seconds went: encoding this node's frame, sending it to the peers, waiting for theirs, and
    /// merging what arrived. `D92` added them because the total was not enough to act on — 4.25 s/step is a
    /// number, and 4.25 s/step of *waiting* is a different problem from 4.25 s/step of *encoding*.
    public var encodeSeconds = 0.0
    public var sendSeconds = 0.0
    public var receiveSeconds = 0.0
    public var reduceSeconds = 0.0

    public init(
        reduces: Int = 0, termsSent: Int = 0, termsReceived: Int = 0, bytesSent: Int = 0,
        bytesReceived: Int = 0, seconds: Double = 0.0, encodeSeconds: Double = 0.0,
        sendSeconds: Double = 0.0, receiveSeconds: Double = 0.0, reduceSeconds: Double = 0.0
    ) {
        self.reduces = reduces
        self.termsSent = termsSent
        self.termsReceived = termsReceived
        self.bytesSent = bytesSent
        self.bytesReceived = bytesReceived
        self.seconds = seconds
        self.encodeSeconds = encodeSeconds
        self.sendSeconds = sendSeconds
        self.receiveSeconds = receiveSeconds
        self.reduceSeconds = reduceSeconds
    }
}

public enum ShardExchange {
    /// Merge terms from every node, collapsing retries and refusing contradictions.
    ///
    /// A duplicate is allowed only when it is **bit-identical** to the copy already held: that is what a
    /// resend after a timeout looks like, and it is harmless. A duplicate that differs means two nodes
    /// disagree about a term they both claim to have computed, which no amount of care downstream can
    /// repair — so it is refused here, where the diagnosis is still possible.
    public static func merge(
        _ contributions: [ExpertContribution], indices: [[Int]]
    ) throws -> [ExpertContribution] {
        var byKey: [String: ExpertContribution] = [:]
        for contribution in contributions {
            let key = "\(contribution.token):\(contribution.expert)"
            if let previous = byKey[key] {
                guard previous.matchesExactly(contribution) else {
                    throw ShardExchangeError.duplicateWithDifferentBits(key: key)
                }
                continue
            }
            byKey[key] = contribution
        }
        let missing = OrderedReduction.selectedKeys(indices).subtracting(byKey.keys)
        guard missing.isEmpty else {
            throw ShardExchangeError.incomplete(missing: missing.sorted())
        }
        return byKey.values.sorted(by: OrderedReduction.precedes)
    }

    /// Send this node's terms to every peer, receive theirs, verify, and reduce.
    ///
    /// `own` is a value, so a retry is a resend of the same bytes rather than a recomputation — and a
    /// peer that already received them merges the duplicate instead of double-counting it.
    @discardableResult
    public static func allReduce(
        own: [ExpertContribution], peers: [any ContributionTransport],
        indices: [[Int]], tokens: Int, hiddenSize: Int, policy: ExchangePolicy = .default,
        ledger: ExchangeLedger? = nil
    ) throws -> [Float] {
        let startedAt = DispatchTime.now().uptimeNanoseconds
        var termsSent = 0, termsReceived = 0, bytesSent = 0, bytesReceived = 0
        // Where the time went, so the next change is aimed rather than guessed (`D92`). A segment is measured
        // around the work itself; the total stays the measurement of record and still includes retries.
        var encodeSeconds = 0.0, sendSeconds = 0.0, receiveSeconds = 0.0, reduceSeconds = 0.0
        func since(_ mark: UInt64) -> Double {
            Double(DispatchTime.now().uptimeNanoseconds &- mark) / 1e9
        }
        defer {
            ledger?.record(
                termsSent: termsSent, termsReceived: termsReceived, bytesSent: bytesSent,
                bytesReceived: bytesReceived,
                seconds: since(startedAt), encodeSeconds: encodeSeconds, sendSeconds: sendSeconds,
                receiveSeconds: receiveSeconds, reduceSeconds: reduceSeconds
            )
        }
        guard policy.attempts >= 1 else { throw ShardExchangeError.noAttempts }
        var mark = DispatchTime.now().uptimeNanoseconds
        let frame = try ContributionWire.encode(own, tokens: tokens, hiddenSize: hiddenSize)
        encodeSeconds += since(mark)

        var received: [ExpertContribution] = []
        var attempt = 1
        while true {
            do {
                // The policy is the caller's, and it has to reach the transport or it means nothing.
                for peer in peers { peer.applyTimeout(milliseconds: policy.receiveTimeoutMilliseconds) }
                mark = DispatchTime.now().uptimeNanoseconds
                for peer in peers { try peer.send(frame) }
                sendSeconds += since(mark)
                // The same frame goes to every peer, so what left this node is that frame times the peers.
                termsSent += own.count * peers.count
                bytesSent += frame.count * peers.count
                received = []
                mark = DispatchTime.now().uptimeNanoseconds
                for peer in peers {
                    let answer = try peer.receive()
                    bytesReceived += answer.count
                    let decoded = try ContributionWire.decode(answer)
                    termsReceived += decoded.contributions.count
                    // A frame is validated against itself in `decode`; this is the check only the
                    // receiver can make, and without it a peer's geometry reaches `accumulate`, whose
                    // width invariant is a precondition — a crash, from the network (`DC-083`).
                    guard decoded.tokens == tokens, decoded.hiddenSize == hiddenSize else {
                        throw ContributionWireError.geometryMismatch(
                            tokens: decoded.tokens, hiddenSize: decoded.hiddenSize,
                            expectedTokens: tokens, expectedHiddenSize: hiddenSize
                        )
                    }
                    received += decoded.contributions
                }
                receiveSeconds += since(mark)
                break
            } catch let error as ContributionTransportError {
                // Only a clean timeout is retryable: `midFrame` means the stream is desynchronised.
                if case .timedOut(midFrame: false) = error, attempt < policy.attempts {
                    attempt += 1
                    continue
                }
                throw error
            }
        }

        mark = DispatchTime.now().uptimeNanoseconds
        let merged = try merge(own + received, indices: indices)
        let accumulated = try OrderedReduction.accumulate(merged, tokens: tokens, hiddenSize: hiddenSize)
        reduceSeconds += since(mark)
        return accumulated
    }
}
