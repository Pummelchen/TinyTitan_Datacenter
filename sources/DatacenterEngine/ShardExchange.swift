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
    case noAttempts

    public var description: String {
        switch self {
        case .incomplete(let missing):
            return "the reduction is missing \(missing.count) selected term(s), starting at \(missing.first ?? "-"): "
                + "a run that summed fewer experts would be quietly wrong"
        case .duplicateWithDifferentBits(let key):
            return "term \(key) arrived twice with different bits, so one of the two is not what this node computed"
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
        indices: [[Int]], tokens: Int, hiddenSize: Int, policy: ExchangePolicy = .default
    ) throws -> [Float] {
        guard policy.attempts >= 1 else { throw ShardExchangeError.noAttempts }
        let frame = try ContributionWire.encode(own, tokens: tokens, hiddenSize: hiddenSize)

        var received: [ExpertContribution] = []
        var attempt = 1
        while true {
            do {
                // The policy is the caller's, and it has to reach the transport or it means nothing.
                for peer in peers { peer.applyTimeout(milliseconds: policy.receiveTimeoutMilliseconds) }
                for peer in peers { try peer.send(frame) }
                received = []
                for peer in peers { received += try ContributionWire.decode(try peer.receive()) }
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

        return OrderedReduction.accumulate(
            try merge(own + received, indices: indices), tokens: tokens, hiddenSize: hiddenSize
        )
    }
}
