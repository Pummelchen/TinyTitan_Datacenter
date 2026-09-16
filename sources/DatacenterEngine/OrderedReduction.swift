import Foundation

/// One expert's contribution to one token's output, before it is summed into place.
///
/// The all-reduce carries **these**, not per-node partial sums, and that is the whole content of the
/// reduction contract (`D17`, I2). Floating-point addition is not associative, so a node that
/// pre-sums the experts it owns produces a different low-bit pattern from the sequence a single node
/// computes. Shipping the terms and ordering them canonically makes the N-node sum **the same
/// sequence of additions** as the 1-node sum — which is what "bit-identical to the M1 baseline"
/// means, and the only way it can be true.
/// What can be wrong with a set of terms, in the terms' own vocabulary (`DC-083`).
public enum ReductionError: Swift.Error, CustomStringConvertible, Equatable {
    case negativeGeometry(tokens: Int, hiddenSize: Int)
    case geometryOverflows(tokens: Int, hiddenSize: Int)
    case tokenOutsideRange(token: Int, tokens: Int)
    case widthMismatch(width: Int, hiddenSize: Int)

    public var description: String {
        switch self {
        case .negativeGeometry(let tokens, let hiddenSize):
            return "a reduction over \(tokens)x\(hiddenSize) has no shape"
        case .geometryOverflows(let tokens, let hiddenSize):
            return "\(tokens)x\(hiddenSize) does not fit in this machine's address space"
        case .tokenOutsideRange(let token, let tokens):
            return "a term is for token \(token), which is outside \(tokens)"
        case .widthMismatch(let width, let hiddenSize):
            return "a term has width \(width) where the hidden size is \(hiddenSize)"
        }
    }
}

public struct ExpertContribution: Sendable, Equatable {
    public let token: Int
    public let expert: Int
    /// The expert's `down` projection for this token, before the routing weight is applied.
    public let values: [Float]
    /// The routing weight for this (token, expert) pair.
    public let scale: Float

    public init(token: Int, expert: Int, values: [Float], scale: Float) {
        self.token = token
        self.expert = expert
        self.values = values
        self.scale = scale
    }
}

/// The deterministic reduction an N-node run performs, and the reason it can be bit-identical to a
/// one-node run.
///
/// The single-node path accumulates a token's experts in **ascending expert id** order, with the
/// routing weight applied before each addition. This type reproduces exactly that sequence from
/// contributions produced anywhere, so:
///
/// - **arrival order does not matter** — the terms are sorted by their canonical key, so a slow node
///   cannot change the answer, and neither can a retry;
/// - **ring order does not matter** — the brief asks for a "fixed ring order, never arrival order",
///   and ordering by `(token, expert)` is strictly stronger: the ring can be renumbered, nodes can
///   be added or replaced, and the bits do not move;
/// - **partitioning does not matter** — which node owns which expert changes who *computes* a term,
///   never the order the terms are added in.
///
/// The one thing that does matter and cannot be recovered is a **missing** contribution: a run that
/// reduced seven of a token's eight experts would be quietly wrong, so absence has to be an error at
/// the transport layer (`DC-009`) rather than something this function papers over. It cannot tell
/// "no contribution" from "a contribution of zero" — that is the caller's contract to keep.
public enum OrderedReduction {
    /// The canonical order key: token ascending, then expert id ascending.
    ///
    /// Tokens are independent accumulators, so their relative order does not change any result; the
    /// key includes the token so the sequence is fully specified and reproducible in one pass.
    @inline(__always)
    public static func precedes(_ one: ExpertContribution, _ other: ExpertContribution) -> Bool {
        one.token == other.token ? one.expert < other.expert : one.token < other.token
    }

    /// The `(token, expert)` keys a router's selection requires a reduction to have seen.
    public static func selectedKeys(_ indices: [[Int]]) -> Set<String> {
        var keys: Set<String> = []
        for (token, experts) in indices.enumerated() {
            for expert in experts { keys.insert("\(token):\(expert)") }
        }
        return keys
    }

    /// Whether a reduction saw **exactly** the terms the router selected — no missing term and no
    /// duplicate.
    ///
    /// This is the guard `D17` requires and the reduction itself cannot provide: it cannot tell a
    /// missing contribution from a contribution of zero, so a node that never answered would otherwise
    /// produce a smaller sum that still looks like a number. Every sharded run must call this before it
    /// reduces, including on the node-failure path (`DC-043`) — where the answer is to fail the run.
    public static func isComplete(_ contributions: [ExpertContribution], indices: [[Int]]) -> Bool {
        var seen: Set<String> = []
        for contribution in contributions {
            let key = "\(contribution.token):\(contribution.expert)"
            if seen.contains(key) { return false }
            seen.insert(key)
        }
        return seen == selectedKeys(indices)
    }

    /// Sum contributions into a `[tokens, hiddenSize]` output in the canonical order.
    ///
    /// The multiplication is applied before the addition, to the same expression shape the
    /// single-node path uses — `D17` is only meaningful if both sites round identically, and a
    /// compiler that contracted one into an FMA and not the other would break bit-identity in a way
    /// only a test like this one can see.
    ///
    /// The invariants **throw** rather than preconditing (`DC-083`). They were preconditions, which is
    /// a crash, and this function consumes terms decoded from the network: a peer that declared a
    /// different geometry could reach it. The wire now refuses that frame earlier and by name, and this
    /// is the second lock on the same door — a public API that takes data from elsewhere should not have
    /// a crash as its error handling.
    public static func accumulate(
        _ contributions: [ExpertContribution], tokens: Int, hiddenSize: Int
    ) throws -> [Float] {
        guard tokens >= 0, hiddenSize >= 0 else {
            throw ReductionError.negativeGeometry(tokens: tokens, hiddenSize: hiddenSize)
        }
        let (count, overflowed) = tokens.multipliedReportingOverflow(by: hiddenSize)
        guard !overflowed else {
            throw ReductionError.geometryOverflows(tokens: tokens, hiddenSize: hiddenSize)
        }
        var output = [Float](repeating: 0, count: count)
        for contribution in contributions.sorted(by: precedes) {
            guard contribution.token >= 0, contribution.token < tokens else {
                throw ReductionError.tokenOutsideRange(token: contribution.token, tokens: tokens)
            }
            guard contribution.values.count == hiddenSize else {
                throw ReductionError.widthMismatch(
                    width: contribution.values.count, hiddenSize: hiddenSize
                )
            }
            let base = contribution.token * hiddenSize
            for index in 0..<hiddenSize {
                output[base + index] = output[base + index] + contribution.values[index] * contribution.scale
            }
        }
        return output
    }
}
