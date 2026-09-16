import Foundation

/// One expert's contribution to one token's output, before it is summed into place.
///
/// The all-reduce carries **these**, not per-node partial sums, and that is the whole content of the
/// reduction contract (`D17`, I2). Floating-point addition is not associative, so a node that
/// pre-sums the experts it owns produces a different low-bit pattern from the sequence a single node
/// computes. Shipping the terms and ordering them canonically makes the N-node sum **the same
/// sequence of additions** as the 1-node sum — which is what "bit-identical to the M1 baseline"
/// means, and the only way it can be true.
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

    /// Sum contributions into a `[tokens, hiddenSize]` output in the canonical order.
    ///
    /// The multiplication is applied before the addition, to the same expression shape the
    /// single-node path uses — `D17` is only meaningful if both sites round identically, and a
    /// compiler that contracted one into an FMA and not the other would break bit-identity in a way
    /// only a test like this one can see.
    public static func accumulate(
        _ contributions: [ExpertContribution], tokens: Int, hiddenSize: Int
    ) -> [Float] {
        var output = [Float](repeating: 0, count: tokens * hiddenSize)
        for contribution in contributions.sorted(by: precedes) {
            precondition(
                contribution.token >= 0 && contribution.token < tokens,
                "contribution for token \(contribution.token), which is outside \(tokens)"
            )
            precondition(
                contribution.values.count == hiddenSize,
                "contribution width \(contribution.values.count) is not the hidden size \(hiddenSize)"
            )
            let base = contribution.token * hiddenSize
            for index in 0..<hiddenSize {
                output[base + index] = output[base + index] + contribution.values[index] * contribution.scale
            }
        }
        return output
    }
}
