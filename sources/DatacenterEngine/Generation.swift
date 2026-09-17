import Foundation

/// Greedy decoding, and the reason it lives in the engine rather than in a caller.
///
/// I3 says discrete decisions must match the reference **exactly**, checked separately
/// from any numeric tolerance. The sampled token is the most consequential discrete
/// decision in the model: a 1-ULP difference in the logits can flip it, and from that
/// point the two continuations are unrelated even though every per-tensor check still
/// looks green. So sampling is written once, here, with a stated tie-break, and every
/// generated token is recorded in the trace's discrete section where the differ compares
/// it as an index set rather than as a number.
///
/// M0 has **no KV cache**: each step re-runs the whole sequence. That is deliberate —
/// a cache is a second numeric path through attention, and M0's job is to establish one
/// correct path before there are two. Speed is M1's problem, and the milestone says so.
public enum Greedy {
    /// The index of the largest value, with ties going to the **lowest index**.
    ///
    /// The tie-break is part of the contract (D5): numpy's `argmax` returns the first
    /// occurrence, and a `>` comparison here gives the same answer. A tie is not exotic —
    /// it happens whenever two logits are equal, which quantisation makes more likely, not
    /// less.
    public static func argmax(_ values: [Float], offset: Int, width: Int) -> Int {
        precondition(offset >= 0 && width > 0 && offset + width <= values.count, "argmax window out of range")
        var bestIndex = offset
        var bestValue = values[offset]
        for index in 1..<width {
            let value = values[offset + index]
            if value > bestValue {
                bestValue = value
                bestIndex = offset + index
            }
        }
        return bestIndex - offset
    }

    /// The gap between the best and the second-best logit.
    ///
    /// This is the number that separates the two ways a token can differ between two numeric
    /// paths. A **marginal** choice — a margin of a few ULP — was never really decided, and a
    /// path that differs in the last bit is entitled to move it; that is I3's warning made
    /// concrete. A large margin that flips is a defect, because no rounding explains it.
    public static func margin(_ values: [Float], offset: Int, width: Int) -> Float {
        precondition(offset >= 0 && width > 0 && offset + width <= values.count, "margin window out of range")
        var best = values[offset]
        var second: Float = -Float.infinity
        for index in 1..<width {
            let value = values[offset + index]
            if value > best {
                second = best
                best = value
            } else if value > second {
                second = value
            }
        }
        return best - second
    }
}

public struct Generation {
    public let prompt: [Int]
    public let generated: [Int]
    /// Seconds per forward pass, one per step, for the record rather than for a claim.
    public let secondsPerStep: [Double]
    /// The **margin** between the two highest logits at each step, one per generated token.
    ///
    /// A marginal argmax flip and a bug look identical from the token ids alone, and they mean
    /// opposite things: the first is what I3 warns about — a numeric-path difference moving a
    /// decision that was never decided — and the second is a defect. The margin is what tells
    /// them apart. It is recorded rather than judged, so a divergence can be read afterwards.
    public let margins: [Float]

    public var tokens: [Int] { prompt + generated }

    /// The captured tensors of the **last** forward pass, which saw the whole sequence —
    /// so a trace of a generation is comparable between two implementations element by
    /// element, not just by its token ids.
    public let captured: [TraceWriter.Tensor]

    /// Phase timings over the decode steps, present only when profiling was asked for.
    ///
    /// Nil means **not measured**, which is not the same as zero — the distinction `ForwardResult.profile`
    /// draws, kept here because this is the struct the throughput gate's per-node metrics are built from, and
    /// a phase reported as 0.000 s would claim a measurement nobody took (`D88`).
    public let profile: ProfileReport?

    /// What the decoded-layer cache did, or nil when no cache was used.
    ///
    /// Recorded per node because `DC-052`'s done-when is exactly this: the cache budget and what it held,
    /// measured rather than assumed. Nil means **no cache**, not a cache that held nothing.
    public let layerCache: LayerCacheMetrics?

    public init(
        prompt: [Int], generated: [Int], secondsPerStep: [Double], captured: [TraceWriter.Tensor],
        margins: [Float] = [], profile: ProfileReport? = nil, layerCache: LayerCacheMetrics? = nil
    ) {
        self.margins = margins
        self.prompt = prompt
        self.generated = generated
        self.secondsPerStep = secondsPerStep
        self.captured = captured
        self.profile = profile
        self.layerCache = layerCache
    }
}
