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
}

public struct Generation {
    public let prompt: [Int]
    public let generated: [Int]
    /// Seconds per forward pass, one per step, for the record rather than for a claim.
    public let secondsPerStep: [Double]

    public var tokens: [Int] { prompt + generated }

    /// The captured tensors of the **last** forward pass, which saw the whole sequence —
    /// so a trace of a generation is comparable between two implementations element by
    /// element, not just by its token ids.
    public let captured: [TraceWriter.Tensor]

    public init(prompt: [Int], generated: [Int], secondsPerStep: [Double], captured: [TraceWriter.Tensor]) {
        self.prompt = prompt
        self.generated = generated
        self.secondsPerStep = secondsPerStep
        self.captured = captured
    }
}
