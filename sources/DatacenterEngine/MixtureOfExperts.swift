import Foundation

/// The mixture of experts' geometry.
public struct MixtureShape: Sendable, Equatable {
    public let hiddenSize: Int
    public let experts: Int
    public let topK: Int
    public let intermediate: Int
    public let sharedIntermediate: Int

    public init(hiddenSize: Int, experts: Int, topK: Int, intermediate: Int, sharedIntermediate: Int) {
        self.hiddenSize = hiddenSize
        self.experts = experts
        self.topK = topK
        self.intermediate = intermediate
        self.sharedIntermediate = sharedIntermediate
    }
}

/// The mixture's weights, by role.
///
/// The experts are **stacked**, which is the checkpoint's own layout: one tensor per
/// projection holding every expert, with gate and up fused. An importer maps names to roles
/// and does not reshape, so this is the layout the kernel reads (L2).
public struct MixtureWeights {
    public let router: [Float]            // [experts, hidden]
    public let sharedGate: [Float]        // [sharedIntermediate, hidden]
    public let sharedUp: [Float]          // [sharedIntermediate, hidden]
    public let sharedDown: [Float]        // [hidden, sharedIntermediate]
    public let sharedScalarGate: [Float]  // [1, hidden]
    /// Where the routed experts come from. The kernel asks for the ones the router chose and
    /// never sees the stack, which is what keeps a 35 B layer's 3.2 GB of experts out of
    /// memory (`DC-032`).
    public let experts: any ExpertWeightProvider

    /// The stacked layout already in memory: what the tiny checkpoints and the golden vectors
    /// use. It goes through the *same* kernel as the streaming path, so the two cannot drift.
    public init(
        router: [Float], gateUp: [Float], down: [Float], sharedGate: [Float], sharedUp: [Float],
        sharedDown: [Float], sharedScalarGate: [Float]
    ) {
        self.init(
            router: router, sharedGate: sharedGate, sharedUp: sharedUp, sharedDown: sharedDown,
            sharedScalarGate: sharedScalarGate, experts: ArrayExpertProvider(gateUp: gateUp, down: down)
        )
    }

    public init(
        router: [Float], sharedGate: [Float], sharedUp: [Float], sharedDown: [Float],
        sharedScalarGate: [Float], experts: any ExpertWeightProvider
    ) {
        self.router = router
        self.sharedGate = sharedGate
        self.sharedUp = sharedUp
        self.sharedDown = sharedDown
        self.sharedScalarGate = sharedScalarGate
        self.experts = experts
    }
}

/// The mixture of experts, in the contract's order (M1).
///
/// Transcribed from `transformers` v5.17.0 `models/qwen3_5_moe/modeling_qwen3_5_moe.py`
/// (`Qwen3_5MoeTopKRouter:884`, `Qwen3_5MoeExperts:845`, `Qwen3_5MoeSparseMoeBlock:903`) and
/// recorded in `docs/reference-qwen36-35b-a3b.md`. The Python contract in
/// `tools/ordered_moe.py` is the bit-exactness target and `MixtureOfExpertsTests` asserts
/// these functions against golden bit patterns emitted from it.
///
/// Two things here are decisions rather than transcriptions, and both are load-bearing:
///
/// - **ties break to the lowest expert index.** `torch.topk` promises nothing about equal
///   probabilities, and I3 needs an index set that can be compared — a set with an undefined
///   order is not comparable. Swift's `sorted(by:)` is not guaranteed stable either, so the
///   comparator is a total order rather than a hope.
/// - **the routed sum accumulates in ascending expert index**, not in top-k rank order. The
///   reference does this, and it is what D4's ring reduction does, so the single-node result
///   and the distributed one agree by construction.
public enum MixtureOfExperts {
    /// The router: `[tokens, topK]` indices and renormalised weights, plus the logits.
    ///
    /// The softmax is fp32 while everything around it is the model's dtype — the router is
    /// the second fp32 island in this model, and I3 is why it stays one.
    public static func router(
        hidden: [Float], tokens: Int, weights: [Float], experts: Int, topK: Int
    ) -> (logits: [Float], indices: [[Int]], weights: [[Float]]) {
        let logits = Ops.orderedMatmul(x: hidden, w: weights, rows: tokens, k: hidden.count / max(tokens, 1), out: experts)
        let probabilities = Ops.softmax(x: logits, rows: tokens, width: experts)

        var indices: [[Int]] = []
        var chosen: [[Float]] = []
        for token in 0..<tokens {
            let base = token * experts
            // A total order: descending probability, then ascending index. `sorted(by:)` is
            // not a stable sort, so the tie-break has to be in the comparator.
            let order = (0..<experts).sorted { left, right in
                let a = probabilities[base + left]
                let b = probabilities[base + right]
                return a == b ? left < right : a > b
            }
            let selection = Array(order.prefix(topK))
            var selected = selection.map { probabilities[base + $0] }
            // Renormalised over the chosen experts, unconditionally: the reference never
            // consults `norm_topk_prob`.
            let total = Ops.orderedSum(selected)
            for index in 0..<selected.count { selected[index] = selected[index] / total }
            indices.append(selection)
            chosen.append(selected)
        }
        return (logits, indices, chosen)
    }

    /// The routed experts, accumulated in ascending expert index.
    ///
    /// The experts are **asked for by index** and read through the provider, so this one
    /// implementation serves both the array-backed path and the SSD-streaming one. That is
    /// deliberate: two implementations would be two chances to accumulate in a different
    /// order, and the order is the thing the contract pins.
    public static func experts(
        hidden: [Float], tokens: Int, provider: any ExpertWeightProvider,
        indices: [[Int]], weights: [[Float]], shape: MixtureShape, profiler: Profiler? = nil
    ) throws -> [Float] {
        let hiddenSize = shape.hiddenSize
        let intermediate = shape.intermediate
        var output = [Float](repeating: 0, count: tokens * hiddenSize)

        // Each expert's (token, rank) pairs, in the order the contract collects them: token
        // ascending, then rank ascending.
        var pairs: [Int: [(token: Int, rank: Int)]] = [:]
        for token in 0..<tokens {
            for rank in 0..<indices[token].count {
                pairs[indices[token][rank], default: []].append((token, rank))
            }
        }

        for expert in pairs.keys.sorted() {
            let assignments = pairs[expert]!
            let gateUp = try provider.gateUp(expert: expert, shape: shape)
            let down = try provider.down(expert: expert, shape: shape)
            profiler?.mark("mix.read")
            var current = [Float](repeating: 0, count: assignments.count * hiddenSize)
            for (position, assignment) in assignments.enumerated() {
                for index in 0..<hiddenSize {
                    current[position * hiddenSize + index] = hidden[assignment.token * hiddenSize + index]
                }
            }
            profiler?.mark("mix.gather")
            let fused = Ops.orderedMatmul(
                x: current, w: gateUp, rows: assignments.count, k: hiddenSize, out: 2 * intermediate
            )
            profiler?.mark("mix.gateup")
            var activated = [Float](repeating: 0, count: assignments.count * intermediate)
            for row in 0..<assignments.count {
                for index in 0..<intermediate {
                    let gate = fused[row * 2 * intermediate + index]
                    let up = fused[row * 2 * intermediate + intermediate + index]
                    activated[row * intermediate + index] = Ops.silu(gate) * up
                }
            }
            profiler?.mark("mix.act")
            let projected = Ops.orderedMatmul(
                x: activated, w: down, rows: assignments.count, k: intermediate, out: hiddenSize
            )
            profiler?.mark("mix.down")
            for (position, assignment) in assignments.enumerated() {
                let scale = weights[assignment.token][assignment.rank]
                for index in 0..<hiddenSize {
                    let target = assignment.token * hiddenSize + index
                    output[target] = output[target] + projected[position * hiddenSize + index] * scale
                }
            }
            profiler?.mark("mix.acc")
        }
        return output
    }

    /// The whole block: the routed sum plus the shared expert, gated by its own scalar.
    ///
    /// The shared expert is **added**, not ranked: it is not part of the router's top-k, and
    /// its gate is a sigmoid of a projection of the hidden state.
    public static func block(
        hidden: [Float], tokens: Int, weights: MixtureWeights, shape: MixtureShape, profiler: Profiler? = nil
    ) throws -> (output: [Float], indices: [[Int]], weights: [[Float]]) {
        let hiddenSize = shape.hiddenSize
        let shared = Ops.orderedMatmul(
            x: {
                let gate = Ops.orderedMatmul(
                    x: hidden, w: weights.sharedGate, rows: tokens, k: hiddenSize, out: shape.sharedIntermediate
                )
                let up = Ops.orderedMatmul(
                    x: hidden, w: weights.sharedUp, rows: tokens, k: hiddenSize, out: shape.sharedIntermediate
                )
                var activated = [Float](repeating: 0, count: gate.count)
                for index in 0..<gate.count { activated[index] = Ops.silu(gate[index]) * up[index] }
                return activated
            }(),
            w: weights.sharedDown, rows: tokens, k: shape.sharedIntermediate, out: hiddenSize
        )
        profiler?.mark("mix.shared")
        let (_, indices, chosen) = router(
            hidden: hidden, tokens: tokens, weights: weights.router, experts: shape.experts, topK: shape.topK
        )
        profiler?.mark("mix.router")
        let routed = try experts(
            hidden: hidden, tokens: tokens, provider: weights.experts,
            indices: indices, weights: chosen, shape: shape, profiler: profiler
        )
        profiler?.mark("mix.experts")
        let scalar = Ops.orderedMatmul(
            x: hidden, w: weights.sharedScalarGate, rows: tokens, k: hiddenSize, out: 1
        )
        var output = [Float](repeating: 0, count: routed.count)
        for token in 0..<tokens {
            let gate = Ops.sigmoid(scalar[token])
            for index in 0..<hiddenSize {
                let position = token * hiddenSize + index
                output[position] = routed[position] + gate * shared[position]
            }
        }
        profiler?.mark("mix.combine")
        return (output, indices, chosen)
    }
}
