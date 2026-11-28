import Foundation

/// Summing a sharded MoE's contributions without changing the answer.
///
/// The reference's `moe_phase2_down_reduce_k8` is a **fixed k = 8** reduction:
///
/// ```
/// partial[sg] = float(routing_w[sg]) * value
/// acc(float)  = float(residual[d])
/// acc        += partial[0] … partial[7]      // in slot order, fp32
/// y[d]        = half(acc)
/// ```
///
/// Three properties of it are what make sharding safe, and each is load-bearing:
///
/// 1. **k is fixed at 8**, not the number of experts a node happens to hold. So a node that owns two of
///    the eight slots still presents eight partials, six of them zero.
/// 2. **The partials are fp32**, not fp16 — fp16, bf16 and int8 all break bit-exactness (`D168`), which is
///    why the wire carries fp32.
/// 3. **The sum is ordered by slot.** Reordering the additions changes the result, so slots travel with the
///    values and are never renumbered (`D154`).
///
/// Given those, the exchange is exact and not merely close. A contribution is *zero-padded across all k
/// slots*, and IEEE addition of zero is exact — `0 + x == x`, with no rounding — so accumulating every
/// node's padded partials element-wise introduces **no error at all**. What remains is the ordered k-sum,
/// which is then the same operation over the same eight values on every node. The sharded result is
/// therefore **bit-identical to the single-node one by construction**, not by tolerance.
public enum ShardReduce {
    /// The reference's k for `moe_phase2_down_reduce_k8`.
    public static let k8 = 8

    public enum Error: Swift.Error, Equatable {
        /// A contribution was not padded to the full k, so some slot would be silently missing from the sum
        /// rather than explicitly zero. Treated as an error, not a shortfall: a node that omits its zeros is
        /// indistinguishable from one that forgot to send them.
        case contributionNotPadded(expected: Int, got: Int)
        /// Two contributions disagreed about how many dimensions they carry.
        case dimensionMismatch(expected: Int, got: Int)
    }

    /// Sum per-slot partials across nodes, element-wise, in fp32.
    ///
    /// `contributions[n]` is node *n*'s partial vector: `k` values for one dimension, zero where it owns no
    /// slot. Element-wise addition of zero-padded vectors is **exact**, so this step cannot perturb the
    /// result no matter how the nodes are partitioned or in what order their replies arrive.
    public static func accumulate(_ contributions: [[Float]], k: Int = k8) throws -> [Float] {
        guard !contributions.isEmpty else { return [Float](repeating: 0, count: k) }
        var out = [Float](repeating: 0, count: k)
        for contribution in contributions {
            guard contribution.count == k else {
                throw Error.contributionNotPadded(expected: k, got: contribution.count)
            }
            for slot in 0..<k { out[slot] += contribution[slot] }
        }
        return out
    }

    /// The reference's ordered sum, reproduced exactly: `partial[0] + … + partial[k-1]`, fp32, in slot order.
    ///
    /// Written as a loop that cannot be reordered rather than a `reduce`, so the association is visible in the
    /// source and cannot be changed by a later refactor that looks equivalent.
    public static func orderedSum(_ partial: [Float], start: Float = 0) -> Float {
        var acc = start
        for slot in partial.indices { acc += partial[slot] }
        return acc
    }

    /// Accumulate padded contributions and reduce them in slot order — the whole exchange, arithmetic only.
    public static func reduce(_ contributions: [[Float]], residual: Float, k: Int = k8) throws -> Float {
        orderedSum(try accumulate(contributions, k: k), start: residual)
    }

    /// Accumulate over `dimensions` dimensions laid out `[slot][dimension]`.
    ///
    /// This is the shape the reply carries: each node sends all k slots for every dimension, so a receiver
    /// never has to know the partition to rebuild the full sum.
    public static func accumulateRows(_ contributions: [[[Float]]],
                                      dimensions: Int,
                                      k: Int = k8) throws -> [[Float]] {
        var out = [[Float]](repeating: [Float](repeating: 0, count: dimensions), count: k)
        for rows in contributions {
            guard rows.count == k else {
                throw Error.contributionNotPadded(expected: k, got: rows.count)
            }
            for slot in 0..<k {
                guard rows[slot].count == dimensions else {
                    throw Error.dimensionMismatch(expected: dimensions, got: rows[slot].count)
                }
                for d in 0..<dimensions { out[slot][d] += rows[slot][d] }
            }
        }
        return out
    }

    /// The final per-dimension values, each summed in slot order over the accumulated rows.
    public static func reduceRows(_ contributions: [[[Float]]],
                                  dimensions: Int,
                                  residuals: [Float],
                                  k: Int = k8) throws -> [Float] {
        let rows = try accumulateRows(contributions, dimensions: dimensions, k: k)
        var y = [Float](repeating: 0, count: dimensions)
        for d in 0..<dimensions {
            var acc = d < residuals.count ? residuals[d] : 0
            for slot in 0..<k { acc += rows[slot][d] }
            y[d] = acc
        }
        return y
    }
}
