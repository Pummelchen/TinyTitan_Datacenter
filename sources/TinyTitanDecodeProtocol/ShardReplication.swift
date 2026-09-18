import Foundation

/// Which experts to hold on **every** node, and why it is the only speed lever that cannot change the answer.
///
/// `D166`/`D168`: sharding divides the expert reads, but the exchange that replaces them costs bytes and
/// latency, and the deciding term is **bytes** — 16.7 ms of a per-step exchange, which no plan can reduce
/// except by *replication*. An expert every node holds is an expert that never crosses the wire, so it is
/// removed from the exchange by construction rather than by scheduling.
///
/// Replication is also the **only** such lever that keeps bit-exactness. Every expert a node reads locally
/// contributes its partial to the same fixed k = 8 fp32 sum in the same slot order; an expert it does not read
/// is a zero it adds, which is exact (`ShardReduce`). So the arithmetic is untouched whether an expert is held
/// once or four times — where halving the wire with a lower precision would change the numbers, which `D168`
/// rules out.
///
/// What it costs is device space, and that is what bounds it: at 1,769,472 bytes per expert per layer, an
/// expert replicated everywhere is **4 × 40 × 1,769,472 ≈ 283 MB** of the 2.83 GB cache that `D178` measured as
/// this node's optimum. Replication and the resident expert cache compete for the same memory, so the set is
/// chosen to be small and to earn its place.
public enum ShardReplication {
    /// Bytes of one expert's packed weights, from the install `D175` verified:
    /// `packed_experts/layer_NN.bin` is 452,984,832 B for 256 experts.
    public static let expertBytes = 1_769_472

    public enum Error: Swift.Error, Equatable {
        case negativeCount(Int)
        /// More experts were requested than the model has. Clamping silently would replicate *fewer* experts
        /// than asked for while reporting success, which is the failure mode that makes a wire budget wrong.
        case countExceedsModel(requested: Int, experts: Int)
        case countMismatch(routingCounts: Int, experts: Int)
    }

    /// The experts to replicate, chosen from **observed** routing rather than assumed uniformity.
    ///
    /// The most-routed experts are the ones that would otherwise cross the wire most often, so they are the
    /// ones whose replication removes the most bytes. Ties are broken by **expert index**, so two nodes given
    /// the same counts choose the same set — a replication decision that differed per node would make
    /// `isLocal` disagree across the cluster, and one node would wait for an expert another had decided it
    /// already held.
    public static func replicatedExperts(routingCounts: [Int],
                                         experts: Int,
                                         count: Int) throws -> [Int] {
        guard count >= 0 else { throw Error.negativeCount(count) }
        guard routingCounts.count == experts else {
            throw Error.countMismatch(routingCounts: routingCounts.count, experts: experts)
        }
        guard count <= experts else {
            throw Error.countExceedsModel(requested: count, experts: experts)
        }
        guard count > 0 else { return [] }

        // Descending by count, then ascending by expert index. `sorted(by:)` is not a stable sort in Swift, so
        // the index tie-break is written into the predicate rather than relied on to survive.
        let ranked = (0..<experts).sorted { lhs, rhs in
            routingCounts[lhs] == routingCounts[rhs]
                ? lhs < rhs
                : routingCounts[lhs] > routingCounts[rhs]
        }
        return ranked.prefix(count).sorted()
    }

    /// What replicating `count` experts removes from one node's per-step exchange, **as a model**.
    ///
    /// The honest form of this number needs the real routing distribution, which is why the selection above
    /// takes observed counts rather than assuming uniformity. This helper answers the sizing question — "what
    /// is R worth?" — and is labelled a model so it is never quoted as a measurement.
    ///
    /// - Parameters:
    ///   - routedPerLayer: the model's top-k (8 for a 35B-A3B).
    ///   - layers: 40.
    ///   - share: the fraction of routed slots that land on replicated experts. Under uniform routing this is
    ///     `count / experts`; under real routing it is the sum of the chosen experts' frequencies, and that is
    ///     the number `replicatedExperts` is chosen to maximise.
    public static func modelledWireSavingBytes(count: Int,
                                               experts: Int,
                                               routedPerLayer: Int,
                                               layers: Int,
                                               share: Double) -> Double {
        guard count > 0, experts > 0 else { return 0 }
        // Routed slots per step that no longer cross the wire, times the bytes each one would have carried.
        let slotsAvoided = Double(routedPerLayer) * share * Double(layers)
        return slotsAvoided * Double(expertBytes)
    }

    /// The uniform-routing share, which is the *assumption* `D166`'s byte model rests on and this file exists
    /// to replace with observed counts. Kept separate and named so that using it is a visible choice.
    public static func uniformShare(count: Int, experts: Int) -> Double {
        guard experts > 0 else { return 0 }
        return Double(count) / Double(experts)
    }

    /// Resident cost of replicating `count` experts on one node: the cache it takes from the budget that
    /// `D178` measured an optimum within.
    public static func residentBytes(count: Int, layers: Int) -> Int {
        count * layers * expertBytes
    }
}
