import Foundation

/// One decoded expert slice, as the bank stores it.
private struct BankSlice {
    let values: [Float]
    var bytes: Int { values.count * 4 }
}

/// The `(layer, expert, projection)` a resident slice belongs to.
///
/// The projection is part of the key because a fused gate/up slice and a down slice are **different
/// arrays**: sharing one entry would return the wrong weights, and the only thing that would notice is a
/// wrong number in a trace.
private struct BankKey: Hashable {
    enum Projection: Hashable { case gateUp, down }
    let layer: Int
    let expert: Int
    let projection: Projection
}

/// The expert slices a **whole generation** has resident, across every layer.
///
/// `D97`/`DC-119`. The cache this replaces was correct and useless: `ExpertSlotCache` was built inside the
/// layer-load path and dropped with the layer's other weights, so a generation asking for the same experts
/// on the next token asked a cache that no longer existed. `D31` measured a hit rate of **0 at every size**
/// and it was read as "experts are never reused"; the truth is that they could not be, because the store's
/// lifetime was one layer load. This type moves the store to the generation and keeps the *policy* the same
/// — least-recently-used, a byte budget, and counters, because "the cache helped" is a claim that has to be
/// reproducible.
///
/// **Why a byte budget rather than a slot count.** The old capacity was in experts *per layer*, which made
/// the resident bytes a function of the layer count and the model's geometry. A bank shared by every layer
/// has no such natural unit: what it must not exceed is memory. So `SHARD_EXPERT_BANK_MB` is spent directly,
/// eviction is by bytes, and the size knob means what it says on a model of any shape.
///
/// **Bit-exactness is not at risk, and that is the point of the type.** A cached slice is the same array the
/// uncached path would have decoded, so a hit and a miss produce the same values; the trace digest is
/// unchanged whether the bank is empty, warm or full. `ExpertBankTests` asserts the values a hit returns are
/// the ones the loader produced, not merely a same-shaped array.
///
/// **What it measured, and what it is for now.** Built and measured on the real 35 B-A3B (`D98`): a decode
/// token asks for 773 slices of 12.5 MB, so one token's working set is ~9.7 GB across both projections, and a
/// 537 MB bank has a reuse distance **nine times** its capacity. The alternated A/B found **0.0% hits and
/// identical elements read at 0, 512 and 1024 MB**, the bank on being slightly slower in both pairs. So this
/// type is not the win it was built to be — what it *is* is the instrument that proved the capacity cannot be
/// the lever on an 8 GB node, and the **staging area the preload writes into**: with the fan-out of `DC-118`
/// the same 512 MB is worth **1.35x** on the step, while the bank alone is still a loss (`D101`). `D31`'s "hit rate 0
/// at every size" was right about the workload as well as the lifetime; the lifetime fix is what made the two
/// distinguishable.
///
/// **Thread-safe, deliberately.** The lock is taken to look a slice up and to publish one, never across the
/// load itself — so a caller that fans misses across threads (`DC-118`, the shape `D94` used for the
/// dequantiser) serialises only its bookkeeping, not its I/O or its dequantisation. Two threads that miss
/// the same key may both load it; the second insert wins and the first array is dropped. That is waste, not
/// corruption, and it is the correct trade while a duplicate load is cheaper than a held lock.
public final class ExpertBank: @unchecked Sendable {
    /// The bytes this bank may hold. Zero means "hold nothing", which is the uncached path.
    public let budgetBytes: Int
    /// An optional ceiling on resident slices **per projection**, from `SHARD_EXPERT_SLOTS`.
    public let sliceCap: Int?

    private let lock = NSLock()
    private var slices: [BankKey: BankSlice] = [:]
    private var order: [BankKey] = []
    private var bytesHeld = 0
    private var perProjection: [BankKey.Projection: Int] = [:]
    private var counters = ExpertProviderMetrics()

    public init(budgetBytes: Int, sliceCap: Int? = nil) {
        self.budgetBytes = max(0, budgetBytes)
        self.sliceCap = sliceCap
    }

    /// The bytes resident right now. Read under the lock, because a measurement is not a race.
    public var residentBytes: Int {
        lock.lock(); defer { lock.unlock() }
        return bytesHeld
    }

    /// How many slices are resident right now.
    public var residentSlices: Int {
        lock.lock(); defer { lock.unlock() }
        return slices.count
    }

    /// What the bank has been asked for, and what it had to read.
    ///
    /// `peakResidentExperts` keeps the meaning it has always had — the largest number of slices resident for
    /// **one projection** — because M1's gate reads it and a number whose unit changes silently is worse than
    /// no number. The two projections are counted separately for the same reason `sliceCap` is applied to
    /// each: a fused gate/up and a down are different arrays, and the old capacity was per projection too.
    public var metrics: ExpertProviderMetrics {
        lock.lock(); defer { lock.unlock() }
        var snapshot = counters
        snapshot.peakResidentExperts = max(
            snapshot.peakResidentExperts, max(perProjection[.gateUp] ?? 0, perProjection[.down] ?? 0)
        )
        return snapshot
    }

    /// A slice plus what it cost, so a per-layer caller can count *its* requests while the bank counts the
    /// generation's. Two questions, two counters: summing a shared bank's totals over forty layers is how a
    /// per-layer metric becomes a number four thousand per cent too large.
    struct Outcome {
        let values: [Float]
        let wasHit: Bool
        let elementsLoaded: Int
        /// The bank's own peak for one projection, so a per-layer caller can report the shared ceiling
        /// without taking a second lock to ask. Resident *peak* is a property of the bank, not of a layer.
        let residentPeak: Int
    }

    /// A fused gate/up slice, from the bank when it is resident and from `load` when it is not.
    ///
    /// The load runs **outside** the lock: holding it across a disk read would make a parallel preload
    /// serial and defeat the point of having one.
    func gateUp(layer: Int, expert: Int, load: () throws -> [Float]) rethrows -> Outcome {
        try slice(layer: layer, expert: expert, projection: .gateUp, load: load)
    }

    /// The down projection of an expert, on the same terms as `gateUp`.
    func down(layer: Int, expert: Int, load: () throws -> [Float]) rethrows -> Outcome {
        try slice(layer: layer, expert: expert, projection: .down, load: load)
    }

    private func slice(
        layer: Int, expert: Int, projection: BankKey.Projection, load: () throws -> [Float]
    ) rethrows -> Outcome {
        let key = BankKey(layer: layer, expert: expert, projection: projection)
        if let hit = take(key) { return hit }
        let values = try load()
        let peak = publish(key, values)
        return Outcome(values: values, wasHit: false, elementsLoaded: values.count, residentPeak: peak)
    }

    private func residentPeak() -> Int {
        max(perProjection[.gateUp] ?? 0, perProjection[.down] ?? 0)
    }

    private func take(_ key: BankKey) -> Outcome? {
        lock.lock(); defer { lock.unlock() }
        counters.requests += 1
        guard let slice = slices[key] else {
            counters.misses += 1
            return nil
        }
        counters.hits += 1
        order.removeAll { $0 == key }
        order.append(key)
        return Outcome(
            values: slice.values, wasHit: true, elementsLoaded: 0, residentPeak: residentPeak()
        )
    }

    private func publish(_ key: BankKey, _ values: [Float]) -> Int {
        lock.lock(); defer { lock.unlock() }
        counters.elementsRead += values.count
        guard budgetBytes > 0 else { return residentPeak() }
        let slice = BankSlice(values: values)
        if let existing = slices[key] {
            bytesHeld -= existing.bytes
        } else {
            perProjection[key.projection, default: 0] += 1
        }
        slices[key] = slice
        bytesHeld += slice.bytes
        order.removeAll { $0 == key }
        order.append(key)
        evict()
        return residentPeak()
    }

    private func overCap(_ projection: BankKey.Projection) -> Bool {
        guard let sliceCap else { return false }
        return (perProjection[projection] ?? 0) > sliceCap
    }

    /// Evict least-recently-used until both ceilings hold — the byte budget, and the per-projection slice cap
    /// when `SHARD_EXPERT_SLOTS` asked for one. The cap is per projection because that is what it always meant:
    /// experts per layer, per projection.
    private func evict() {
        while bytesHeld > budgetBytes || overCap(.gateUp) || overCap(.down) {
            let victim: BankKey?
            if overCap(.gateUp), let key = order.first(where: { $0.projection == .gateUp }) {
                victim = key
            } else if overCap(.down), let key = order.first(where: { $0.projection == .down }) {
                victim = key
            } else {
                victim = order.first
            }
            guard let victim else { break }
            order.removeAll { $0 == victim }
            if let slice = slices.removeValue(forKey: victim) {
                bytesHeld -= slice.bytes
                perProjection[victim.projection, default: 0] -= 1
            }
        }
    }
}
