import DatacenterIR
import Foundation

/// A decoded decoder layer, held so it does not have to be decoded again.
///
/// `loadLayer` builds one of these by dequantising the install's int4 codes into fp32; `D88` measured that this
/// costs **30.5% of a cached step**, that it happens on every token, and that the bytes it decodes never change.
/// The type exists so those bytes can be kept.
struct DecodedLayer {
    let weights: [TensorRole: [Float]]
    let gdn: GatedDeltaNetWeights?
    let feedForward: Qwen3_5Forward.FeedForward

    /// The bytes this layer occupies as fp32 reals, counted rather than estimated.
    ///
    /// Counted, because the budget has to be a measurement: the same arithmetic spelled out as
    /// `3·inter·hidden·4` per expert is how a 14.5 GB change once shipped past 98 green tests
    /// (`SlotBudgetTests`). What is held here is what is counted here.
    var bytes: Int {
        var total = 0
        for values in weights.values { total += values.count * MemoryLayout<Float>.size }
        // The mixture's expert slot banks are *not* counted: their arrays are the provider's, and the provider
        // reports its own bytes. What is counted is what this type owns.
        if case .dense(let gate, let up, let down) = feedForward {
            total += (gate.count + up.count + down.count) * MemoryLayout<Float>.size
        }
        return total
    }
}

/// What a layer cache did, for `metrics.json`. Nil on a generation means **no cache was used**, which is not
/// the same as a cache that held nothing.
public struct LayerCacheMetrics: Sendable, Equatable {
    public var budgetBytes = 0
    public var bytesHeld = 0
    public var layersHeld = 0
    public var layerCount = 0
    public var hits = 0
    public var misses = 0

    public init(
        budgetBytes: Int = 0, bytesHeld: Int = 0, layersHeld: Int = 0, layerCount: Int = 0,
        hits: Int = 0, misses: Int = 0
    ) {
        self.budgetBytes = budgetBytes
        self.bytesHeld = bytesHeld
        self.layersHeld = layersHeld
        self.layerCount = layerCount
        self.hits = hits
        self.misses = misses
    }
}

/// Decoded layers held across the steps of one generation.
///
/// **Why a set of layers and not an LRU.** A decode sweeps every layer in order, once per token, so a
/// least-recently-used policy has a **zero** hit rate by construction — the layer evicted is always the one
/// about to be asked for. Every step revisits every layer, so holding *any* layer across steps pays on every
/// step, and the layers to hold are simply the ones that fit. That is also why the budget is spent here rather
/// than in the expert slot bank: `D31` measured that bank's hit rate at **zero at every size**, while a layer
/// held here is a hit on every step by construction.
///
/// A cache is created per generation and owned by it, so a test can hand in a budget of zero and see the
/// uncached path, and so no mutable state lives on the (immutable, `Sendable`) forward pass.
public final class LayerWeightCache {
    private var layers: [Int: DecodedLayer] = [:]
    private var bytes = 0
    private var hits = 0
    private var misses = 0
    public let budgetBytes: Int
    public let layerCount: Int

    public init(budgetBytes: Int, layerCount: Int) {
        self.budgetBytes = max(0, budgetBytes)
        self.layerCount = layerCount
    }

    /// The layer at `index`, decoded by `load` only when it is not already held.
    ///
    /// Internal, unlike the type: `DecodedLayer` is the engine's own assembly of a layer, and a caller of the
    /// public API has no business constructing one — it hands in a budget and reads the metrics.
    ///
    /// A layer that does not fit the remaining budget is still returned, and is simply not kept — a miss that
    /// is not cached is a miss next step, which is honest and bounded, rather than a budget quietly exceeded.
    func layer(_ index: Int, load: () throws -> DecodedLayer) rethrows -> DecodedLayer {
        if let held = layers[index] {
            hits += 1
            return held
        }
        misses += 1
        let decoded = try load()
        let cost = decoded.bytes
        if cost > 0, bytes + cost <= budgetBytes {
            layers[index] = decoded
            bytes += cost
        }
        return decoded
    }

    public var metrics: LayerCacheMetrics {
        LayerCacheMetrics(
            budgetBytes: budgetBytes, bytesHeld: bytes, layersHeld: layers.count, layerCount: layerCount,
            hits: hits, misses: misses
        )
    }
}
