import Foundation

/// Where a mixture's expert weights come from, one expert at a time.
///
/// This is the interface the whole of M1's memory story hangs on. A layer's experts are
/// `[experts, 2·intermediate, hidden]` and `[experts, hidden, intermediate]`: for the 35 B
/// model that is 805 M parameters per layer, 3.2 GB in fp32 and 1.6 GB even in bf16, against
/// about 4.5 GB of usable memory per node. So the *only* workable shape is to read the experts
/// the router actually chose and nothing else, which means the kernel must ask for them by
/// index rather than being handed a stack.
///
/// Two things follow from the interface, and both are deliberate:
///
/// - **the read order is the reduction order.** `MixtureOfExperts` asks in ascending expert
///   index, which is the order the contract accumulates in and the order D4's ring reduction
///   will use. A cache that reordered reads to be sequential on disk would change the
///   arithmetic, so it does not.
/// - **a provider is a protocol, not a file.** The array-backed one is what the tests and the
///   tiny checkpoints use; the source-backed one streams row ranges; the counting one wraps
///   either to measure. The kernel cannot tell them apart, which is what makes the bit-identity
///   between them a meaningful test rather than a coincidence.
public protocol ExpertWeightProvider {
    /// `[2·intermediate, hidden]`, the gate in the first half and the up in the second.
    func gateUp(expert: Int, shape: MixtureShape) throws -> [Float]
    /// `[hidden, intermediate]`.
    func down(expert: Int, shape: MixtureShape) throws -> [Float]
}

/// Raised when a source hands back a width the mixture's geometry does not agree with — a
/// misread stack would otherwise be multiplied as if it were the right matrix.
public enum ExpertProviderError: Swift.Error, CustomStringConvertible {
    case unexpectedWidth(tensor: String, expert: Int, got: Int, expected: Int)

    public var description: String {
        switch self {
        case .unexpectedWidth(let tensor, let expert, let got, let expected):
            return "expert \(expert) of \(tensor) has \(got) values, expected \(expected)"
        }
    }
}

/// Reads one expert's slices out of a stacked tensor, by row range.
///
/// The checkpoint stores the experts stacked with the **expert as the leading axis** —
/// `[experts, 2·intermediate, hidden]` and `[experts, hidden, intermediate]` — so one expert is
/// exactly **one row** of that tensor, and the reader's row is the expert's whole projection.
/// That is why this is a row range of length one: it is not a coincidence of the test fixture
/// but the layout the vendor ships, and it is what makes an expert fetch a single seek.
///
/// The width is checked against the mixture's own geometry before the values are used. A stack
/// whose leading axis were *not* the expert would still have rows, and every row would be the
/// wrong matrix — a failure that multiplies cleanly and means nothing.
///
/// Note what this does **not** do: it does not read every expert in the tensor to pick one out.
/// That is the difference between this and the array path it replaces, and it is the whole
/// point of `DC-032`.
public struct StackedExpertProvider: ExpertWeightProvider {
    private let source: any WeightSource
    private let gateUpName: String
    private let downName: String

    public init(source: any WeightSource, gateUpName: String, downName: String) {
        self.source = source
        self.gateUpName = gateUpName
        self.downName = downName
    }

    public func gateUp(expert: Int, shape: MixtureShape) throws -> [Float] {
        let values = try source.rowsStreaming(named: gateUpName, range: expert..<(expert + 1))
        let expected = 2 * shape.intermediate * shape.hiddenSize
        guard values.count == expected else {
            throw ExpertProviderError.unexpectedWidth(
                tensor: gateUpName, expert: expert, got: values.count, expected: expected
            )
        }
        return values
    }

    public func down(expert: Int, shape: MixtureShape) throws -> [Float] {
        let values = try source.rowsStreaming(named: downName, range: expert..<(expert + 1))
        let expected = shape.hiddenSize * shape.intermediate
        guard values.count == expected else {
            throw ExpertProviderError.unexpectedWidth(
                tensor: downName, expert: expert, got: values.count, expected: expected
            )
        }
        return values
    }
}

/// A provider over arrays already in memory — the tiny checkpoints, and the tests that pin the
/// streaming path against the non-streaming one.
public struct ArrayExpertProvider: ExpertWeightProvider {
    private let gateUpStack: [Float]
    private let downStack: [Float]

    public init(gateUp: [Float], down: [Float]) {
        self.gateUpStack = gateUp
        self.downStack = down
    }

    public func gateUp(expert: Int, shape: MixtureShape) throws -> [Float] {
        let rows = 2 * shape.intermediate
        let width = shape.hiddenSize
        let start = expert * rows * width
        return Array(gateUpStack[start..<(start + rows * width)])
    }

    public func down(expert: Int, shape: MixtureShape) throws -> [Float] {
        let rows = shape.hiddenSize
        let width = shape.intermediate
        let start = expert * rows * width
        return Array(downStack[start..<(start + rows * width)])
    }
}

/// What a provider actually did, so the gate can be a number rather than an assurance.
///
/// M1's gate asks for "a measured cache hit rate" and for resident memory to stay inside a
/// budget; neither is observable from the outside of a working forward pass, so the provider
/// counts.
public struct ExpertProviderMetrics: Sendable, Equatable {
    /// Expert slices requested, i.e. one per (expert, projection) read.
    public var requests: Int = 0
    /// Requests served from a slot without touching the source.
    public var hits: Int = 0
    /// Requests that had to go to the source — the SSD traffic, in other words.
    public var misses: Int = 0
    /// **Elements** read from the source, in fp32. The unit is named because getting it wrong is
    /// easy and invisible: this counted elements while the counting provider counted rows, and a
    /// figure that is off by the row width still looks like a plausible number of bytes.
    ///
    /// The brief's currency is *bytes read per token*, which is this times the bytes per element
    /// the file stores — two bytes for a bf16 shard, four for the fp32 in memory. The trace tool
    /// reports both, because they differ by a factor of two and the SSD cares about the first.
    public var elementsRead: Int = 0
    /// The largest number of expert slices resident at once, per projection.
    public var peakResidentExperts: Int = 0

    public init() {}

    public var hitRate: Double {
        requests == 0 ? 0 : Double(hits) / Double(requests)
    }
}

/// A bounded cache of expert slices, with the counters M1's gate needs.
///
/// The policy is written down rather than implied, because each choice is visible in the
/// arithmetic or in the timings:
///
/// - **exactly the requested expert is fetched**, never a neighbourhood: prefetching neighbours
///   would read bytes the model did not ask for, and on a 1,500 MB/s SSD the bytes are the
///   budget;
/// - **eviction is least-recently-used**, and the capacity is in *experts*, so the resident
///   bytes are `capacity × (2·intermediate·hidden + hidden·intermediate) × 4`;
/// - **the counters are part of the type**, because "the cache helped" is a claim that has to
///   be reproducible.
///
/// The cache belongs to one layer: the forward pass builds it as it loads the layer and drops
/// it with the layer's other weights, which is the "per-layer ID→slot bank" the design calls
/// for. That is also why a hit is possible at all inside a single token — several of the
/// token's chosen experts can repeat across positions, and a generation asks for the same
/// experts again on the next token only if the layer is still resident, which it is not.
public final class ExpertSlotCache: ExpertWeightProvider {
    private let upstream: any ExpertWeightProvider
    private let capacity: Int
    private var gateUpSlots: [Int: [Float]] = [:]
    private var downSlots: [Int: [Float]] = [:]
    private var gateUpOrder: [Int] = []
    private var downOrder: [Int] = []
    public private(set) var metrics = ExpertProviderMetrics()

    public init(upstream: any ExpertWeightProvider, capacity: Int) {
        precondition(capacity >= 1, "a cache with no slots is not a cache")
        self.upstream = upstream
        self.capacity = capacity
    }

    private func fetch(
        _ slots: inout [Int: [Float]], _ order: inout [Int], _ expert: Int,
        _ load: () throws -> [Float]
    ) rethrows -> [Float] {
        metrics.requests += 1
        if let cached = slots[expert] {
            metrics.hits += 1
            order.removeAll { $0 == expert }
            order.append(expert)
            return cached
        }
        metrics.misses += 1
        let loaded = try load()
        slots[expert] = loaded
        order.append(expert)
        metrics.elementsRead += loaded.count
        if order.count > capacity {
            let evicted = order.removeFirst()
            slots.removeValue(forKey: evicted)
        }
        metrics.peakResidentExperts = max(metrics.peakResidentExperts, slots.count)
        return loaded
    }

    public func gateUp(expert: Int, shape: MixtureShape) throws -> [Float] {
        try fetch(&gateUpSlots, &gateUpOrder, expert) { try upstream.gateUp(expert: expert, shape: shape) }
    }

    public func down(expert: Int, shape: MixtureShape) throws -> [Float] {
        try fetch(&downSlots, &downOrder, expert) { try upstream.down(expert: expert, shape: shape) }
    }
}

/// Counts what a provider read, without caching anything. The instrument for the test that
/// says the streaming path reads only the chosen experts.
public final class CountingExpertProvider: ExpertWeightProvider {
    private let upstream: any ExpertWeightProvider
    public private(set) var requested: [Int] = []
    /// Elements read, to match `ExpertProviderMetrics.elementsRead`.
    public private(set) var elementsRead: Int = 0

    public init(upstream: any ExpertWeightProvider) {
        self.upstream = upstream
    }

    public func gateUp(expert: Int, shape: MixtureShape) throws -> [Float] {
        requested.append(expert)
        let values = try upstream.gateUp(expert: expert, shape: shape)
        elementsRead += values.count
        return values
    }

    public func down(expert: Int, shape: MixtureShape) throws -> [Float] {
        let values = try upstream.down(expert: expert, shape: shape)
        elementsRead += values.count
        return values
    }
}
