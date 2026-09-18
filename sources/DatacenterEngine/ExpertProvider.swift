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
    /// Whether this provider can serve an expert at all.
    ///
    /// A node owns a subset of the experts while the router selects across all of them, so the expert
    /// path asks before it reads: an unowned expert is **skipped**, not zeroed. Skipping is only safe
    /// because every sharded run checks the reduction for completeness first
    /// (`OrderedReduction.isComplete`) — that check is what `D17` means by "absence has to be loud".
    /// A provider that serves everything, which is every single-node one, keeps the default.
    func serves(_ expert: Int) -> Bool
    /// Fetch these experts ahead of the loop that will ask for them (`DC-118`).
    ///
    /// A **hint**, and the default is to do nothing, because most providers have nowhere to put the bytes.
    /// The one that matters is the slot cache in front of a generation-scoped bank: the reads are latency-bound
    /// (1.08 GB/step at 0.83 GB/s, 520 small `pread`s issued one at a time) and fanning them across threads is
    /// the only way to stop paying for them one at a time. A preload that fails is not an error — the call that
    /// follows raises it, which is where the failure belongs.
    func preload(experts: [Int], shape: MixtureShape)
}

extension ExpertWeightProvider {
    public func serves(_ expert: Int) -> Bool { true }
    public func preload(experts: [Int], shape: MixtureShape) {}
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

/// The expert provider one layer sees, backed by the generation's bank.
///
/// This was a bounded per-layer cache, and the doc comment it replaced admitted the consequence in its own
/// words: "a generation asks for the same experts again on the next token only if the layer is still
/// resident, which it is not". `DC-119` moved the *store* to `ExpertBank`, whose lifetime is the generation,
/// and left this type as the per-layer face of it — so the provider protocol, the mixture's call sites and
/// the sharded path are all unchanged, and the only thing that is different is that a slice fetched for
/// layer 17 on one token can still be there on the next.
///
/// The counters now come from the bank, because with a shared store a per-layer count would answer a
/// question nobody asked ("did *this* layer hit?" rather than "did the generation hit?"). `metrics` is
/// therefore the same snapshot from every layer, which is what a measurement wants.
public final class ExpertSlotCache: ExpertWeightProvider {
    private let upstream: any ExpertWeightProvider
    private let bank: ExpertBank?
    private let layer: Int

    /// What **this layer** was asked for, and what this layer had to read.
    ///
    /// Per layer rather than the bank's total, deliberately: `ForwardResult.expertMetrics` carries one entry
    /// per mixture layer and a caller sums them, so a shared snapshot here would be counted forty times over.
    /// The generation-wide totals live on the bank and reach `metrics.json` through `Generation.experts`.
    public private(set) var metrics = ExpertProviderMetrics()

    public init(upstream: any ExpertWeightProvider, bank: ExpertBank?, layer: Int) {
        self.upstream = upstream
        self.bank = bank
        self.layer = layer
    }

    public func gateUp(expert: Int, shape: MixtureShape) throws -> [Float] {
        guard let bank else {
            metrics.requests += 1
            metrics.misses += 1
            let values = try upstream.gateUp(expert: expert, shape: shape)
            metrics.elementsRead += values.count
            return values
        }
        let outcome = try bank.gateUp(layer: layer, expert: expert) {
            try upstream.gateUp(expert: expert, shape: shape)
        }
        count(outcome)
        return outcome.values
    }

    public func down(expert: Int, shape: MixtureShape) throws -> [Float] {
        guard let bank else {
            metrics.requests += 1
            metrics.misses += 1
            let values = try upstream.down(expert: expert, shape: shape)
            metrics.elementsRead += values.count
            return values
        }
        let outcome = try bank.down(layer: layer, expert: expert) {
            try upstream.down(expert: expert, shape: shape)
        }
        count(outcome)
        return outcome.values
    }

    /// Fetch this layer's chosen experts ahead of the loop that asks for them (`DC-118`).
    ///
    /// Two conditions, and both are load-bearing. The bank must have a budget, because otherwise the
    /// preloaded bytes are discarded and the loop reads them **again** — twice the work rather than half the
    /// latency. And there must be more than one thread, because the point is the fan-out.
    ///
    /// The warm reads deliberately **do not count as requests**: the request is the loop's, and the loop will
    /// find each slice resident and count a *hit*. Counting the warm-up too would double every request and
    /// every miss, and `ForwardResult.expertMetrics` is summed over layers by callers.
    public func preload(experts: [Int], shape: MixtureShape) {
        guard let bank, bank.budgetBytes > 0, DecodeThreads.count > 1 else { return }
        let wanted = experts.filter { upstream.serves($0) }
        guard wanted.count > 1 else { return }
        nonisolated(unsafe) let target = self
        DispatchQueue.concurrentPerform(iterations: wanted.count) { index in
            _ = try? target.warm(expert: wanted[index], shape: shape)
        }
    }

    /// Both projections of one expert, into the bank, without counting the *requests*.
    ///
    /// The volume is a different question from the request, and this was wrong at first: a warm read that
    /// counted nothing hid the bytes it had really read, so `expertElementsRead` went to zero and
    /// `ExpertProviderTests`/`SourceBytesTests` — whose whole point is that traffic is measured — failed with
    /// "the fixture must route experts, or this test proves nothing". The request belongs to the loop; the
    /// **volume belongs to whoever touched the disk**, which is this path when the loop finds the slice
    /// resident.
    private func warm(expert: Int, shape: MixtureShape) throws {
        guard let bank else { return }
        let up = try bank.gateUp(layer: layer, expert: expert) {
            try upstream.gateUp(expert: expert, shape: shape)
        }
        metrics.elementsRead += up.elementsLoaded
        let downOutcome = try bank.down(layer: layer, expert: expert) {
            try upstream.down(expert: expert, shape: shape)
        }
        metrics.elementsRead += downOutcome.elementsLoaded
    }

    private func count(_ outcome: ExpertBank.Outcome) {
        metrics.requests += 1
        if outcome.wasHit {
            metrics.hits += 1
        } else {
            metrics.misses += 1
            metrics.elementsRead += outcome.elementsLoaded
        }
        // The ceiling is the bank's, not this layer's: with a shared store a per-layer peak would be a number
        // nobody can act on, and M1's gate reads this field to check the budget is respected.
        metrics.peakResidentExperts = max(metrics.peakResidentExperts, outcome.residentPeak)
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
