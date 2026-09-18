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
    /// The gate-up projection's **product**, from the stored int4 form, when this provider can compute it
    /// without materialising the weights (`D109`).
    ///
    /// `nil` means "not this provider", and the caller falls back to `gateUp` followed by the contract matmul.
    /// The default is `nil`, so `ArrayExpertProvider`, the counting wrapper and every test keep the old path
    /// with no code of their own.
    func gateUpProduct(x: [Float], rows: Int, expert: Int, shape: MixtureShape) throws -> [Float]?
    /// The down projection's product, on the same terms as `gateUpProduct`.
    func downProduct(x: [Float], rows: Int, expert: Int, shape: MixtureShape) throws -> [Float]?
    /// Read these experts' **stored** forms ahead of the loop, the packed counterpart of `preload` (`D109`).
    ///
    /// `preload` warms a bank of fp32 slices; when the fused path is on there are no fp32 slices to warm, so
    /// the fan-out has to warm the packed slab cache instead or the reads fall back to one at a time.
    func preloadPacked(experts: [Int], shape: MixtureShape)
    /// Whether `gateUpProduct`/`downProduct` can actually serve this provider (`D110`).
    ///
    /// `preload` has to choose between two destinations — a bank of fp32 slices and a cache of packed slabs —
    /// and it can only choose correctly if it knows which one this provider has. Asking the *capability*
    /// rather than testing the switch is what lets an array-backed provider keep the bank path it has always
    /// had while an install takes the packed one.
    var servesPacked: Bool { get }
}

extension ExpertWeightProvider {
    public func serves(_ expert: Int) -> Bool { true }
    public func preload(experts: [Int], shape: MixtureShape) {}
    public func gateUpProduct(x: [Float], rows: Int, expert: Int, shape: MixtureShape) throws -> [Float]? { nil }
    public func downProduct(x: [Float], rows: Int, expert: Int, shape: MixtureShape) throws -> [Float]? { nil }
    public func preloadPacked(experts: [Int], shape: MixtureShape) {}
    public var servesPacked: Bool { false }
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

    /// **The fused path** (`D109`): the slab goes to the device kernel, which dequantises as it multiplies.
    ///
    /// `nil` is the answer whenever the device cannot be used — no Metal, the switch off — and also when the
    /// source has no stored form, which is what keeps this honest for a checkpoint. The output width is
    /// checked exactly as `gateUp`/`down` check the input width, and for the same reason: a misread stack
    /// would otherwise be multiplied as if it were the right matrix.
    public func gateUpProduct(x: [Float], rows: Int, expert: Int, shape: MixtureShape) throws -> [Float]? {
        try product(x: x, rows: rows, expert: expert, name: gateUpName, out: 2 * shape.intermediate)
    }

    public func downProduct(x: [Float], rows: Int, expert: Int, shape: MixtureShape) throws -> [Float]? {
        try product(x: x, rows: rows, expert: expert, name: downName, out: shape.hiddenSize)
    }

    /// `D101`'s fan-out, against the **packed** slabs rather than a bank of `[Float]`.
    ///
    /// The budget check is load-bearing and was learned the hard way: with `SHARD_SLAB_CACHE_MB=0` there is
    /// nowhere to keep what this reads, so the loop reads every slab a **second** time — the fused arm
    /// measured 14.86 GB of reads against the control's 8.46 GB, which is the whole of its regression. A
    /// preload with no cache is not a hidden read, it is a duplicated one.
    public func preloadPacked(experts: [Int], shape: MixtureShape) {
        guard MetalInt4Matmul.enabled, MetalInt4Matmul.isAvailable,
            source.packedCacheBudget > 0, !experts.isEmpty
        else { return }
        nonisolated(unsafe) let target = self
        // **One task per (expert, projection), not per expert** (`D113`). A slab is three `pread`s — codes,
        // scales, zeros — so fanning over eight experts gave eight threads each issuing six *sequential*
        // reads; the fan-out is the only concurrency the device sees, and it was half what the layer asked
        // for. Sixteen tasks of three reads each is the same bytes with twice the requests in flight.
        let names = [gateUpName, downName]
        DispatchQueue.concurrentPerform(iterations: experts.count * names.count) { index in
            let range = experts[index / names.count]..<(experts[index / names.count] + 1)
            _ = try? target.source.packedRows(named: names[index % names.count], range: range)
        }
    }

    /// An install has a packed cache only when the knob gives it one; without it a preload would read bytes
    /// that nothing keeps, which is a duplicated read rather than a hidden one (`D110`).
    public var servesPacked: Bool { MetalInt4Matmul.isAvailable && source.packedCacheBudget > 0 }

    private func product(x: [Float], rows: Int, expert: Int, name: String, out: Int) throws -> [Float]? {
        guard MetalInt4Matmul.enabled, MetalInt4Matmul.isAvailable else { return nil }
        guard let packed = try source.packedRows(named: name, range: expert..<(expert + 1)) else { return nil }
        let product = try MetalInt4Matmul.matmul(
            payload: packed.payload, entry: packed.entry, rowCount: packed.payloadRows, x: x, rows: rows
        )
        guard product.count == rows * out else {
            throw ExpertProviderError.unexpectedWidth(
                tensor: name, expert: expert, got: product.count, expected: rows * out
            )
        }
        return product
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

    /// The counters are written from the preload's threads as well as the loop's (`D101`), so the writes are
    /// guarded. Reads are not, for the same reason `InstallFile.ReadState` gives: the metrics are asked for
    /// between forwards, when no worker is running.
    private let metricsLock = NSLock()

    /// What this layer's loop asked for, for `DC-121`, and the shape it asked with.
    ///
    /// Written only by the single-threaded loop and read only as a snapshot taken on that same thread inside
    /// `prefetchPredicted`, so it needs no lock of its own — which is worth stating because the last two rounds
    /// have both been about a shared slot that was read from somewhere else.
    private var lastRequested: [Int] = []
    private var lastShape: MixtureShape?

    /// One serial queue for prefetching, shared by every layer of every forward: the point is one background
    /// thread, not one per layer.
    private static let prefetchQueue = DispatchQueue(label: "datacenter.expert.prefetch")

    /// `SHARD_EXPERT_PREDICT=1` turns `DC-121` on — **off by default, because it was measured and it loses.**
    ///
    /// The reasoning was sound and the measurement disagreed: a layer's routing is correlated between tokens, the
    /// reads were on the critical path, so issuing them early should have hidden them behind the attention. It did
    /// not. Alternated on one binary: **1.859 / 1.816 s/step with it on against 1.854 / 1.721 with it off**, and
    /// `mix.read` *larger* with it on (0.830 / 0.813 against 0.790 / 0.760). The digest was identical throughout.
    ///
    /// The reading is that `D101`'s fan-out had already stopped this being a latency problem: the device is now
    /// **saturated**, so a second stream of reads does not fill a gap, it takes bandwidth from the first. That is
    /// `DC-121`'s real finding, and it moves the next lever from *hiding* the read to **reading fewer bytes** —
    /// which is a bank that holds slices in their packed int4 form rather than dequantised.
    static let predictionEnabled = ProcessInfo.processInfo.environment["SHARD_EXPERT_PREDICT"] == "1"

    private func note(_ expert: Int, shape: MixtureShape) {
        if lastShape == nil { lastShape = shape }
        if !lastRequested.contains(expert) { lastRequested.append(expert) }
    }

    private func addElementsRead(_ count: Int) {
        metricsLock.lock(); metrics.elementsRead += count; metricsLock.unlock()
    }

    public init(upstream: any ExpertWeightProvider, bank: ExpertBank?, layer: Int) {
        self.upstream = upstream
        self.bank = bank
        self.layer = layer
    }

    public func gateUp(expert: Int, shape: MixtureShape) throws -> [Float] {
        note(expert, shape: shape)
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
        note(expert, shape: shape)
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

    /// Remember what the loop asked for, so the *next* token can be anticipated (`DC-121`).
    ///
    /// `DC-118` hides the reads behind **other reads** by fanning the misses out. This hides them behind *compute*:
    /// a layer's routing is strongly correlated between consecutive tokens, so the experts this layer wanted last
    /// time are issued **now**, on a background thread, before the attention work below them, and the loop that
    /// follows finds them resident. A wrong guess costs only the read the loop would have made anyway.
    ///
    /// A **serial** background thread rather than a fan-out is deliberate: the reads are I/O-bound and the eight
    /// cores are busy with the layer's own arithmetic, so a second dispatch wave here would compete for the very
    /// resource it is trying to stay off the critical path of.
    /// `force` exists so the mechanism can be tested while the default stays off: a switch read once from the
    /// environment cannot be turned on for one test, and a mechanism that is never exercised is a mechanism that
    /// rots.
    public func prefetchPredicted(force: Bool = false) {
        guard force || Self.predictionEnabled, let bank, bank.budgetBytes > 0, DecodeThreads.count > 1 else {
            return
        }
        let predicted = lastRequested
        guard predicted.count > 1, let shape = lastShape else { return }
        nonisolated(unsafe) let target = self
        Self.prefetchQueue.async {
            for expert in predicted { _ = try? target.warm(expert: expert, shape: shape) }
        }
    }

    /// Wait for the background prefetch, so a test can assert what it warmed instead of sleeping on it.
    func waitForPrefetch() { Self.prefetchQueue.sync {} }

    /// The fused products, on their way past the bank.
    ///
    /// There is nothing for the bank to hold here: the whole point is that the fp32 slice never exists. The
    /// counters are kept honest rather than skipped — a fused fetch **is** a request and a miss (it went to
    /// the source), and the elements read are the weights the kernel walked, computed from the geometry
    /// exactly as the fp32 path counts them — so the gate's hit-rate and traffic numbers mean the same thing
    /// in both arms.
    public func gateUpProduct(x: [Float], rows: Int, expert: Int, shape: MixtureShape) throws -> [Float]? {
        guard let product = try upstream.gateUpProduct(x: x, rows: rows, expert: expert, shape: shape) else {
            return nil
        }
        countFused()
        return product
    }

    public func downProduct(x: [Float], rows: Int, expert: Int, shape: MixtureShape) throws -> [Float]? {
        guard let product = try upstream.downProduct(x: x, rows: rows, expert: expert, shape: shape) else {
            return nil
        }
        countFused()
        return product
    }

    /// The fused path counts a **request** and nothing else.
    ///
    /// There is no bank to hit or miss, and `elementsRead` must mean "bytes that came off the source" rather
    /// than "weights this call walked" — the first version counted every request as a read and reported
    /// 13,824 elements read on a forward whose slabs were all served from the packed cache. The traffic
    /// measure for this mode is the source's own `bytesReadFromSource` and the slab cache's counters, which
    /// is where the disk was actually touched.
    private func countFused() {
        metrics.requests += 1
    }

    /// Fetch this layer's chosen experts ahead of the loop that asks for them (`DC-118`).
    ///
    /// Two conditions, and both are load-bearing. The bank must have a budget, because otherwise the
    /// preloaded bytes are discarded and the loop reads them **again** — twice the work rather than half the
    /// latency. And there must be more than one thread, because the point is the fan-out.
    ///
    /// Under the fused path there is no bank to warm, so the fan-out warms the **packed slab cache** instead
    /// (`D109`) — the same threads, the same hint, a different destination, and without it the reads would
    /// fall back to one at a time.
    public func preload(experts: [Int], shape: MixtureShape) {
        if MetalInt4Matmul.enabled, upstream.servesPacked {
            upstream.preloadPacked(experts: experts.filter { upstream.serves($0) }, shape: shape)
            return
        }
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
        addElementsRead(up.elementsLoaded)
        let downOutcome = try bank.down(layer: layer, expert: expert) {
            try upstream.down(expert: expert, shape: shape)
        }
        addElementsRead(downOutcome.elementsLoaded)
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
