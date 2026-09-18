import CryptoKit
import DatacenterIR
import Foundation

/// A weight source: the two operations every backend offers, and the only two the forward
/// passes use.
///
/// There are two implementations — a checkpoint's `safetensors` and a quantized install —
/// and the forward pass is written against this rather than against either, so the same
/// arithmetic runs on both and a difference between them is the quantization's.
/// A leading-axis row range of an int4 tensor **exactly as stored**, for a consumer that decodes it
/// itself — the device kernels above all.
///
/// `payload` is the layout `InstallFile.dequantizeInt4` takes: every row's codes, then every row's
/// scales, then every row's zeros. `payloadRows` is the number of **payload** rows in the range, which is
/// *not* `range.count` for a stacked expert tensor — one expert of `[256, 1024, 2048]` is 1,024 payload
/// rows, and confusing the two is what `rows`' own comment calls out as a bug that still looks like a
/// tensor.
public struct PackedInt4Rows: Sendable {
    public let payload: Data
    public let entry: InstallFile.Entry
    public let payloadRows: Int
    /// The tensor's name when this is a **whole dense tensor** rather than an expert's row range (`D116`).
    ///
    /// It is the identity of a weight that does not change between steps, which is what lets
    /// `MetalInt4Matmul` map it on the device once instead of copying it on every call. A row range of a
    /// stacked expert tensor leaves this nil: those bytes are staged by the slab cache and may be evicted, so
    /// a mapping of them would dangle.
    public let key: String?

    public init(payload: Data, entry: InstallFile.Entry, payloadRows: Int, key: String? = nil) {
        self.payload = payload
        self.entry = entry
        self.payloadRows = payloadRows
        self.key = key
    }
}

/// A row range of a **non-int4** tensor, exactly as stored — bf16, above all, which is how the LM head and
/// the smaller dense tensors are held (`D109`).
///
/// `data` is a slice of the source's cached whole-tensor payload, so fetching many ranges of one tensor
/// costs one read and no copies. `dtype` and `width` travel with it because a consumer has to know how to
/// widen the row and how many elements a row holds.
public struct StoredRows: Sendable {
    public let data: Data
    public let dtype: String
    public let rowCount: Int
    public let width: Int

    public init(data: Data, dtype: String, rowCount: Int, width: Int) {
        self.data = data
        self.dtype = dtype
        self.rowCount = rowCount
        self.width = width
    }
}

public protocol WeightSource {
    /// A whole tensor, in fp32.
    func tensor(named name: String) throws -> [Float]
    /// A row range, in fp32. The embedding and the tied head are read this way so a
    /// `[248320, 2048]` matrix is never materialised to use one row of it.
    func rows(named name: String, range: Range<Int>) throws -> [Float]
    /// A row range for a consumer that will not read it again soon, so its pages are not worth
    /// keeping: the routed expert slabs. Defaults to the ordinary read, which is what an install
    /// already does — its payload is read uncached by construction.
    func rowsStreaming(named name: String, range: Range<Int>) throws -> [Float]
    /// The **stored** bytes of a leading-axis row range, when this source can hand them over without
    /// decoding them (`D108`).
    ///
    /// A quantized source can, and that is the point: `dequantizeInt4` materialises four bytes per weight
    /// where the file holds half a byte, and a device kernel consumes the packed form directly. The
    /// default is `nil`, so every source keeps working and a caller falls back to `rows` — a checkpoint
    /// has no packed form at all.
    func packedRows(named name: String, range: Range<Int>) throws -> PackedInt4Rows?
    /// The **whole** tensor in its stored form, when the source has one (`D111`).
    ///
    /// `packedRows` takes a leading-axis range; a dense projection is wanted entire, and its row count lives in
    /// the source's own manifest rather than in the caller, so the caller has nothing to pass. This is the
    /// accessor that lets a dense path avoid decoding a weight it is only going to multiply.
    func packedTensor(named name: String) throws -> PackedInt4Rows?
    /// The **stored** bytes of a row range for a dtype that is not int4 — bf16 above all, which is how the
    /// LM head is held (`D109`).
    ///
    /// `rows` decodes these to `Float` through a whole-tensor read that never touches the payload cache, so
    /// the head's 1.017 GB came off the device on **every** token. Here the whole tensor is fetched through
    /// `payload`, which caches it, and the range is a slice of the cached bytes with no copy and no decode.
    /// The default is `nil`, so a source that holds only fp32 keeps the old path.
    func storedRows(named name: String, range: Range<Int>) throws -> StoredRows?
    /// Payload bytes this source has **actually read**, when it counts them.
    ///
    /// Zero means **not counted**, not "nothing was read": the dense family loads its weights at
    /// open time and holds them, so it has no live counter to ask, and a caller must not present
    /// that as zero traffic. That confusion is exactly what made `expert_bytes_from_ssd` an
    /// estimate — `elementsRead * 2`, which is 3.5x too large for a 4-bit install.
    var bytesReadFromSource: Int { get }
    /// Where the time goes inside this source, when it counts it.
    var sourceTiming: SourceTiming { get }
    /// The payload cache's counters; a source without one reports zeroes.
    var payloadCacheMetrics: PayloadCacheMetrics { get }
    /// Whole-tensor requests per tensor, most-requested first — the `DC-106` repeat-read audit.
    var payloadRequestCounts: [(name: String, count: Int)] { get }
    /// The **stored-form** cache's budget in bytes, when this source has one (`D109`).
    ///
    /// A preload of packed rows is only useful if the bytes it reads are kept: with nowhere to put them the
    /// loop reads the same slab again, which is work done twice rather than latency hidden. The default is
    /// zero, so a source with no such cache (a checkpoint) makes `preloadPacked` a no-op.
    var packedCacheBudget: Int { get }
}

/// A source's own account of its cost: bytes read, and the seconds spent reading, verifying and
/// unpacking them. Fractions of 1.0 sum to the time the caller spent inside the source.
/// What the whole-tensor payload cache did (`DC-106`), on the `SourceTiming` pattern: a value the
/// caller collects rather than four accessors to keep in step.
public struct PayloadCacheMetrics: Sendable, Equatable {
    /// Whole-tensor payload bytes that had to come from disk. Row-range reads are not counted here:
    /// residency cannot eliminate them, so a second forward driving this to **zero** is the claim.
    public var bytesRead = 0
    /// Whole-tensor reads the cache served instead.
    public var hits = 0
    /// What the cache is holding, in bytes.
    public var bytesHeld = 0

    /// The **packed slab** cache's counters (`DC-120`), reported here rather than in a second struct because a
    /// caller asks one question of both: how much of this run's payload had to come off the device.
    ///
    /// `slabHits` are expert row ranges served from memory **without touching the disk at all**, which is the
    /// quantity `D105` identified as the lever — the device is saturated, so the only way to make the read
    /// cheaper is not to make it.
    public var slabHits = 0
    public var slabMisses = 0
    public var slabBytesHeld = 0

    public init(
        bytesRead: Int = 0, hits: Int = 0, bytesHeld: Int = 0,
        slabHits: Int = 0, slabMisses: Int = 0, slabBytesHeld: Int = 0
    ) {
        self.bytesRead = bytesRead
        self.hits = hits
        self.bytesHeld = bytesHeld
        self.slabHits = slabHits
        self.slabMisses = slabMisses
        self.slabBytesHeld = slabBytesHeld
    }
}

public struct SourceTiming: Sendable {
    public var counted = false
    public var bytes = 0
    /// How much of `bytes` was spent verifying rather than using. Both numbers, because a total that
    /// folds verification in cannot answer "how much did the model actually need?".
    public var verifiedBytes = 0
    public var readSeconds = 0.0
    public var digestSeconds = 0.0
    public var unpackSeconds = 0.0

    public init() {}

    public var accountedSeconds: Double { readSeconds + digestSeconds + unpackSeconds }
}

extension WeightSource {
    public var bytesReadFromSource: Int { 0 }
    public var sourceTiming: SourceTiming { SourceTiming() }
    public var payloadCacheMetrics: PayloadCacheMetrics { PayloadCacheMetrics() }
    public var payloadRequestCounts: [(name: String, count: Int)] { [] }
    public var packedCacheBudget: Int { 0 }
}

extension WeightSource {
    public func rowsStreaming(named name: String, range: Range<Int>) throws -> [Float] {
        try rows(named: name, range: range)
    }

    /// A checkpoint stores no packed form, and neither does a source that decodes on the way in, so the
    /// default is to have nothing to hand over.
    public func packedRows(named name: String, range: Range<Int>) throws -> PackedInt4Rows? { nil }

    /// A source that holds only fp32 has nothing stored to hand over either.
    public func storedRows(named name: String, range: Range<Int>) throws -> StoredRows? { nil }

    public func packedTensor(named name: String) throws -> PackedInt4Rows? { nil }
}

extension SafetensorsFile: WeightSource {
    public func tensor(named name: String) throws -> [Float] { try float32(name) }
    public func rows(named name: String, range: Range<Int>) throws -> [Float] { try float32(name, rows: range) }
}

/// Reads an install written by `tools/quantize.py`.
///
/// The dequantization here has to be **bit-identical** to the Python one: the packed codes,
/// the group scales and the zero points are integers and exact arithmetic, so there is no
/// rounding to disagree about — which is why the two implementations can be compared byte
/// for byte rather than by tolerance.
public struct InstallFile: WeightSource {
    public struct Entry: Decodable, Sendable {
        public var name: String
        public var role: String
        public var quant: String
        public var shape: [Int]
        public var padded_columns: Int
        /// A digest per **leading-axis slab**, when the install was written with them. Optional, so
        /// a schema-1 install still loads and is checked the expensive way.
        public var slab_sha256: [String]?
        public var group: Int
        public var dtype: String
        public var offset: Int
        public var nbytes: Int
        public var sha256: String
    }

    public struct Manifest: Decodable, Sendable {
        public var schema: Int
        public var family: String
        public var passes: [String]
        public var policy_files: [String]
        public var spec: IRSpec
        public var tensors: [Entry]
    }

    public enum Error: Swift.Error, CustomStringConvertible {
        case unknownTensor(String)
        case badHeader(String)
        case digestMismatch(String)

        public var description: String {
            switch self {
            case .unknownTensor(let name): return "the install has no tensor '\(name)'"
            case .badHeader(let detail): return "install header: \(detail)"
            case .digestMismatch(let name): return "\(name): the payload digest does not match the manifest"
            }
        }
    }

    public let manifest: Manifest
    private let blob: UncachedFile
    private let entries: [String: Entry]

    /// The names whose digest this instance has already checked.
    ///
    /// A reference box rather than a `var`, because `WeightSource` conformance is non-mutating and
    /// an expert tensor is fetched again and again: re-hashing a five-hundred-megabyte payload on
    /// every read of it would be its own disaster, and a struct cannot hold that memo itself.
    private final class ReadState {
        var names: Set<String> = []
        /// Payload bytes this instance has read. Exposed so a test can assert that reading one
        /// expert of a stacked tensor does not cost the whole stack, which is the difference
        /// between streaming and not streaming.
        var bytesRead = 0
        /// Of those bytes, how many were read to verify a digest rather than to be used. Verification
        /// reads were **not counted at all** until this existed, which is how a "measured" 3.06 GB
        /// understated a forward's traffic by about half.
        var bytesVerified = 0
        /// Where the time went. `mix.read` was 65% of a real forward and the disk is measured at
        /// ~1 GB/s, so the caller needs to know whether those seconds are the device or the
        /// unpacking — a distinction arithmetic cannot settle.
        var readSeconds = 0.0
        var digestSeconds = 0.0
        var unpackSeconds = 0.0
        /// Whole-tensor payload bytes (`DC-106`): the dense backbone's share, and the figure a second
        /// forward must drive to zero. Row-range reads are deliberately not here.
        var payloadBytesRead = 0
        /// How many times each tensor's payload was asked for, whether or not the disk was touched.
        /// `DC-106`'s measurement found a run reading more than one forward's worth of payload per step,
        /// which can only mean a tensor was asked for twice; this is what names it.
        var payloadRequests: [String: Int] = [:]
        /// The cache's budget, read once from the environment so a measurement is a series of processes.
        /// `SHARD_SLAB_CACHE_MB`, the packed slab cache's budget (`DC-120`). **Zero by default**, like every other
        /// cache in this engine: a measurement decides, and `D89`/`D98`/`D105` all ended with a default of zero
        /// after the measurement disagreed with the reasoning.
        static let slabCacheBudgetValue = InstallFile.slabCacheBudget(
            environment: ProcessInfo.processInfo.environment
        )

        static let payloadCacheBudgetValue = InstallFile.payloadCacheBudget(
            environment: ProcessInfo.processInfo.environment
        )

        /// Counting is guarded because `DC-118` fans expert misses across threads: two `+=` on the same
        /// counter lose one of them, and a lost read under-reports the very cost that change exists to
        /// reduce.
        ///
        /// **The reads are deliberately unlocked**, and that is a statement about *when* they happen rather
        /// than a hope: `sourceTiming` and `payloadCacheMetrics` are asked between forwards, after every
        /// dispatched read has returned, so there is no concurrent writer at that moment. A lock on the read
        /// side would be paid for on every measured forward and buy nothing.
        private let lock = NSLock()

        func addRead(seconds: Double, bytes: Int) {
            lock.lock(); readSeconds += seconds; bytesRead += bytes; lock.unlock()
        }

        func addVerified(bytes: Int) {
            lock.lock(); bytesVerified += bytes; lock.unlock()
        }

        func addDigest(seconds: Double) {
            lock.lock(); digestSeconds += seconds; lock.unlock()
        }

        func addUnpack(seconds: Double) {
            lock.lock(); unpackSeconds += seconds; lock.unlock()
        }

        /// Two counters, two meanings, and they must not be conflated: a **request** is counted on the way
        /// *past* the cache, so a hit is still visible, and **bytes** are counted only when the disk was
        /// touched (`DC-106`). One method for both recorded a request named "" for every read, which is how
        /// this surfaced — `InstallCacheTests` compares per-name counts across two forwards.
        func notePayloadRequest(_ name: String) {
            lock.lock(); payloadRequests[name, default: 0] += 1; lock.unlock()
        }

        func addPayloadBytes(_ count: Int) {
            lock.lock(); payloadBytesRead += count; lock.unlock()
        }
    }

    /// The whole-tensor payload cache (`DC-106`). A **class**, because `InstallFile` is a struct and a
    /// cache is state that outlives one call — the same reason `ReadState` is one. Least-recently-used,
    /// bounded in bytes, and holding the payload exactly as stored.
    /// The **packed** slab cache (`DC-120`): expert row ranges, held as the bytes they were read as.
    ///
    /// `PayloadCache` below holds *whole tensors* and cannot help here — a stacked expert tensor is gigabytes, and
    /// the unit of reuse is one expert's row range, not the tensor. This holds those ranges keyed by name and
    /// range **as stored**: four bits per weight rather than the thirty-two the decoder produces, so a byte budget
    /// buys four times as many slabs as a cache of `[Float]` would.
    ///
    /// A hit skips the three `pread`s altogether, which is the point: `D105` measured the device as **saturated**
    /// by `D101`'s fan-out, so a read cannot be hidden, only avoided.
    ///
    /// Locked, because the reader is called from the expert fan-out (`D101`) and the head's blocks (`D102`).
    final class SlabCache: @unchecked Sendable {
        private let lock = NSLock()
        private let budget: Int
        private var payloads: [String: Data] = [:]
        private var order: [String] = []
        private var bytes = 0
        private var hits = 0
        private var misses = 0
        private var wiredBytes = 0

        /// Whether the bank is **wired** (`D122`), on by default; `SHARD_SLAB_WIRED=0` turns it off so the two
        /// are compared on one binary.
        ///
        /// This is the one ingredient of the reference's residency recipe this engine had never tried. Its
        /// reader `mlock`s every slot, and the reason is visible in this engine's own measurements: a bigger
        /// cache kept regressing phases that never touch it (`D115`, `D118`, `D119`), which is what a *wired*
        /// versus *reclaimable* difference looks like from the outside. Anonymous pages that the kernel may
        /// compress or swap are not a cache, they are a suggestion — and under pressure the kernel takes them
        /// back exactly when the cache was supposed to be earning its keep.
        static var wired: Bool {
            ProcessInfo.processInfo.environment["SHARD_SLAB_WIRED"] != "0"
        }

        init(budget: Int) { self.budget = budget }

        /// What this cache may hold, so a caller can decide whether a preload has anywhere to put its bytes.
        var budgetBytes: Int { budget }

        func value(for key: String) -> Data? {
            guard budget > 0 else { return nil }
            lock.lock(); defer { lock.unlock() }
            guard let payload = payloads[key] else { misses += 1; return nil }
            hits += 1
            order.removeAll { $0 == key }
            order.append(key)
            return payload
        }

        func store(_ payload: Data, named key: String) {
            guard budget > 0, payload.count <= budget else { return }
            lock.lock(); defer { lock.unlock() }
            if payloads[key] == nil { bytes += payload.count }
            payloads[key] = payload
            order.removeAll { $0 == key }
            order.append(key)
            if Self.wired, let base = payload.withUnsafeBytes({ $0.baseAddress }),
                mlock(base, payload.count) == 0
            {
                wiredBytes += payload.count
            }
            while bytes > budget, let victim = order.first {
                order.removeFirst()
                guard let evicted = payloads.removeValue(forKey: victim) else { continue }
                bytes -= evicted.count
                // Unwire **before** releasing the storage, and only what was actually wired: an `mlock` that
                // failed (the wired limit is finite) must not be answered with a `munlock`, which would take
                // the count negative and silently shrink the limit for everything after it.
                if Self.wired, wiredBytes >= evicted.count,
                    let base = evicted.withUnsafeBytes({ $0.baseAddress })
                {
                    munlock(base, evicted.count)
                    wiredBytes -= evicted.count
                }
            }
        }

        var metrics: PayloadCacheMetrics {
            lock.lock(); defer { lock.unlock() }
            return PayloadCacheMetrics(slabHits: hits, slabMisses: misses, slabBytesHeld: bytes)
        }
    }

    final class PayloadCache {
        let budget: Int
        private var payloads: [String: Data] = [:]
        private var order: [String] = []
        private(set) var bytes = 0
        private(set) var hits = 0

        init(budget: Int) { self.budget = budget }

        var names: Set<String> { Set(payloads.keys) }

        func value(for name: String) -> Data? {
            guard budget > 0, let cached = payloads[name] else { return nil }
            hits += 1
            if let position = order.firstIndex(of: name) {
                order.remove(at: position)
                order.append(name)
            }
            return cached
        }

        func store(_ data: Data, named name: String) {
            guard budget > 0, data.count <= budget else { return }
            bytes += data.count - (payloads[name]?.count ?? 0)
            if payloads[name] == nil { order.append(name) }
            payloads[name] = data
            while bytes > budget, let oldest = order.first {
                order.removeFirst()
                if let dropped = payloads.removeValue(forKey: oldest) { bytes -= dropped.count }
            }
        }
    }

    private let state = ReadState()
    private let payloadCache = PayloadCache(budget: ReadState.payloadCacheBudgetValue)
    private let slabCache = SlabCache(budget: ReadState.slabCacheBudgetValue)

    /// Whether to verify each slab's digest **on every read**. Off by default, and that default is a
    /// measurement: on the real 35 B model the verification cost **21.04 s of a 39.8 s forward**, 82%
    /// of the expert fetch, because it re-hashes every expert payload the model reads. Integrity is
    /// established out of band — the manifest carries a digest per tensor and per slab and
    /// `tools/quantize.py verify` checks the whole install — so the hot path is the wrong place for
    /// it. `SHARD_VERIFY_SLABS` is not a switch; a caller asks explicitly, which keeps the capability
    /// tested instead of deleting it.
    public let verifySlabs: Bool

    /// Whether a tensor's whole payload is hashed **the first time it is read**. Off by default, for
    /// the same reason as `verifySlabs` and with a measurement behind it: the embedding is 1.02 GB and
    /// reading five of its rows was verifying all of it, which is 1.64 s in a 19.8 s forward, and the
    /// dense tensors together are ~3 GB of first-touch verification. Integrity is established out of
    /// band — `install.json` carries a digest per tensor, and `tools/quantize.py verify` checks the
    /// whole install uncached at a 34 MB peak. `verifyOnFirstUse: true` or `verify: true` restores it,
    /// and both paths are tested.
    public let verifyOnFirstUse: Bool

    /// Payload bytes read through this instance.
    public var bytesRead: Int { state.bytesRead }

    /// The whole-tensor payload counters — the instrument `DC-106` is accepted against.
    public var payloadCacheMetrics: PayloadCacheMetrics {
        PayloadCacheMetrics(
            bytesRead: state.payloadBytesRead, hits: payloadCache.hits, bytesHeld: payloadCache.bytes,
            slabHits: slabCache.metrics.slabHits, slabMisses: slabCache.metrics.slabMisses,
            slabBytesHeld: slabCache.metrics.slabBytesHeld
        )
    }

    /// The names the cache is holding, so a test can assert that a stacked expert bank never lands in it.
    var payloadCacheNames: Set<String> { payloadCache.names }

    /// `WeightSource`: the same counter, so a caller can report measured traffic rather than an
    /// estimate derived from element counts.
    public var bytesReadFromSource: Int { bytesRead }

    /// `WeightSource`: what the packed slab cache is allowed to hold (`D109`), so a caller can tell whether
    /// a stored-form preload has anywhere to put its bytes.
    public var packedCacheBudget: Int { slabCache.budgetBytes }

    /// `WeightSource`: what the reading, verifying and unpacking cost, in seconds.
    public var sourceTiming: SourceTiming {
        var timing = SourceTiming()
        timing.counted = true
        timing.bytes = state.bytesRead
        timing.verifiedBytes = state.bytesVerified
        timing.readSeconds = state.readSeconds
        timing.digestSeconds = state.digestSeconds
        timing.unpackSeconds = state.unpackSeconds
        return timing
    }

    /// Open an install.
    ///
    /// `verify` is **false** by default, and that is a deliberate change from the first version,
    /// which hashed the entire payload on open. That made opening a 20 GB install a 20 GB read:
    /// slow everywhere, and on the 8 GB development node the read filled the page cache, memory
    /// pressure grew swap, and free disk fell from 17 GB to 2.96 GB in half a minute. Integrity
    /// is not weakened by moving it — each payload is checked the first time it is read, so a
    /// tampered tensor still cannot produce plausible numbers (I6) — and `verifyAll()` restores
    /// the eager check for a gate that wants to make it explicit.
    public init(
        url: URL, verify: Bool = false, verifySlabs: Bool = false, verifyOnFirstUse: Bool = false
    ) throws {
        self.verifySlabs = verifySlabs
        self.verifyOnFirstUse = verifyOnFirstUse
        let manifestData = try Data(contentsOf: url.appendingPathComponent("install.json"))
        let manifest = try JSONDecoder().decode(Manifest.self, from: manifestData)
        self.manifest = manifest
        // Read through `UncachedFile`, not `mmap`: the payload is larger than the machine, and
        // pages cached from it evict everything useful and turn a read into swap pressure.
        //
        // **`SHARD_INSTALL_CACHED=1` asks the kernel to keep the pages anyway** (`D111`), and the two cases are
        // not the same access pattern. The hazard `UncachedFile` records is a **sequential scan of the whole
        // file** — a 20 GB verification — whose pages are never revisited and which evicts everything useful on
        // the way past. A decode step is the opposite: a **582 MB working set of expert slabs that repeats every
        // token**, which is what a buffer cache is for, and whose pages are clean and evictable rather than
        // anonymous. The switch exists so the two can be measured against each other on one binary.
        self.blob = try UncachedFile(
            url: url.appendingPathComponent("data.bin"), uncached: !Self.readThroughCacheEnabled
        )
        var entries: [String: Entry] = [:]
        for entry in manifest.tensors { entries[entry.name] = entry }
        self.entries = entries
        if verify { try verifyAll() }
    }

    /// Check every payload digest, in bounded windows. For a gate rather than for normal use.
    public func verifyAll() throws {
        for entry in manifest.tensors where !(try digestMatches(entry)) {
            throw Error.digestMismatch(entry.name)
        }
    }

    /// Read an entry, hash it, and count both — then hand the buffer back so a caller that needs the
    /// payload does not read it a second time.
    ///
    /// Reading it twice was the old behaviour: `digestMatches` read the whole entry to hash it and
    /// `payload` read it again to return it, which doubled the dense traffic of every forward. The
    /// verification read is also **counted** now; it was invisible to `bytesRead`, so the metric
    /// reported the use-traffic and called it the total.
    @discardableResult
    private func verifyRead(_ entry: Entry) throws -> Data {
        let readStarted = DispatchTime.now().uptimeNanoseconds
        let payload = try blob.readData(offset: entry.offset, byteCount: entry.nbytes)
        state.addRead(
            seconds: Double(DispatchTime.now().uptimeNanoseconds &- readStarted) / 1e9, bytes: payload.count
        )
        state.addVerified(bytes: payload.count)
        let digestStarted = DispatchTime.now().uptimeNanoseconds
        let digest = SHA256.hash(data: payload).map { String(format: "%02x", $0) }.joined()
        state.addDigest(seconds: Double(DispatchTime.now().uptimeNanoseconds &- digestStarted) / 1e9)
        guard digest == entry.sha256 else { throw Error.digestMismatch(entry.name) }
        state.names.insert(entry.name)
        return payload
    }

    private func digestMatches(_ entry: Entry) throws -> Bool {
        if state.names.contains(entry.name) { return true }
        guard entry.offset + entry.nbytes <= blob.byteCount else { return false }
        _ = try verifyRead(entry)
        return true
    }

    /// The entry's payload exactly as stored, for a caller that wants to decode it itself — the
    /// Metal kernels, or a test comparing two decoders on the same bytes. Checked like any other
    /// read, and counted.
    func rawPayload(named name: String) throws -> Data {
        try payload(try entry(name))
    }

    public func entry(_ name: String) throws -> Entry {
        guard let entry = entries[name] else { throw Error.unknownTensor(name) }
        return entry
    }

    /// A bounded read that is counted, so the cost of a row range is observable rather than
    /// asserted.
    private func readCounted(offset: Int, byteCount: Int) throws -> Data {
        let started = DispatchTime.now().uptimeNanoseconds
        let data = try blob.readData(offset: offset, byteCount: byteCount)
        state.addRead(
            seconds: Double(DispatchTime.now().uptimeNanoseconds &- started) / 1e9, bytes: data.count
        )
        return data
    }

    /// The payload cache's budget, from `SHARD_DENSE_CACHE_MB` (0 disables it, for an A/B measurement).
    ///
    /// The dense backbone is read and dequantised on **every forward**: the forward walks every layer for
    /// every token, so the same ~20 MB per layer of norms, attention, router and shared-expert weights is
    /// fetched again per token. Measured on the real install, the whole-tensor payload is **0.796 GB per
    /// forward** (2.83 GB of non-expert tensors, less the 1.02 GB embedding and 1.02 GB head, which are
    /// read a row at a time) — small enough to hold, and the default budget covers it.
    ///
    /// It caches the **packed** bytes rather than the dequantised fp32, and that is the whole point: 0.74 GB
    /// of that 0.796 is int4, which is eight times larger decoded, so caching the decoded form would need
    /// ~6 GB against this node's ~4.5 GB budget — the `DC-091` mistake a second time. Packed bytes cost what
    /// the disk costs, and the dequantisation is arithmetic the CPU was doing anyway.
    ///
    /// The stacked expert banks never reach here: they are read a row range at a time, so 18 GB of experts
    /// cannot fill this. `InstallCacheTests` asserts that rather than trusting it.
    /// The slab cache's budget in bytes, from `SHARD_SLAB_CACHE_MB` when it is sane.
    ///
    /// Refused rather than clamped, for the reason `bankBudgetBytes` gives: a budget nobody could hold is a node
    /// that swaps, and a silent clamp hides the typo that caused it.
    static func slabCacheBudget(environment: [String: String]) -> Int {
        // **128 MiB by default.** Two measurements put it there, and the second changed the answer. `D110`
        // swept 256/512/768/1024 with the device still bypassed and found 256 best (1.465, 1.467, 1.483,
        // 1.667 s), because the resident bytes cost more in memory pressure than the reads they save — that is
        // `D106`'s verdict on a cache this node cannot afford, and it is why the default is the smallest size
        // that pays rather than the largest that fits. `D112` turned the kernel's buffer cache on (`D111`),
        // which caches the same slabs in *clean, evictable* pages instead of anonymous ones, and the optimum
        // moved down: three alternated pairs put **128 at 0.930 s against 256 at 0.957**, and a finer sweep
        // found 64/96/128/192 at 0.951, 0.943, 0.944, 0.953 — flat between 96 and 128. **Zero is worse than
        // any of them** (1.595 s) for a reason worth keeping: with no cache at all `preloadPacked` declines,
        // so the fan-out disappears and the loop reads every slab itself.
        // **The unit is bytes**, and the first version of this line returned the bare literal `256` — a
        // 256-**byte** budget, which refused every 1.2 MB slab. The default then looked like a cache that
        // never held anything while the env-var path, which multiplies, worked: `slab_cache_bytes_held` was 0
        // and the step was 2.52 s instead of 1.47. It is written as the multiplication for that reason.
        //
        // **Absent and invalid are different answers.** An unset knob takes the measured default; a knob that
        // is set to something unparseable or out of range is **refused** as zero, because a silent fallback
        // hides the typo that caused it — the same rule the bank's budget and every other budget here follow.
        guard let raw = environment["SHARD_SLAB_CACHE_MB"] else { return 128 * 1_048_576 }
        guard let megabytes = Int(raw), megabytes >= 0, megabytes <= 1 << 20 else { return 0 }
        return megabytes * 1_048_576
    }

    static func payloadCacheBudget(environment: [String: String]) -> Int {
        // **2048 MiB, not 1024** (`D109`). The budget has to hold the layer payload (~995 MiB) *and* the LM
        // head (~970 MiB) at once, because both are re-read every token and an LRU that holds one evicts the
        // other, so the two trade device reads forever. At 1024 MiB the head could not be held at all, which
        // is why it was read off the device on every step; `storedRows` is what puts it through this cache.
        // `SHARD_DENSE_CACHE_MB=1024` restores the old ceiling for an A/B.
        let defaultMegabytes = 2048
        guard let raw = environment["SHARD_DENSE_CACHE_MB"], let megabytes = Int(raw) else {
            return defaultMegabytes * 1_048_576
        }
        guard megabytes >= 0, megabytes <= 1 << 20 else { return defaultMegabytes * 1_048_576 }
        return megabytes * 1_048_576
    }

    /// Whole-tensor requests per tensor, most-requested first. A tensor here with a count above one was
    /// asked for more than once in this source's life, which within one forward is a repeat read.
    public var payloadRequestCounts: [(name: String, count: Int)] {
        state.payloadRequests.sorted { left, right in
            left.value == right.value ? left.key < right.key : left.value > right.value
        }.map { (name: $0.key, count: $0.value) }
    }

    private func payload(_ entry: Entry) throws -> Data {
        state.notePayloadRequest(entry.name)
        if let cached = payloadCache.value(for: entry.name) { return cached }
        let data: Data
        if verifyOnFirstUse {
            data = try verifyRead(entry)
        } else {
            data = try readCounted(offset: entry.offset, byteCount: entry.nbytes)
        }
        // Counted on the way *past* the cache, so the metric is disk traffic and a hit is invisible —
        // which is exactly what a second forward has to show.
        state.addPayloadBytes(data.count)
        payloadCache.store(data, named: entry.name)
        return data
    }

    public func tensor(named name: String) throws -> [Float] {
        let entry = try entry(name)
        let data = try payload(entry)
        guard entry.dtype == "int4" else {
            return try Self.decodeRaw(data, dtype: entry.dtype, elementCount: entry.shape.reduce(1, *))
        }
        // **The CPU decoder, measured rather than assumed** (`D109`). This path was pointed at
        // `MetalUnpack` on the theory that the device must win, and it lost: `load` went 353 → 534 ms/step,
        // because a dense projection is a *few large* tensors and the device path copies the payload into
        // buffers and the fp32 result back out for each one, where `dequantizeInt4` is already threaded
        // across the cores (`D94`) on data that is in cache. The GPU unpack stays where `D59` measured it
        // winning: the many small expert slabs in `rows`.
        return try Self.dequantizeInt4(data, entry: entry)
    }

    public func rows(named name: String, range: Range<Int>) throws -> [Float] {
        let entry = try entry(name)
        guard entry.shape.count >= 2 else { return try tensor(named: name) }
        // One row is the **leading axis**, whatever the rank: a token of the embedding, or one
        // expert of a stacked expert tensor. The row's width is the product of the rest.
        let width = entry.shape.dropFirst().reduce(1, *)
        let rowCount = entry.shape[0]
        guard range.lowerBound >= 0, range.upperBound <= rowCount else {
            throw Error.badHeader("row range \(range) is outside '\(name)'")
        }
        if entry.dtype != "int4" {
            // Sliced from the stored bytes: one row of the embedding should not cost a
            // gigabyte of decoding.
            if verifyOnFirstUse {
                guard try digestMatches(entry) else { throw Error.digestMismatch(entry.name) }
            }
            let stride = Self.elementSize(entry.dtype) * width
            let start = entry.offset + range.lowerBound * stride
            let slice = try blob.readData(offset: start, byteCount: range.count * stride)
            return try Self.decodeRaw(slice, dtype: entry.dtype, elementCount: range.count * width)
        }
        // Packed codes cannot be sliced as bytes without re-deriving the group layout, and the
        // layout is section-major, so the rows come out of **three** ranges rather than out of a
        // whole-tensor decode. That distinction is the difference between streaming and not: the
        // real model's expert tensor is `[256, 1024, 2048]`, so decoding it to return one expert
        // is 537 M parameters and two gigabytes of `Float` on a node with four and a half.
        //
        // Since `D108` the packed payload is also a **product**: `packedRows` hands the same bytes over
        // undecoded so a device kernel can consume them, and this path decodes them for the CPU. One read,
        // two consumers, and the section layout is computed in one place.
        let packed = try int4RowsPayload(named: name, entry: entry, range: range)
        let unpackStarted = DispatchTime.now().uptimeNanoseconds
        // The decoder is *chosen*, not replaced. The GPU path has existed since `D10` and has been called
        // by nothing but its own tests, because it used to disagree with the scalar one on a row whose
        // final group is partly filled (`DC-087`). That reproducer passes on the widened grid now, so this
        // is the last step `DC-033` was waiting for -- and it stays behind a flag because every recorded
        // digest was produced by the scalar path, so a changed default would move all of them at once.
        // `SHARD_GPU_UNPACK=1` selects it; the real-model trace digest is what verifies it.
        let decoded: [Float]
        if Self.gpuUnpackEnabled, MetalUnpack.isAvailable {
            decoded = try MetalUnpack.unpack(
                payload: packed.payload, entry: entry, rowCount: packed.payloadRows
            )
        } else {
            decoded = try Self.dequantizeInt4(packed.payload, entry: entry, rowCount: packed.payloadRows)
        }
        state.addUnpack(seconds: Double(DispatchTime.now().uptimeNanoseconds &- unpackStarted) / 1e9)
        return decoded
    }

    /// The **stored** payload of a leading-axis row range, with the packed slab cache consulted first.
    ///
    /// This is the read half of `rows`' int4 branch, split out so that `packedRows` can offer the same
    /// bytes to a device kernel without decoding them (`D108`). It returns the three sections
    /// concatenated — every row's codes, then every row's scales, then every row's zeros — which is the
    /// layout both decoders take and the layout `InstallFile.int4Layout` describes.
    func int4RowsPayload(
        named name: String, entry: Entry, range: Range<Int>
    ) throws -> (payload: Data, payloadRows: Int) {
        let totalRows = entry.shape.dropLast().reduce(1, *)
        guard entry.padded_columns % entry.group == 0, entry.padded_columns % 2 == 0 else {
            throw Error.badHeader("\(name): group \(entry.group) does not divide \(entry.padded_columns)")
        }
        // The range is in **leading-axis entries** — one expert of a stack, not one payload row —
        // so it has to be translated, and getting that wrong returns the first expert's first
        // *row* while still looking like a tensor. One entry spans `inner` payload rows.
        let inner = entry.shape.dropFirst().dropLast().reduce(1, *)
        let payloadRows = range.count * inner
        let first = range.lowerBound * inner
        let groupsPerRow = entry.padded_columns / entry.group
        let codesPerRow = entry.padded_columns / 2
        let codesBytes = totalRows * codesPerRow
        let scalesBytes = totalRows * groupsPerRow * 4
        // `DC-120`: a packed row range that has been read before is served from memory, and the three `pread`s
        // below are skipped entirely. The key is the tensor and the range, because that is what the expert provider
        // asks for and what the router repeats.
        let slabKey = "\(name)#\(range.lowerBound)-\(range.upperBound)"
        if let cached = slabCache.value(for: slabKey) { return (cached, payloadRows) }
        let codes = try readCounted(
            offset: entry.offset + first * codesPerRow, byteCount: payloadRows * codesPerRow
        )
        let scales = try readCounted(
            offset: entry.offset + codesBytes + first * groupsPerRow * 4, byteCount: payloadRows * groupsPerRow * 4
        )
        let zeros = try readCounted(
            offset: entry.offset + codesBytes + scalesBytes + first * groupsPerRow, byteCount: payloadRows * groupsPerRow
        )
        // Verify what was read rather than the whole tensor: these bytes are already in hand, where
        // the entry digest reads 537 MB of expert stack to hand back one expert.
        //
        // Three things had to be right, and each was wrong in turn:
        //
        // 1. The slab index **is** the leading-axis index, so a request for one expert is one slab —
        //    not `range.count` payload rows. Guarding on `range.count % inner == 0` is `1 % 32` for a
        //    stacked tensor, which skipped the branch and left every read paying the entry digest.
        // 2. This function had **two** whole-entry checks on the path — one before the reads and one
        //    after — so removing either alone left the other rejecting any tamper in the entry.
        // 3. The slice offsets are **local to the buffers just read** (`index * inner`), while the
        //    digest to compare against is indexed **globally** (`range.lowerBound + index`). Using
        //    the global index for both sliced past the end of a one-expert buffer and trapped.
        // Off by default: see `verifySlabs`. The fallback matters as much as the loop — with
        // verification off, falling through to `digestMatches` would read and hash the **whole**
        // entry to answer a question about one expert, which is the `DC-088` bug in reverse.
        //
        // The clock starts *inside* the branch: started outside it, the phase booked the cost of the
        // branch test itself (4.2e-08 s) as verification, and a test that pins "off costs nothing"
        // caught it. A phase has to mean what it says.
        if verifySlabs {
            let digestStarted = DispatchTime.now().uptimeNanoseconds
            if let slabs = entry.slab_sha256, inner > 0, range.lowerBound + range.count <= slabs.count {
                for index in 0..<range.count {
                    let low = index * inner, high = low + inner
                    var hasher = SHA256()
                    hasher.update(data: codes[(low * codesPerRow)..<(high * codesPerRow)])
                    hasher.update(data: scales[(low * groupsPerRow * 4)..<(high * groupsPerRow * 4)])
                    hasher.update(data: zeros[(low * groupsPerRow)..<(high * groupsPerRow)])
                    let digest = hasher.finalize().map { String(format: "%02x", $0) }.joined()
                    guard digest == slabs[range.lowerBound + index] else { throw Error.digestMismatch(entry.name) }
                }
            } else {
                guard try digestMatches(entry) else { throw Error.digestMismatch(entry.name) }
            }
            state.addDigest(seconds: Double(DispatchTime.now().uptimeNanoseconds &- digestStarted) / 1e9)
        }
        // The concatenation is inside the unpack timing on purpose: `codes + scales + zeros` copies
        // the payload before the decoder sees it, and a copy of the payload is part of what the
        // caller experiences as "fetching an expert". A device consumer pays the same copy, and it is
        // still the unpack phase rather than the read.
        let unpackStarted = DispatchTime.now().uptimeNanoseconds
        let payload = codes + scales + zeros
        state.addUnpack(seconds: Double(DispatchTime.now().uptimeNanoseconds &- unpackStarted) / 1e9)
        slabCache.store(payload, named: slabKey)
        return (payload, payloadRows)
    }

    /// `WeightSource.packedRows`: the bytes `rows` would decode, handed over undecoded (`D108`).
    ///
    /// The counters are the read path's own — `readCounted` and the slab cache — so a device consumer's
    /// traffic is measured exactly as a CPU one's is, and `SHARD_SLAB_CACHE_MB` is the knob that decides
    /// how much of it is memory. A non-int4 tensor, or one without a leading axis, has no packed form to
    /// offer and answers `nil`, which is what sends the caller back to `rows`.
    public func packedRows(named name: String, range: Range<Int>) throws -> PackedInt4Rows? {
        let entry = try entry(name)
        guard entry.dtype == "int4", entry.shape.count >= 2 else { return nil }
        guard range.lowerBound >= 0, range.upperBound <= entry.shape[0] else {
            throw Error.badHeader("row range \(range) is outside '\(name)'")
        }
        let packed = try int4RowsPayload(named: name, entry: entry, range: range)
        return PackedInt4Rows(payload: packed.payload, entry: entry, payloadRows: packed.payloadRows)
    }

    /// `WeightSource.storedRows`: the stored bf16/fp32 bytes of a row range, through the payload cache.
    ///
    /// The whole tensor goes through `payload`, which is the difference that matters for the head: `rows`
    /// reads the slice straight from the blob with `F_NOCACHE` and **never caches it**, so `[248320, 2048]`
    /// of bf16 — 1.017 GB — came off the device on every token (`D109`). Here the first request reads and
    /// caches it and every later one is a slice of memory. `SHARD_DENSE_CACHE_MB` has to be large enough to
    /// hold it beside the layer payload, or the LRU will trade the two.
    ///
    /// A row range of a non-int4 tensor is contiguous — the leading axis is the outermost — so the slice is
    /// exact and free. int4 is `packedRows`' business: its sections are not contiguous, so there is no
    /// byte range that means one row.
    /// `WeightSource.packedTensor`: the whole tensor, which for a dense projection is the only range there is.
    ///
    /// **The cached whole-tensor payload, not a row-range read** (`D111`). A dense int4 tensor's three sections
    /// are contiguous in the file in exactly the order the layout wants, and `payload(entry)` is where the
    /// install already keeps them; routing this through `int4RowsPayload` sent **7.5 GB a step** back to the
    /// device for bytes that were resident — a re-read disguised as a decode-avoidance. The cache is what makes
    /// this an improvement rather than a trade.
    public func packedTensor(named name: String) throws -> PackedInt4Rows? {
        let entry = try entry(name)
        guard entry.dtype == "int4", entry.shape.count >= 2 else { return nil }
        let data = try payload(entry)
        // **The key is content-addressed, not just the name** (`D116`). A name is unique within one install
        // and says nothing across two: mapping `linear.in_qkv` from one install and then reading it for
        // another would answer with the first one's weights, which is the `D115` bug — a key coarser than the
        // thing it caches — one level further out. The manifest already carries the payload's digest, so the
        // key says exactly which bytes are mapped.
        return PackedInt4Rows(
            payload: data, entry: entry, payloadRows: entry.shape.dropLast().reduce(1, *),
            key: "\(name)#\(entry.sha256.prefix(16))"
        )
    }

    public func storedRows(named name: String, range: Range<Int>) throws -> StoredRows? {
        let entry = try entry(name)
        guard entry.dtype != "int4", entry.shape.count >= 2 else { return nil }
        let width = entry.shape.dropFirst().reduce(1, *)
        guard range.lowerBound >= 0, range.upperBound <= entry.shape[0] else {
            throw Error.badHeader("row range \(range) is outside '\(name)'")
        }
        let data = try payload(entry)
        let stride = Self.elementSize(entry.dtype) * width
        let start = range.lowerBound * stride
        let end = start + range.count * stride
        guard end <= data.count else {
            throw Error.badHeader("\(name): rows \(range) need \(end) bytes, the payload has \(data.count)")
        }
        return StoredRows(data: data[start..<end], dtype: entry.dtype, rowCount: range.count, width: width)
    }

    static func elementSize(_ dtype: String) -> Int {
        switch dtype {
        case "bf16", "fp16": return 2
        case "fp32": return 4
        default: return 1
        }
    }

    /// Decode `count` elements with `transform`, across threads when the work warrants it (`D99`).
    ///
    /// This is a **map**: every element is a pure function of its own bytes, so splitting it into disjoint
    /// ranges cannot change a value — which is why this is the cheapest parallelism in the engine and why
    /// `decodeRaw`'s callers (the LM head's bf16 weights, above all) were paying a single core for a
    /// 508-million-element conversion on every token.
    private static func mapElements(
        count: Int, into values: inout [Float], _ transform: @escaping @Sendable (Int) -> Float
    ) {
        guard DecodeThreads.wantsParallelism(work: count) else {
            for index in 0..<count { values[index] = transform(index) }
            return
        }
        let stride = max(1, (count + DecodeThreads.count - 1) / DecodeThreads.count)
        let blocks = (count + stride - 1) / stride
        // `body` needs no `nonisolated(unsafe)`: it is `@Sendable` already, and the only unsafe part is the
        // destination pointer below, whose ranges are disjoint by construction.
        let body = transform
        values.withUnsafeMutableBufferPointer { out in
            nonisolated(unsafe) let destination = out.baseAddress!
            DispatchQueue.concurrentPerform(iterations: blocks) { block in
                let first = block * stride
                let last = min(first + stride, count)
                for index in first..<last { destination[index] = body(index) }
            }
        }
    }

    static func decodeRaw(_ data: Data, dtype: String, elementCount: Int) throws -> [Float] {
        var values = [Float](repeating: 0, count: elementCount)
        try data.withUnsafeBytes { raw in
            nonisolated(unsafe) let base = raw.baseAddress!
            switch dtype {
            case "bf16":
                Self.mapElements(count: elementCount, into: &values) { index in
                    let word = base.loadUnaligned(fromByteOffset: index * 2, as: UInt16.self)
                    return Float(bitPattern: UInt32(UInt16(littleEndian: word)) << 16)
                }
            case "fp16":
                Self.mapElements(count: elementCount, into: &values) { index in
                    let word = base.loadUnaligned(fromByteOffset: index * 2, as: UInt16.self)
                    return Float(Float16(bitPattern: UInt16(littleEndian: word)))
                }
            case "fp32":
                Self.mapElements(count: elementCount, into: &values) { index in
                    let word = base.loadUnaligned(fromByteOffset: index * 4, as: UInt32.self)
                    return Float(bitPattern: UInt32(littleEndian: word))
                }
            default:
                throw Error.badHeader("unknown stored dtype '\(dtype)'")
            }
        }
        return values
    }

    /// The packed format, decoded exactly as `tools/quantize.py` encodes it.
    ///
    /// Two codes per byte, **low nibble first**; codes are signed four-bit values, so `0b1000`
    /// is -8 and not 8 — reading it as 8 shifts a whole group by sixteen steps and still
    /// produces plausible weights. Each group of `group` codes shares one fp32 scale and one
    /// int4 zero point, and the reconstruction is `(code - zero) * scale`.
    /// Dequantize `rowCount` payload rows — the whole tensor, or a range of it.
    ///
    /// The layout is **section-major**: every row's codes, then every row's scales, then every
    /// row's zeros. That is what makes a row range decodable without the rest of the tensor, since
    /// each section is row-contiguous; it is also the thing to get wrong, because the wrong row
    /// count still produces plausible weights.
    /// The three sections of an int4 payload, in bytes, for a given number of rows.
    struct Int4Layout {
        let rows: Int
        let columns: Int
        let padded: Int
        let group: Int
        let groups: Int
        let codeBytes: Int
        let scaleBytes: Int
    }

    static func int4Layout(entry: Entry, rowCount: Int?, payloadBytes: Int) throws -> Int4Layout {
        // A stacked expert tensor is `[experts, rows, columns]` and its payload was quantized
        // with the leading axis flattened, so the codes describe `experts x rows` rows.
        let rows = rowCount ?? entry.shape.dropLast().reduce(1, *)
        let columns = entry.shape.last ?? 0
        let padded = entry.padded_columns
        let group = entry.group
        // Each condition says which one failed: the first version reported "group does not divide
        // padded" for a failure of the even-width rule, which sent a reader looking at the wrong
        // number.
        guard group > 0 else {
            throw Error.badHeader("\(entry.name): group is \(group), which cannot divide anything")
        }
        guard padded % group == 0 else {
            throw Error.badHeader("\(entry.name): group \(group) does not divide padded width \(padded)")
        }
        guard padded % 2 == 0 else {
            // Two codes share a byte, so an odd padded width has no defined packing.
            throw Error.badHeader("\(entry.name): padded width \(padded) is odd, and two codes share a byte")
        }
        let groups = padded / group
        let codeBytes = rows * padded / 2
        let scaleBytes = rows * groups * 4
        guard payloadBytes == codeBytes + scaleBytes + rows * groups else {
            throw Error.badHeader("\(entry.name): payload is \(payloadBytes) bytes, the layout needs \(codeBytes + scaleBytes + rows * groups)")
        }
        return Int4Layout(
            rows: rows, columns: columns, padded: padded, group: group, groups: groups,
            codeBytes: codeBytes, scaleBytes: scaleBytes
        )
    }

    /// Unpack four-bit codes into `Float`.
    ///
    /// The fast path, and the reason it is free: the **only** floating-point operation here is one
    /// multiply, `Float(code - zero) * scale`, with no summation anywhere. There is no
    /// accumulation order to preserve, so a vector formulation is bit-identical by construction
    /// rather than by measurement — `Float(code - zero)` is exact because the integers are tiny,
    /// and `SIMD4<Float> * scalar` rounds once per lane exactly as the scalar multiply does.
    ///
    /// The win comes from the other direction: the scale and the zero point are per *group* (sixty
    /// four values), and the scalar loop reloaded both for every element.
    /// Whether the int4 unpack runs on the GPU. **On where there is a GPU**, and the switch turns it off.
    ///
    /// It was opt-in for two rounds, on the argument that a default is a policy. That argument holds for a
    /// change that could move a digest; it does not hold here, because this one cannot. The GPU decoder is
    /// asserted bit-identical to the scalar one over a grid that includes the partly-filled final groups
    /// `DC-087` used to disagree on, on a real fixture install tensor, on the partial-row payload the row
    /// path actually assembles, and across buffer-cache reuse (`MetalUnpackTests`); and end to end, the real
    /// 35 B trace is `b0d382dbabf36df0…` with the decoder either way, `trace_diff` reporting IDENTICAL.
    ///
    /// It is also **faster**: the phase it lives in is the largest in the forward (`mix.read`, 7.23 s of a
    /// 19 s profile, of which the unpack is 4.53 s), and the GPU path takes about 2 s off a five-token
    /// trace. Leaving a verified-faster path switched off would mean measuring M3 against a slower engine
    /// than the repository has. `SHARD_GPU_UNPACK=0` restores the scalar path, which is how the two are
    /// compared.
    ///
    /// A host with no GPU falls back to the scalar path, so CI runners and a machine without Metal are
    /// unaffected.
    public static let gpuUnpackEnabled = ProcessInfo.processInfo.environment["SHARD_GPU_UNPACK"] != "0"

    /// Whether the payload file is read **through** the kernel's buffer cache rather than with `F_NOCACHE`
    /// (`D111`). **On by default, and measured**; `SHARD_INSTALL_CACHED=0` restores the bypass.
    ///
    /// The doc comment on `UncachedFile` is about a sequential scan of a file larger than the machine; a decode
    /// step is a repeating working set, which is the case a buffer cache is designed for. What has to be true
    /// for this to be a win rather than the panic `D58` records is that the cached pages are **clean and
    /// evictable** — they are, since the file is opened read-only and never written — so the kernel reclaims
    /// them instead of growing swap. The measurement is what decides it.
    public static let readThroughCacheEnabled =
        ProcessInfo.processInfo.environment["SHARD_INSTALL_CACHED"] != "0"

    /// How many threads a dequantisation may use.
    ///
    /// `load` was **30.5% of a cached step** and **47.9% of a cluster step** (`D88`, `D93`), and it is the same
    /// constants decoded again on every token — replicated work, which is exactly what the cluster's ratio is
    /// made of. It ran on **one core of an eight-core machine**. The rows are independent, so the loop is
    /// spread; `SHARD_DECODE_THREADS=1` selects the single-threaded path, which is how the two are compared on
    /// one binary rather than across builds (`D62`).
    public static let decodeThreadCount: Int = DecodeThreads.count

    static func dequantizeInt4(_ data: Data, entry: Entry, rowCount: Int? = nil) throws -> [Float] {
        let layout = try int4Layout(entry: entry, rowCount: rowCount, payloadBytes: data.count)
        var values = [Float](repeating: 0, count: layout.rows * layout.columns)
        // The rows are independent: a row's values are a function of that row's codes, scales and zeros and of
        // nothing else. So the row loop is spread across the machine's cores — which changes **which thread**
        // computes a value and not how the value is computed, the same rule the GPU matmul is held to (`D63`).
        // `SHARD_DECODE_THREADS=1` restores the single-threaded path, and it exists so the two can be compared
        // on one binary instead of across builds: `D62` is what happens when that is not done.
        let shape = layout
        // No `nonisolated(unsafe)` here: an `Int` is `Sendable`, so the annotation would be a claim about
        // the type that the type does not need — and the clean scratch build's warning scan said so, having
        // been invisible to every incremental build up to that point.
        let rows = shape.rows
        data.withUnsafeBytes { raw in
            let base = raw.baseAddress!
            nonisolated(unsafe) let codePointer = base
            nonisolated(unsafe) let scalePointer = base + shape.codeBytes
            nonisolated(unsafe) let zeroPointer = base + shape.codeBytes + shape.scaleBytes
            values.withUnsafeMutableBufferPointer { destination in
                nonisolated(unsafe) let out = destination.baseAddress!
                let decodeRow: @Sendable (Int) -> Void = { row in
                    let rowCodes = codePointer + row * (shape.padded / 2)
                    let rowGroups = row * shape.groups
                    let rowValues = row * shape.columns
                    var index = 0
                    while index < shape.columns {
                        let groupIndex = rowGroups + index / shape.group
                        let scale = Self.flushed(Float(bitPattern: UInt32(littleEndian: scalePointer.loadUnaligned(fromByteOffset: groupIndex * 4, as: UInt32.self))))
                        let rawZero = Int(zeroPointer.loadUnaligned(fromByteOffset: groupIndex, as: UInt8.self))
                        let zero = rawZero >= 128 ? rawZero - 256 : rawZero
                        // The last group of a padded tensor holds fewer real values than the group
                        // size, so the vector loop is bounded by what is actually stored.
                        let inGroup = min(shape.group, shape.columns - index)
                        var offset = 0
                        // Eight codes per load, but **only when the group is a multiple of eight**:
                        // a wider block must not straddle two groups, because the scale and the zero
                        // point change at the boundary. The four-wide loop below is the general case
                        // and its arithmetic is the same one multiply per element.
                        let scaleGroup = SIMD4<Float>(repeating: scale)
                        if shape.group % 8 == 0 {
                            while offset + 8 <= inGroup {
                                let word = rowCodes.loadUnaligned(fromByteOffset: (index + offset) / 2, as: UInt32.self)
                                let low = Self.withoutSignedZero(SIMD4<Float>(
                                    Float(signed(Int(word & 0x0F)) - zero),
                                    Float(signed(Int((word >> 4) & 0x0F)) - zero),
                                    Float(signed(Int((word >> 8) & 0x0F)) - zero),
                                    Float(signed(Int((word >> 12) & 0x0F)) - zero)
                                ) * scaleGroup)
                                let high = Self.withoutSignedZero(SIMD4<Float>(
                                    Float(signed(Int((word >> 16) & 0x0F)) - zero),
                                    Float(signed(Int((word >> 20) & 0x0F)) - zero),
                                    Float(signed(Int((word >> 24) & 0x0F)) - zero),
                                    Float(signed(Int((word >> 28) & 0x0F)) - zero)
                                ) * scaleGroup)
                                let at = rowValues + index + offset
                                out[at + 0] = low[0]
                                out[at + 1] = low[1]
                                out[at + 2] = low[2]
                                out[at + 3] = low[3]
                                out[at + 4] = high[0]
                                out[at + 5] = high[1]
                                out[at + 6] = high[2]
                                out[at + 7] = high[3]
                                offset += 8
                            }
                        }
                        while offset + 4 <= inGroup {
                            // Two bytes carry four codes, low nibble first — the order is the format.
                            let pair = rowCodes.loadUnaligned(fromByteOffset: (index + offset) / 2, as: UInt16.self)
                            let lanes = SIMD4<Float>(
                                Float(signed(Int(pair & 0x0F)) - zero),
                                Float(signed(Int((pair >> 4) & 0x0F)) - zero),
                                Float(signed(Int((pair >> 8) & 0x0F)) - zero),
                                Float(signed(Int((pair >> 12) & 0x0F)) - zero)
                            )
                            let product = Self.withoutSignedZero(lanes * SIMD4<Float>(repeating: scale))
                            out[rowValues + index + offset + 0] = product[0]
                            out[rowValues + index + offset + 1] = product[1]
                            out[rowValues + index + offset + 2] = product[2]
                            out[rowValues + index + offset + 3] = product[3]
                            offset += 4
                        }
                        while offset < inGroup {
                            let position = index + offset
                            let byte = rowCodes.loadUnaligned(fromByteOffset: position / 2, as: UInt8.self)
                            let nibble = position % 2 == 0 ? (byte & 0x0F) : (byte >> 4)
                            out[rowValues + position] = Self.withoutSignedZero(
                                Float(signed(Int(nibble)) - zero) * scale
                            )
                            offset += 1
                        }
                        index += inGroup
                    }
                }
                if Self.decodeThreadCount > 1, rows > 1 {
                    DispatchQueue.concurrentPerform(iterations: rows, execute: decodeRow)
                } else {
                    for row in 0..<rows { decodeRow(row) }
                }
            }
        }
        return values
    }

    /// `D34`: **a zero is `+0.0`**, on every path.
    ///
    /// `DC-087`'s residue was one shape out of twenty-four where the GPU and the CPU disagreed on a
    /// single index, and the whole of the difference was the **sign of zero**: the CPU produced
    /// `0x80000000` where the GPU produced `0x00000000`. It is not an arithmetic difference — every
    /// non-zero value agrees — it is a difference in what a zero *is*, and bit-identity has no opinion
    /// until the contract does.
    ///
    /// So it does: a computed zero carries no sign. Adding `+0.0` is the normalisation, because IEEE
    /// round-to-nearest maps `-0.0 + 0.0` to `+0.0` and leaves every other value, including an infinity,
    /// exactly as it was. It costs one addition per value and it makes the GPU's behaviour the
    /// definition rather than the anomaly — the same move as `D11`, one bit over.
    @inline(__always)
    static func withoutSignedZero(_ value: Float) -> Float {
        // Written as a branch on purpose. The first version was `value + 0`, which the **Metal** compiler
        // folded away under its default fast-math flags — leaving nine values of 195 as `-0.0` — and a
        // release build of this file is entitled to do the same. A comparison cannot be folded, and the
        // literal result is `+0.0` by definition. The NaN clause is the other half: `+ 0` preserves a NaN
        // payload while the GPU canonicalises, and a zero code times an infinite scale is an indeterminate
        // form the contract has to answer rather than a rounding question either side may decide.
        if value.isNaN { return Float.nan }
        if value == 0 { return 0 }
        return value
    }

    /// The same rule for a lane group, because the vector paths compute four or eight at a time — and the
    /// additive idiom (`+ SIMD4(repeating: 0)`) that first stood in for this neither canonicalises a NaN
    /// nor survives a compiler that folds it. The rule is a rule; it gets one implementation.
    @inline(__always)
    static func withoutSignedZero(_ value: SIMD4<Float>) -> SIMD4<Float> {
        SIMD4(
            withoutSignedZero(value[0]), withoutSignedZero(value[1]),
            withoutSignedZero(value[2]), withoutSignedZero(value[3])
        )
    }

    /// `D11`: a denormal scale is read as zero, on both the CPU and (already) the GPU.
    ///
    /// Measured on the real 35 B install: **0.722705 %** of 523,304,960 scales are denormal, and one
    /// first-layer tensor is 41 % denormal by itself. Rather than keep every kernel away from the
    /// whole expert path, the contract *defines* the flush — `tools/quantize.py` does the same — so a
    /// comparison between the two stays exact. The values lost are ~1e-38 against weights of ~1e-1.
    @inline(__always)
    static func flushed(_ scale: Float) -> Float {
        (scale != 0 && abs(scale) < Float.leastNormalMagnitude) ? 0 : scale
    }

    /// A four-bit code as a signed integer: two's complement in four bits.
    @inline(__always)
    private static func signed(_ nibble: Int) -> Int {
        let value = nibble & 0x0F
        return value >= 8 ? value - 16 : value
    }

    /// The definition: one element at a time, in index order. Kept because a fast formulation is
    /// only trustworthy while something independent says it agrees.
    static func dequantizeInt4Scalar(_ data: Data, entry: Entry, rowCount: Int? = nil) throws -> [Float] {
        let layout = try int4Layout(entry: entry, rowCount: rowCount, payloadBytes: data.count)
        let rows = layout.rows
        let columns = layout.columns
        let padded = layout.padded
        let group = layout.group
        let groups = layout.groups
        let codeBytes = layout.codeBytes
        let scaleBytes = layout.scaleBytes

        var values = [Float](repeating: 0, count: rows * columns)
        data.withUnsafeBytes { raw in
            let base = raw.baseAddress!
            let codes = base
            let scales = base + codeBytes
            let zeros = base + codeBytes + scaleBytes
            for row in 0..<rows {
                let rowCodes = codes + row * (padded / 2)
                for index in 0..<padded {
                    let byte = rowCodes.loadUnaligned(fromByteOffset: index / 2, as: UInt8.self)
                    let nibble = index % 2 == 0 ? (byte & 0x0F) : (byte >> 4)
                    let code = Int(nibble >= 8 ? Int(nibble) - 16 : Int(nibble))
                    let groupIndex = row * groups + index / group
                    let scale = scales.loadUnaligned(fromByteOffset: groupIndex * 4, as: UInt32.self)
                    let zero = zeros.loadUnaligned(fromByteOffset: groupIndex, as: UInt8.self)
                    let zeroValue = Int(zero >= 128 ? Int(zero) - 256 : Int(zero))
                    if index < columns {
                        // The same `D34` normalisation as the vector path: a zero carries no sign and a NaN
                        // is canonical. This scalar variant exists so the Metal kernel has something to be
                        // compared against, and it was the one place the rule was missing — which the
                        // comparison found, two shapes out of sixty, one index each.
                        values[row * columns + index] = Self.withoutSignedZero(
                            Float(code - zeroValue)
                                * Self.flushed(Float(bitPattern: UInt32(littleEndian: scale)))
                        )
                    }
                }
            }
        }
        return values
    }
}
