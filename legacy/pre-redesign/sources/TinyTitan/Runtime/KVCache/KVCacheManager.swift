import Foundation
import Darwin
import Metal

/// Which attention variant a layer runs. Qwen 3.6 interleaves 30
/// gated-DeltaNet linear-attention layers with 10 full-attention layers;
/// linear layers keep a fixed-size recurrent state (owned by
/// `GDNStateManager`) instead of per-token K/V rows. Sourced from
/// `ArchConfig.fullAttentionLayerMask` (0 = swa, 1 = full, 2 = linear).
public enum LayerKind: Sendable { case swa, full, linear }

/// A read view the attention kernels bind. `offset` stays 0; ring-enabled SWA
/// layers expose the physical start slot for diagnostics while kernels map
/// logical positions with the supplied ring capacity.
/// unchecked-invariant: every stored property is a `let`. The type is only
/// @unchecked because MTLBuffer is not Sendable; the struct itself is a
/// read-only descriptor of a range, and callers that write through it are
/// serialised by whoever owns the buffer.
public struct KVView: @unchecked Sendable {
    public let buffer: MTLBuffer
    /// Byte offset of logical position 0. Zero for slot 0 -- and so always zero
    /// when there is a single slot -- and the slot's region base otherwise.
    public let offset: Int
    /// Bytes per token, including affine metadata for quantized storage.
    public let stride: Int
    /// Number of valid positions written so far (== `position`), i.e. an
    /// exclusive bound: attention reads `[0, validTokenCount)`.
    ///
    /// Stated as a count because that is what it is and what every kernel
    /// iterates (`p < seqLen`). It is deliberately *not* "the index of the last
    /// valid token", which is `validTokenCount - 1`: the decode path passes
    /// `position + 1` and prefill passes `startPosition + t` precisely so the
    /// just-written token is included. The `keyView(layer:)` /
    /// `valueView(layer:)` convenience overloads default to `position`, which
    /// before `advance()` is one *fewer* than that — correct for a caller reading
    /// everything but the token it is about to write, and not a bound to copy
    /// into a new call site without knowing which of the two it wants.
    public let validTokenCount: Int
    public let precision: KVCachePrecision
    public let valueBytes: Int
    public let groupSize: Int
}

/// Per-layer K/V storage for the decode loop.
///
/// One K buffer and one V buffer per layer, allocated once in `init` — the
/// decode hot path never allocates. Linear storage sizes every layer for
/// `maxContext`; ring storage caps SWA layers to their physical capacity
/// while full-attention layers remain linear. With more than one slot the
/// layer buffer holds `slots` contiguous regions of that same capacity, one per
/// concurrent sequence, and every accessor takes the slot whose region it means.
///
/// The K/V projection GEMV writes straight into the slot returned by
/// `kSlot`/`vSlot` (no separate `kv_write` kernel); the runner then norms +
/// optionally RoPE's each slot in place. `advance()` bumps the cursor once
/// both are written.
///
/// 8 GB rule: storage is bounded by per-layer physical capacity, allocated
/// once. `reset()` returns physical pages to the OS via `MADV_DONTNEED` so a
/// finished generation does not keep its KV resident into the next turn.
public final class KVCacheManager {
    public let config: ArchConfig
    public let maxContext: Int
    public let fp16RingEnabled: Bool
    public let precision: KVCachePrecision
    /// Retained so `reserve` can allocate; init-only allocation was the previous
    /// invariant and growth deliberately relaxes it.
    private let device: MTLDevice

    private var kBuffers: [MTLBuffer]
    private var vBuffers: [MTLBuffer]
    private let strides:  [Int]         // bytes per token, per layer
    private let kinds:    [LayerKind]
    private var capacityTokens: [Int]
    private let valueBytes: [Int]

    /// How many independent sequences share these stores. Each slot owns its own
    /// contiguous region of every layer buffer and its own cursor, so a batched
    /// decode step writes one token per slot without aliasing. Defaults to one:
    /// at `slots == 1` every offset here is exactly what it was before slots
    /// existed, which is what keeps the single-sequence path byte-identical.
    public let slots: Int

    /// Hard cap, so a bad argument cannot ask for an absurd allocation.
    public static let maximumSlots = 8

    /// Logical token cursor per slot. `position` is slot 0's -- the
    /// single-sequence case every existing caller uses.
    private var positions: [Int]

    /// Slot 0's cursor.
    public var position: Int { positions[0] }

    private static let fp16Size = 2
    public static let quantizationGroupSize = 64

    public init(device: MTLDevice,
                config: ArchConfig,
                maxContext: Int,
                slots: Int = 1,
                fp16RingEnabled: Bool = false,
                precision: KVCachePrecision = .fp16,
                slidingWindow: Int? = nil,
                maxPrefillChunkTokens: Int = 128) throws {
        precondition(maxContext > 0, "maxContext must be positive")
        precondition(maxPrefillChunkTokens > 0, "maxPrefillChunkTokens must be positive")
        precondition(slots > 0 && slots <= Self.maximumSlots,
                     "slots must be between 1 and \(Self.maximumSlots)")
        self.device = device
        self.config = config
        self.maxContext = maxContext
        self.slots = slots
        self.positions = Array(repeating: 0, count: slots)
        self.precision = precision
        let ringEnabled = fp16RingEnabled
        self.fp16RingEnabled = ringEnabled

        let swaStride  = config.numKVHeads     * config.headDim     * Self.fp16Size
        let fullStride = config.numFullKVHeads * config.fullHeadDim  * Self.fp16Size
        let swaCapacity = min(maxContext,
                              max(1, (slidingWindow ?? config.slidingWindow) + maxPrefillChunkTokens))

        var ks: [MTLBuffer] = []
        var vs: [MTLBuffer] = []
        var st: [Int] = []
        var kd: [LayerKind] = []
        var caps: [Int] = []
        var valueByteCounts: [Int] = []
        ks.reserveCapacity(config.numLayers)
        vs.reserveCapacity(config.numLayers)
        st.reserveCapacity(config.numLayers)
        kd.reserveCapacity(config.numLayers)
        caps.reserveCapacity(config.numLayers)
        valueByteCounts.reserveCapacity(config.numLayers)

        // Linear-attention layers keep no per-token K/V rows; they share one
        // page-sized placeholder so the parallel arrays stay non-optional.
        var linearPlaceholder: MTLBuffer? = nil

        for layer in 0..<config.numLayers {
            let maskValue = config.fullAttentionLayerMask[layer]
            if maskValue == 2 {
                let placeholder: MTLBuffer
                if let existing = linearPlaceholder {
                    placeholder = existing
                } else {
                    guard let made = device.makeBuffer(length: Int(getpagesize()),
                                                       options: .storageModeShared) else {
                        throw ModelError.residentBufferWrapFailed
                    }
                    made.label = "kv.linear-placeholder"
                    linearPlaceholder = made
                    placeholder = made
                }
                ks.append(placeholder)
                vs.append(placeholder)
                st.append(0)
                kd.append(.linear)
                caps.append(0)
                valueByteCounts.append(0)
                continue
            }
            let isFull = maskValue != 0
            let fp16Stride = isFull ? fullStride : swaStride
            let elements = fp16Stride / Self.fp16Size
            let layout = Self.rowLayout(elements: elements, precision: precision)
            let stride = layout.stride
            // Linear layers start at `initialCapacityTokens` and grow on demand
            // rather than reserving `maxContext` up front. At 262144 tokens a
            // full reservation is 512 MiB per layer, 20 GiB across 40 -- lazily
            // touched, so it barely shows in RSS, but on a 24 GB machine the
            // mappings alone cost throughput: the same 25-token prompt measured
            // 13.80 s of decode at maxContext 8192 against 22.44 s at 262144.
            // Growing keeps a short conversation at a short conversation's cost
            // while leaving the advertised limit reachable.
            let capacity = ringEnabled && !isFull
                ? swaCapacity
                : min(maxContext, Self.initialCapacityTokens)
            let length = slots * capacity * stride

            guard let kBuf = device.makeBuffer(length: length, options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            kBuf.label = "kv.K.layer\(layer)"
            ks.append(kBuf)

            guard let vBuf = device.makeBuffer(length: length, options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            vBuf.label = "kv.V.layer\(layer)"
            vs.append(vBuf)

            st.append(stride)
            kd.append(isFull ? .full : .swa)
            caps.append(capacity)
            valueByteCounts.append(layout.valueBytes)
        }

        self.kBuffers = ks
        self.vBuffers = vs
        self.strides  = st
        self.kinds    = kd
        self.capacityTokens = caps
        self.valueBytes = valueByteCounts
    }

    /// Tokens each linear layer is sized for before any growth.
    ///
    /// 8192 because it measured as fast as any smaller reservation and holds an
    /// ordinary conversation without a single grow. Capacity doubles from here.
    public static let initialCapacityTokens = 8_192

    /// Worst-case bytes `slots` sequences' K/V stores can occupy at
    /// `maxContext`, without allocating anything.
    ///
    /// Worst case means every slot grown to the advertised context: growth is
    /// lazy, so this is the ceiling a batched session must be able to hold, not
    /// what it maps at load. It mirrors the layout `init` builds, and a test
    /// compares the two against each other so a storage-format change cannot
    /// leave the formula quietly wrong.
    public static func worstCaseBytes(config: ArchConfig,
                                      maxContext: Int,
                                      precision: KVCachePrecision = .fp16,
                                      slots: Int = 1,
                                      fp16RingEnabled: Bool = false,
                                      slidingWindow: Int? = nil,
                                      maxPrefillChunkTokens: Int = 128) -> Int {
        precondition(maxContext > 0, "maxContext must be positive")
        precondition(slots > 0, "slots must be positive")
        var total = 0
        for layer in 0..<config.numLayers {
            let maskValue = config.fullAttentionLayerMask[layer]
            if maskValue == 2 { continue }
            let isFull = maskValue != 0
            let elements = isFull
                ? config.numFullKVHeads * config.fullHeadDim
                : config.numKVHeads * config.headDim
            let stride = rowLayout(elements: elements, precision: precision).stride
            let capacity: Int
            if fp16RingEnabled && !isFull {
                capacity = min(maxContext,
                               max(1, (slidingWindow ?? config.slidingWindow)
                                   + maxPrefillChunkTokens))
            } else {
                capacity = maxContext
            }
            // One K buffer and one V buffer per layer.
            total += slots * capacity * stride * 2
        }
        return total
    }

    /// Grows linear layers so every one can hold `tokens`, copying what is
    /// already stored.
    ///
    /// Must be called before writing at a position beyond the current capacity.
    /// Ring-backed SWA layers are never grown -- their capacity is the window and
    /// is deliberate. Capacity doubles, so a conversation reaching the advertised
    /// 262144 limit pays five copies in total rather than one per token.
    ///
    /// Buffers are `storageModeShared`, so the copy is a plain `memcpy`; callers
    /// fetch buffers through the accessors at use time and never cache them
    /// across tokens, which is what makes swapping them safe here.
    public func reserve(tokens: Int) throws {
        try reserve(tokens: tokens, slot: 0)
    }

    /// Grows the stores so slot `slot` can hold `tokens`, copying every slot's
    /// live rows. All slots share one allocation per layer, so growth is
    /// all-or-nothing even though only one cursor motivated it.
    public func reserve(tokens: Int, slot: Int) throws {
        validateSlot(slot)
        let needed = min(max(tokens, 1), maxContext)
        for layer in 0..<kinds.count {
            guard kinds[layer] != .linear else { continue }
            if fp16RingEnabled && kinds[layer] == .swa { continue }
            let current = capacityTokens[layer]
            guard current < needed else { continue }
            var target = current
            while target < needed { target *= 2 }
            target = min(target, maxContext)

            let stride = strides[layer]
            let length = slots * target * stride
            guard let newK = device.makeBuffer(length: length, options: .storageModeShared),
                  let newV = device.makeBuffer(length: length, options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            newK.label = "kv.K.layer\(layer)"
            newV.label = "kv.V.layer\(layer)"
            // Each slot's live rows move from its old region to its new one.
            // At one slot this is the original single `memcpy` from the base.
            for s in 0..<slots {
                let usedBytes = min(current, positions[s]) * stride
                guard usedBytes > 0 else { continue }
                memcpy(newK.contents() + s * target * stride,
                       kBuffers[layer].contents() + s * current * stride, usedBytes)
                memcpy(newV.contents() + s * target * stride,
                       vBuffers[layer].contents() + s * current * stride, usedBytes)
            }
            kBuffers[layer] = newK
            vBuffers[layer] = newV
            capacityTokens[layer] = target
        }
    }

    public func layerKind(_ layer: Int) -> LayerKind { kinds[layer] }

    /// Bytes per token for `layer` (K and V share the same stride).
    public func stride(layer: Int) -> Int { strides[layer] }

    /// Physical token capacity for `layer`. Ring-enabled SWA layers can be
    /// smaller than `maxContext`; full layers and ring-off storage stay linear.
    public func capacity(layer: Int) -> Int { capacityTokens[layer] }

    public func ringCapacity(layer: Int) -> Int {
        guard fp16RingEnabled, kinds[layer] == .swa else { return 0 }
        return capacityTokens[layer]
    }

    /// Total bytes of the K buffer for `layer`, across all slots.
    public func bufferLength(layer: Int) -> Int {
        return slots * capacityTokens[layer] * strides[layer]
    }

    /// First token index of slot `slot`'s region within a layer buffer.
    private func regionBase(layer: Int, slot: Int) -> Int {
        validateSlot(slot)
        return slot * capacityTokens[layer]
    }

    /// Write target for this layer's K projection at `position` in `slot`.
    public func kSlot(layer: Int, position: Int,
                      slot: Int = 0) -> (buffer: MTLBuffer, offset: Int) {
        precondition(kinds[layer] != .linear, "linear layers have no KV slots")
        validateRange(start: position, count: 1)
        let token = regionBase(layer: layer, slot: slot)
            + physicalSlot(layer: layer, position: position)
        return (kBuffers[layer], token * strides[layer])
    }

    /// Write target for this layer's V projection at `position` in `slot`.
    /// Always distinct from `kSlot`: K runs per-head k_norm + RoPE while V is
    /// normed differently, so the two buffers cannot alias.
    public func vSlot(layer: Int, position: Int,
                      slot: Int = 0) -> (buffer: MTLBuffer, offset: Int) {
        precondition(kinds[layer] != .linear, "linear layers have no KV slots")
        validateRange(start: position, count: 1)
        let token = regionBase(layer: layer, slot: slot)
            + physicalSlot(layer: layer, position: position)
        return (vBuffers[layer], token * strides[layer])
    }

    public func kRange(layer: Int, start: Int, count: Int,
                       slot: Int = 0) -> (buffer: MTLBuffer, offset: Int, stride: Int) {
        validateRange(start: start, count: count)
        validateContiguousPhysicalRange(layer: layer, start: start, count: count)
        let token = regionBase(layer: layer, slot: slot)
            + physicalSlot(layer: layer, position: start)
        return (kBuffers[layer], token * strides[layer], strides[layer])
    }

    public func vRange(layer: Int, start: Int, count: Int,
                       slot: Int = 0) -> (buffer: MTLBuffer, offset: Int, stride: Int) {
        validateRange(start: start, count: count)
        validateContiguousPhysicalRange(layer: layer, start: start, count: count)
        let token = regionBase(layer: layer, slot: slot)
            + physicalSlot(layer: layer, position: start)
        return (vBuffers[layer], token * strides[layer], strides[layer])
    }

    public func keyView(layer: Int) -> KVView {
        keyView(layer: layer, slot: 0, validTokenCount: positions[0])
    }

    public func keyView(layer: Int, validTokenCount: Int) -> KVView {
        keyView(layer: layer, slot: 0, validTokenCount: validTokenCount)
    }

    public func keyView(layer: Int, slot: Int, validTokenCount: Int) -> KVView {
        validateSlot(slot)
        validateValidTokenCount(validTokenCount)
        return makeView(buffer: kBuffers[layer], layer: layer,
                        offset: regionBase(layer: layer, slot: slot) * strides[layer],
                        validTokenCount: validTokenCount)
    }

    public func valueView(layer: Int) -> KVView {
        valueView(layer: layer, slot: 0, validTokenCount: positions[0])
    }

    func keyBuffer(layer: Int, validTokenCount: Int) -> MTLBuffer {
        keyView(layer: layer, validTokenCount: validTokenCount).buffer
    }

    func valueBuffer(layer: Int, validTokenCount: Int) -> MTLBuffer {
        valueView(layer: layer, validTokenCount: validTokenCount).buffer
    }

    public func valueView(layer: Int, validTokenCount: Int) -> KVView {
        valueView(layer: layer, slot: 0, validTokenCount: validTokenCount)
    }

    public func valueView(layer: Int, slot: Int, validTokenCount: Int) -> KVView {
        validateSlot(slot)
        validateValidTokenCount(validTokenCount)
        return makeView(buffer: vBuffers[layer], layer: layer,
                        offset: regionBase(layer: layer, slot: slot) * strides[layer],
                        validTokenCount: validTokenCount)
    }

    public func keyRangeView(layer: Int, start: Int, count: Int,
                             slot: Int = 0) -> KVView {
        let range = kRange(layer: layer, start: start, count: count, slot: slot)
        return makeView(buffer: range.buffer, layer: layer, offset: range.offset,
                        validTokenCount: count)
    }

    public func valueRangeView(layer: Int, start: Int, count: Int,
                               slot: Int = 0) -> KVView {
        let range = vRange(layer: layer, start: start, count: count, slot: slot)
        return makeView(buffer: range.buffer, layer: layer, offset: range.offset,
                        validTokenCount: count)
    }

    /// Advance slot 0's cursor once the current token's K/V are written across
    /// all layers.
    public func advance() { advance(slot: 0, by: 1) }

    public func advance(by count: Int) { advance(slot: 0, by: count) }

    public func advance(slot: Int, by count: Int) {
        validateSlot(slot)
        precondition(count >= 0, "advance count must be non-negative")
        precondition(positions[slot] + count <= maxContext,
                     "advance would exceed maxContext")
        positions[slot] += count
    }

    /// Slot `slot`'s logical cursor.
    public func position(slot: Int) -> Int {
        validateSlot(slot)
        return positions[slot]
    }

    /// Rewind the logical cursor without copying KV. Speculative rows are
    /// append-only and become unreachable immediately; a later pass
    /// overwrites them.
    /// Ring-backed draft KV is safe because MTP verification never rewinds by
    /// more than its two-token proposal depth.
    func rewind(to newPosition: Int) throws {
        try rewind(slot: 0, to: newPosition)
    }

    func rewind(slot: Int, to newPosition: Int) throws {
        validateSlot(slot)
        guard newPosition >= 0, newPosition <= positions[slot] else {
            throw InferenceStateSnapshotError.invalidPosition(newPosition)
        }
        positions[slot] = newPosition
    }

    /// Drop all cached positions and return physical pages to the OS.
    ///
    /// No buffer zeroing — the attention kernels read only `[0, validTokenCount]`,
    /// and `validTokenCount` is now 0. `MADV_DONTNEED` on the page-aligned span
    /// releases resident memory between turns; pages fault back in on next write.
    public func reset() {
        for slot in 0..<slots { positions[slot] = 0 }
        let pageSize = Int(getpagesize())
        var advised = Set<ObjectIdentifier>()
        for layer in 0..<config.numLayers {
            advise(kBuffers[layer], pageSize: pageSize, seen: &advised)
            advise(vBuffers[layer], pageSize: pageSize, seen: &advised)
        }
    }

    /// Reset one slot's cursor without touching the others.
    ///
    /// The rows are neither zeroed nor returned to the OS: a recycled batch slot
    /// overwrites them before they can become visible again (attention reads
    /// only `[0, validTokenCount)`), and the whole-store `reset()` is what
    /// releases pages between generations.
    public func reset(slot: Int) {
        validateSlot(slot)
        positions[slot] = 0
    }

    func snapshotSegmentLengths(at snapshotPosition: Int) throws -> [Int] {
        guard snapshotPosition > 0, snapshotPosition <= maxContext else {
            throw InferenceStateSnapshotError.invalidPosition(snapshotPosition)
        }
        var lengths: [Int] = []
        lengths.reserveCapacity(config.numLayers * 2)
        // One sequence's worth: the snapshot is a prefix of a single slot and is
        // captured/restored at slot 0's region base (offset 0), which is what the
        // prompt cache keys on. A batched slot's payload is a later phase.
        for layer in 0..<config.numLayers where kinds[layer] != .linear {
            let storedTokens = min(snapshotPosition, capacityTokens[layer])
            let (length, overflow) = storedTokens.multipliedReportingOverflow(
                by: strides[layer])
            guard !overflow else { throw InferenceStateSnapshotError.integerOverflow }
            lengths.append(length)
            lengths.append(length)
        }
        return lengths
    }

    func appendSnapshotPayload(to payload: inout Data,
                               segmentLengths: [Int]) throws {
        let expected = try snapshotSegmentLengths(at: position)
        guard segmentLengths == expected else {
            throw InferenceStateSnapshotError.invalidLayout
        }
        var segment = 0
        for layer in 0..<config.numLayers where kinds[layer] != .linear {
            let kLength = segmentLengths[segment]
            payload.append(kBuffers[layer].contents().assumingMemoryBound(to: UInt8.self),
                           count: kLength)
            segment += 1
            let vLength = segmentLengths[segment]
            payload.append(vBuffers[layer].contents().assumingMemoryBound(to: UInt8.self),
                           count: vLength)
            segment += 1
        }
    }

    func restoreSnapshot(position snapshotPosition: Int,
                         segmentLengths: [Int],
                         bytes: UnsafeRawBufferPointer,
                         offset: inout Int) throws {
        // Grow to the snapshot's position *before* computing the expected
        // lengths, because those lengths are a function of capacity:
        // `snapshotSegmentLengths` records `min(position, capacity)`. The saving
        // runner had grown to hold its prefix; a fresh receiver's full-attention
        // layers start at `initialCapacityTokens` (8192), so any snapshot past
        // that produced a different set of lengths and was refused as
        // `invalidLayout`. The restore could therefore never succeed for a long
        // prefix — which is the case the disk tier exists for, and why the
        // feature appeared simply not to work.
        try reserve(tokens: snapshotPosition)
        let expected = try snapshotSegmentLengths(at: snapshotPosition)
        guard segmentLengths == expected else {
            throw InferenceStateSnapshotError.invalidLayout
        }
        reset()
        var segment = 0
        for layer in 0..<config.numLayers where kinds[layer] != .linear {
            let kLength = segmentLengths[segment]
            try copySnapshotSegment(bytes: bytes,
                                    offset: &offset,
                                    length: kLength,
                                    destination: kBuffers[layer])
            segment += 1
            let vLength = segmentLengths[segment]
            try copySnapshotSegment(bytes: bytes,
                                    offset: &offset,
                                    length: vLength,
                                    destination: vBuffers[layer])
            segment += 1
        }
        positions[0] = snapshotPosition
    }

    private func copySnapshotSegment(bytes: UnsafeRawBufferPointer,
                                     offset: inout Int,
                                     length: Int,
                                     destination: MTLBuffer) throws {
        guard length <= destination.length,
              offset >= 0,
              length >= 0,
              offset <= bytes.count - length,
              let source = bytes.baseAddress?.advanced(by: offset) else {
            throw InferenceStateSnapshotError.invalidLayout
        }
        memcpy(destination.contents(), source, length)
        offset += length
    }

    private func validateRange(start: Int, count: Int) {
        precondition(count >= 0, "count must be non-negative")
        precondition(start >= 0, "start must be non-negative")
        precondition(start + count <= maxContext,
                     "range \(start)..<\(start + count) exceeds maxContext \(maxContext)")
    }

    private func makeView(buffer: MTLBuffer, layer: Int, offset: Int,
                          validTokenCount: Int) -> KVView {
        KVView(buffer: buffer, offset: offset, stride: strides[layer],
               validTokenCount: validTokenCount, precision: precision,
               valueBytes: valueBytes[layer], groupSize: Self.quantizationGroupSize)
    }

    private static func rowLayout(elements: Int,
                                  precision: KVCachePrecision) -> (stride: Int, valueBytes: Int) {
        if precision == .fp16 {
            return (elements * fp16Size, elements * fp16Size)
        }
        let packed = (elements * precision.rawValue + 7) / 8
        let alignedPacked = (packed + 1) & ~1
        let groups = (elements + quantizationGroupSize - 1) / quantizationGroupSize
        return (alignedPacked + groups * 2 * fp16Size, alignedPacked)
    }

    private func validateValidTokenCount(_ count: Int) {
        precondition(count >= 0, "validTokenCount must be non-negative")
        precondition(count <= maxContext,
                     "validTokenCount \(count) exceeds maxContext \(maxContext)")
    }

    private func validateSlot(_ slot: Int) {
        precondition(slot >= 0 && slot < slots,
                     "slot \(slot) is out of range 0..<\(slots)")
    }

    private func physicalSlot(layer: Int, position: Int) -> Int {
        precondition(capacityTokens[layer] > 0, "layer has no KV storage")
        return position % capacityTokens[layer]
    }

    private func validateContiguousPhysicalRange(layer: Int, start: Int, count: Int) {
        guard count > 0, fp16RingEnabled, kinds[layer] == .swa else { return }
        let capacity = capacityTokens[layer]
        let physicalStart = start % capacity
        precondition(physicalStart + count <= capacity,
                     "range \(start)..<\(start + count) wraps KV ring capacity \(capacity)")
    }

    private func advise(_ buffer: MTLBuffer, pageSize: Int, seen: inout Set<ObjectIdentifier>) {
        let id = ObjectIdentifier(buffer)
        if seen.contains(id) { return }
        seen.insert(id)
        // MTLBuffer allocations are page-aligned; round the length down to a
        // whole number of pages so we never hand madvise a partial tail page.
        let len = (buffer.length / pageSize) * pageSize
        if len > 0 {
            _ = posix_madvise(buffer.contents(), len, POSIX_MADV_DONTNEED)
        }
    }
}
