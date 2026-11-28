import CoreML
import Foundation
import Metal

/// Switch for the ANE prefill attention path (Track A).
///
/// `on` routes every full-attention layer's prefill attention block through a
/// Core ML sidecar exported by `tools/export_ane_prefill.py`, leaving GDN
/// layers, the MoE, the KV cache format, and all of decode untouched. Output
/// is NOT byte-identical to the GPU path — the sidecar computes in fp16 with
/// a different reduction order (measured ~1% per-layer mean deviation against
/// an fp32 reference) — and it engages only once a prefill reaches one full
/// chunk, so shorter prompts never reach it.
///
/// **On by default** since v4.6: `environmentValue` returns `.on` when
/// `TINYTITAN_PREFILL_ANE` is unset, and a model with no sidecar falls back to the
/// GPU quietly, which is the normal case. `TINYTITAN_PREFILL_ANE=off` opts out.
/// The asymmetry is deliberate and lives in `wasRequestedExplicitly`: someone
/// who asked for `on` and has no sidecar is told, because they meant it; the
/// default must not fail a load over an optional experiment.
///
/// Anything that must be byte-reproducible has to pin this rather than follow
/// the default — `tools/golden-baseline.sh` exports `TINYTITAN_PREFILL_ANE=off` for
/// exactly that reason.
public enum RuntimePrefillANE: String, Codable, Sendable {
    case off
    case on

    /// Whether the caller named the setting, as opposed to taking the
    /// default.
    ///
    /// The two want different failure behaviour. Someone who wrote
    /// `TINYTITAN_PREFILL_ANE=on` and has no sidecar should be told so, with the
    /// export command; a default-on runtime meeting a model that has no
    /// sidecar should quietly use the GPU, because most models do not have
    /// one and failing to load would be absurd.
    public static func wasRequestedExplicitly(
        _ environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Bool {
        environment["TINYTITAN_PREFILL_ANE"] != nil
    }

    public static func environmentValue(
        _ environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> RuntimePrefillANE {
        guard let raw = environment["TINYTITAN_PREFILL_ANE"] else { return .on }
        guard let value = RuntimePrefillANE(rawValue: raw) else {
            throw PrefillError.chunkedUnsupported(
                "unsupported TINYTITAN_PREFILL_ANE '\(raw)'; allowed: off, on")
        }
        return value
    }
}

/// Transfers a loaded `MLModel` out of a background load task.
///
/// unchecked-invariant: the box is created inside the loading task, handed to
/// exactly one awaiting consumer, and never mutated; the prefill loop that
/// consumes it is single-flight, so no two threads ever hold the same model.
private struct LoadedModelBox: @unchecked Sendable {
    let model: MLModel
}

/// Runs the full-attention prefill block on the Neural Engine.
///
/// One multifunction `.mlpackage` per full-attention layer holds functions
/// `h0, h4096, ...` sharing a single fp16 weight set; the function name is
/// the KV-history length, which in this runtime is always the chunk-aligned
/// `startPosition`. The block consumes the post-input-norm hidden chunk and a
/// token-major fp16 K/V history, and produces the attention branch output
/// plus the chunk's cache-layout K/V — the same bytes the GPU path stages
/// before quantizing into the cache, so `copyPrefillKVToCache` is reused
/// verbatim and decode sees an ordinary cache.
///
/// Design constants mirror the exporter and are validated against its
/// manifest at load; a mismatch fails closed rather than computing nonsense.
///
/// unchecked-invariant: driven exclusively by the single-flight prefill loop
/// of one runner; buffers and lazy caches are never touched concurrently.
final class ANEPrefillAttention: @unchecked Sendable {
    /// The geometry the sidecar's Core ML graph was built for. The exporter
    /// records it and `init` refuses a sidecar that does not match the model:
    /// a mismatch computes a *different* attention, and plausibly.
    struct SidecarGeometry: Decodable {
        let family: String
        let hiddenSize: Int
        let numHeads: Int
        let numKVHeads: Int
        let headDim: Int
        let chunkTokens: Int
        let fullAttentionLayers: [Int]?
    }

    struct SidecarMetadata: Decodable {
        let version: Int
        let family: String
        let chunkTokens: Int
        let histories: [Int]
        let layers: [Int]
        /// Present from the generalized exporter on; absent in a sidecar built
        /// while the graph was hard-coded to the 35B-A3B geometry.
        let geometry: SidecarGeometry?
        /// SHA-256 of the `model_weights.bin` the sidecar was exported from,
        /// copied out of that model's install receipt at export time.
        let weightsSha256: String?
        /// True only when the exporter watched the Neural Engine compile every
        /// variant and saw no compiler error. Core ML reports such a failure on
        /// the native stderr and still exits 0, so a sidecar from an exporter
        /// without this flag may silently run the whole prefill on the CPU.
        let aneCompileVerified: Bool?
        /// True when the exporter built this sidecar for a family whose
        /// full-attention layers pick keys with a sparse indexer, so the
        /// runtime has to fold that selection into the mask. Absent on a
        /// sidecar for a dense family; the runtime refuses to load a
        /// sparse-indexed model's sidecar that does not record it.
        let selectionFolded: Bool?
    }

    static let expectedVersion = 1
    /// -30000 underflows fp16 exp() exactly like -inf without putting
    /// infinity arithmetic into the ANE graph (whose fused SDPA op NaNs).
    static let maskNegative = Float16(-30000)

    /// Which mask the ANE path feeds the sidecar.
    ///
    /// `folded` is the shipped behaviour: a sparse-indexed model's key
    /// selection is folded into the additive mask, which is the only way the
    /// ANE computes the attention the model actually computes. `causal` is a
    /// **verification control, not a tuning knob** — it feeds the causal-only
    /// mask, which is *wrong* for such a model past its dense-exact window. It
    /// exists so an A/B can show the fold is load-bearing by measuring the
    /// causal arm diverge from the GPU path, rather than asserting it.
    enum MaskMode: String {
        case folded
        case causal

        static func environmentValue(
            _ environment: [String: String] = ProcessInfo.processInfo.environment
        ) throws -> MaskMode {
            guard let raw = environment["TINYTITAN_ANE_MASK"] else { return .folded }
            guard let mode = MaskMode(rawValue: raw) else {
                throw PrefillError.chunkedUnsupported(
                    "unsupported TINYTITAN_ANE_MASK '\(raw)'; allowed: folded, causal")
            }
            return mode
        }
    }

    /// The sidecar directory for a configured prefill chunk.
    ///
    /// A model may carry one sidecar per chunk width — `ane_prefill-1024`
    /// beside the historical `ane_prefill` (4,096) — because the width that
    /// wins depends on the prompt: 4,096 for long ones, a smaller chunk to
    /// reach the band below it at all. The configured chunk picks the
    /// directory; `init` then insists the sidecar found there was built for
    /// exactly that chunk, so a nearer width is refused rather than run.
    static func sidecarDirectory(modelDirectory: URL,
                                 configChunkTokens: Int) -> URL {
        let specific = modelDirectory.appendingPathComponent(
            "ane_prefill-\(configChunkTokens)", isDirectory: true)
        let meta = specific.appendingPathComponent("ane_prefill.json")
        if FileManager.default.fileExists(atPath: meta.path) {
            return specific
        }
        return modelDirectory.appendingPathComponent("ane_prefill",
                                                     isDirectory: true)
    }

    let chunkTokens: Int
    let histories: Set<Int>
    let coveredLayers: Set<Int>
    let maxPromptTokens: Int
    /// True when this model's full-attention layers select keys with a QSA
    /// indexer, so every chunk past the dense-exact window must be fed a mask
    /// with that selection folded in rather than the causal mask alone.
    let requiresSelection: Bool
    /// The visible-key count below which dense attention *is* the model's
    /// selection, from the model's own indexer geometry. A chunk that needs a
    /// selection and does not get one is refused rather than attended densely.
    let exactVisibleKeys: Int?
    /// Which mask the sidecar is fed; `.causal` only as a verification control.
    let maskMode: MaskMode

    /// Shared-mode staging: the GPU blits `normed` in, Core ML writes the
    /// three outputs back via output backings, the GPU quantizes K/V into the
    /// cache from the same memory, and the shadow append memcpys from it.
    let stagingNormed: MTLBuffer
    let stagingOut: MTLBuffer
    let stagingK: MTLBuffer
    let stagingV: MTLBuffer

    private let hiddenSize: Int
    private let kvDim: Int
    private let packageDir: URL
    private let compiledDir: URL
    /// At most one loaded model at a time. Each loaded function pins an
    /// E5RT/ANE inference arena (the h4096 variant's score tensors alone are
    /// ~1 GB); keeping 20 of them resident during a long prefill pressured
    /// the 8 GiB expert slot cache out of RAM and collapsed the decode that
    /// followed to ~2 tok/s. One-at-a-time bounds the ANE footprint to a
    /// single context at ~0.5 s reload cost per layer-chunk.
    private var residentModel: (layer: Int, history: Int, model: MLModel)?
    /// In-flight load of the *next* layer's model, started as soon as this
    /// layer's prediction returns so the ~0.5 s load overlaps the GPU's MoE
    /// stage instead of serializing in front of the next prediction. At most
    /// one is outstanding, which keeps the one-resident-arena rule intact:
    /// the preloaded model only becomes resident when `model(layer:history:)`
    /// adopts it, and that is the same moment the previous one is dropped.
    private var preloaded: (layer: Int, history: Int, task: Task<LoadedModelBox, Error>)?
    private var masks: [Int: MLMultiArray] = [:]
    private var maskStorage: [Int: UnsafeMutableRawPointer] = [:]
    /// The folded masks, one per history window, beside the causal ones: for a
    /// sparse-indexed model the mask depends on the *layer* as well, so the
    /// buffer is rewritten per covered layer. One chunk of one layer is live
    /// at a time in the chunk loop, so this is bounded by the same 33–134 MB
    /// per window the causal masks cost, not by layer count.
    private var selectionMasks: [Int: MLMultiArray] = [:]
    private var selectionMaskStorage: [Int: UnsafeMutableRawPointer] = [:]
    /// One all-`-30000` row per history window, the fold's reset.
    private var selectionNegativeRow: [Int: UnsafeMutableRawPointer] = [:]
    /// Token-major fp16 K/V rows per layer, at absolute prompt positions, so
    /// later chunks can attend to exact-precision history without
    /// re-dequantizing the cache. Allocated on the first append (single-chunk
    /// prompts never pay for it) and reused across requests.
    private var shadowK: [Int: UnsafeMutableRawPointer] = [:]
    private var shadowV: [Int: UnsafeMutableRawPointer] = [:]
    private(set) var shadowTokens = 0
    private var loggedFallback = false
    private var loggedCausalMask = false

    /// - Parameter weightsSha256: the model's own recorded `model_weights.bin`
    ///   digest, taken from its install receipt. A sidecar exported from
    ///   different weights computes plausible-looking but wrong attention, and
    ///   nothing downstream would catch it — so the binding is checked here
    ///   and fails closed. Nil skips the check (no receipt available) and says
    ///   so, rather than silently trusting.
    /// - Parameters family, fullAttentionLayerMask: the model the sidecar is
    ///   being loaded for. The sidecar's recorded geometry must match it; a
    ///   mismatch is refused rather than run, because a graph built for another
    ///   width or head count computes a different attention and says nothing.
    /// - Parameter configChunkTokens: the runtime's configured prefill chunk.
    ///   It selects *which* sidecar directory is loaded, and the one that is
    ///   found must be built for exactly this chunk — the gate below is a
    ///   contract with the graph's fixed shapes, so a nearer one is not usable.
    /// - Parameter maskMode: `.folded` everywhere except a verification run;
    ///   nil reads `TINYTITAN_ANE_MASK` (the tests pass it explicitly).
    init(modelDirectory: URL, device: MTLDevice,
         hiddenSize: Int, kvDim: Int, weightsSha256: String?,
         family: ModelFamily, fullAttentionLayerMask: [UInt8],
         sparseIndexer: SparseIndexerConfig,
         configChunkTokens: Int,
         maskMode: MaskMode? = nil) throws {
        self.maskMode = try maskMode ?? MaskMode.environmentValue()
        let dir = Self.sidecarDirectory(modelDirectory: modelDirectory,
                                        configChunkTokens: configChunkTokens)
        let metaURL = dir.appendingPathComponent("ane_prefill.json")
        guard FileManager.default.fileExists(atPath: metaURL.path) else {
            throw PrefillError.chunkedUnsupported(
                "TINYTITAN_PREFILL_ANE=on but \(metaURL.path) is missing; run "
                + "tools/export_ane_prefill.py --model \(modelDirectory.path) "
                + "--chunk \(configChunkTokens) for this model first")
        }
        let meta = try JSONDecoder().decode(
            SidecarMetadata.self, from: Data(contentsOf: metaURL))
        guard meta.version == Self.expectedVersion else {
            throw PrefillError.chunkedUnsupported(
                "ANE prefill sidecar version \(meta.version) != supported \(Self.expectedVersion); re-export")
        }
        // The graph's shapes are fixed by its chunk, so only a sidecar built
        // for the configured chunk can be fed. This is also what makes a
        // multi-width model safe: the chunk-specific directory is preferred,
        // and a fallback to the default one is refused here when it does not
        // match.
        guard meta.chunkTokens == configChunkTokens else {
            throw PrefillError.chunkedUnsupported(
                "ANE prefill sidecar in \(dir.lastPathComponent) is built for a "
                + "\(meta.chunkTokens)-token chunk but the runtime's prefill "
                + "chunk is \(configChunkTokens); export one for this width "
                + "(tools/export_ane_prefill.py --chunk \(configChunkTokens)) "
                + "or configure --prefill-chunk \(meta.chunkTokens)")
        }
        // A sparse-indexed model is served by folding its indexer's selection
        // into the mask the sidecar is fed (`fillSelectionMask`): the graph's
        // mask input is an *arbitrary* additive mask, so no part of the graph
        // changes and the same sidecar works either way. What has to be true is
        // that the runtime actually folds — dense attention matches the
        // selection only through `keptBlocks * compressRatio +
        // (compressRatio - 1)` visible keys (2,051 for the shipped Qwen 3.8
        // geometry), and the smallest chunk the ANE accepts is a full 4,096,
        // already past it. A sidecar that does not record the contract is
        // refused rather than trusted, because a causal-only mask attends to
        // keys the model drops, silently and with plausible output.
        let exactVisibleKeys = sparseIndexer.enabled
            ? QSAExactness(sparseIndexer).maximumExactVisibleKeys : nil
        if sparseIndexer.enabled {
            guard meta.selectionFolded == true else {
                throw PrefillError.chunkedUnsupported(
                    "ANE prefill sidecar does not record a folded sparse "
                    + "selection (selectionFolded is missing or false); with a "
                    + "causal-only mask the sidecar would attend to keys this "
                    + "model's indexer drops past \(exactVisibleKeys ?? 0) visible "
                    + "keys. Re-export with tools/export_ane_prefill.py.")
            }
        }
        // One geometry per sidecar. The graph's weights, head split, rope and
        // GQA expansion are all built from these numbers, so a sidecar that
        // disagrees with the model is not "close enough": it computes a
        // different attention. Refusing sends the runner to the GPU path.
        guard let geometry = meta.geometry else {
            throw PrefillError.chunkedUnsupported(
                "ANE prefill sidecar records no geometry (it predates the "
                + "generalized exporter); re-export it for this model")
        }
        guard geometry.family == family.rawValue,
              meta.family == family.rawValue else {
            throw PrefillError.chunkedUnsupported(
                "ANE prefill sidecar is for family '\(geometry.family)' but this "
                + "model is '\(family.rawValue)'; re-export it for this model")
        }
        guard geometry.hiddenSize == hiddenSize,
              geometry.numKVHeads * geometry.headDim == kvDim,
              geometry.chunkTokens == meta.chunkTokens else {
            throw PrefillError.chunkedUnsupported(
                "ANE prefill sidecar geometry (hidden \(geometry.hiddenSize), "
                + "\(geometry.numKVHeads)x\(geometry.headDim) kv, chunk "
                + "\(geometry.chunkTokens)) does not match this model (hidden "
                + "\(hiddenSize), kvDim \(kvDim), chunk \(meta.chunkTokens))")
        }
        for layer in meta.layers {
            guard layer >= 0, layer < fullAttentionLayerMask.count,
                  fullAttentionLayerMask[layer] == 1 else {
                throw PrefillError.chunkedUnsupported(
                    "ANE prefill sidecar covers layer \(layer), which is not a "
                    + "full-attention layer of this model")
            }
        }
        // Issue #7: a sidecar whose variants the ANE refused to compile loads
        // fine and then runs the whole prefill on the CPU, ~38x slower than the
        // GPU path. Only an exporter that verified the compilation sets this.
        // Refusing here means the GPU path is used instead (or, with
        // TINYTITAN_PREFILL_ANE=on, the load fails with the reason).
        guard meta.aneCompileVerified == true else {
            throw PrefillError.chunkedUnsupported(
                "ANE prefill sidecar does not record a verified ANE compilation "
                + "(aneCompileVerified is missing or false); the Neural Engine "
                + "may have refused it and prefill would run on the CPU at ~38x "
                + "the GPU cost. Re-export with tools/export_ane_prefill.py.")
        }
        if let weightsSha256, let exported = meta.weightsSha256 {
            guard exported.lowercased() == weightsSha256.lowercased() else {
                throw PrefillError.chunkedUnsupported(
                    "ANE prefill sidecar was exported from different weights "
                    + "(sidecar \(exported.prefix(12))..., model "
                    + "\(weightsSha256.prefix(12))...); re-export it for this model")
            }
        } else {
            // stderr for the same reason as the fallback notice below:
            // stdout is the generated text.
            FileHandle.standardError.write(Data(
                ("TinyTitan ane-prefill: sidecar/weights binding unverified "
                 + "(no receipt digest available); a stale sidecar would not "
                 + "be detected\n").utf8))
        }
        self.chunkTokens = meta.chunkTokens
        self.histories = Set(meta.histories)
        self.coveredLayers = Set(meta.layers)
        self.maxPromptTokens = (meta.histories.max() ?? 0) + meta.chunkTokens
        self.requiresSelection = sparseIndexer.enabled
        self.exactVisibleKeys = exactVisibleKeys
        self.hiddenSize = hiddenSize
        self.kvDim = kvDim
        self.packageDir = dir
        self.compiledDir = dir.appendingPathComponent("compiled-v\(meta.version)")
        try FileManager.default.createDirectory(
            at: compiledDir, withIntermediateDirectories: true)

        let halfBytes = MemoryLayout<Float16>.stride
        func staging(_ elements: Int, _ label: String) throws -> MTLBuffer {
            guard let made = device.makeBuffer(length: elements * halfBytes,
                                               options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            made.label = label
            return made
        }
        self.stagingNormed = try staging(meta.chunkTokens * hiddenSize,
                                         "ane.staging.normed")
        self.stagingOut = try staging(meta.chunkTokens * hiddenSize,
                                      "ane.staging.out")
        self.stagingK = try staging(meta.chunkTokens * kvDim, "ane.staging.k")
        self.stagingV = try staging(meta.chunkTokens * kvDim, "ane.staging.v")
    }

    deinit {
        for pointer in maskStorage.values { pointer.deallocate() }
        for pointer in selectionMaskStorage.values { pointer.deallocate() }
        for pointer in selectionNegativeRow.values { pointer.deallocate() }
        for pointer in shadowK.values { pointer.deallocate() }
        for pointer in shadowV.values { pointer.deallocate() }
    }

    /// Whether this chunk can run on the ANE. Continuity matters: a chunk at
    /// a nonzero start needs the shadow rows of every earlier chunk, so a
    /// resumed or partially GPU-processed prefill falls back for the rest of
    /// the request instead of attending to a hole.
    func eligibleChunk(startPosition: Int, tokenCount: Int,
                       configChunkTokens: Int) -> Bool {
        // A short prompt is one partial chunk; padding it to 4,096 costs
        // ~2 s of ANE work against under a second on the GPU, so the ANE
        // serves only full chunks and the continuation chunks of long
        // prompts — the workload it wins by 26x.
        let fullOrContinuation = tokenCount == chunkTokens || startPosition > 0
        guard configChunkTokens == chunkTokens,
              fullOrContinuation,
              tokenCount <= chunkTokens,
              startPosition % chunkTokens == 0,
              histories.contains(startPosition) else {
            if !loggedFallback {
                loggedFallback = true
                // stderr, not stdout: stdout carries generated tokens, and a
                // diagnostic written there lands in the middle of the model's
                // output. Harmless while ANE prefill was opt-in and this
                // never fired; corrupting once it became the default, which
                // is how the golden baselines caught it.
                FileHandle.standardError.write(Data(
                    ("TinyTitan ane-prefill fallback: chunk at \(startPosition) "
                     + "(+\(tokenCount)) outside sidecar coverage "
                     + "(chunk \(chunkTokens), max prompt \(maxPromptTokens)); "
                     + "using the GPU path\n").utf8))
            }
            return false
        }
        if startPosition == 0 {
            shadowTokens = 0
            return true
        }
        return shadowTokens == startPosition
    }

    private func model(layer: Int, history: Int) async throws -> MLModel {
        if let cached = residentModel,
           cached.layer == layer, cached.history == history {
            return cached.model
        }
        if let pending = preloaded, pending.layer == layer,
           pending.history == history {
            preloaded = nil
            // Drop the old arena only once the new model is in hand, then
            // adopt it — never two resident at once for longer than the
            // handover itself.
            let loaded = try await pending.task.value.model
            residentModel = (layer, history, loaded)
            return loaded
        }
        // A preload for a different layer is now useless; await and discard it
        // rather than leaking an arena behind the resident one.
        if let stale = preloaded {
            preloaded = nil
            _ = try? await stale.task.value
        }
        residentModel = nil
        let compiled = try await compiledModelURL(layer: layer)
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndNeuralEngine
        configuration.functionName = "h\(history)"
        let loaded = try MLModel(contentsOf: compiled,
                                 configuration: configuration)
        residentModel = (layer, history, loaded)
        return loaded
    }

    /// The on-disk compiled model for `layer`, compiling it from the package
    /// on first use and whenever the package is newer.
    private func compiledModelURL(layer: Int) async throws -> URL {
        let package = packageDir.appendingPathComponent("layer_\(layer).mlpackage")
        let compiled = compiledDir.appendingPathComponent("layer_\(layer).mlmodelc")
        let fm = FileManager.default
        func modifiedDate(_ url: URL) -> Date {
            (try? fm.attributesOfItem(atPath: url.path)[.modificationDate]
             as? Date) ?? .distantPast
        }
        if !fm.fileExists(atPath: compiled.path)
            || modifiedDate(compiled) < modifiedDate(package) {
            guard fm.fileExists(atPath: package.path) else {
                throw PrefillError.chunkedUnsupported(
                    "ANE prefill sidecar is missing \(package.lastPathComponent)")
            }
            let temporary = try await MLModel.compileModel(at: package)
            _ = try? fm.removeItem(at: compiled)
            try fm.moveItem(at: temporary, to: compiled)
        }
        return compiled
    }

    /// The compile-and-load half of `model(layer:history:)`, without touching
    /// `residentModel` — safe to run detached for a preload.
    private func loadModel(layer: Int, history: Int) async throws -> MLModel {
        let compiled = try await compiledModelURL(layer: layer)
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndNeuralEngine
        configuration.functionName = "h\(history)"
        return try MLModel(contentsOf: compiled, configuration: configuration)
    }

    /// Drops everything prefill allocated. Called when prefill hands over to
    /// decode so nothing ANE-side competes with the expert cache for
    /// residency; cheap no-op when nothing is loaded.
    ///
    /// The Core ML model is not the only thing that has to go. Decode runs
    /// entirely on the GPU and never reads the shadow rows or the masks, but
    /// they stay allocated for the life of this object unless dropped here:
    /// ~33 MB of shadow per covered layer plus one mask per history window
    /// (33 MB at h0, 100 MB at h8192) is roughly half a gigabyte taken from
    /// the expert slot cache for the whole of decode. Releasing only the
    /// model left that behind and cost measured decode throughput -- the
    /// bounded cache is the entire reason a model larger than RAM runs at
    /// all, so prefill scratch must not outlive prefill.
    ///
    /// Masks are rebuilt on the next prefill (a few tens of ms of fills
    /// against a prefill measured in minutes); shadow rows are per-request by
    /// construction, so `shadowTokens` resets with them and a later chunk
    /// correctly falls back rather than attending to a freed history.
    func releaseModels() {
        residentModel = nil
        preloaded?.task.cancel()
        preloaded = nil
        // Release the borrowing MLMultiArrays before the storage they point
        // at: they are built with `deallocator: nil`, so this dictionary owns
        // the memory.
        masks.removeAll()
        for pointer in maskStorage.values { pointer.deallocate() }
        maskStorage.removeAll()
        selectionMasks.removeAll()
        for pointer in selectionMaskStorage.values { pointer.deallocate() }
        selectionMaskStorage.removeAll()
        for pointer in selectionNegativeRow.values { pointer.deallocate() }
        selectionNegativeRow.removeAll()
        for pointer in shadowK.values { pointer.deallocate() }
        for pointer in shadowV.values { pointer.deallocate() }
        shadowK.removeAll()
        shadowV.removeAll()
        shadowTokens = 0
    }

    /// Starts loading `layer`'s model for `history` in the background, if it
    /// is not already resident or in flight. Called right after a prediction
    /// returns, so the load runs while the caller encodes and executes the
    /// layer's MoE stage on the GPU.
    func preload(layer: Int, history: Int) {
        if let cached = residentModel,
           cached.layer == layer, cached.history == history { return }
        if let pending = preloaded,
           pending.layer == layer, pending.history == history { return }
        preloaded?.task.cancel()
        preloaded = (layer, history, Task { [self] in
            LoadedModelBox(model: try await loadModel(layer: layer,
                                                      history: history))
        })
    }

    private func mask(history: Int) throws -> MLMultiArray {
        if let cached = masks[history] { return cached }
        let total = history + chunkTokens
        let count = chunkTokens * total
        let storage = UnsafeMutableRawPointer.allocate(
            byteCount: count * MemoryLayout<Float16>.stride,
            alignment: 16_384)
        let values = storage.bindMemory(to: Float16.self, capacity: count)
        for row in 0..<chunkTokens {
            let base = row * total
            let allowed = history + row + 1
            for column in 0..<allowed { values[base + column] = 0 }
            for column in allowed..<total {
                values[base + column] = Self.maskNegative
            }
        }
        let array = try MLMultiArray(
            dataPointer: storage,
            shape: [1, 1, NSNumber(value: chunkTokens), NSNumber(value: total)],
            dataType: .float16,
            strides: [NSNumber(value: count), NSNumber(value: count),
                      NSNumber(value: total), 1],
            deallocator: nil)
        maskStorage[history] = storage
        masks[history] = array
        return array
    }

    /// The additive mask for one chunk of one layer: `-30000` on every key the
    /// query must not read, `0` on every key it may.
    ///
    /// For a sparse-indexed model the causal mask alone is not the attention
    /// this model computes: past the dense-exact window the QSA indexer drops
    /// keys, and the GPU path reads exactly the kept ones. The sidecar's mask
    /// input is *arbitrary*, so folding the selection in here makes the ANE's
    /// softmax the GPU's gather — `exp(-30000)` underflows to zero in fp16
    /// exactly as an omitted key contributes nothing — and the graph does not
    /// change at all.
    ///
    /// The buffer is per history window and rewritten for every prediction:
    /// the selection is per layer *and* per prompt, so a cached fill cannot be
    /// reused across layers or requests. Refilling is one memcpy and at most
    /// `selectionWidth` stores per row — tens of milliseconds against a
    /// prefill measured in minutes.
    ///
    /// Rows past `tokenCount` are the chunk's padding (zero queries), which an
    /// all-masked row averages uniformly — finite, and discarded by `predict`.
    ///
    /// Internal rather than private: `ANEPrefillAttentionTests` checks the fold
    /// against a hand-built selection, and the fold is exactly the arithmetic a
    /// wrong mask would silently get wrong.
    func selectionMask(history: Int, tokenCount: Int,
                       selection: QSASelection) throws -> MLMultiArray {
        let halfBytes = MemoryLayout<Float16>.stride
        let total = history + chunkTokens
        let count = chunkTokens * total
        let array: MLMultiArray
        if let cached = selectionMasks[history] {
            array = cached
        } else {
            let storage = UnsafeMutableRawPointer.allocate(
                byteCount: count * halfBytes, alignment: 16_384)
            let negatives = UnsafeMutableRawPointer.allocate(
                byteCount: total * halfBytes, alignment: 16_384)
            let negativeValues = negatives.bindMemory(to: Float16.self,
                                                      capacity: total)
            for column in 0..<total { negativeValues[column] = Self.maskNegative }
            array = try MLMultiArray(
                dataPointer: storage,
                shape: [1, 1, NSNumber(value: chunkTokens), NSNumber(value: total)],
                dataType: .float16,
                strides: [NSNumber(value: count), NSNumber(value: count),
                          NSNumber(value: total), 1],
                deallocator: nil)
            selectionMaskStorage[history] = storage
            selectionNegativeRow[history] = negatives
            selectionMasks[history] = array
        }
        let storage = selectionMaskStorage[history]!
        let negativeRow = selectionNegativeRow[history]!.bindMemory(
            to: Float16.self, capacity: total)
        let values = storage.bindMemory(to: Float16.self, capacity: count)
        let indices = selection.indices.contents().bindMemory(
            to: UInt32.self, capacity: max(1, tokenCount * selection.indexStride))
        let counts = selection.counts.contents().bindMemory(
            to: UInt32.self, capacity: max(1, tokenCount))
        for row in 0..<chunkTokens {
            let base = row * total
            memcpy(UnsafeMutableRawPointer(values + base),
                   UnsafeRawPointer(negativeRow), total * halfBytes)
            guard row < tokenCount else { continue }
            // The compacted ascending selection the GPU's attention gathers:
            // the same keys, so the same softmax.
            let written = min(Int(counts[row]), selection.indexStride)
            let rowIndices = indices + row * selection.indexStride
            for slot in 0..<written {
                let key = Int(rowIndices[slot])
                guard key < total else { continue }
                values[base + key] = 0
            }
        }
        return array
    }

    private func wrap(_ buffer: MTLBuffer, rows: Int,
                      columns: Int) throws -> MLMultiArray {
        try MLMultiArray(dataPointer: buffer.contents(),
                         shape: [NSNumber(value: rows), NSNumber(value: columns)],
                         dataType: .float16,
                         strides: [NSNumber(value: columns), 1],
                         deallocator: nil)
    }

    private func wrapShadow(_ storage: UnsafeMutableRawPointer,
                            rows: Int) throws -> MLMultiArray {
        try MLMultiArray(dataPointer: storage,
                         shape: [NSNumber(value: rows), NSNumber(value: kvDim)],
                         dataType: .float16,
                         strides: [NSNumber(value: kvDim), 1],
                         deallocator: nil)
    }

    /// Runs one layer's attention block. `stagingNormed` must already hold
    /// the chunk's post-norm hidden rows; results land in `stagingOut` /
    /// `stagingK` / `stagingV` (real `tokenCount` rows; padding discarded).
    ///
    /// - Parameter selection: this layer's QSA key selection for this chunk, or
    ///   nil where the model has no indexer or every visible key is kept. A
    ///   sparse-indexed model past its dense-exact window must supply one: the
    ///   causal mask would otherwise attend to keys the model drops, so a
    ///   missing selection there is refused rather than run.
    func predict(layer: Int, history: Int, tokenCount: Int,
                 selection: QSASelection?) async throws {
        let halfBytes = MemoryLayout<Float16>.stride
        if tokenCount < chunkTokens {
            // Padded rows must be zeros: zero queries attend uniformly and
            // produce finite garbage that is discarded, whereas stale staging
            // bytes could push fp16 out of range.
            let start = tokenCount * hiddenSize * halfBytes
            let length = (chunkTokens - tokenCount) * hiddenSize * halfBytes
            memset(stagingNormed.contents().advanced(by: start), 0, length)
        }
        let maskFeature: MLMultiArray
        if let selection, maskMode == .folded {
            maskFeature = try selectionMask(history: history,
                                            tokenCount: tokenCount,
                                            selection: selection)
        } else {
            if requiresSelection, maskMode == .causal, let exact = exactVisibleKeys,
               history + tokenCount > exact, !loggedCausalMask {
                loggedCausalMask = true
                // stderr for the same reason as the fallback notice: stdout is
                // the generated text.
                FileHandle.standardError.write(Data(
                    ("TinyTitan ane-prefill: TINYTITAN_ANE_MASK=causal feeds the "
                     + "causal-only mask, which is WRONG for this sparse-indexed "
                     + "model past \(exact) visible keys; verification control "
                     + "only\n").utf8))
            }
            // No selection is only correct while every visible key is kept.
            // Past that the GPU path gathers the indexer's choice and the ANE
            // has to be fed the same one; a missing selection there is a caller
            // bug, not permission to attend densely.
            if requiresSelection, maskMode == .folded, let exact = exactVisibleKeys,
               history + tokenCount > exact {
                throw PrefillError.chunkedUnsupported(
                    "ANE prefill has no QSA selection for the chunk at "
                    + "\(history)+\(tokenCount) tokens, where this model's "
                    + "indexer drops keys past \(exact) visible ones; refusing to "
                    + "attend densely")
            }
            maskFeature = try mask(history: history)
        }
        var features: [String: MLMultiArray] = [
            "normed": try wrap(stagingNormed, rows: chunkTokens,
                               columns: hiddenSize),
            "mask": maskFeature,
        ]
        if history > 0 {
            guard let kShadow = shadowK[layer], let vShadow = shadowV[layer] else {
                throw PrefillError.chunkedUnsupported(
                    "ANE prefill shadow missing for layer \(layer) at history \(history)")
            }
            features["k_hist"] = try wrapShadow(kShadow, rows: history)
            features["v_hist"] = try wrapShadow(vShadow, rows: history)
        }
        let provider = try MLDictionaryFeatureProvider(
            dictionary: features.mapValues { MLFeatureValue(multiArray: $0) })
        let options = MLPredictionOptions()
        options.outputBackings = [
            "out": try wrap(stagingOut, rows: chunkTokens, columns: hiddenSize),
            "k_new": try wrap(stagingK, rows: chunkTokens, columns: kvDim),
            "v_new": try wrap(stagingV, rows: chunkTokens, columns: kvDim),
        ]
        let model = try await model(layer: layer, history: history)
        let result = try await model.prediction(from: provider, options: options)
        // Output backings are best-effort; copy back any output Core ML chose
        // to allocate elsewhere.
        try copyIfNotBacked(result, name: "out", buffer: stagingOut,
                            elements: chunkTokens * hiddenSize)
        try copyIfNotBacked(result, name: "k_new", buffer: stagingK,
                            elements: chunkTokens * kvDim)
        try copyIfNotBacked(result, name: "v_new", buffer: stagingV,
                            elements: chunkTokens * kvDim)
    }

    private func copyIfNotBacked(_ result: MLFeatureProvider, name: String,
                                 buffer: MTLBuffer, elements: Int) throws {
        guard let array = result.featureValue(for: name)?.multiArrayValue else {
            throw PrefillError.chunkedUnsupported(
                "ANE prefill output '\(name)' missing from prediction")
        }
        array.withUnsafeBytes { bytes in
            guard let base = bytes.baseAddress,
                  base != buffer.contents() else { return }
            memcpy(buffer.contents(), base,
                   elements * MemoryLayout<Float16>.stride)
        }
    }

    /// Saves the chunk's K/V rows for later chunks. Partial chunks are always
    /// the last chunk of a prompt, so their rows can never be history and the
    /// shadow (33 MB per layer) is never allocated for single-chunk prompts.
    func appendShadow(layer: Int, startPosition: Int, tokenCount: Int) {
        guard tokenCount == chunkTokens else { return }
        let rowBytes = kvDim * MemoryLayout<Float16>.stride
        let capacityBytes = maxPromptTokens * rowBytes
        if shadowK[layer] == nil {
            shadowK[layer] = .allocate(byteCount: capacityBytes, alignment: 16_384)
            shadowV[layer] = .allocate(byteCount: capacityBytes, alignment: 16_384)
        }
        let offset = startPosition * rowBytes
        let length = tokenCount * rowBytes
        memcpy(shadowK[layer]!.advanced(by: offset),
               stagingK.contents(), length)
        memcpy(shadowV[layer]!.advanced(by: offset),
               stagingV.contents(), length)
    }

    /// Marks the chunk's shadow rows visible to the next chunk. Called once
    /// after every covered layer appended, so a thrown mid-chunk error leaves
    /// `shadowTokens` behind `startPosition` and the next attempt falls back
    /// to the GPU instead of attending to partial history.
    func finishChunk(startPosition: Int, tokenCount: Int) {
        shadowTokens = tokenCount == chunkTokens
            ? startPosition + tokenCount : 0
        if tokenCount < chunkTokens {
            // A partial chunk is the prompt's last: decode is next.
            releaseModels()
        }
    }
}
