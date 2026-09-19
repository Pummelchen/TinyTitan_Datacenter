import Foundation
import Metal

/// Streaming callbacks from `runRawCompletion`. `.prefill` reports monotonic
/// producer-defined prompt progress; scalar replay reports per token, while a
/// prefill-capable producer may report per internal chunk. `.token` fires per
/// decoded non-stop token; `.tail` carries the detokenizer flush remainder at a
/// stop boundary.
public enum RawDecodeProgress: Sendable {
    case prefill(done: Int, total: Int)
    case token(index: Int, id: Int32, delta: String)
    case tail(String)
}

public enum RawCompletionStart: Sendable, Equatable {
    case reset
    case resume(cachedPromptTokens: Int)
}

public struct RawDecodeResult: Sendable {
    public let prefillTokens: Int
    public let cachedPromptTokens: Int
    public let computedPrefillTokens: Int
    public let prefillSeconds: Double
    public let newTokens: Int
    public let decodeSeconds: Double
    public let reason: StopReason
    public let kvPosition: Int
    public let kvBackedTokenIDs: [Int32]
    public let uncommittedBoundaryTokenIDs: [Int32]
}

/// Preallocated per-generation buffers (two 512 KiB vocab buffers plus a token
/// slot) and sampler. A warm session reuses them for every token, avoiding
/// per-token Metal buffer allocation.
///
/// unchecked-invariant: the buffers and sampler are exclusively owned by one
/// generation at a time — the single-in-flight guard upstream is the contract.
public struct RawCompletionScratch: @unchecked Sendable {
    let logits: MTLBuffer
    let probs: MTLBuffer
    let outToken: MTLBuffer
    let sampler: Sampler

    public init(context: MetalContext, vocab: Int, logitSoftcap: Float = 0.0) throws {
        guard let logits = context.device.makeBuffer(length: vocab * MemoryLayout<Float16>.size,
                                                     options: .storageModeShared),
              let probs = context.device.makeBuffer(length: vocab * MemoryLayout<Float16>.size,
                                                    options: .storageModeShared),
              let outToken = context.device.makeBuffer(length: MemoryLayout<UInt32>.size,
                                                       options: .storageModeShared)
        else {
            throw ModelError.residentBufferWrapFailed
        }
        self.logits = logits
        self.probs = probs
        self.outToken = outToken
        self.sampler = try Sampler(context: context, vocab: vocab,
                                   logitSoftcap: logitSoftcap)
    }
}

extension GenerationConfig {
    /// A pure-greedy config can use the fused head's GPU argmax
    /// (`RealForwardRunner.lastGreedyToken`) instead of sampling from the
    /// logits buffer. Anything else needs real logits.
    public var isPureGreedy: Bool {
        temperature == 0 && presencePenalty == 0 && repetitionPenalty == 1
    }

}

/// Raw-completion prefill + decode loop shared by the CLI and the Mac app.
/// Consumes pre-encoded `promptIds` (BOS + verbatim encode upstream — no chat
/// template). Stop handling, detokenizer flush ordering, and history append
/// ordering are shared by both front ends.
///
/// When the producer runs the fused lm_head (`RealForwardRunner` default) the
/// logits buffer is never written; the loop then requires a pure-greedy config
/// and reads `lastGreedyToken`. Callers with sampling configs must construct
/// the runner with `forceLogitsHead: true`.
/// lint:allow-long the generation loop: continuation validation, the prefill
/// mode switch, then token-by-token decode with stop matching and progress
/// reporting. The loop body reads and writes the same half-dozen pieces of
/// decode state on every iteration, so splitting it would thread that state
/// back through parameters on every call.
public func runRawCompletion(producer: any LogitProducer,
                             tokenizer: GFTokenizer,
                             promptIds: [Int32],
                             config: GenerationConfig,
                             context: MetalContext,
                             scratch: RawCompletionScratch,
                             prefillConfig: PrefillRuntimeConfig = .defaultChunked,
                             start: RawCompletionStart = .reset,
                             slot: Int = 0,
                             shouldStop: () -> Bool = { false },
                             onProgress: (RawDecodeProgress) -> Void) async throws -> RawDecodeResult {
    if let mtp = producer as? StreamingMTPDecoder {
        // One draft decoder drafts for one sequence; a batched slot has no MTP
        // state of its own yet (see the plan's MTP note).
        guard slot == 0 else {
            throw GeneratorError.invalidGenerationConfig(
                "the MTP decode path is single-sequence; slot \(slot) is not supported")
        }
        // A grammar is a per-token contract with the sampler, and the MTP path
        // drafts several tokens ahead of it; its own sampling never consults a
        // mask. Serving a schema through MTP would emit unconstrained tokens,
        // so the request takes the ordinary path instead.
        guard config.constraint == nil else {
            throw GeneratorError.invalidGenerationConfig(
                "constrained decoding does not support the MTP decode path")
        }
        return try await runStreamingMTPCompletion(
            decoder: mtp,
            tokenizer: tokenizer,
            promptIds: promptIds,
            config: config,
            scratch: scratch,
            prefillConfig: prefillConfig,
            start: start,
            shouldStop: shouldStop,
            onProgress: onProgress)
    }
    try config.validate()
    guard !promptIds.isEmpty else {
        throw GeneratorError.emptyPrompt
    }
    let fusedRunner = producer as? RealForwardRunner
    let fusedGreedy = fusedRunner?.usesFusedGreedyHead == true
    guard !fusedGreedy || config.isPureGreedy else {
        throw PrefillError.unsupportedPrefillSeed(
            "the fused-head producer cannot serve this sampling configuration; use a logits head")
    }
    // The fused head picks its token without ever writing the logits buffer a
    // mask would edit, so a constrained request must take the logits path.
    guard !fusedGreedy || config.constraint == nil else {
        throw GeneratorError.invalidGenerationConfig(
            "constrained decoding needs the logits head; the fused greedy head cannot be masked")
    }

    let cachedPromptTokens: Int
    switch start {
    case .reset:
        cachedPromptTokens = 0
    case .resume(let count):
        guard count > 0, count < promptIds.count else {
            throw GeneratorError.invalidContinuation(
                "cached prompt token count must be greater than zero and less than the effective prompt")
        }
        guard producer is any ContinuableLogitProducer else {
            throw GeneratorError.invalidContinuation(
                "producer does not support continuation")
        }
        cachedPromptTokens = count
    }
    let computedPrefillTokens = promptIds.count - cachedPromptTokens

    var detok = GFDetokenizer(tokenizer: tokenizer)
    var history = Array(promptIds.prefix(cachedPromptTokens))
    history.reserveCapacity(promptIds.count + config.maxNewTokens)

    if let context = producer as? any ContextWindowReporting {
        // A resume already occupies `cachedPromptTokens` KV rows, so only the
        // uncached prompt plus the response is new work — it must fit the
        // remaining capacity. Algebraically this is the final-KV-position
        // bound (`promptIds.count + maxNewTokens <= maxContext`); written in
        // remaining-capacity form so near-maxContext continuations are not
        // over-rejected (R9).
        let newRows = (promptIds.count - cachedPromptTokens) + config.maxNewTokens
        let remainingCapacity = context.maxContext - cachedPromptTokens
        if newRows > remainingCapacity {
            throw GeneratorError.contextOverflow(prompt: promptIds.count,
                                                 maxNew: config.maxNewTokens,
                                                 maxContext: context.maxContext)
        }
    }
    switch start {
    case .reset:
        await producer.resetSequence(slot: slot)
    case .resume:
        // Re-derive the conformance rather than force-cast on the guard 30
        // lines above: a trap here would take down the server process, and the
        // invariant is far enough away to be broken by an unrelated edit.
        guard let continuable = producer as? any ContinuableLogitProducer else {
            throw GeneratorError.invalidContinuation(
                "producer does not support continuation")
        }
        try continuable.prepareForContinuation(expectedPosition: cachedPromptTokens)
    }
    let prefillStart = Date()
    var position = cachedPromptTokens
    var prefillSeed: PrefillSeed?
    let prefillTokens = promptIds[cachedPromptTokens...]
    // Only slot 0 can use chunked prefill: that path writes slot 0's KV region
    // (slot-aware chunked prefill is not implemented yet). Another slot prefills
    // by running its prompt through the decode step, which is slot-aware. Slot 0
    // keeps the chunked fast path, so the single-sequence behaviour is unchanged.
    let prefillMode: PrefillRuntimeConfig.Mode = prefillConfig.mode
    switch prefillMode {
    case .chunked where producer is any ChunkedPrefillRunner:
        // lint:allow-force the `where` clause one line above is the guard; a
        // producer without the conformance falls through to plain `.chunked`.
        let chunked = producer as! any ChunkedPrefillRunner
        let mode: PrefillOutputMode = fusedGreedy ? .greedyIfAvailable : .logits
        let result = try await chunked.prefillChunked(tokens: prefillTokens,
                                                      startPosition: position,
                                                      slot: slot,
                                                      outputMode: mode,
                                                      config: prefillConfig,
                                                      into: scratch.logits) { done in
            onProgress(.prefill(done: cachedPromptTokens + done, total: promptIds.count))
        }
        if mode == .logits, result.seed != .logitsWritten {
            throw PrefillError.unsupportedPrefillSeed(
                "RawCompletion chunked prefill requested logits but producer returned \(result.seed)")
        }
        if case .greedyToken = result.seed, !config.isPureGreedy {
            throw PrefillError.unsupportedPrefillSeed(
                "RawCompletion chunked prefill returned a greedy token for a sampling config")
        }
        position = result.newPosition
        prefillSeed = result.seed
        history.append(contentsOf: prefillTokens)
    case .chunked:
        throw PrefillError.chunkedUnsupported(
            PrefillError.chunkedRequiresChunkedRunnerReason)
    case .off:
        for t in prefillTokens {
            try Task.checkCancellation()
            try await producer.produce(token: t, position: position, slot: slot,
                                       into: scratch.logits)
            position += 1
            history.append(t)
            onProgress(.prefill(done: position, total: promptIds.count))
        }
    }

    let decodeStart = Date()
    let prefillSeconds = decodeStart.timeIntervalSince(prefillStart)
    // The scratch sampler persists across generations; its incremental
    // repetition-penalty history is per-generation (R25).
    scratch.sampler.resetPenaltyHistory()
    var stopMatcher = StreamingStopMatcher(stops: config.stopStrings)
    var generated = 0
    var reason: StopReason = .maxTokens
    var uncommittedBoundaryTokenIDs: [Int32] = []
    var toolCallMarkers = ToolCallMarkerCounter()
    var loopMark = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)

    /// THE LOOKAHEAD'S CARRY (D411). The token this stage needs at step N is its peer's token from step N-1, which
    /// the peer emitted at the END of its own step N-1 - so it can be fetched at the end of this iteration instead of
    /// blocking at the start of the next one. D410 measured that blocking order at 207 ms of every 295 ms step on
    /// the first stage, against 87.5 ms of its own work.
    var pendingIncoming: Int32? = nil

    while true {
        try Task.checkCancellation()

        let tokenID: Int32
        let tSample = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        if generated == 0, let seed = prefillSeed {
            switch seed {
            case .greedyToken(let token):
                tokenID = Int32(bitPattern: token)
            case .logitsWritten:
                tokenID = try sampleOnce(scratch: scratch, context: context,
                                     history: history, config: config, position: generated,
                                     timing: fusedRunner)
            }
        } else if fusedGreedy {
            tokenID = Int32(bitPattern: fusedRunner!.lastGreedyToken)
        } else {
            tokenID = try sampleOnce(scratch: scratch, context: context,
                                 history: history, config: config, position: generated,
                                 timing: fusedRunner)
        }
        fusedRunner?.totalLoopSampleNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tSample
        generated += 1
        // The mask was built from the state before this token; move the
        // grammar over it now, so the next position's mask is the next
        // position's. A rejection here cannot be the model's fault -- the
        // sampler only ever saw allowed ids -- so it is reported, never
        // shrugged off, exactly like an out-of-range id.
        if let constraint = config.constraint, !constraint.observe(tokenID) {
            throw GeneratorError.constrainedDecodeViolation(id: tokenID)
        }
        uncommittedBoundaryTokenIDs = [tokenID]
        toolCallMarkers.observe(tokenID,
                                start: tokenizer.toolCallStartID,
                                end: tokenizer.toolCallEndID)

        if tokenizer.stopTokenIDs.contains(tokenID) || config.extraStopTokens.contains(tokenID) {
            if tokenID == tokenizer.endOfTurnID {
                // The stop token says the turn ended, not why: `<|im_end|>`
                // closes both a prose answer and a tool call. `toolCalls` is
                // the reason the callers branch on, so it has to come from the
                // turn's own tokens.
                reason = toolCallMarkers.isCompleteToolTurn ? .toolCalls : .endOfTurn
            } else {
                reason = .eos
            }
            let tail = stopMatcher.push(detok.flush()) + stopMatcher.finish()
            if !tail.isEmpty { onProgress(.tail(tail)) }
            break
        }

        let delta = try detok.push(tokenID)
        let visible = stopMatcher.push(delta)
        let tProgress = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        onProgress(.token(index: generated - 1, id: tokenID, delta: visible))
        fusedRunner?.totalLoopProgressNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tProgress

        let hitStopString = stopMatcher.isStopped || shouldStop()
        let hitMax = generated >= config.maxNewTokens
        if hitStopString || hitMax {
            let tail = stopMatcher.push(detok.flush()) + stopMatcher.finish()
            if !tail.isEmpty { onProgress(.tail(tail)) }
            if hitStopString {
                // A configured stop string truncates output; the caller's
                // external stop signal reports `.external` instead (R35).
                reason = stopMatcher.isStopped ? .stopString : .external
            } else {
                reason = .maxTokens
            }
            break
        }

        history.append(tokenID)
        // Everything since the previous produce returned that is neither the
        // sampler nor the callback: stop matching, detokenizing, bookkeeping.
        if let fusedRunner {
            let now = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            fusedRunner.totalLoopOtherNanos &+= now - loopMark
        }
        // THE RING'S BACKWARD EDGE, USED (D361). A stage that SAMPLES publishes the token it chose; a stage that
        // EMBEDS replaces the token it would have used with the one that came back. Only one of the two hooks is
        // non-nil on any given stage - the last stage connects back and the first one listens - so calling both is
        // safe and neither needs a role check.
        //
        // The order matters and is the reason this is one line rather than two: the sink carries what was SAMPLED,
        // so it must run before the source overwrites `tokenID`. Doing it the other way would publish the token
        // this stage was told to use as though it had chosen it.
        // A CAST RATHER THAN A PROTOCOL MEMBER, deliberately. `producer` is `any LogitProducer`, and putting the two
        // hooks on that protocol would oblige every conformer and every fake to implement them - which is precisely
        // the cost D359 recorded, where a protocol change left a stale fake breaking the suite for four rounds.
        // These two properties belong to the ring, not to the act of producing logits.
        var stepToken = tokenID
        if let ring = producer as? RealForwardRunner {
            // THE SINK STAYS EARLY AND THE SOURCE MOVES TO THE END OF THE ITERATION (D411). The sink must run before
            // anything can overwrite `tokenID`, because it carries what this stage SAMPLED - and at this point
            // `tokenID` is still the token the sampler chose at the end of the previous iteration, so publishing it
            // here is both correct and as early as the peer could possibly want it.
            // RELAY THE HEAD'S CHOICE, DO NOT RE-SAMPLE IT (D426). A stage's own sample is meaningless to a chain -
            // only the head's token decides anything, and every other stage's job on the reverse edge is to pass it
            // along. Publishing `tokenID` here made a middle stage emit a token it had chosen itself, so the first
            // stage received something that had been through a sampler it did not need: the extra hop then cost a
            // whole stage step (48.7 ms) instead of a wire crossing (2.4 ms for 12 KB), which is the 88 ms D425
            // measured. When this stage was TOLD a token, that token is the one to forward.
            //
            // The head still publishes its own sample, because it is the stage that chooses and `pendingIncoming` is
            // nil there. The first stage never publishes at all, having no sink.
            // THE HEAD PUBLISHES HERE; A RELAYING STAGE PUBLISHES ON RECEIPT (D428). A stage with no source is the
            // one that chooses, so its own sample is the token the chain needs and there is nothing to wait for. A
            // stage with a source has nothing of its own to publish - it forwards what it is given, below, the
            // moment it is given it.
            if ring.nextTokenSource == nil { ring.nextTokenSink?(tokenID, 0) }
            // The token to produce with arrived during the PREVIOUS iteration's work (below), not now.
            if let carried = pendingIncoming {
                stepToken = carried; pendingIncoming = nil
            } else if let source = ring.nextTokenSource {
                // THE FIRST DECODE STEP HAS NOTHING CARRIED (D432). The lookahead fetches at the END of an iteration,
                // so on the very first one there is no previous iteration to have fetched - and a first stage would
                // produce from its OWN sampler's token instead of the one the head chose. The trace showed exactly
                // that: A published its position-5 state BEFORE it was ever told ` Paris`, so the state it sent
                // downstream was computed from a token nobody selected. Token 1 stayed right because it comes from
                // the prefill's logits; every token after it was computed from that wrong state.
                let first = source(position)
                if first >= 0 { stepToken = first }
            }
        }
        try await producer.produce(token: stepToken, position: position, slot: slot,
                                   into: scratch.logits)
        // FETCH FOR THE NEXT STEP, WHILE THE PEER IS STILL WORKING ON THIS ONE. This is the whole reordering: the
        // receive is the same call it always was, moved to the other end of the iteration, so that the wait overlaps
        // the peer's step rather than preceding this stage's own.
        if let ring = producer as? RealForwardRunner, let source = ring.nextTokenSource {
            let incoming = source(position + 1)
            // -1 is the sentinel for "nothing has come back yet" (D361), and it is distinguishable from 0
            // because 0 is a legitimate token id.
            if incoming >= 0 {
                pendingIncoming = incoming
                // FORWARD IT NOW, NOT NEXT ITERATION (D428). Waiting until the top of the next loop held the head's
                // token for a whole stage step before the stage behind us saw it - which is the 59 ms D427 measured
                // as residual. A stage whose sink is nil (the first) makes this a no-op, and the head never reaches
                // this branch because it has no source.
                ring.nextTokenSink?(incoming, 0)
            }
        }
        loopMark = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        position += 1
        uncommittedBoundaryTokenIDs.removeAll(keepingCapacity: true)
    }

    return RawDecodeResult(prefillTokens: promptIds.count,
                           cachedPromptTokens: cachedPromptTokens,
                           computedPrefillTokens: computedPrefillTokens,
                           prefillSeconds: prefillSeconds,
                           newTokens: generated,
                           decodeSeconds: Date().timeIntervalSince(decodeStart),
                           reason: reason,
                           kvPosition: position,
                           kvBackedTokenIDs: history,
                           uncommittedBoundaryTokenIDs: uncommittedBoundaryTokenIDs)
}

private func runStreamingMTPCompletion(
    decoder: StreamingMTPDecoder,
    tokenizer: GFTokenizer,
    promptIds: [Int32],
    config: GenerationConfig,
    scratch: RawCompletionScratch,
    prefillConfig: PrefillRuntimeConfig,
    start: RawCompletionStart,
    shouldStop: () -> Bool,
    onProgress: (RawDecodeProgress) -> Void
) async throws -> RawDecodeResult {
    try config.validate()
    guard config.isPureGreedy else { throw StreamingMTPError.greedyOnly }
    guard case .reset = start else {
        throw GeneratorError.invalidContinuation(
            "MTP continuation snapshots are not yet persisted; start a fresh request")
    }
    guard !promptIds.isEmpty else { throw GeneratorError.emptyPrompt }

    let prefillStart = Date()
    var boundary = try await decoder.prepare(
        promptIds: promptIds,
        config: config,
        prefillConfig: prefillConfig,
        logits: scratch.logits) { done in
            onProgress(.prefill(done: done, total: promptIds.count))
        }
    let decodeStart = Date()
    let prefillSeconds = decodeStart.timeIntervalSince(prefillStart)

    var detok = GFDetokenizer(tokenizer: tokenizer)
    var stopMatcher = StreamingStopMatcher(stops: config.stopStrings)
    var generated = 0
    var reason: StopReason = .maxTokens
    var backedHistory = promptIds
    var uncommitted: [Int32] = []
    var pending: [(token: Int32, backed: Bool)] = [(boundary, false)]
    var toolCallMarkers = ToolCallMarkerCounter()

    decodeLoop: while true {
        while !pending.isEmpty {
            try Task.checkCancellation()
            let item = pending.removeFirst()
            boundary = item.token
            generated += 1
            // Mirrors the scalar loop: `uncommitted` holds the last emitted
            // token iff advance has not yet committed it to the target KV
            // (R10). A token reported backed by the batch is already in the
            // KV, so it never sits uncommitted.
            uncommitted = item.backed ? [] : [item.token]
            toolCallMarkers.observe(item.token,
                                    start: tokenizer.toolCallStartID,
                                    end: tokenizer.toolCallEndID)

            if tokenizer.stopTokenIDs.contains(item.token)
                || config.extraStopTokens.contains(item.token) {
                // Same classification as the scalar loop above.
                if item.token == tokenizer.endOfTurnID {
                    reason = toolCallMarkers.isCompleteToolTurn ? .toolCalls : .endOfTurn
                }
                else { reason = .eos }
                let tail = stopMatcher.push(detok.flush()) + stopMatcher.finish()
                if !tail.isEmpty { onProgress(.tail(tail)) }
                break decodeLoop
            }
            let visible = stopMatcher.push(try detok.push(item.token))
            onProgress(.token(index: generated - 1, id: item.token, delta: visible))
            let hitStop = stopMatcher.isStopped || shouldStop()
            let hitMax = generated >= config.maxNewTokens
            if hitStop || hitMax {
                let tail = stopMatcher.push(detok.flush()) + stopMatcher.finish()
                if !tail.isEmpty { onProgress(.tail(tail)) }
                if hitStop {
                    reason = stopMatcher.isStopped ? .stopString : .external
                } else {
                    reason = .maxTokens
                }
                break decodeLoop
            }
        }

        // The boundary is reported backed only after the advance that commits
        // it to the target KV succeeds (R10): "reported backed" strictly means
        // "committed by a completed advance", so the final boundary token is
        // always accounted for in `kvBackedTokenIDs`.
        let batch = try await decoder.advance(boundaryToken: boundary)
        backedHistory.append(boundary)
        pending = batch.tokenIDs.enumerated().map { index, token in
            (token, index < batch.backedPrefixCount)
        }
        if batch.backedPrefixCount > 0 {
            backedHistory.append(contentsOf: batch.tokenIDs.prefix(batch.backedPrefixCount))
        }
    }

    return RawDecodeResult(
        prefillTokens: promptIds.count,
        cachedPromptTokens: 0,
        computedPrefillTokens: promptIds.count,
        prefillSeconds: prefillSeconds,
        newTokens: generated,
        decodeSeconds: Date().timeIntervalSince(decodeStart),
        reason: reason,
        kvPosition: decoder.targetPosition,
        kvBackedTokenIDs: backedHistory,
        uncommittedBoundaryTokenIDs: uncommitted)
}

/// Samples one token id.
///
/// `timing` is the runner whose `TINYTITAN_KERNEL_STATS` timeline this command
/// buffer joins, when the producer is one. Without it the sampler's GPU span
/// is invisible to the role summary *and* to the gap accounting, so it lands
/// inside the `head_logits->embed` transition and inflates what reads as idle.
/// That is not hypothetical: it hid a 15.45 ms/token Top-K kernel until the
/// gap was traced by hand.
private func sampleOnce(scratch: RawCompletionScratch, context: MetalContext,
                        history: [Int32], config: GenerationConfig, position: Int,
                        timing: RealForwardRunner? = nil) throws -> Int32 {
    guard let cb = context.queue.makeCommandBuffer() else {
        throw ModelError.residentBufferWrapFailed
    }
    try scratch.sampler.sample(commandBuffer: cb, logits: scratch.logits, probs: scratch.probs,
                               history: history, config: config, position: position,
                               outToken: scratch.outToken)
    cb.commit(); cb.waitUntilCompleted()
    timing?.recordKernelGPU(role: "sample", cb)
    // Read after completion, while the row max is still the one this dispatch
    // wrote. A row with no finite logit leaves the sampler's in-range fallback
    // in `outToken`; returning it would report a broken model as a valid token,
    // and the generation would then feed that token back and loop on it.
    guard scratch.sampler.lastRowHadFiniteLogit else {
        throw GeneratorError.degenerateLogitsRow
    }
    return try validatedToken(scratch.outToken.contents().load(as: UInt32.self),
                              vocab: scratch.sampler.vocab)
}

/// Checks a sampled id against the vocabulary before anything uses it.
///
/// The sampler's contract is an in-range id and its tests pin that, but an
/// out-of-range one has been seen intermittently under full-suite GPU load and
/// never identified. A token id indexes the embedding table and extends the KV
/// history, so an unchecked one is silent corruption exactly like the traps this
/// audit converted into reports; this makes it a named error instead, with the
/// id in it, so the next occurrence identifies itself.
func validatedToken(_ raw: UInt32, vocab: Int) throws -> Int32 {
    guard raw < UInt32(vocab) else {
        throw GeneratorError.samplerReturnedOutOfRangeToken(id: raw, vocab: vocab)
    }
    return Int32(bitPattern: raw)
}
