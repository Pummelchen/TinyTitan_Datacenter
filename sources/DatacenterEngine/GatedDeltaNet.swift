import Foundation

/// The Gated DeltaNet's geometry, as the kernel needs it.
///
/// Kept separate from the IR's `ModelConfig` — which carries optional fields for every
/// family — because this is a numeric kernel's parameter block and unwrapping optionals in
/// its inner loops would be both slower and less clear. The forward pass adapts one to the
/// other once, at the boundary, and the tests pin both.
public struct GatedDeltaNetShape: Sendable, Equatable {
    public let hiddenSize: Int
    public let keyHeads: Int
    public let valueHeads: Int
    public let keyHeadDim: Int
    public let valueHeadDim: Int
    public let convKernel: Int
    public let eps: Float

    public init(
        hiddenSize: Int, keyHeads: Int, valueHeads: Int, keyHeadDim: Int, valueHeadDim: Int,
        convKernel: Int, eps: Float
    ) {
        self.hiddenSize = hiddenSize
        self.keyHeads = keyHeads
        self.valueHeads = valueHeads
        self.keyHeadDim = keyHeadDim
        self.valueHeadDim = valueHeadDim
        self.convKernel = convKernel
        self.eps = eps
    }

    public var keyDim: Int { keyHeads * keyHeadDim }
    public var valueDim: Int { valueHeads * valueHeadDim }
    public var convDim: Int { keyDim * 2 + valueDim }
}

/// The layer's weights, by role. The forward pass fills this from the IR, so nothing here
/// knows a tensor name.
public struct GatedDeltaNetWeights: Sendable {
    public let inQKV: [Float]     // [convDim, hidden]
    public let inZ: [Float]       // [valueDim, hidden]
    public let inB: [Float]       // [valueHeads, hidden]
    public let inA: [Float]       // [valueHeads, hidden]
    public let conv: [Float]      // [convDim, 1, kernel]
    public let aLog: [Float]      // [valueHeads], stored fp32 in the checkpoint
    public let dtBias: [Float]    // [valueHeads]
    public let norm: [Float]      // [valueHeadDim], stored fp32
    public let outProj: [Float]   // [hidden, valueDim]

    /// The three int4 projections in their **stored** form, when the source had one (`D111`).
    ///
    /// When one of these is set its `[Float]` twin is **empty**, because the decode it would hold is exactly
    /// the cost this avoids. `projection` below is the only thing allowed to read either, and it refuses to
    /// multiply an empty array — a silently empty weight would multiply as zeros and still look like a tensor.
    public let packedInQKV: PackedInt4Rows?
    public let packedInZ: PackedInt4Rows?
    public let packedOut: PackedInt4Rows?

    public init(
        inQKV: [Float], inZ: [Float], inB: [Float], inA: [Float], conv: [Float],
        aLog: [Float], dtBias: [Float], norm: [Float], outProj: [Float],
        packedInQKV: PackedInt4Rows? = nil, packedInZ: PackedInt4Rows? = nil,
        packedOut: PackedInt4Rows? = nil
    ) {
        self.packedInQKV = packedInQKV
        self.packedInZ = packedInZ
        self.packedOut = packedOut
        self.inQKV = inQKV
        self.inZ = inZ
        self.inB = inB
        self.inA = inA
        self.conv = conv
        self.aLog = aLog
        self.dtBias = dtBias
        self.norm = norm
        self.outProj = outProj
    }
}

/// The layer, in the contract's order.
///
/// Transcribed from `transformers` v5.17.0 `models/qwen3_5/modeling_qwen3_5.py`
/// (`Qwen3_5GatedDeltaNet:550`, `torch_chunk_gated_delta_rule:301`, `l2norm:294`,
/// `Qwen3_5RMSNormGated:225`) and recorded step by step in
/// `docs/reference-qwen35-2b.md`. The Python contract in `tools/ordered_qwen35.py` is the
/// bit-exactness target, and `GatedDeltaNetTests` asserts these functions against golden
/// bit patterns emitted from it.
///
/// Every contraction is an ascending-index fp32 accumulation with no reassociation and no
/// fused multiply-add; every transcendental is evaluated in double and rounded to `Float`.
public enum GatedDeltaNet {
    public static let chunkSize = 64

    /// `F.softplus` with torch's default threshold, in the contract's precision.
    @inline(__always)
    public static func softplus(_ x: Float) -> Float {
        let scaled = x * 1.0
        if scaled > 20.0 { return x }
        return Float(log1p(exp(Double(scaled))) / 1.0)
    }

    /// `x * rsqrt((x·x).sum + eps)`, with the sum in ascending order. The contract states
    /// `1/sqrt`, since `rsqrt` is not required to be correctly rounded.
    static func l2norm(_ values: [Float], count: Int, width: Int, eps: Float, rowOffset: Int) -> [Float] {
        var result = [Float](repeating: 0, count: count * width)
        for row in 0..<count {
            let base = rowOffset + row * width
            var squares: [Float] = []
            squares.reserveCapacity(width)
            for index in 0..<width {
                let value = values[base + index]
                squares.append(value * value)
            }
            let total = Ops.orderedSum(squares)
            let inverse = Float(1) / (total + eps).squareRoot()
            for index in 0..<width {
                result[row * width + index] = values[base + index] * inverse
            }
        }
        return result
    }

    /// `causal_conv1d_fn:270` — depthwise, left-padded by `kernel - 1`, truncated, then
    /// activated. Output `s` reads `x[s + k - (kernel - 1)]` through tap `k`, so nothing
    /// later than `s` contributes.
    public static func depthwiseCausalConv(
        _ x: [Float], channels: Int, length: Int, weight: [Float], kernel: Int, activate: Bool
    ) -> [Float] {
        var out = [Float](repeating: 0, count: channels * length)
        for channel in 0..<channels {
            let tapBase = channel * kernel
            for position in 0..<length {
                var accumulator: Float = 0
                for tap in 0..<kernel {
                    let source = position + tap - (kernel - 1)
                    if source >= 0 {
                        accumulator = accumulator + weight[tapBase + tap] * x[channel * length + source]
                    }
                }
                out[channel * length + position] = activate ? Ops.silu(accumulator) : accumulator
            }
        }
        return out
    }

    /// The gated RMSNorm, whose two orderings are its whole point: the normalised value is
    /// rounded — in the bf16 oracle — *before* the weight multiply, and the gate is
    /// activated *after* it. In an all-fp32 contract the first rounding does not exist,
    /// which is recorded rather than glossed.
    public static func gatedRMSNorm(
        hidden: [Float], gate: [Float], weight: [Float], rows: Int, width: Int, eps: Float
    ) -> [Float] {
        var out = [Float](repeating: 0, count: rows * width)
        for row in 0..<rows {
            let base = row * width
            var squares: [Float] = []
            squares.reserveCapacity(width)
            for index in 0..<width {
                let value = hidden[base + index]
                squares.append(value * value)
            }
            let variance = Ops.orderedSum(squares) / Float(width)
            let inverse = Float(1) / (variance + eps).squareRoot()
            for index in 0..<width {
                out[base + index] = weight[index] * (hidden[base + index] * inverse) * Ops.silu(gate[base + index])
            }
        }
        return out
    }

    /// Forward substitution for a unit lower triangular system, rows eliminated in order.
    static func solveUnitLower(lower: [Float], rhs: [Float], rows: Int, columns: Int) -> [Float] {
        var solution = [Float](repeating: 0, count: rows * columns)
        for row in 0..<rows {
            for column in 0..<columns { solution[row * columns + column] = rhs[row * columns + column] }
            for previous in 0..<row {
                let factor = lower[row * rows + previous]
                for column in 0..<columns {
                    solution[row * columns + column] =
                        solution[row * columns + column] - factor * solution[previous * columns + column]
                }
            }
        }
        return solution
    }

    /// `torch_chunk_gated_delta_rule:301`, for one (batch, head) pair.
    ///
    /// `query`, `key` are `[length, keyDim]`, `value` is `[length, valueDim]`, `decay` and
    /// `beta` are `[length]`. Returns the output and the final state `[keyDim, valueDim]`.
    static func chunkedDeltaRule(
        query: [Float], key: [Float], value: [Float], decay: [Float], beta: [Float],
        length: Int, keyDim: Int, valueDim: Int, useL2Norm: Bool
    ) -> (output: [Float], state: [Float]) {
        let eps: Float = 1e-6
        var q = useL2Norm ? l2norm(query, count: length, width: keyDim, eps: eps, rowOffset: 0) : query
        var k = useL2Norm ? l2norm(key, count: length, width: keyDim, eps: eps, rowOffset: 0) : key
        let scaling = Float(1) / Float(keyDim).squareRoot()
        for index in 0..<q.count { q[index] = q[index] * scaling }

        let padding = (chunkSize - length % chunkSize) % chunkSize
        let total = length + padding
        let chunks = total / chunkSize
        if padding > 0 {
            q.append(contentsOf: [Float](repeating: 0, count: padding * keyDim))
            k.append(contentsOf: [Float](repeating: 0, count: padding * keyDim))
        }
        var v = value
        var betaPadded = beta
        var decayPadded = decay
        if padding > 0 {
            v.append(contentsOf: [Float](repeating: 0, count: padding * valueDim))
            betaPadded.append(contentsOf: [Float](repeating: 0, count: padding))
            decayPadded.append(contentsOf: [Float](repeating: 0, count: padding))
        }

        var valueBeta = [Float](repeating: 0, count: total * valueDim)
        var keyBeta = [Float](repeating: 0, count: total * keyDim)
        for position in 0..<total {
            let scale = betaPadded[position]
            for index in 0..<valueDim { valueBeta[position * valueDim + index] = v[position * valueDim + index] * scale }
            for index in 0..<keyDim { keyBeta[position * keyDim + index] = k[position * keyDim + index] * scale }
        }

        // Cumulative decay: a left-to-right prefix sum inside each chunk.
        var cumulative = [Float](repeating: 0, count: total)
        for chunk in 0..<chunks {
            var running: Float = 0
            for position in 0..<chunkSize {
                running = running + decayPadded[chunk * chunkSize + position]
                cumulative[chunk * chunkSize + position] = running
            }
        }

        var queryRotated = [Float](repeating: 0, count: total * keyDim)
        var keyRotated = [Float](repeating: 0, count: total * keyDim)
        var newValues = [Float](repeating: 0, count: total * valueDim)
        var keyCumDecay = [Float](repeating: 0, count: total * keyDim)
        var intraChunkAttn = [Float](repeating: 0, count: chunks * chunkSize * chunkSize)
        var chunkDecay = [Float](repeating: 0, count: chunks)

        for chunk in 0..<chunks {
            let base = chunk * chunkSize
            var pairwise = [Float](repeating: 0, count: chunkSize * chunkSize)
            for row in 0..<chunkSize {
                for column in 0..<chunkSize {
                    // The strictly upper triangle is masked before the exponential, so no
                    // future position contributes and no overflow can occur.
                    if column > row {
                        pairwise[row * chunkSize + column] = Ops.exp32(-Float.infinity)
                    } else {
                        pairwise[row * chunkSize + column] =
                            Ops.exp32(cumulative[base + row] - cumulative[base + column])
                    }
                }
            }

            // The chunk's slices, so the shared 2-D ordered matmul can be used unchanged.
            func chunkSlice(_ source: [Float], width: Int) -> [Float] {
                var slice = [Float](repeating: 0, count: chunkSize * width)
                for row in 0..<chunkSize {
                    for index in 0..<width {
                        slice[row * width + index] = source[(base + row) * width + index]
                    }
                }
                return slice
            }

            let keyChunk = chunkSlice(k, width: keyDim)
            let keyBetaChunk = chunkSlice(keyBeta, width: keyDim)
            let queryChunk = chunkSlice(q, width: keyDim)
            let valueBetaChunk = chunkSlice(valueBeta, width: valueDim)

            var utSystem = MetalMatmul.ordered(
                x: keyBetaChunk, w: keyChunk, rows: chunkSize, k: keyDim, out: chunkSize
            )
            for index in 0..<utSystem.count { utSystem[index] = utSystem[index] * pairwise[index] }

            var intra = MetalMatmul.ordered(
                x: queryChunk, w: keyChunk, rows: chunkSize, k: keyDim, out: chunkSize
            )
            for index in 0..<intra.count { intra[index] = intra[index] * pairwise[index] }
            for index in 0..<intra.count { intraChunkAttn[chunk * chunkSize * chunkSize + index] = intra[index] }

            var decayedKeyBeta = [Float](repeating: 0, count: chunkSize * keyDim)
            for row in 0..<chunkSize {
                let factor = Ops.exp32(cumulative[base + row])
                for index in 0..<keyDim {
                    decayedKeyBeta[row * keyDim + index] = keyBetaChunk[row * keyDim + index] * factor
                }
            }

            let solvedValues = solveUnitLower(
                lower: utSystem, rhs: valueBetaChunk, rows: chunkSize, columns: valueDim
            )
            let solvedKeys = solveUnitLower(
                lower: utSystem, rhs: decayedKeyBeta, rows: chunkSize, columns: keyDim
            )
            for index in 0..<solvedValues.count { newValues[base * valueDim + index] = solvedValues[index] }
            for index in 0..<solvedKeys.count { keyCumDecay[base * keyDim + index] = solvedKeys[index] }

            let lastDecay = cumulative[base + chunkSize - 1]
            chunkDecay[chunk] = Ops.exp32(lastDecay)
            for row in 0..<chunkSize {
                let queryFactor = Ops.exp32(cumulative[base + row])
                let keyFactor = Ops.exp32(lastDecay - cumulative[base + row])
                for index in 0..<keyDim {
                    queryRotated[(base + row) * keyDim + index] = q[(base + row) * keyDim + index] * queryFactor
                    keyRotated[(base + row) * keyDim + index] = k[(base + row) * keyDim + index] * keyFactor
                }
            }
        }

        var state = [Float](repeating: 0, count: keyDim * valueDim)
        var output = [Float](repeating: 0, count: total * valueDim)

        for chunk in 0..<chunks {
            let base = chunk * chunkSize
            // stateᵀ, because orderedMatmul contracts the last axis of both operands.
            var stateTransposed = [Float](repeating: 0, count: valueDim * keyDim)
            for row in 0..<keyDim {
                for column in 0..<valueDim {
                    stateTransposed[column * keyDim + row] = state[row * valueDim + column]
                }
            }

            func slice(_ source: [Float], width: Int) -> [Float] {
                var result = [Float](repeating: 0, count: chunkSize * width)
                for row in 0..<chunkSize {
                    for index in 0..<width {
                        result[row * width + index] = source[(base + row) * width + index]
                    }
                }
                return result
            }

            let solved = slice(newValues, width: valueDim)
            let keyCum = slice(keyCumDecay, width: keyDim)
            let queryChunk = slice(queryRotated, width: keyDim)

            let predicted = MetalMatmul.ordered(
                x: keyCum, w: stateTransposed, rows: chunkSize, k: keyDim, out: valueDim
            )
            var vNew = [Float](repeating: 0, count: chunkSize * valueDim)
            for index in 0..<vNew.count { vNew[index] = solved[index] - predicted[index] }

            var vNewTransposed = [Float](repeating: 0, count: valueDim * chunkSize)
            for row in 0..<chunkSize {
                for index in 0..<valueDim {
                    vNewTransposed[index * chunkSize + row] = vNew[row * valueDim + index]
                }
            }

            let inter = MetalMatmul.ordered(
                x: queryChunk, w: stateTransposed, rows: chunkSize, k: keyDim, out: valueDim
            )
            let intraChunk = [Float](
                intraChunkAttn[(chunk * chunkSize * chunkSize)..<((chunk + 1) * chunkSize * chunkSize)]
            )
            let within = MetalMatmul.ordered(
                x: intraChunk, w: vNewTransposed, rows: chunkSize, k: chunkSize, out: valueDim
            )
            for index in 0..<within.count {
                output[base * valueDim + index] = inter[index] + within[index]
            }

            // state = state * chunkDecay + keyᵀ @ vNew
            let keyRotatedChunk = slice(keyRotated, width: keyDim)
            var keyTransposed = [Float](repeating: 0, count: keyDim * chunkSize)
            for row in 0..<chunkSize {
                for index in 0..<keyDim {
                    keyTransposed[index * chunkSize + row] = keyRotatedChunk[row * keyDim + index]
                }
            }
            let update = MetalMatmul.ordered(
                x: keyTransposed, w: vNewTransposed, rows: keyDim, k: chunkSize, out: valueDim
            )
            let decayFactor = chunkDecay[chunk]
            for index in 0..<state.count { state[index] = state[index] * decayFactor + update[index] }
        }

        var trimmed = [Float](repeating: 0, count: length * valueDim)
        for index in 0..<trimmed.count { trimmed[index] = output[index] }
        return (trimmed, state)
    }

    /// The whole layer: projection, conv, split, gates, the delta rule, the gated norm and
    /// the output projection. `hidden` is `[batch, length, hiddenSize]`.
    /// The state a Gated DeltaNet layer carries between tokens.
    ///
    /// Two pieces, and both are needed: the convolution's **window** — the last `kernel - 1` raw
    /// projections per channel, because the causal conv is depthwise with a kernel of four and
    /// would otherwise see zeros where history belongs — and the **recurrent state** `[heads,
    /// keyHeadDim, valueHeadDim]`, which is what `Qwen3_5MoeGatedDeltaNet.forward` threads
    /// through `initial_state`/`output_final_state`.
    ///
    /// The layout follows `causal_conv1d_update`: `conv` is `[channels, kernel - 1]` per batch
    /// element, oldest first, which is exactly the window the reference concatenates its new
    /// input onto.
    public final class State {
        public var conv: [Float]
        public var recurrent: [Float]

        public init(batch: Int, convDim: Int, kernel: Int, valueHeads: Int, keyHeadDim: Int, valueHeadDim: Int) {
            self.conv = [Float](repeating: 0, count: batch * convDim * (kernel - 1))
            self.recurrent = [Float](
                repeating: 0, count: batch * valueHeads * keyHeadDim * valueHeadDim
            )
        }
    }

    /// One token through the layer, carrying `state` forward — the decode path.
    ///
    /// `D8`: the chunked rule is authoritative for what the model *is*, and this implements the
    /// recurrent form of the same recurrence, which agrees with it to about 1e-7 relative rather
    /// than to the bit, because the chunked rule groups its sums over 64 positions and this
    /// accumulates a step at a time. That is a second numeric path by decision, not by accident,
    /// and it is checked against the sequence path rather than assumed equal to it.
    ///
    /// The conv follows `causal_conv1d_update:252`: concatenate the window with the new input,
    /// take the convolution with no padding, keep the **last** output, and store the last
    /// `kernel - 1` inputs as the new window.
    /// A projection's product, from the **stored** form when the layer has one (`D111`).
    ///
    /// `values` is empty exactly when `packed` is set: the decode is the cost this avoids, so a layer that can
    /// serve the packed bytes never materialises them. If the device is unavailable or refuses — a shape the
    /// kernel cannot take — the stored bytes are decoded on the CPU rather than multiplying nothing, so the
    /// empty array can never reach `Ops.orderedMatmul`.
    private static func projection(
        x: [Float], rows: Int, k: Int, out: Int, values: [Float], packed: PackedInt4Rows?
    ) -> [Float] {
        if MetalInt4Matmul.enabled, let packed, values.isEmpty {
            if let product = try? MetalInt4Matmul.matmul(
                payload: packed.payload, entry: packed.entry, rowCount: packed.payloadRows, x: x, rows: rows
            ), product.count == rows * out {
                return product
            }
            if let decoded = try? InstallFile.dequantizeInt4(
                packed.payload, entry: packed.entry, rowCount: packed.payloadRows
            ) {
                return MetalMatmul.ordered(x: x, w: decoded, rows: rows, k: k, out: out)
            }
        }
        return MetalMatmul.ordered(x: x, w: values, rows: rows, k: k, out: out)
    }

    public static func decodeStep(
        hidden: [Float], weights: GatedDeltaNetWeights, shape: GatedDeltaNetShape, state: State
    ) -> [Float] {
        let keyDim = shape.keyDim
        let valueDim = shape.valueDim
        let convDim = shape.convDim
        let heads = shape.valueHeads
        let keyHeads = shape.keyHeads
        let headK = shape.keyHeadDim
        let headV = shape.valueHeadDim
        let kernel = shape.convKernel
        let window = kernel - 1
        let eps = shape.eps

        let mixed = projection(
            x: hidden, rows: 1, k: shape.hiddenSize, out: convDim,
            values: weights.inQKV, packed: weights.packedInQKV
        )

        // The window, oldest first, then the new projection: the reference's `torch.cat`.
        var convolved = [Float](repeating: 0, count: convDim)
        // **One buffer for every channel** (`D109`). This was allocated *inside* the loop below — 8,192
        // times per layer and 245,760 times in a decode step — for a four-element array that is fully
        // overwritten before it is read, so the allocator was paying for work the loop does not need.
        // The values cannot change: every one of `0..<kernel` is written each iteration.
        var samples = [Float](repeating: 0, count: kernel)
        for channel in 0..<convDim {
            for index in 0..<window { samples[index] = state.conv[channel * window + index] }
            samples[window] = mixed[channel]
            var total: Float = 0
            for tap in 0..<kernel {
                // The conv weight is `[convDim, 1, kernel]`, and tap `k` reads `samples[k]`.
                total += weights.conv[channel * kernel + tap] * samples[tap]
            }
            convolved[channel] = Ops.silu(total)
            // The state keeps the last `kernel - 1` inputs, which is the window shifted by one.
            for index in 0..<window { state.conv[channel * window + index] = samples[index + 1] }
        }

        let z = projection(
            x: hidden, rows: 1, k: shape.hiddenSize, out: valueDim,
            values: weights.inZ, packed: weights.packedInZ
        )
        let b = MetalMatmul.ordered(x: hidden, w: weights.inB, rows: 1, k: shape.hiddenSize, out: heads)
        let a = MetalMatmul.ordered(x: hidden, w: weights.inA, rows: 1, k: shape.hiddenSize, out: heads)

        // Per head, as everywhere: writing the gates once per position gives the last head's
        // values to all of them, which is wrong in a way that still produces plausible numbers.
        var beta = [Float](repeating: 0, count: heads)
        var decay = [Float](repeating: 0, count: heads)
        for head in 0..<heads {
            beta[head] = Ops.sigmoid(b[head])
            decay[head] = -Ops.exp32(weights.aLog[head]) * softplus(a[head] + weights.dtBias[head])
        }

        // Query, key and value for this position, with the key heads repeated up to the value
        // head count — the grouped-query step `DC-038` fixed in the sequence path.
        let headRepeats = max(heads / max(keyHeads, 1), 1)
        var query = [Float](repeating: 0, count: heads * headK)
        var key = [Float](repeating: 0, count: heads * headK)
        var value = [Float](repeating: 0, count: heads * headV)
        for head in 0..<keyHeads {
            for repeatIndex in 0..<headRepeats {
                let target = (head * headRepeats + repeatIndex) * headK
                for component in 0..<headK {
                    query[target + component] = convolved[head * headK + component]
                    key[target + component] = convolved[keyDim + head * headK + component]
                }
            }
        }
        for component in 0..<valueDim { value[component] = convolved[2 * keyDim + component] }

        if true {
            // `use_qk_l2norm_in_kernel`, then the unconditional scaling by 1/sqrt(head_dim) —
            // the line whose absence leaves the output out by a constant factor.
            query = l2norm(query, count: heads, width: headK, eps: 1e-6, rowOffset: 0)
            key = l2norm(key, count: heads, width: headK, eps: 1e-6, rowOffset: 0)
            let scale = 1 / Float(headK).squareRoot()
            for index in 0..<query.count { query[index] *= scale }
        }

        // The recurrence, one step: decay the state, correct it towards this token's value, and
        // read the output from it. Transcribed from `torch_recurrent_gated_delta_rule:440`.
        var core = [Float](repeating: 0, count: heads * headV)
        for head in 0..<heads {
            let stateBase = head * headK * headV
            let decayHead = Ops.exp32(decay[head])
            for index in 0..<(headK * headV) { state.recurrent[stateBase + index] *= decayHead }

            // `kv_mem = (state * k).sum(dim=-2)`: over the key dimension, in ascending order.
            var kvMemory = [Float](repeating: 0, count: headV)
            for kIndex in 0..<headK {
                let keyValue = key[head * headK + kIndex]
                for vIndex in 0..<headV {
                    kvMemory[vIndex] += state.recurrent[stateBase + kIndex * headV + vIndex] * keyValue
                }
            }
            var delta = [Float](repeating: 0, count: headV)
            for vIndex in 0..<headV {
                delta[vIndex] = (value[head * headV + vIndex] - kvMemory[vIndex]) * beta[head]
            }
            for kIndex in 0..<headK {
                let keyValue = key[head * headK + kIndex]
                for vIndex in 0..<headV {
                    state.recurrent[stateBase + kIndex * headV + vIndex] += keyValue * delta[vIndex]
                }
            }
            for kIndex in 0..<headK {
                let queryValue = query[head * headK + kIndex]
                for vIndex in 0..<headV {
                    core[head * headV + vIndex] += state.recurrent[stateBase + kIndex * headV + vIndex] * queryValue
                }
            }
        }

        let normalised = gatedRMSNorm(hidden: core, gate: z, weight: weights.norm, rows: heads, width: headV, eps: eps)
        return projection(
            x: normalised, rows: 1, k: valueDim, out: shape.hiddenSize,
            values: weights.outProj, packed: weights.packedOut
        )
    }

    public static func layer(
        hidden: [Float], weights: GatedDeltaNetWeights, shape: GatedDeltaNetShape, batch: Int, length: Int,
        record: ((String, [Float], [Int]) -> Void)? = nil
    ) -> [Float] {
        let keyDim = shape.keyDim
        let valueDim = shape.valueDim
        let convDim = shape.convDim
        let heads = shape.valueHeads
        let keyHeads = shape.keyHeads
        let headK = shape.keyHeadDim
        let headV = shape.valueHeadDim

        // `Qwen3_5MoeGatedDeltaNet.forward:645` — grouped-query style:
        //
        //     if self.num_v_heads // self.num_k_heads > 1:
        //         query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        //         key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        //
        // More value heads than key heads means each key head serves several value heads, and
        // the query and key heads have to be repeated consecutively before the rule runs. The
        // MoE family has sixteen key heads to thirty-two value heads; with the counts equal
        // this is the identity, which is why the 2 B model never exercised it. Without it the
        // per-head slice below indexes past the end of the key and pairs the wrong heads.
        let headRepeats = max(heads / max(keyHeads, 1), 1)
        // The reference does not check this: a value head count that is not a multiple would
        // silently repeat the wrong number of times there.
        precondition(
            keyHeads * headRepeats == heads,
            "the value head count must be a multiple of the key head count"
        )
        let expandedKeyDim = heads * headK

        var output = [Float](repeating: 0, count: batch * length * shape.hiddenSize)

        // The projection is computed for the whole batch, then the conv runs per sequence
        // because it is causal along the sequence axis.
        var mixed = projection(
            x: hidden, rows: batch * length, k: shape.hiddenSize, out: convDim,
            values: weights.inQKV, packed: weights.packedInQKV
        )
        var convolved = [Float](repeating: 0, count: batch * length * convDim)
        for index in 0..<batch {
            var channels = [Float](repeating: 0, count: convDim * length)
            for position in 0..<length {
                for channel in 0..<convDim {
                    channels[channel * length + position] = mixed[(index * length + position) * convDim + channel]
                }
            }
            let filtered = depthwiseCausalConv(
                channels, channels: convDim, length: length, weight: weights.conv,
                kernel: shape.convKernel, activate: true
            )
            for position in 0..<length {
                for channel in 0..<convDim {
                    convolved[(index * length + position) * convDim + channel] = filtered[channel * length + position]
                }
            }
        }
        mixed = convolved

        let z = projection(
            x: hidden, rows: batch * length, k: shape.hiddenSize, out: valueDim,
            values: weights.inZ, packed: weights.packedInZ
        )
        let b = MetalMatmul.ordered(
            x: hidden, w: weights.inB, rows: batch * length, k: shape.hiddenSize, out: heads
        )
        let a = MetalMatmul.ordered(
            x: hidden, w: weights.inA, rows: batch * length, k: shape.hiddenSize, out: heads
        )

        for index in 0..<batch {
            var query = [Float](repeating: 0, count: length * expandedKeyDim)
            var key = [Float](repeating: 0, count: length * expandedKeyDim)
            var value = [Float](repeating: 0, count: length * valueDim)
            // Per position *and* per head: the reference computes `beta` and `g` for every
            // value head, and each head is its own recurrence. Writing these as one value
            // per position gives the last head's gates to all of them, which is wrong in a
            // way that still produces plausible numbers.
            var decay = [Float](repeating: 0, count: length * heads)
            var beta = [Float](repeating: 0, count: length * heads)

            for position in 0..<length {
                let row = (index * length + position) * convDim
                // The conv output is still laid out with one block per *key* head; the
                // repeat happens on the way in, exactly where the reference does it.
                for head in 0..<keyHeads {
                    for repeatIndex in 0..<headRepeats {
                        let target = position * expandedKeyDim + (head * headRepeats + repeatIndex) * headK
                        for component in 0..<headK {
                            query[target + component] = mixed[row + head * headK + component]
                            key[target + component] = mixed[row + keyDim + head * headK + component]
                        }
                    }
                }
                for component in 0..<valueDim {
                    value[position * valueDim + component] = mixed[row + 2 * keyDim + component]
                }
                let headBase = (index * length + position) * heads
                for head in 0..<heads {
                    beta[position * heads + head] = Ops.sigmoid(b[headBase + head])
                    // g = -exp(A_log) * softplus(a + dt_bias), in fp32.
                    decay[position * heads + head] =
                        -Ops.exp32(weights.aLog[head]) * softplus(a[headBase + head] + weights.dtBias[head])
                }
            }

            // The rule is run per head, because the heads are independent recurrences.
            var core = [Float](repeating: 0, count: length * heads * headV)
            for head in 0..<heads {
                var headQuery = [Float](repeating: 0, count: length * headK)
                var headKey = [Float](repeating: 0, count: length * headK)
                var headValue = [Float](repeating: 0, count: length * headV)
                for position in 0..<length {
                    for component in 0..<headK {
                        headQuery[position * headK + component] = query[position * expandedKeyDim + head * headK + component]
                        headKey[position * headK + component] = key[position * expandedKeyDim + head * headK + component]
                    }
                    for component in 0..<headV {
                        headValue[position * headV + component] = value[position * valueDim + head * headV + component]
                    }
                }
                var headDecay = [Float](repeating: 0, count: length)
                var headBeta = [Float](repeating: 0, count: length)
                for position in 0..<length {
                    headDecay[position] = decay[position * heads + head]
                    headBeta[position] = beta[position * heads + head]
                }
                let (headOutput, _) = chunkedDeltaRule(
                    query: headQuery, key: headKey, value: headValue, decay: headDecay, beta: headBeta,
                    length: length, keyDim: headK, valueDim: headV, useL2Norm: true
                )
                for position in 0..<length {
                    for component in 0..<headV {
                        core[(position * heads + head) * headV + component] = headOutput[position * headV + component]
                    }
                }
            }

            // Two points, both opt-in through `record`: the rule's output before the gated norm, and the
            // gated norm's output before the projection. They split the tail in two, which is what a
            // divergence localised to "the attention half" needs next.
            record?("delta_core", core, [length * heads, headV])

            let normalised = gatedRMSNorm(
                hidden: core, gate: z[index * length * valueDim..<(index + 1) * length * valueDim].map { $0 },
                weight: weights.norm, rows: length * heads, width: headV, eps: shape.eps
            )
            record?("gated_norm_out", normalised, [length * heads, headV])

            let projected = projection(
                x: normalised, rows: length, k: valueDim, out: shape.hiddenSize,
                values: weights.outProj, packed: weights.packedOut
            )
            for index2 in 0..<projected.count {
                output[index * length * shape.hiddenSize + index2] = projected[index2]
            }
        }
        return output
    }
}
