import Foundation
import Metal

/// The int4 matmul with the dequantisation **inside** it: `x @ dequantizeInt4(w)ᵀ`, and the fp32 weight
/// array the split path materialises never exists.
///
/// `D107` measured the same fusion on the CPU and it lost **4.7x** — two tight loops (an eight-wide unpack
/// and a four-wide matmul) beat one clever loop — but it also bounded the prize: on a CPU the unpack is
/// **7% of a step**, so even a faster fused kernel had almost nothing to win. On a **GPU** the question is
/// different in kind rather than degree. The unpack runs there already (`D59`), so the fp32 slab is not a
/// compute cost, it is **traffic**: the engine materialises about **6.5 GB of `Float` per token** into a
/// Metal buffer, hands it back to the CPU as a Swift array, and the matmul reads it again. Fusing removes
/// the write and the read-back entirely; the kernel reads the packed codes it needs and writes `out`
/// floats.
///
/// **Bit-exactness is not a hope here, it is the same three rules the other two kernels already obey.**
/// The dequantised value is built exactly as `MetalUnpack`'s kernel builds it — one multiply, the `D34`
/// zero-sign/NaN normalisation, `D11`'s denormal-scale flush — and the accumulation is
/// `accumulator + metal::fma(x, w, 0.0f)`, which `D61` measured to be the only spelling that reproduces
/// the contract's separate product and add under every math mode. One thread per output, `k` ascending,
/// so nothing is reassociated.
///
/// **One boundary is a property of the device, not of this kernel.** An Apple GPU flushes a denormal
/// *result* to zero, so a product of two small normals is zero here and not on the CPU; `MetalMatmul` has
/// the same behaviour and no `MTLMathMode` changes it. It cannot be reached through the weights (a weight
/// is never denormal) and it is pinned by a named test rather than left implicit — the comment on
/// `options.mathMode` below has the measurement.
public enum MetalInt4Matmul {
    public enum Error: Swift.Error {
        case noDevice
        case compileFailed(String)
        case commandFailed(String)
        case shapeMismatch(String)
    }

    /// Whether this host has a GPU. CI runners have none, so every test skips on this.
    public static var isAvailable: Bool { MTLCreateSystemDefaultDevice() != nil }

    /// Whether the **routed experts** take the fused path. **On by default, and measured** (`D110`): with a
    /// 256 MiB packed slab cache the step is **1.465 s against the split path's 1.539** over five alternated
    /// pairs on one binary, digest unchanged, and at **lower** peak RSS (3.04 against 3.13 GB). The kernel had
    /// to be made 3.6-4x faster first (`D110`'s wide loads) and the cache had to be sized *down* to 256 MiB —
    /// 512 MiB and above cost more in memory pressure than they save, which is `D106`'s finding again.
    /// `SHARD_GPU_INT4_EXPERTS=0` restores the split path.
    public static let enabled = ProcessInfo.processInfo.environment["SHARD_GPU_INT4_EXPERTS"] != "0"

    private struct Pipeline {
        let device: any MTLDevice
        let queue: any MTLCommandQueue
        let state: any MTLComputePipelineState
    }

    private static let pipeline: Result<Pipeline, Error> = {
        guard let device = MTLCreateSystemDefaultDevice() else { return .failure(.noDevice) }
        guard let queue = device.makeCommandQueue() else {
            return .failure(.commandFailed("could not make a command queue"))
        }
        let options = MTLCompileOptions()
        // `.relaxed`, the same non-fast setting the unpack and the matmul use. `D61` measured that
        // `accumulator + metal::fma(x, w, 0)` is exact under `.fast`, `.relaxed` and `.safe` alike, and the
        // **one** place this kernel can disagree with the CPU is not a math mode at all: an Apple GPU
        // flushes a denormal *result* to zero and `.safe` does not change that (measured — see
        // `MetalInt4MatmulTests.testADenormalProductFlushesOnTheGpu`, which pins it for this kernel and for
        // `MetalMatmul`). It is unreachable through the weights, which are never denormal — the smallest
        // non-zero code is 1 and a denormal scale is flushed to zero first — so it needs a product of two
        // small normals, which is the model's regime never. The mode costs nothing that matters; the
        // boundary is a property of the device and is documented rather than hidden behind a flag.
        // **`.safe`, not `.relaxed`** (`D110`): this kernel's contract is the *order* of its additions, and
        // `.relaxed` permits reassociation. `D61` measured that `.safe` still contracts `a*b+c` into one `fma`
        // — which is why the `fma(x, w, 0)` spelling is still exact — so nothing is given up but the licence
        // to reorder.
        options.mathMode = .safe
        do {
            let library = try device.makeLibrary(source: shader, options: options)
            guard let function = library.makeFunction(name: "int4_matmul") else {
                return .failure(.compileFailed("the library has no 'int4_matmul' function"))
            }
            return .success(
                Pipeline(
                    device: device, queue: queue,
                    state: try device.makeComputePipelineState(function: function)
                )
            )
        } catch {
            return .failure(.compileFailed("\(error)"))
        }
    }()

    private static let buffers = MetalBufferCache()

    /// `x` is `[rows, k]` and the payload is the int4 form of an `[out, k]` matrix, `out` being the
    /// layout's row count — one expert's projection in the engine's own use. The result is `[rows, out]`.
    ///
    /// This is `InstallFile.dequantizeInt4` followed by `Ops.orderedMatmul`, with the array in between
    /// removed; `MetalInt4MatmulTests` asserts the bit patterns over a grid of shapes rather than the
    /// intent.
    public static func matmul(
        payload: Data, entry: InstallFile.Entry, rowCount: Int? = nil, x: [Float], rows: Int
    ) throws -> [Float] {
        let layout = try InstallFile.int4Layout(entry: entry, rowCount: rowCount, payloadBytes: payload.count)
        let out = layout.rows
        let k = layout.columns
        guard rows >= 0, k >= 0, out >= 0 else {
            throw Error.shapeMismatch("negative shape \(rows)x\(k)x\(out)")
        }
        guard x.count == rows * k else {
            throw Error.shapeMismatch("x has \(x.count) values, \(rows)x\(k) needs \(rows * k)")
        }
        let values = rows * out
        guard values > 0 else { return [] }
        // `k == 0` cannot happen for a real tensor, but `Ops.orderedMatmul` answers it with zeros and
        // this must be indistinguishable from that. A kernel with an empty loop would answer the same;
        // returning early keeps the grid non-degenerate.
        if k == 0 { return [Float](repeating: 0, count: values) }

        let pipeline = try Self.pipeline.current()
        let zeroBytes = layout.rows * layout.groups
        return try Self.buffers.withBuffers(
            [layout.codeBytes, layout.scaleBytes, zeroBytes, rows * k * 4, values * 4],
            device: pipeline.device
        ) { cached in
            let codes = cached[0], scales = cached[1], zeros = cached[2], xBuffer = cached[3], outBuffer = cached[4]
            payload.copyBytes(to: codes.contents().assumingMemoryBound(to: UInt8.self), from: 0..<layout.codeBytes)
            payload.copyBytes(
                to: scales.contents().assumingMemoryBound(to: UInt8.self),
                from: layout.codeBytes..<(layout.codeBytes + layout.scaleBytes)
            )
            payload.copyBytes(
                to: zeros.contents().assumingMemoryBound(to: UInt8.self),
                from: (layout.codeBytes + layout.scaleBytes)..<(layout.codeBytes + layout.scaleBytes + zeroBytes)
            )
            x.withUnsafeBytes { source in
                if let base = source.baseAddress, rows * k > 0 {
                    xBuffer.contents().copyMemory(from: base, byteCount: rows * k * 4)
                }
            }

            var dims = SIMD4<UInt32>(UInt32(rows), UInt32(k), UInt32(out), UInt32(layout.group))
            var extra = SIMD4<UInt32>(UInt32(layout.padded), UInt32(layout.groups), 0, 0)
            guard let command = pipeline.queue.makeCommandBuffer(),
                  let encoder = command.makeComputeCommandEncoder()
            else { throw Error.commandFailed("could not make a command buffer") }

            encoder.setComputePipelineState(pipeline.state)
            encoder.setBuffer(codes, offset: 0, index: 0)
            encoder.setBuffer(scales, offset: 0, index: 1)
            encoder.setBuffer(zeros, offset: 0, index: 2)
            encoder.setBuffer(xBuffer, offset: 0, index: 3)
            encoder.setBuffer(outBuffer, offset: 0, index: 4)
            encoder.setBytes(&dims, length: MemoryLayout<SIMD4<UInt32>>.size, index: 5)
            encoder.setBytes(&extra, length: MemoryLayout<SIMD4<UInt32>>.size, index: 6)
            let width = pipeline.state.threadExecutionWidth
            encoder.dispatchThreads(
                MTLSize(width: values, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1)
            )
            encoder.endEncoding()
            command.commit()
            command.waitUntilCompleted()
            if let error = command.error { throw Error.commandFailed("\(error)") }

            let raw = outBuffer.contents().bindMemory(to: Float.self, capacity: values)
            return Array(UnsafeBufferPointer(start: raw, count: values))
        }
    }

    private static let shader = """
    #include <metal_stdlib>
    using namespace metal;

    // One thread per output element, `k` ascending, so the accumulation order is the contract's.
    //
    // The scale and the zero point are loaded **once per group** rather than per element: they are
    // constant across a group's `group` codes, and the group loop is what makes the fused form cheap
    // instead of the per-element group-index arithmetic that sank the CPU fusion (`D107`).
    kernel void int4_matmul(device const uchar *codes [[buffer(0)]],
                            device const float *scales [[buffer(1)]],
                            device const uchar *zeros [[buffer(2)]],
                            device const float *x [[buffer(3)]],
                            device float *out [[buffer(4)]],
                            constant uint4 &dims [[buffer(5)]],
                            constant uint4 &extra [[buffer(6)]],
                            uint gid [[thread_position_in_grid]]) {
        uint rows = dims.x, k = dims.y, outputs = dims.z, group = dims.w;
        uint padded = extra.x;
        uint row = gid / outputs;
        uint column = gid % outputs;
        if (row >= rows || column >= outputs) { return; }

        uint groupsPerRow = padded / group;
        device const uchar *rowCodes = codes + column * (padded / 2);
        device const float *rowScales = scales + column * groupsPerRow;
        device const uchar *rowZeros = zeros + column * groupsPerRow;
        device const float *xRow = x + row * k;

        float accumulator = 0.0f;
        uint groupIndex = 0;
        // **Wide loads** (`D110`). The first version read one `uchar` per element: lane `l` and lane `l + 1`
        // walk rows `padded / 2` bytes apart, so every load fetch used one byte of a cache line and the kernel
        // measured **4.3 GB/s on hardware that does ~100**. A `uint4` carries **32 codes**, and because a row's
        // codes are contiguous each such load uses every byte it fetches. The arithmetic below is unchanged —
        // same nibble order, same `fma`, ascending `k` — so this is a load-shape change and not a numeric one.
        bool rowAligned = ((column * (padded / 2)) % 16) == 0;
        for (uint base = 0; base < k; base += group, ++groupIndex) {
            uint span = min(group, k - base);
            // `D11`: a denormal scale reads as zero, exactly as `tools/quantize.py` and both Swift paths
            // do. The comparison is against the smallest **normal** float by its bit pattern, so no
            // rounding of a decimal literal can move the boundary. If the device flushes the load to
            // zero anyway (Metal's fast-math modes may), the value is already zero and the result is the
            // same — the branch is written so both behaviours agree.
            float scale = rowScales[groupIndex];
            if (scale != 0.0f && fabs(scale) < as_type<float>(0x00800000u)) { scale = 0.0f; }
            int zero = int(rowZeros[groupIndex]);
            if (zero >= 128) { zero -= 256; }
            uint offset = 0;
            if (rowAligned && (base % 32) == 0) {
                while (offset + 32 <= span) {
                    // A `uint4` is 16 bytes and therefore 32 four-bit codes, little-endian within each word.
                    uint4 word = *((const device uint4 *)(rowCodes + (base + offset) / 2));
                    uint words[4] = { word.x, word.y, word.z, word.w };
                    // **Not unrolled, on purpose** (`D110`). With `#pragma unroll` the optimiser is free to
                    // reassociate this float chain under `.relaxed`, and it did: one output in a 33-wide test
                    // came back 1 ULP off. An accumulator chain written out is the contract; a tree-reduced
                    // one is a different sum. The loop bound is a compile-time constant so the compiler may
                    // still unroll it — but it must not *reassociate* it, and only the sequential form is
                    // entitled to that.
                    for (uint w = 0; w < 4; ++w) {
                        for (uint b = 0; b < 8; ++b) {
                            uint nibble = (words[w] >> (4 * b)) & 0x0F;
                            int code = nibble >= 8 ? int(nibble) - 16 : int(nibble);
                            float value = float(code - zero) * scale;
                            // `D34`, and the same two branches `MetalUnpack`'s kernel uses for the same reason:
                            // a computed zero carries no sign, and the one indeterminate case (a zero code
                            // times an infinite scale) is the canonical quiet NaN.
                            if (value == 0.0f) { value = as_type<float>(0u); }
                            if (isnan(value)) { value = as_type<float>(0x7FC00000u); }
                            accumulator = accumulator + metal::fma(xRow[base + offset + w * 8 + b], value, 0.0f);
                        }
                    }
                    offset += 32;
                }
            }
            // The tail: a group whose start is not 16-byte aligned, or fewer than 32 codes left in it. The
            // scalar path keeps the same arithmetic, so a shape the wide path cannot take is slower and not
            // different.
            while (offset < span) {
                uint index = base + offset;
                uchar byte = rowCodes[index / 2];
                uint nibble = (index % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
                int code = nibble >= 8 ? int(nibble) - 16 : int(nibble);
                float value = float(code - zero) * scale;
                if (value == 0.0f) { value = as_type<float>(0u); }
                if (isnan(value)) { value = as_type<float>(0x7FC00000u); }
                accumulator = accumulator + metal::fma(xRow[index], value, 0.0f);
                offset += 1;
            }
        }
        out[row * outputs + column] = accumulator;
    }
    """
}

private extension Result where Failure == MetalInt4Matmul.Error {
    func current() throws -> Success {
        switch self {
        case .success(let value): return value
        case .failure(let error): throw error
        }
    }
}
