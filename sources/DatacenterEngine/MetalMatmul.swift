import Foundation
import Metal

/// The contract's matmul, on the GPU: `x @ wᵀ`, one multiply and one add per output, ascending `k`.
///
/// **`D61` settled the arithmetic before this was written**, which is the only reason the kernel can be as
/// simple as it is. `ordered_matmul` is defined as one rounding per multiply and one per add — "no FMA" —
/// and the measurement found that **no** `MTLMathMode` keeps those apart: `.fast`, `.relaxed` and `.safe`
/// all contract `a * b + acc` into a single `fma`, and `.safe` only keeps a product in its own statement
/// apart from the following add. The spelling that reproduces the contract under every mode is
/// `accumulator + metal::fma(x, w, 0.0f)`, and that is what this kernel does — so it can share `.relaxed`
/// with the unpack rather than forcing a second, slower pipeline.
///
/// One thread per output element, accumulating over `k` ascending: there is no reduction and nothing to
/// reassociate, which is what makes a GPU kernel bit-exact against a sequential loop at all. The vector
/// on the CPU runs across the **output** dimension for the same reason (`Ops.orderedMatmulVectorized`).
public enum MetalMatmul {
    public enum Error: Swift.Error {
        case noDevice
        case compileFailed(String)
        case commandFailed(String)
        case shapeMismatch(String)
    }

    /// Whether this host has a GPU. CI runners have none, so the tests skip there.
    public static var isAvailable: Bool { MTLCreateSystemDefaultDevice() != nil }

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
        // `.relaxed` is the same non-fast setting the unpack uses, and `D61` measured that the kernel's
        // arithmetic does not depend on it: `fma(x, w, 0)` keeps the product and the add apart under every
        // mode, so this choice cannot change a result.
        options.mathMode = .relaxed
        do {
            let library = try device.makeLibrary(source: shader, options: options)
            guard let function = library.makeFunction(name: "matmul") else {
                return .failure(.compileFailed("the library has no 'matmul' function"))
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

    /// `x` is `[rows, k]` and `w` is `[out, k]` — the layout `Ops.orderedMatmul` takes and every checkpoint
    /// uses — and the result is `[rows, out]`, `x @ wᵀ`.
    public static func matmul(x: [Float], w: [Float], rows: Int, k: Int, out: Int) throws -> [Float] {
        guard rows >= 0, k >= 0, out >= 0 else {
            throw Error.shapeMismatch("negative shape \(rows)x\(k)x\(out)")
        }
        guard x.count == rows * k else {
            throw Error.shapeMismatch("x has \(x.count) values, \(rows)x\(k) needs \(rows * k)")
        }
        guard w.count == out * k else {
            throw Error.shapeMismatch("w has \(w.count) values, \(out)x\(k) needs \(out * k)")
        }
        let values = rows * out
        guard values > 0 else { return [] }
        if k == 0 { return [Float](repeating: 0, count: values) }

        let pipeline = try Self.pipeline.current()
        return try Self.buffers.withBuffers(
            [rows * k * 4, out * k * 4, values * 4], device: pipeline.device
        ) { cached in
            let xBuffer = cached[0], wBuffer = cached[1], outBuffer = cached[2]
            x.withUnsafeBytes { source in
                if let base = source.baseAddress { xBuffer.contents().copyMemory(from: base, byteCount: rows * k * 4) }
            }
            w.withUnsafeBytes { source in
                if let base = source.baseAddress { wBuffer.contents().copyMemory(from: base, byteCount: out * k * 4) }
            }

            var dims = SIMD4<UInt32>(UInt32(rows), UInt32(k), UInt32(out), 0)
            guard let command = pipeline.queue.makeCommandBuffer(),
                  let encoder = command.makeComputeCommandEncoder()
            else { throw Error.commandFailed("could not make a command buffer") }

            encoder.setComputePipelineState(pipeline.state)
            encoder.setBuffer(xBuffer, offset: 0, index: 0)
            encoder.setBuffer(wBuffer, offset: 0, index: 1)
            encoder.setBuffer(outBuffer, offset: 0, index: 2)
            encoder.setBytes(&dims, length: MemoryLayout<SIMD4<UInt32>>.size, index: 3)
            let width = min(pipeline.state.threadExecutionWidth, values)
            encoder.dispatchThreads(
                MTLSize(width: values, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: max(width, 1), height: 1, depth: 1)
            )
            encoder.endEncoding()
            command.commit()
            command.waitUntilCompleted()
            if let error = command.error { throw Error.commandFailed("\(error)") }

            let raw = outBuffer.contents().bindMemory(to: Float.self, capacity: values)
            return Array(UnsafeBufferPointer(start: raw, count: values))
        }
    }

    /// Whether large matmuls run on the GPU. **Off by default**, and `SHARD_GPU_MATMUL=1` turns it on.
    ///
    /// The arithmetic is settled — `D61` measured it, and `MetalMatmulTests` asserts this kernel is
    /// bit-identical to `Ops.orderedMatmul` over 288 shapes, the head's real shape and cache reuse — and the
    /// real trace is `b0d382dbabf36df0…` either way. What is *not* settled is that it is ever faster, and the
    /// measurement says it is not (`D63`): with the conditions alternated rather than run in sequence, every
    /// phase that uses it is slower — `attn.core` 4.05 → 5.55 s, `mix.gateup` 1.17 → 1.62 s, `head` 1.74 →
    /// 1.86 s — while `mix.read`, which no matmul touches, is unchanged, so the difference is the kernel and
    /// not the machine.
    ///
    /// The cause is the access pattern, not the arithmetic: one thread per output keeps `k` in its inner
    /// loop, so adjacent threads walk `w` rows **8 KB apart** — a warp touches thirty-two cache lines to use
    /// four bytes of each. The CPU's vectorised path walks `k` contiguously inside one output and reuses `x`.
    /// A threadgroup-tiled kernel would fix that, and tiling cannot change a result because it changes
    /// *which thread* accumulates and not the order. Until that kernel exists, this one is opt-in: a kernel
    /// that is slower is not a default.
    public static let enabled = ProcessInfo.processInfo.environment["SHARD_GPU_MATMUL"] == "1"

    /// Below this much work a dispatch costs more than it saves, which the unpack learned the expensive way
    /// (`D59`): its first version was *slower* than the scalar path it replaced. The figure is deliberately
    /// conservative — the CPU does about 1.4 G multiply-adds a second, so a dispatch's overhead is worth
    /// roughly a hundred thousand of them, and this is ten times that. It is what leaves the DeltaNet's
    /// per-head contractions and the decode path's one-row projections on the CPU without a list of special
    /// cases: they are simply below the line.
    public static let minimumWork = 1_000_000

    /// `Ops.orderedMatmul`, or the GPU equivalent when one is available, asked for, and worth it.
    ///
    /// This is the *only* place the choice is made, which is why it lives beside the kernel rather than in
    /// `Ops`: `Ops` **is** the definition of the contract and should not know about a device, and a second
    /// chooser somewhere else would be a second answer to the same question.
    public static func ordered(x: [Float], w: [Float], rows: Int, k: Int, out: Int) -> [Float] {
        if enabled, isAvailable, rows * k * out >= minimumWork,
            let product = try? matmul(x: x, w: w, rows: rows, k: k, out: out)
        {
            return product
        }
        return Ops.orderedMatmul(x: x, w: w, rows: rows, k: k, out: out)
    }

    private static let shader = """
    #include <metal_stdlib>
    using namespace metal;

    // One thread per output value, `k` accumulated in ascending order, one rounding for the product and one
    // for the add. `metal::fma(x, w, 0.0f)` is the correctly-rounded product whatever the compiler would
    // prefer to do with `x * w` followed by an add: `D61` measured that every math mode this SDK offers
    // contracts that pair into a single `fma`, which is a different number.
    kernel void matmul(device const float *x [[buffer(0)]],
                       device const float *w [[buffer(1)]],
                       device float *out [[buffer(2)]],
                       constant uint4 &dims [[buffer(3)]],
                       uint gid [[thread_position_in_grid]]) {
        uint rows = dims.x, k = dims.y, columns = dims.z;
        uint row = gid / columns;
        uint column = gid % columns;
        if (row >= rows) { return; }
        float accumulator = 0.0f;
        for (uint index = 0; index < k; ++index) {
            accumulator = accumulator + metal::fma(x[row * k + index], w[column * k + index], 0.0f);
        }
        out[gid] = accumulator;
    }
    """
}

private extension Result where Failure == MetalMatmul.Error {
    func current() throws -> Success {
        switch self {
        case .success(let value): return value
        case .failure(let error): throw error
        }
    }
}
