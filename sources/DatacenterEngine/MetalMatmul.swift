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

    /// The threadgroup's width in outputs. The shader's `TILE` has to equal it, and a mismatch would be a
    /// wrong answer rather than a slow one. Nothing can read a `#define` back out of a compiled library, so
    /// the guard is not an assertion here but the grid: `MetalMatmulTests` compares bit patterns over shapes
    /// whose `out` and `k` are not multiples of the tile, which is exactly where a mismatch would show.
    static let tile = 32

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
            // One threadgroup per 32 outputs and per row; the last group in a row is partly active, and the
            // kernel returns early for the lanes past the end rather than reading past `w`.
            let tile = MetalMatmul.tile
            encoder.dispatchThreadgroups(
                MTLSize(width: (out + tile - 1) / tile, height: rows, depth: 1),
                threadsPerThreadgroup: MTLSize(width: tile, height: 1, depth: 1)
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
    /// real trace is `b0d382dbabf36df0…` either way. **One boundary is now known and stated rather than
    /// assumed (`D108`):** an Apple GPU flushes a denormal *result* to zero, so a product of two small normals
    /// is a different number here than on the CPU, for this kernel and for every other one on the device (the
    /// math mode does not change it). The 288 shapes do not reach it because their values are O(1); real
    /// activations and weights do not either. `MetalInt4MatmulTests.testADenormalProductFlushesOnTheGpu` is
    /// where it is pinned. What is *not* settled is that it is ever faster, and the
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

    // A threadgroup-tiled matmul with the **contract's order preserved**.
    //
    // The first version had one thread per output with `k` in its inner loop: thread `column` and thread
    // `column + 1` then read `w` rows `k * 4` bytes apart — 8 KB at the head's width — so a warp touched
    // thirty-two cache lines to use four bytes of each. Measured, it was slower than the CPU (`D63`).
    //
    // Here each threadgroup owns `TILE` consecutive outputs for one row of `x`, and loads `w`'s tile for the
    // current `k` **cooperatively and coalesced** into threadgroup memory: each of the `TILE` rows of the tile
    // is `K_TILE` contiguous floats, which is one cache line at TILE=32. Each thread then accumulates its own
    // output over that tile — still **ascending in k**, tile after tile — so the sequence of roundings is
    // identical to the sequential loop and to the CPU. Tiling changes *which thread* adds, not the order in
    // which the adds happen, which is why the bit-exactness assertion carries over unchanged.
    //
    // The tail is **skipped rather than zero-padded**, and that is not a micro-optimisation: adding a
    // zero-valued term to a `-0.0` accumulator gives `+0.0`, and this project asserts zero signs bit for bit
    // (`D34`). A padded kernel would be right for every value and wrong for the sign of zero.
    #define TILE 32
    #define K_TILE 32

    kernel void matmul(device const float *x [[buffer(0)]],
                       device const float *w [[buffer(1)]],
                       device float *out [[buffer(2)]],
                       constant uint4 &dims [[buffer(3)]],
                       uint2 group [[threadgroup_position_in_grid]],
                       uint2 lane [[thread_position_in_threadgroup]]) {
        uint rows = dims.x, k = dims.y, columns = dims.z;
        uint row = group.y;
        uint column = group.x * TILE + lane.x;
        // **Predicated, never an early return.** A `return` here leaves the threadgroup before the barriers
        // below, and a barrier that some lanes have exited is undefined — which is not a theory: the first
        // version returned, and the grid failed with the first element of every short row correct and the
        // rest wrong, because the surviving lane was reading a tile row that its owner never loaded. Each
        // lane loads and reads **its own row** of the tile, so an inactive lane simply owns an unread one.
        bool active = row < rows && column < columns;

        // One extra column of padding so that the per-thread stride through a row of the tile is not a
        // multiple of the bank count; it changes which bank is read, never which value.
        threadgroup float tile[TILE][K_TILE + 1];

        float accumulator = 0.0f;
        for (uint base = 0; base < k; base += K_TILE) {
            uint span = min((uint)K_TILE, k - base);
            // The load is the whole point, and it is a transpose: lane `l` fills **column `l`** of every row
            // of the tile, so the lanes of a warp read `w[row * k + base + 0..TILE-1]` — TILE contiguous
            // floats, one 128-byte line — instead of each walking a different row 8 KB away. Loading each
            // lane's own row instead would be *correct* and exactly the stride this kernel exists to remove.
            // The guard is on the column, not on `active`: a lane past `span` has no column to fill, and a
            // row past `columns` has nothing to fill it from.
            for (uint r = 0; r < TILE; ++r) {
                if (group.x * TILE + r < columns && lane.x < span) {
                    tile[r][lane.x] = w[(group.x * TILE + r) * k + base + lane.x];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (active) {
                for (uint index = 0; index < span; ++index) {
                    // `metal::fma(x, w, 0)` is the correctly-rounded product whatever the compiler would
                    // prefer to do with `x * w` followed by an add: `D61` measured that every math mode this
                    // SDK offers contracts that pair into a single `fma`, which is a different number.
                    accumulator = accumulator + metal::fma(x[row * k + base + index], tile[lane.x][index], 0.0f);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (active) { out[row * columns + column] = accumulator; }
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
