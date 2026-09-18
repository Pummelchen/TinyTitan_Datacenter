import Foundation
import Metal

/// The contract's matmul over **bf16 weights**, which is how the install stores the LM head and the
/// smaller dense tensors.
///
/// The point is not arithmetic, it is traffic. `InstallFile.decodeRaw` turns a bf16 weight into `Float` at
/// four bytes where the file holds two, and the head is `[248320, 2048]`: **1.017 GB stored, 2.034 GB
/// decoded, per token** (`D109`). The conversion is exact — a bf16 is the top sixteen bits of an fp32 — so
/// a kernel that widens in a register produces the same number while the slab never exists.
///
/// The accumulation is `MetalMatmul`'s exactly: one threadgroup per `TILE` consecutive outputs, each thread
/// accumulating its own output **ascending in k**, `accumulator + metal::fma(x, w, 0.0f)` because `D61`
/// measured that to be the only spelling that keeps the contract's product and add apart. Widening a bf16
/// is a shift and cannot round, so the only arithmetic that can differ from the CPU path is the one that
/// does not (`MetalBf16MatmulTests` asserts the bit patterns over a grid).
public enum MetalBf16Matmul {
    public enum Error: Swift.Error {
        case noDevice
        case compileFailed(String)
        case commandFailed(String)
        case shapeMismatch(String)
    }

    /// Whether this host has a GPU. CI runners have none, so every test skips on this.
    public static var isAvailable: Bool { MTLCreateSystemDefaultDevice() != nil }

    /// Whether the LM head's blocks take the device path. **On by default, and measured** (`D109`):
    /// `head` fell **255 → 118 ms/step** and the step **1.718 → 1.580 s** in an alternated A/B on one binary,
    /// with the trace digest unchanged. `SHARD_GPU_BF16_HEAD=0` restores the CPU path.
    public static let enabled = ProcessInfo.processInfo.environment["SHARD_GPU_BF16_HEAD"] != "0"

    /// The threadgroup's width in outputs; the shader's `TILE` must equal it, and a mismatch would be a
    /// wrong answer rather than a slow one — so the grid the tests walk includes outputs that are not
    /// multiples of it.
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
        // The same non-fast setting the other kernels use. `D61` measured the accumulation exact under
        // `.fast`, `.relaxed` and `.safe` alike, and widening a bf16 is exact under all three.
        options.mathMode = .relaxed
        do {
            let library = try device.makeLibrary(source: shader, options: options)
            guard let function = library.makeFunction(name: "bf16_matmul") else {
                return .failure(.compileFailed("the library has no 'bf16_matmul' function"))
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

    /// `x` is `[rows, k]` and `w` is `[out, k]` of **little-endian bf16**, the layout the install stores and
    /// `InstallFile.decodeRaw` widens. The result is `[rows, out]`.
    public static func matmul(x: [Float], w: Data, rows: Int, k: Int, out: Int) throws -> [Float] {
        guard rows >= 0, k >= 0, out >= 0 else {
            throw Error.shapeMismatch("negative shape \(rows)x\(k)x\(out)")
        }
        guard x.count == rows * k else {
            throw Error.shapeMismatch("x has \(x.count) values, \(rows)x\(k) needs \(rows * k)")
        }
        guard w.count == out * k * 2 else {
            throw Error.shapeMismatch("w has \(w.count) bytes, \(out)x\(k) of bf16 needs \(out * k * 2)")
        }
        let values = rows * out
        guard values > 0 else { return [] }
        if k == 0 { return [Float](repeating: 0, count: values) }

        let pipeline = try Self.pipeline.current()
        return try Self.buffers.withBuffers(
            [rows * k * 4, out * k * 2, values * 4], device: pipeline.device
        ) { cached in
            let xBuffer = cached[0], wBuffer = cached[1], outBuffer = cached[2]
            x.withUnsafeBytes { source in
                if let base = source.baseAddress, rows * k > 0 {
                    xBuffer.contents().copyMemory(from: base, byteCount: rows * k * 4)
                }
            }
            w.withUnsafeBytes { source in
                if let base = source.baseAddress, out * k * 2 > 0 {
                    wBuffer.contents().copyMemory(from: base, byteCount: out * k * 2)
                }
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
            let tile = MetalBf16Matmul.tile
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

    private static let shader = """
    #include <metal_stdlib>
    using namespace metal;

    // `MetalMatmul`'s tiled body with a **bf16 weight**. The tile load is the whole difference: lane `l`
    // fills column `l` of every row of the tile from `w[row * k + base + l]`, so the lanes of a warp read
    // contiguous bf16 — the coalescing the first GPU matmul lacked (`D63`).
    //
    // The widening is `as_type<float>(uint(bits) << 16)`: a bf16 is the top half of an fp32, so this is a
    // move and cannot round. The accumulation is unchanged — ascending `k`, `fma(x, w, 0)` — and the tail
    // is **skipped rather than zero-padded**, because adding a zero term to a `-0.0` accumulator gives
    // `+0.0` and this project asserts zero signs bit for bit (`D34`).
    #define TILE 32
    #define K_TILE 32

    kernel void bf16_matmul(device const float *x [[buffer(0)]],
                            device const ushort *w [[buffer(1)]],
                            device float *out [[buffer(2)]],
                            constant uint4 &dims [[buffer(3)]],
                            uint2 group [[threadgroup_position_in_grid]],
                            uint2 lane [[thread_position_in_threadgroup]]) {
        uint rows = dims.x, k = dims.y, columns = dims.z;
        uint row = group.y;
        uint column = group.x * TILE + lane.x;
        // Predicated, never an early return: a barrier that some lanes have exited is undefined.
        bool active = row < rows && column < columns;

        threadgroup float tile[TILE][K_TILE + 1];

        float accumulator = 0.0f;
        for (uint base = 0; base < k; base += K_TILE) {
            uint span = min((uint)K_TILE, k - base);
            for (uint r = 0; r < TILE; ++r) {
                if (group.x * TILE + r < columns && lane.x < span) {
                    uint bits = uint(w[(group.x * TILE + r) * k + base + lane.x]);
                    tile[r][lane.x] = as_type<float>(bits << 16);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (active) {
                for (uint index = 0; index < span; ++index) {
                    accumulator = accumulator + metal::fma(x[row * k + base + index], tile[lane.x][index], 0.0f);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (active) { out[row * columns + column] = accumulator; }
    }
    """
}

private extension Result where Failure == MetalBf16Matmul.Error {
    func current() throws -> Success {
        switch self {
        case .success(let value): return value
        case .failure(let error): throw error
        }
    }
}
