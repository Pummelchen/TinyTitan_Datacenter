import Foundation
import Metal

/// The first Metal kernel: unpacking four-bit codes.
///
/// `D10` chose this one deliberately. It is element-wise, its only floating-point operation is one
/// multiply, and it has no summation at all — so the "one accumulator per output, `k` ascending"
/// rule does not even apply to it, which makes it the kernel least able to argue with the contract
/// and the right place to prove the toolchain, the harness and the fast-math discipline.
///
/// **Fast math is off, and that is not a detail.** Metal compiles with it on by default, which is
/// the licence to reassociate and to fuse `a * b + c` into an FMA; a kernel whose source arithmetic
/// matches the CPU one can therefore produce different bits. `D10` requires `fastMathEnabled =
/// false`, and the test compares against the scalar op bit for bit rather than by tolerance.
///
/// The three sections are separate buffers rather than one with offsets, because a `device const
/// float *` must be aligned and the codes section's length is `rows * padded / 2`, which is not
/// always a multiple of four.
public enum MetalUnpack {
    public enum Error: Swift.Error {
        case noDevice
        case compileFailed(String)
        case commandFailed(String)
        case unexpectedLayout(String)
    }

    /// Whether this host has a GPU. CI runners have none, so every test skips on this.
    public static var isAvailable: Bool { MTLCreateSystemDefaultDevice() != nil }

    private struct Pipeline {
        let device: MTLDevice
        let queue: MTLCommandQueue
        let state: MTLComputePipelineState
    }

    private static let pipeline: Result<Pipeline, Error> = {
        guard let device = MTLCreateSystemDefaultDevice() else { return .failure(.noDevice) }
        guard let queue = device.makeCommandQueue() else {
            return .failure(.commandFailed("could not make a command queue"))
        }
        let options = MTLCompileOptions()
        // `D10`: Metal's default is fast math, which is the licence to reassociate and to fuse
        // `a * b + c` into an FMA. `mathMode = .relaxed` is the non-fast setting (`fastMathEnabled`
        // is deprecated as of macOS 15). For **this** kernel the setting cannot change a result —
        // there is no add and no reduction, only one multiply — but the GEMM kernel that follows
        // must not assume that, and will have to measure whether `.relaxed` is enough or whether
        // the accumulation needs to be guarded differently.
        options.mathMode = .relaxed
        do {
            let library = try device.makeLibrary(source: shader, options: options)
            guard let function = library.makeFunction(name: "unpack4") else {
                return .failure(.compileFailed("the library has no 'unpack4' function"))
            }
            return .success(Pipeline(device: device, queue: queue, state: try device.makeComputePipelineState(function: function)))
        } catch {
            return .failure(.compileFailed("\(error)"))
        }
    }()

    /// The same values as `InstallFile.dequantizeInt4`, on the GPU.
    public static func unpack(payload: Data, entry: InstallFile.Entry, rowCount: Int? = nil) throws -> [Float] {
        let pipeline = try Self.pipeline.current()
        let layout = try InstallFile.int4Layout(entry: entry, rowCount: rowCount, payloadBytes: payload.count)
        let values = layout.rows * layout.columns
        guard values > 0 else { return [] }

        let codes = payload.subdata(in: 0..<layout.codeBytes)
        let scales = payload.subdata(in: layout.codeBytes..<(layout.codeBytes + layout.scaleBytes))
        let zeros = payload.subdata(
            in: (layout.codeBytes + layout.scaleBytes)..<(layout.codeBytes + layout.scaleBytes + layout.rows * layout.groups)
        )

        func buffer(_ bytes: Data, _ label: String) throws -> MTLBuffer {
            guard let buffer = pipeline.device.makeBuffer(bytes: [UInt8](bytes), length: max(bytes.count, 1), options: .storageModeShared) else {
                throw Error.commandFailed("could not make the \(label) buffer")
            }
            return buffer
        }

        var params = SIMD4<UInt32>(UInt32(layout.rows), UInt32(layout.columns), UInt32(layout.padded), UInt32(layout.group))
        guard let out = pipeline.device.makeBuffer(length: values * 4, options: .storageModeShared) else {
            throw Error.commandFailed("could not make the output buffer")
        }
        guard let command = pipeline.queue.makeCommandBuffer(),
              let encoder = command.makeComputeCommandEncoder()
        else { throw Error.commandFailed("could not make a command buffer") }

        encoder.setComputePipelineState(pipeline.state)
        encoder.setBuffer(try buffer(codes, "codes"), offset: 0, index: 0)
        encoder.setBuffer(try buffer(scales, "scales"), offset: 0, index: 1)
        encoder.setBuffer(try buffer(zeros, "zeros"), offset: 0, index: 2)
        encoder.setBuffer(out, offset: 0, index: 3)
        encoder.setBytes(&params, length: MemoryLayout<SIMD4<UInt32>>.size, index: 4)
        let width = pipeline.state.threadExecutionWidth
        encoder.dispatchThreads(
            MTLSize(width: values, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1)
        )
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        if let error = command.error { throw Error.commandFailed("\(error)") }

        let raw = out.contents().bindMemory(to: Float.self, capacity: values)
        return Array(UnsafeBufferPointer(start: raw, count: values))
    }

    private static let shader = """
    #include <metal_stdlib>
    using namespace metal;

    // One thread per output value: no reduction, no accumulator, nothing to reassociate.
    kernel void unpack4(device const uchar *codes [[buffer(0)]],
                        device const float *scales [[buffer(1)]],
                        device const uchar *zeros [[buffer(2)]],
                        device float *out [[buffer(3)]],
                        constant uint4 &params [[buffer(4)]],
                        uint gid [[thread_position_in_grid]]) {
        uint rows = params.x, columns = params.y, padded = params.z, group = params.w;
        uint row = gid / columns;
        uint column = gid % columns;
        uint groupsPerRow = padded / group;
        uint groupIndex = row * groupsPerRow + column / group;
        float scale = scales[groupIndex];
        int zero = int(zeros[groupIndex]);
        if (zero >= 128) { zero -= 256; }
        uchar byte = codes[row * (padded / 2) + column / 2];
        uint nibble = (column % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
        int code = nibble >= 8 ? int(nibble) - 16 : int(nibble);
        out[gid] = float(code - zero) * scale;
    }
    """
}

private extension Result where Failure == MetalUnpack.Error {
    func current() throws -> Success {
        switch self {
        case .success(let value): return value
        case .failure(let error): throw error
        }
    }
}
