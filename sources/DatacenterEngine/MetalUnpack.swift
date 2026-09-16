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
        let rawScales = payload.subdata(in: layout.codeBytes..<(layout.codeBytes + layout.scaleBytes))
        // `D11` on the GPU, which this kernel did not do: a denormal scale reads as zero, exactly as the
        // scalar path and `tools/quantize.py` both do. The divergence `DC-087` recorded as "the sign of
        // zero" was therefore two things — the sign, and a rule this path had never implemented at all.
        // Flushing here is one pass per dispatch instead of a branch per value, and it routes through the
        // same `flushed` the CPU uses, so the rule cannot drift between them.
        var flushedScales = [Float](repeating: 0, count: layout.rows * layout.groups)
        rawScales.withUnsafeBytes { raw in
            for index in 0..<flushedScales.count {
                let bits = raw.loadUnaligned(fromByteOffset: index * 4, as: UInt32.self)
                flushedScales[index] = InstallFile.flushed(
                    Float(bitPattern: UInt32(littleEndian: bits))
                )
            }
        }
        let scales = Data(bytes: flushedScales, count: flushedScales.count * 4)
        let zeros = payload.subdata(
            in: (layout.codeBytes + layout.scaleBytes)..<(layout.codeBytes + layout.scaleBytes + layout.rows * layout.groups)
        )

        func buffer(_ bytes: Data, _ label: String) throws -> any MTLBuffer {
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
        // `D34`: a zero carries no sign. `DC-087`'s residue was one index of one shape out of
        // twenty-four where this kernel produced `+0.0` and the scalar path produced `-0.0` — and, once
        // the scalar path was normalised, the same disagreement in the other direction, which is why
        // both sides define it rather than one matching the other. `+ 0.0f` is exact here because fast
        // math is off (above): it maps `-0.0` to `+0.0` and leaves every other value alone, and LLVM may
        // only fold it away under `nsz`, which off is not.
        float value = float(code - zero) * scale;
        // `D34`, and the reason these are branches rather than `value + 0.0f`: the Metal compiler folded
        // that addition away — correctly, under the fast-math flags it defaults to, whatever the intent of
        // the options here — so nine values out of 195 still came out as `-0.0` while the scalar path
        // produced `+0.0`. A branch on a comparison cannot be folded, and the replacement is written from
        // the **bit pattern** so that no arithmetic rule can reinterpret it. A zero code times an infinite
        // scale is the only NaN this kernel can make, and the canonical quiet NaN is the agreed value.
        if (value == 0.0f) { value = as_type<float>(0u); }
        if (isnan(value)) { value = as_type<float>(0x7FC00000u); }
        out[gid] = value;
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
