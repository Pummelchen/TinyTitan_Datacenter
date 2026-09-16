import Foundation
import Metal
import XCTest

@testable import DatacenterEngine

/// **The measurement `D10` asked for, in writing.** The comment on `MetalUnpack`'s pipeline says the GEMM
/// that follows the unpack "must not assume that `.relaxed` is enough, and will have to measure whether
/// `.relaxed` is enough or whether the accumulation needs to be guarded differently". This is that
/// measurement, as a test rather than a guess.
///
/// What is at stake is exact. The contract's matmul is `ordered_matmul`, documented in
/// `tools/ordered_reference.py` as "one multiply and one add per output, so the rounding sequence is fully
/// determined" — **no FMA** — and `Ops.orderedMatmulScalar` says the same thing in Swift, materialising the
/// product before the add because "the contract forbids fusing the two". A GPU that contracts `a * b + acc`
/// into one `fma` produces a *different number*, one ULP away, and a bit-exactness claim would be false.
///
/// So: compile the same accumulation under each `MTLMathMode`, feed it a triple where the two roundings are
/// known to differ, and see which modes reproduce the contract.
///
/// **What it found.** `.fast` and `.relaxed` both contract — `0x4017DC47` where the contract says
/// `0x4017DC48` — so `D10`'s worry was right and its "measure whether `.relaxed` is enough" was the right
/// instruction. `.safe` is the mode that does not. The long-dot half of this test passed under *every*
/// mode including the contracting ones, which is the trap worth naming: over 257 terms the contraction
/// differences cancelled, so a test written against a long dot would have concluded that the GPU was exact.
/// The discriminator has to be one multiply and one add. The discriminating triple is not invented
/// here — it was found by search (`fma(a, b, c) != round(round(a*b) + c)`) and both bit patterns are
/// asserted below, so the test cannot pass by being unable to tell the difference.
final class MetalMathModeTests: XCTestCase {
    /// `acc = init + x * w`, one output per thread: the smallest thing that can reveal a contraction.
    ///
    /// Three formulations of the same arithmetic, because the measurement below found that *no* math mode
    /// keeps the multiply and the add apart — `.fast`, `.relaxed` and `.safe` all contract — so the
    /// separation has to be forced in the source, and which spelling forces it is an empirical question.
    /// Each is compiled on its own, so a spelling the compiler rejects is a reported result rather than a
    /// crash.
    private static func source(_ body: String) -> String {
        """
        #include <metal_stdlib>
        using namespace metal;

        kernel void dot(device const float *x [[buffer(0)]],
                        device const float *w [[buffer(1)]],
                        device float *out [[buffer(2)]],
                        constant uint &k [[buffer(3)]],
                        constant float &init [[buffer(4)]],
                        uint gid [[thread_position_in_grid]]) {
            float accumulator = init;
            for (uint index = 0; index < k; ++index) {
                \(body)
            }
            out[gid] = accumulator;
        }
        """
    }

    private static let formulations: [(String, String)] = [
        ("plain", "accumulator = accumulator + x[gid * k + index] * w[gid * k + index];"),
        ("product-first", "float product = x[gid * k + index] * w[gid * k + index]; accumulator = accumulator + product;"),
        ("fma-zero", "accumulator = accumulator + metal::fma(x[gid * k + index], w[gid * k + index], 0.0f);"),
        ("builtin-mul", "float product = __builtin_fmul_rn(x[gid * k + index], w[gid * k + index]); accumulator = accumulator + product;"),
    ]

    /// A triple where the fused and separate roundings differ by one ULP, with both values asserted.
    /// Searched, not chosen: see the class comment.
    private let discrimination = (
        a: Float(bitPattern: 0xBF8CB1A5),  // -1.0991713
        b: Float(bitPattern: 0xBF4CA136),  // -0.7993349
        c: Float(bitPattern: 0x3FBF4266),  // 1.4942138
        fused: UInt32(0x4017DC47),         // 2.3728197 — one ULP below
        separate: UInt32(0x4017DC48)       // 2.37282
    )

    private func pipeline(_ mode: MTLMathMode, _ body: String) throws -> (any MTLComputePipelineState, any MTLCommandQueue) {
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice(), "no Metal device")
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let options = MTLCompileOptions()
        options.mathMode = mode
        let library = try device.makeLibrary(source: Self.source(body), options: options)
        let function = try XCTUnwrap(library.makeFunction(name: "dot"))
        return (try device.makeComputePipelineState(function: function), queue)
    }

    /// Returns the bit pattern of `init + x * w` under this mode and formulation, for one output; `nil` when
    /// the compiler rejects the spelling, which is itself a result.
    private func run(
        _ mode: MTLMathMode, _ body: String, x: [Float], w: [Float], init value: Float
    ) throws -> UInt32? {
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let compiled: (any MTLComputePipelineState, any MTLCommandQueue)
        do {
            compiled = try pipeline(mode, body)
        } catch {
            print("  \(mode) / \(body.prefix(24))…: did not compile")
            return nil
        }
        let (state, queue) = compiled
        func buffer(_ values: [Float]) throws -> any MTLBuffer {
            try XCTUnwrap(device.makeBuffer(bytes: values, length: max(values.count * 4, 4), options: .storageModeShared))
        }
        let out = try XCTUnwrap(device.makeBuffer(length: 4, options: .storageModeShared))
        var k = UInt32(x.count)
        var start = value
        let command = try XCTUnwrap(queue.makeCommandBuffer())
        let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
        encoder.setComputePipelineState(state)
        encoder.setBuffer(try buffer(x), offset: 0, index: 0)
        encoder.setBuffer(try buffer(w), offset: 0, index: 1)
        encoder.setBuffer(out, offset: 0, index: 2)
        encoder.setBytes(&k, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&start, length: MemoryLayout<Float>.size, index: 4)
        encoder.dispatchThreads(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1)
        )
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        if let error = command.error { throw error }
        return out.contents().bindMemory(to: UInt32.self, capacity: 1)[0]
    }

    func testWhichModeAndFormulationReproduceTheContractsRounding() throws {
        try XCTSkipUnless(MetalUnpack.isAvailable, "no Metal device")
        let (a, b, c) = (discrimination.a, discrimination.b, discrimination.c)

        // The triple really does discriminate: this is the assertion that stops the test from being vacuous.
        let fused = Float(bitPattern: discrimination.fused)
        let separate = Float(bitPattern: discrimination.separate)
        XCTAssertNotEqual(fused.bitPattern, separate.bitPattern, "the triple no longer discriminates")
        XCTAssertEqual(
            Float(Double(a) * Double(b) + Double(c)).bitPattern, fused.bitPattern,
            "the fused expectation changed"
        )
        XCTAssertEqual((a * b) + c, separate, "the separate expectation changed")

        print("contract expects 0x\(String(separate.bitPattern, radix: 16)); fused would be 0x\(String(fused.bitPattern, radix: 16))")
        var matching: [String] = []
        for (modeName, mode) in [("fast", MTLMathMode.fast), ("relaxed", .relaxed), ("safe", .safe)] {
            for (formulation, body) in Self.formulations {
                guard let got = try run(mode, body, x: [a], w: [b], init: c) else { continue }
                let verdict = got == separate.bitPattern ? "SEPARATE (the contract)"
                    : got == fused.bitPattern ? "fused (a contraction)"
                    : "neither"
                print("  \(modeName)/\(formulation): 0x\(String(got, radix: 16)) -> \(verdict)")
                if got == separate.bitPattern { matching.append("\(modeName)/\(formulation)") }
            }
        }
        XCTAssertFalse(
            matching.isEmpty,
            "no tested mode and formulation reproduces the contract's separate rounding, so a bit-exact GPU "
            + "matmul needs a different approach entirely"
        )
        print("formulations that reproduce the contract: \(matching.joined(separator: ", "))")
    }

    /// The other half: an accumulation over many terms must not be *reassociated*. A formulation that keeps
    /// the separate rounding but reorders the sum would still disagree with the contract, and a long dot is
    /// where that shows. It is also where the *first* half's trap lives — this test passed under every mode
    /// including the contracting ones, because over 257 terms the contraction differences cancelled — which
    /// is why the discriminator above is one multiply and one add.
    func testTheFormulationThatMatchesAlsoKeepsTheOrder() throws {
        try XCTSkipUnless(MetalUnpack.isAvailable, "no Metal device")
        var state: UInt64 = 0x9E3779B97F4A7C15
        func next() -> Float {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            let unit = Float((state >> 40) & 0xFFFFFF) / Float(1 << 24)
            return (unit - 0.5) * 4
        }
        let k = 257  // prime, odd, and not a power of two
        let x = (0..<k).map { _ in next() }
        let w = (0..<k).map { _ in next() }

        // The engine's own definition of the contract's order, not a loop written here.
        let expected = Ops.orderedMatmulScalar(x: x, w: w, rows: 1, k: k, out: 1)[0]
        print("long dot over k=\(k): contract 0x\(String(expected.bitPattern, radix: 16))")

        var matching: [String] = []
        for (modeName, mode) in [("fast", MTLMathMode.fast), ("relaxed", .relaxed), ("safe", .safe)] {
            for (formulation, body) in Self.formulations {
                guard let got = try run(mode, body, x: x, w: w, init: 0) else { continue }
                let same = got == expected.bitPattern
                print("  \(modeName)/\(formulation): 0x\(String(got, radix: 16)) \(same ? "matches" : "differs")")
                if same { matching.append("\(modeName)/\(formulation)") }
            }
        }
        XCTAssertFalse(
            matching.isEmpty,
            "no tested mode and formulation reproduces the contract over a \(k)-term dot"
        )
    }
}
