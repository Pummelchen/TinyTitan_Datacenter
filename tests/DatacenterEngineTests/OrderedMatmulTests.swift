import XCTest

@testable import DatacenterEngine

/// Whether a faster matmul can keep the contract's bits — the question `DC-033` has to answer
/// before any kernel work, because `I1` and `I2` are defined by the *rounding sequence* and not by
/// the algebra.
///
/// Three things are measured here:
///
/// 1. a formulation that vectorises across the **output** dimension is bit-identical to the scalar
///    contract, for every shape including the awkward ones;
/// 2. a fused multiply-add is **not**, which is why `cblas_sgemm` — every BLAS uses FMA — cannot
///    be used for the ops the gate compares; and
/// 3. what the vector formulation actually buys, in GFLOP/s rather than in adjectives.
final class OrderedMatmulTests: XCTestCase {
    private func random(_ count: Int, seed: UInt64) -> [Float] {
        var state = seed &* 6364136223846793005 &+ 1442695040888963407
        return (0..<count).map { _ in
            state = state &* 6364136223846793005 &+ 1442695040888963407
            let unit = Float(state >> 40) / Float(1 << 24)
            return (unit - 0.5) * 4
        }
    }

    private func bits(_ values: [Float]) -> [UInt32] { values.map(\.bitPattern) }

    /// Every shape, including the ones a fast path is tempted to assume: `out` not a multiple of
    /// four, `k` odd, a single row, a single column.
    func testTheVectorFormulationIsBitIdenticalToTheScalarOne() {
        var compared = 0
        for rows in [1, 2, 3] {
            for k in [1, 2, 3, 7, 16, 17] {
                for out in [1, 2, 3, 4, 5, 8, 9, 15] {
                    let x = random(rows * k, seed: UInt64(rows * 1000 + k * 10 + out))
                    let w = random(out * k, seed: UInt64(rows * 2000 + k * 20 + out))
                    let scalar = Ops.orderedMatmulScalar(x: x, w: w, rows: rows, k: k, out: out)
                    let vector = Ops.orderedMatmulVectorized(x: x, w: w, rows: rows, k: k, out: out)
                    XCTAssertEqual(
                        bits(vector), bits(scalar),
                        "rows \(rows) k \(k) out \(out): the vector formulation changed the bits"
                    )
                    compared += 1
                }
            }
        }
        XCTAssertGreaterThan(compared, 100, "the grid must actually cover the shapes")
    }

    /// The measurement that settles the BLAS question. `Float.addingProduct` is a genuine fused
    /// multiply-add — one rounding instead of two — which is what every BLAS kernel does
    /// internally, so this is what a `cblas_sgemm` swap would cost in bit terms.
    ///
    /// It is a **rate**, not a yes-or-no, and that is the point: on most inputs the fused and
    /// ordered sums agree to the last bit, so a spot check proves nothing. What matters is how
    /// often they differ, because a gate compares every value.
    func testFusedMultiplyAddDisagreesWithTheContractOnSomeInputs() {
        var differing = 0
        let trials = 256
        for trial in 0..<trials {
            let k = 32 + (trial % 33)
            let x = random(k, seed: UInt64(trial) &* 2 &+ 1)
            let w = random(k, seed: UInt64(trial) &* 2 &+ 2)
            var ordered: Float = 0
            var fused: Float = 0
            for index in 0..<k {
                ordered = ordered + x[index] * w[index]
                fused = fused.addingProduct(x[index], w[index])
            }
            if ordered.bitPattern != fused.bitPattern { differing += 1 }
        }
        print("fma vs the contract: \(differing) of \(trials) dot products differ in the last bit")
        XCTAssertGreaterThan(
            differing, 0,
            "if no input differed, the compiler would be free to fuse and BLAS would be usable, "
                + "and the contract-vector tests would be passing for the wrong reason"
        )
    }

    /// A report, not a gate: the numbers go into the decision record, and a test that asserted a
    /// speedup would fail on a busy machine for reasons that have nothing to do with the kernel.
    func testMatmulThroughputReport() {
        let n = 256
        let x = random(n * n, seed: 1)
        let w = random(n * n, seed: 2)
        let macs = Double(n) * Double(n) * Double(n)

        func time(_ body: () -> [Float]) -> Double {
            _ = body()  // warm
            let start = DispatchTime.now().uptimeNanoseconds
            let result = body()
            let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
            XCTAssertEqual(result.count, n * n)
            return elapsed
        }

        let scalar = time { Ops.orderedMatmulScalar(x: x, w: w, rows: n, k: n, out: n) }
        let vector = time { Ops.orderedMatmulVectorized(x: x, w: w, rows: n, k: n, out: n) }
        print(String(
            format: "matmul %dx%dx%d: scalar %.1f GFLOP/s (%.3f s), vector %.1f GFLOP/s (%.3f s), speedup %.2fx",
            n, n, n, 2 * macs / scalar / 1e9, scalar, 2 * macs / vector / 1e9, vector, scalar / vector
        ))
    }

    /// Where the per-token time actually goes, which decides what to optimise next.
    ///
    /// The intuition is that a Mixture-of-Experts model is matmul-bound, and on this engine it is
    /// not: the gate measured 52.2 s per token for 6.9 GFLOP of matmul, and 6.9 GFLOP at the
    /// scalar rate above is well under a second. So the report below measures the *other* loop —
    /// unpacking four-bit codes into `Float` — which is what runs once per expert parameter.
    func testWhereTheTimeGoesReport() throws {
        // One real expert tensor's worth: `[256, 1024, 2048]` is 537 M values, so this is a
        // thirty-second of the stack and enough to see the rate.
        let rows = 8192
        let columns = 2048
        let padded = 2048
        let group = 64
        let payload = try makeInt4Payload(rows: rows, columns: columns, padded: padded, group: group)
        let entry = try makeEntry(rows: rows, columns: columns, padded: padded, group: group)

        let start = DispatchTime.now().uptimeNanoseconds
        let values = try InstallFile.dequantizeInt4(payload, entry: entry)
        let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9

        XCTAssertEqual(values.count, rows * columns)
        let rate = Double(rows * columns) / elapsed / 1e6
        print(String(
            format: "dequantize %d values (%.1f MB int4): %.3f s, %.1f M values/s; "
                + "one 35 B token dequantizes ~3.45 B values, so ~%.1f s per token of unpacking",
            rows * columns, Double(payload.count) / 1e6, elapsed, rate, 3.45e9 / (rate * 1e6)
        ))
    }

    private func makeInt4Payload(rows: Int, columns: Int, padded: Int, group: Int) throws -> Data {
        let groups = rows * (padded / group)
        // A deterministic pattern rather than zeros, so nothing averages out to something a
        // shorter loop would also produce.
        return Data((0..<(rows * (padded / 2) + groups * 4 + groups)).map { UInt8($0 % 251) })
    }

    private func makeEntry(rows: Int, columns: Int, padded: Int, group: Int) throws -> InstallFile.Entry {
        let json = """
        {"name":"synthetic","role":"expert.stack_gate_up","quant":"int4","shape":[\(rows),\(columns)],
         "padded_columns":\(padded),"group":\(group),"dtype":"int4","offset":0,"nbytes":0,"sha256":""}
        """
        return try JSONDecoder().decode(InstallFile.Entry.self, from: Data(json.utf8))
    }
}
