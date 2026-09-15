import XCTest

@testable import DatacenterEngine

/// The engine's ops against the numeric contract, asserted as **bit patterns**.
///
/// A tolerance would pass while the engine quietly accumulated in a different order, and
/// the whole point of the contract is that the order is fixed. Closeness is checked
/// elsewhere, against the semantic oracle; here the only acceptable answer is equality.
final class ContractVectorTests: XCTestCase {
    struct Vector: Decodable {
        var shape: [Int]
        var bits: [UInt32]

        var floats: [Float] { bits.map { Float(bitPattern: $0) } }
    }

    struct Matmul: Decodable { var x: Vector; var w: Vector; var out: Vector }
    struct Single: Decodable { var x: Vector; var out: Vector }
    struct Norm: Decodable { var x: Vector; var weight: Vector; var eps: Double; var out: Vector }
    struct Rope: Decodable { var positions: [Double]; var head_dim: Int; var theta: Double; var cos: Vector; var sin: Vector }
    struct ApplyRope: Decodable {
        var x: Vector; var cos: Vector; var sin: Vector; var out: Vector
        enum CodingKeys: String, CodingKey { case x, cos, sin, out }
    }

    struct Fixture: Decodable {
        var matmul: Matmul
        var sum: Single
        var rms_norm: Norm
        var silu: Single
        var sigmoid: Single
        var softmax: Single
        var rope: Rope
        var apply_rope: ApplyRope

        enum CodingKeys: String, CodingKey {
            case matmul, sum, silu, sigmoid, softmax, rope
            case rms_norm = "rms_norm"
            case apply_rope = "apply_rope"
        }
    }

    private func fixture() throws -> Fixture {
        let url = try XCTUnwrap(
            Bundle.module.url(forResource: "contract-vectors", withExtension: "json", subdirectory: "Fixtures")
        )
        return try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: url))
    }

    /// Report the first differing element, because "not equal" on 4096 numbers is not a
    /// diagnosis.
    private func assertSameBits(
        _ actual: [Float], _ expected: Vector, _ label: String, file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(actual.count, expected.bits.count, "\(label): count", file: file, line: line)
        for index in 0..<min(actual.count, expected.bits.count) where actual[index].bitPattern != expected.bits[index] {
            let want = Float(bitPattern: expected.bits[index])
            XCTFail(
                "\(label)[\(index)]: got \(actual[index]) (0x\(String(actual[index].bitPattern, radix: 16))) "
                    + "want \(want) (0x\(String(expected.bits[index], radix: 16)))",
                file: file, line: line
            )
            return
        }
    }

    func testOrderedMatmulMatchesTheContract() throws {
        let vector = try fixture().matmul
        let result = Ops.orderedMatmul(
            x: vector.x.floats, w: vector.w.floats, rows: vector.x.shape[0], k: vector.x.shape[1], out: vector.w.shape[0]
        )
        assertSameBits(result, vector.out, "matmul")
    }

    func testOrderedSumMatchesTheContract() throws {
        let vector = try fixture().sum
        assertSameBits([Ops.orderedSum(vector.x.floats)], vector.out, "sum")
    }

    func testRmsNormMatchesTheContract() throws {
        let vector = try fixture().rms_norm
        let result = Ops.rmsNorm(
            x: vector.x.floats, weight: vector.weight.floats,
            rows: vector.x.shape[0], width: vector.x.shape[1], eps: Float(vector.eps)
        )
        assertSameBits(result, vector.out, "rms_norm")
    }

    func testSiluMatchesTheContractIncludingTheExtremes() throws {
        let vector = try fixture().silu
        assertSameBits(Ops.silu(vector.x.floats), vector.out, "silu")
    }

    func testSigmoidMatchesTheContract() throws {
        let vector = try fixture().sigmoid
        assertSameBits(vector.x.floats.map(Ops.sigmoid), vector.out, "sigmoid")
    }

    func testSoftmaxMatchesTheContract() throws {
        let vector = try fixture().softmax
        let result = Ops.softmax(x: vector.x.floats, rows: vector.x.shape[0], width: vector.x.shape[1])
        assertSameBits(result, vector.out, "softmax")
    }

    func testRopeTablesMatchTheContract() throws {
        let vector = try fixture().rope
        let tables = Ops.ropeTables(headDim: vector.head_dim, positions: vector.positions, theta: vector.theta)
        assertSameBits(tables.cos, vector.cos, "rope.cos")
        assertSameBits(tables.sin, vector.sin, "rope.sin")
    }

    func testApplyRopeMatchesTheContract() throws {
        let vector = try fixture().apply_rope
        let shape = vector.x.shape
        let result = Ops.applyRope(
            x: vector.x.floats, cos: vector.cos.floats, sin: vector.sin.floats,
            tokens: shape[0], heads: shape[1], headDim: shape[2]
        )
        assertSameBits(result, vector.out, "apply_rope")
    }

    /// The contract's transcendentals are defined as "computed in double, rounded to
    /// Float". This is the test that keeps that rule honest: if a future change makes
    /// `exp32` call a float32 exponential, this fails.
    func testTranscendentalsAreTheDoubleRoundedForm() throws {
        let samples: [Float] = [-30, -1.5, -0.25, 0, 0.25, 1.5, 30, 88]
        for value in samples {
            XCTAssertEqual(Ops.exp32(value).bitPattern, Float(exp(Double(value))).bitPattern, "exp(\(value))")
        }
    }
}
