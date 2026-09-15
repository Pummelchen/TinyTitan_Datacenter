import Foundation
import XCTest

@testable import DatacenterEngine

/// Unpacking four-bit codes, where the measurement says the time actually is.
///
/// `D9` measured the matmul at 4.1 GFLOP/s and the unpack at 370.9 M values/s, so a 35 B token
/// spends about 9.3 s unpacking against 1.7 s multiplying. The unpack is the next kernel — and it
/// is a much easier one than the matmul, for a structural reason worth stating: its **only**
/// floating-point operation is one multiply, `Float(code - zero) * scale`, with no summation
/// anywhere. There is no accumulation order to preserve, so a vector formulation is bit-identical
/// by construction rather than by luck, and these tests check the construction anyway.
final class Int4UnpackTests: XCTestCase {
    private func payload(rows: Int, padded: Int, group: Int) -> Data {
        let groups = rows * (padded / group)
        let count = rows * (padded / 2) + groups * 4 + groups
        // Deterministic, and deliberately including every nibble value, every scale byte pattern
        // and every zero-point byte, so sign extension and the zero point's own sign are both
        // exercised rather than hoped for.
        var state: UInt64 = 0x9E3779B97F4A7C15
        var bytes = [UInt8](repeating: 0, count: count)
        for index in 0..<count {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            bytes[index] = UInt8((state >> 33) & 0xFF)
        }
        for index in 0..<min(256, count) { bytes[index] = UInt8(index) }
        return Data(bytes)
    }

    private func entry(rows: Int, columns: Int, padded: Int, group: Int) throws -> InstallFile.Entry {
        let json = """
        {"name":"synthetic","role":"expert.stack_gate_up","quant":"int4","shape":[\(rows),\(columns)],
         "padded_columns":\(padded),"group":\(group),"dtype":"int4","offset":0,"nbytes":0,"sha256":""}
        """
        return try JSONDecoder().decode(InstallFile.Entry.self, from: Data(json.utf8))
    }

    /// The grid, chosen around the places a four-wide loop breaks: `columns` not a multiple of
    /// four, the padded tail, a group of one, and a single row.
    func testTheVectorUnpackIsBitIdenticalToTheScalarOne() throws {
        var compared = 0
        for columns in [1, 3, 4, 5, 6, 16, 17, 63, 64, 65] {
            for group in [1, 4, 8, 64] {
                for rows in [1, 3] {
                    // Swift's `%` keeps the sign of the dividend, so `(-columns) % group` is
                    // negative and `columns + that` can be *smaller* than columns — which made
                    // this grid compare two implementations on an impossible layout and very
                    // nearly sent me hunting a bug in the wrong function. Build the padding
                    // positively.
                    var padded = columns + (group - columns % group) % group
                    if padded % 2 == 1 { padded += group }
                    let entry = try entry(rows: rows, columns: columns, padded: padded, group: group)
                    let body = payload(rows: rows, padded: padded, group: group)
                    let scalar = try InstallFile.dequantizeInt4Scalar(body, entry: entry)
                    let vector = try InstallFile.dequantizeInt4(body, entry: entry)
                    XCTAssertEqual(
                        vector.map(\.bitPattern), scalar.map(\.bitPattern),
                        "columns \(columns) group \(group) rows \(rows): the vector unpack moved a bit"
                    )
                    XCTAssertEqual(vector.count, rows * columns)
                    compared += 1
                }
            }
        }
        XCTAssertGreaterThan(compared, 50, "the grid must actually cover the shapes")
    }

    func testAPayloadOfTheWrongLengthIsStillRefused() throws {
        let entry = try entry(rows: 2, columns: 32, padded: 32, group: 8)
        let body = payload(rows: 2, padded: 32, group: 8)
        XCTAssertThrowsError(try InstallFile.dequantizeInt4(body.dropLast(), entry: entry))
        XCTAssertThrowsError(try InstallFile.dequantizeInt4(body + Data([0]), entry: entry))
    }

    func testAGroupThatDoesNotDivideThePaddedWidthIsRefused() throws {
        let entry = try entry(rows: 1, columns: 32, padded: 32, group: 7)
        let body = payload(rows: 1, padded: 32, group: 8)  // self-consistent payload, bad header
        XCTAssertThrowsError(try InstallFile.dequantizeInt4(body, entry: entry))
    }

    /// `D11`: a denormal scale decodes to zero, on both paths, and they agree.
    ///
    /// Metal flushes denormal operands and the CPU does not, so the contract defines the flush rather
    /// than discovering it — measured at 0.722705 % of the real install's scales. This is the test
    /// that pins the definition, and the fixture cannot: it contains no denormals at all.
    func testADenormalScaleDecodesToZeroOnBothPaths() throws {
        let entry = try entry(rows: 1, columns: 2, padded: 2, group: 2)
        // codes = one byte (two nibbles), scales = one denormal float, zeros = one byte.
        var body = Data([0x51])                                   // nibbles 1 and 5
        body.append(contentsOf: withUnsafeBytes(of: Float(1e-40).bitPattern.littleEndian) { Array($0) })
        body.append(UInt8(0))                                     // zero point 0
        XCTAssertLessThan(Float(1e-40), Float.leastNormalMagnitude, "1e-40 must be denormal for this test")

        let vector = try InstallFile.dequantizeInt4(body, entry: entry)
        let scalar = try InstallFile.dequantizeInt4Scalar(body, entry: entry)
        XCTAssertEqual(vector, [0, 0], "a denormal scale must flush to zero")
        XCTAssertEqual(vector.map(\.bitPattern), scalar.map(\.bitPattern), "the two paths must agree")
        XCTAssertEqual(InstallFile.flushed(Float(1e-40)), 0)
        XCTAssertEqual(InstallFile.flushed(Float(1e-20)), Float(1e-20), "normal scales are untouched")
        XCTAssertEqual(InstallFile.flushed(-Float(1e-20)), -Float(1e-20), "and their sign survives")
    }

    /// A report, not a gate. The rate is what decides whether the unpack is worth more work.
    func testUnpackThroughputReport() throws {
        let rows = 8192
        let columns = 2048
        let group = 64
        let entry = try entry(rows: rows, columns: columns, padded: columns, group: group)
        let body = payload(rows: rows, padded: columns, group: group)

        func time(_ body: () throws -> [Float]) rethrows -> Double {
            _ = try body()
            let start = DispatchTime.now().uptimeNanoseconds
            _ = try body()
            return Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
        }

        let scalar = try time { try InstallFile.dequantizeInt4Scalar(body, entry: entry) }
        let vector = try time { try InstallFile.dequantizeInt4(body, entry: entry) }
        let values = Double(rows * columns)
        print(String(
            format: "unpack %d values: scalar %.1f M values/s (%.3f s), vector %.1f M values/s (%.3f s), speedup %.2fx; "
                + "a 35 B token (3.45 G values) unpacks in %.1f s",
            rows * columns, values / scalar / 1e6, scalar, values / vector / 1e6, vector, scalar / vector,
            3.45e9 / (values / vector)
        ))
    }
}
