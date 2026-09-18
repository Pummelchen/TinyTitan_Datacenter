import Foundation
import XCTest

import DatacenterIR

@testable import DatacenterEngine

/// The **stored-form** accessors: `packedRows` for int4 and `storedRows` for bf16 (`D108`, `D109`).
///
/// Both exist so a device kernel can consume what the file holds instead of what the decoder produces — half
/// a byte per weight rather than four, and two bytes rather than four. The assertions are the same shape as
/// `InstallRowReadTests`: the stored bytes must decode to **exactly** what `rows` returns, or the fast path
/// is a different arithmetic rather than a different representation.
final class InstallStoredFormTests: XCTestCase {
    private func install() throws -> InstallFile {
        let fixture = try XCTUnwrap(
            Bundle.module.url(forResource: "tiny-qwen36", withExtension: nil, subdirectory: "Fixtures")
        )
        return try InstallFile(url: fixture.appendingPathComponent("install"))
    }

    private func firstEntry(_ install: InstallFile, where predicate: (InstallFile.Entry) -> Bool) throws
        -> InstallFile.Entry
    {
        try XCTUnwrap(install.manifest.tensors.first(where: predicate))
    }

    /// An int4 stacked tensor: the packed bytes hand back a payload the decoder turns into the same values,
    /// and the bytes are a quarter of the `[Float]` the ordinary read produces.
    func testPackedRowsDecodeToTheSameValuesAsTheRowRead() throws {
        let install = try install()
        let entry = try firstEntry(install) { $0.dtype == "int4" && $0.shape.count == 3 && $0.shape[0] > 1 }
        let range = 0..<1
        let packed = try XCTUnwrap(
            try install.packedRows(named: entry.name, range: range),
            "an int4 install must hand over packed rows"
        )
        XCTAssertEqual(packed.entry.name, entry.name)
        let decoded = try InstallFile.dequantizeInt4(
            packed.payload, entry: entry, rowCount: packed.payloadRows
        )
        let ordinary = try install.rows(named: entry.name, range: range)
        XCTAssertEqual(decoded.map(\.bitPattern), ordinary.map(\.bitPattern), "packed bytes decoded differently")
        XCTAssertLessThan(packed.payload.count, ordinary.count * 4, "packed must be smaller than fp32")
    }

    /// A bf16 tensor: the stored bytes widen to exactly what the row read returns, and the whole tensor is
    /// read **once** — the second request is a payload-cache hit, which is the difference `D109` is about.
    func testStoredRowsMatchTheRowReadAndAreCached() throws {
        let install = try install()
        let entry = try firstEntry(install) { $0.dtype == "bf16" && $0.shape.count == 2 && $0.shape[0] > 8 }
        let range = 0..<4
        let stored = try XCTUnwrap(
            try install.storedRows(named: entry.name, range: range),
            "a bf16 install must hand over stored rows"
        )
        XCTAssertEqual(stored.dtype, "bf16")
        XCTAssertEqual(stored.rowCount, 4)
        let widened = try InstallFile.decodeRaw(
            stored.data, dtype: stored.dtype, elementCount: stored.rowCount * stored.width
        )
        let ordinary = try install.rows(named: entry.name, range: range)
        XCTAssertEqual(widened.map(\.bitPattern), ordinary.map(\.bitPattern), "stored bytes widened differently")

        let hitsBefore = install.payloadCacheMetrics.hits
        let again = try XCTUnwrap(try install.storedRows(named: entry.name, range: 4..<8))
        XCTAssertFalse(again.data.isEmpty)
        XCTAssertGreaterThan(
            install.payloadCacheMetrics.hits, hitsBefore,
            "the second range of one tensor must be served from the payload cache"
        )
    }

    /// A source with no stored form says so, which is what sends the caller back to `rows`. The array-backed
    /// provider is that case, and the default implementation is what makes it free.
    func testASourceWithoutAStoredFormAnswersNil() throws {
        let entry = try JSONDecoder().decode(InstallFile.Entry.self, from: Data("""
        {"name":"synthetic","role":"expert.stack_gate_up","quant":"int4","shape":[2,8],
         "padded_columns":8,"group":8,"dtype":"int4","offset":0,"nbytes":0,"sha256":""}
        """.utf8))
        let source = ArraySource(rows: [0, 1, 2, 3, 4, 5, 6, 7])
        XCTAssertNil(try source.packedRows(named: "synthetic", range: 0..<1))
        XCTAssertNil(try source.storedRows(named: entry.name, range: 0..<1))
        XCTAssertEqual(source.packedCacheBudget, 0)
    }
}

/// The smallest possible `WeightSource`: it holds fp32 and nothing else, which is exactly the case the two
/// stored-form methods have to answer `nil` for.
private struct ArraySource: WeightSource {
    let rows: [Float]
    func tensor(named name: String) throws -> [Float] { rows }
    func rows(named name: String, range: Range<Int>) throws -> [Float] { Array(rows[range]) }
}
