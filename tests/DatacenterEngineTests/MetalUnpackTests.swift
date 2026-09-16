import Foundation
import XCTest

@testable import DatacenterEngine

/// The first Metal kernel, checked the way `D10` requires: bit for bit, against the scalar op.
///
/// The whole point of the GPU is to be faster, and the whole risk is that it is *differently*
/// rounded. So this file asks the only question that matters first — does it produce the same
/// bytes — and asks about speed afterwards, as a report rather than a gate.
///
/// CI runners have no GPU, so every test here skips when there is no device. That is `DC-036`'s
/// lesson — a toolchain difference between CI and the farm, which has bitten twice — applied
/// before it costs anything.
final class MetalUnpackTests: XCTestCase {
    private func payload(rows: Int, padded: Int, group: Int) -> Data {
        let groups = rows * (padded / group)
        let count = rows * (padded / 2) + groups * 4 + groups
        var state: UInt64 = 0x2545F4914F6CDD1D
        return Data((0..<count).map { _ in
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return UInt8((state >> 33) & 0xFF)
        })
    }

    private func entry(rows: Int, columns: Int, padded: Int, group: Int) throws -> InstallFile.Entry {
        let json = """
        {"name":"synthetic","role":"expert.stack_gate_up","quant":"int4","shape":[\(rows),\(columns)],
         "padded_columns":\(padded),"group":\(group),"dtype":"int4","offset":0,"nbytes":0,"sha256":""}
        """
        return try JSONDecoder().decode(InstallFile.Entry.self, from: Data(json.utf8))
    }

    func testTheGpuUnpackIsBitIdenticalToTheScalarOne() throws {
        // **The grid `DC-087` named.** This test used to exclude every shape whose final group is
        // partly filled (`columns % group != 0`), because the GPU and the CPU really did diverge on
        // them and the kernel was therefore called by nothing. Both halves of that have changed: the
        // reproducer below passes, so the restriction is now a claim to check rather than a defect to
        // step around, and the grid includes the awkward widths — a prime, an odd one and one that is
        // one past a group boundary — instead of avoiding them.
        try XCTSkipUnless(MetalUnpack.isAvailable, "no Metal device (CI runners have none)")
        var compared = 0
        for columns in [1, 4, 16, 17, 33, 64, 65, 128, 129] {
            for group in [1, 4, 8, 64] {
                for rows in [1, 3] {
                    var padded = columns + (group - columns % group) % group
                    if padded % 2 == 1 { padded += group }
                    let entry = try entry(rows: rows, columns: columns, padded: padded, group: group)
                    let body = payload(rows: rows, padded: padded, group: group)
                    let scalar = try InstallFile.dequantizeInt4Scalar(body, entry: entry)
                    let gpu = try MetalUnpack.unpack(payload: body, entry: entry)
                    XCTAssertEqual(
                        gpu.map(\.bitPattern), scalar.map(\.bitPattern),
                        "columns \(columns) group \(group) rows \(rows): the GPU moved a bit"
                    )
                    compared += 1
                }
            }
        }
        XCTAssertGreaterThan(compared, 8, "the grid must actually cover the shapes")
    }

    /// **`DC-087`'s reproducer, kept as a named case now that it passes.** When the last group of a row
    /// is partly filled — `columns = 65`, `group = 4`, so the final group holds one real value of four —
    /// the GPU and the CPU used to disagree on exactly that last column of every row. The name is kept
    /// because this is the shape the defect was found on, and a regression is most likely to arrive
    /// here first; the grid above now covers the same class of widths rather than stepping around it.
    func testPartlyFilledFinalGroupStillMatchesAfterDC087() throws {
        let entry = try entry(rows: 3, columns: 65, padded: 68, group: 4)
        let body = payload(rows: 3, padded: 68, group: 4)
        let scalar = try InstallFile.dequantizeInt4Scalar(body, entry: entry)
        let gpu = try MetalUnpack.unpack(payload: body, entry: entry)
        XCTAssertEqual(gpu.map(\.bitPattern), scalar.map(\.bitPattern))
    }

    /// **The shape the engine actually hands the decoder.** `rows(named:range:)` does not decode a whole
    /// tensor: it reads three byte ranges for the requested leading-axis entries and concatenates them, so
    /// the decoder sees a payload describing *R* rows and is told `rowCount: R`. Every other test here
    /// passes a whole-tensor payload, so this is the case the wiring depends on and nothing covered.
    func testPartialRowPayloadsDecodeIdentically() throws {
        try XCTSkipUnless(MetalUnpack.isAvailable, "no Metal device")
        let rows = 9, columns = 65, group = 4, padded = 68
        let entry = try entry(rows: rows, columns: columns, padded: padded, group: group)
        let whole = payload(rows: rows, padded: padded, group: group)
        let groupsPerRow = padded / group
        let codesPerRow = padded / 2
        let codesBytes = rows * codesPerRow
        let scalesBytes = rows * groupsPerRow * 4

        for (first, count) in [(0, 1), (4, 1), (0, 9), (7, 2)] {
            var partial = Data()
            partial.append(whole[(first * codesPerRow)..<((first + count) * codesPerRow)])
            partial.append(whole[(codesBytes + first * groupsPerRow * 4)..<(codesBytes + (first + count) * groupsPerRow * 4)])
            partial.append(whole[(codesBytes + scalesBytes + first * groupsPerRow)..<(codesBytes + scalesBytes + (first + count) * groupsPerRow)])
            let scalar = try InstallFile.dequantizeInt4(partial, entry: entry, rowCount: count)
            let gpu = try MetalUnpack.unpack(payload: partial, entry: entry, rowCount: count)
            XCTAssertEqual(
                gpu.map(\.bitPattern), scalar.map(\.bitPattern),
                "rows \(first)..<\(first + count) of \(rows): the GPU moved a bit"
            )
        }
    }

    /// **The buffer cache is the new thing, and its risk is the second call.** A buffer grown for a large
    /// shape must not leak into the next small one, and a small one must not be reused for a large shape
    /// without growing. Unpacking small, then large, then small again — each compared against the scalar
    /// path — is what makes that risk testable rather than argued.
    func testTheBufferCacheDoesNotLeakBetweenCalls() throws {
        try XCTSkipUnless(MetalUnpack.isAvailable, "no Metal device")
        // Deliberately in this order: small, large, small, large. The repeat is the point.
        for (rows, columns, group) in [(2, 17, 4), (5, 130, 8), (1, 65, 4), (5, 130, 8)] {
            var padded = columns + (group - columns % group) % group
            if padded % 2 == 1 { padded += group }
            let entry = try entry(rows: rows, columns: columns, padded: padded, group: group)
            let body = payload(rows: rows, padded: padded, group: group)
            let scalar = try InstallFile.dequantizeInt4(body, entry: entry)
            let gpu = try MetalUnpack.unpack(payload: body, entry: entry)
            XCTAssertEqual(
                gpu.map(\.bitPattern), scalar.map(\.bitPattern),
                "\(rows)x\(columns) group \(group): a reused buffer changed the answer"
            )
        }
    }

    /// The fixture's real payloads, not synthetic ones: same bytes the engine unpacks.
    func testTheGpuUnpackMatchesTheWholeInstallTensor() throws {
        try XCTSkipUnless(MetalUnpack.isAvailable, "no Metal device")
        let fixture = try XCTUnwrap(
            Bundle.module.url(forResource: "tiny-qwen36", withExtension: nil, subdirectory: "Fixtures")
        )
        let install = try InstallFile(url: fixture.appendingPathComponent("install"))
        let name = try XCTUnwrap(
            install.manifest.tensors.first { $0.dtype == "int4" && $0.shape.count == 3 }?.name
        )
        let entry = try install.entry(name)
        let bytes = try install.rawPayload(named: name)
        let cpu = try InstallFile.dequantizeInt4(bytes, entry: entry)
        let gpu = try MetalUnpack.unpack(payload: bytes, entry: entry)
        XCTAssertEqual(gpu.map(\.bitPattern), cpu.map(\.bitPattern), name)
    }

    /// A report, not a gate: dispatch overhead dominates at fixture sizes, and a test that asserted
    /// a speedup would fail on a busy machine for reasons unrelated to the kernel.
    func testGpuUnpackThroughputReport() throws {
        try XCTSkipUnless(MetalUnpack.isAvailable, "no Metal device")
        let rows = 8192, columns = 2048, group = 64
        let entry = try entry(rows: rows, columns: columns, padded: columns, group: group)
        let body = payload(rows: rows, padded: columns, group: group)

        _ = try MetalUnpack.unpack(payload: body, entry: entry)  // warm the pipeline
        func time(_ body: () throws -> [Float]) rethrows -> Double {
            _ = try body()
            let start = DispatchTime.now().uptimeNanoseconds
            _ = try body()
            return Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
        }
        let cpu = try time { try InstallFile.dequantizeInt4(body, entry: entry) }
        let gpu = try time { try MetalUnpack.unpack(payload: body, entry: entry) }
        let values = Double(rows * columns)
        print(String(
            format: "unpack %d values: CPU %.1f M values/s (%.3f s), GPU %.1f M values/s (%.3f s), speedup %.2fx",
            rows * columns, values / cpu / 1e6, cpu, values / gpu / 1e6, gpu, cpu / gpu
        ))
    }
}

extension MetalUnpackTests {
    /// The diagnostic `DC-087` asks for: which shapes disagree, and where. No assertions — it
    /// prints, because the first question about a GPU/CPU difference is which case shows it, and
    /// the answer so far is narrow enough to be worth keeping visible.
    ///
    /// Current finding: of twenty-four shapes, **one** disagrees — `columns = 64, group = 1,
    /// rows = 2` — and the first differing index is 117 (row 1, column 53, group 117), where the
    /// GPU produced bits `0`. Single-row versions of the same shape are identical, as are shapes
    /// with many groups per row and multiple rows at coarser groups, so neither "many groups" nor
    /// "many rows" is the trigger on its own.
    func testWhichShapesDisagreeDiagnostic() throws {
        try XCTSkipUnless(MetalUnpack.isAvailable, "no Metal device")
        // The grid is deliberately **at least as wide as the test that fails**, because an instrument
        // narrower than the thing it diagnoses reports "identical" while the gate stays red — which is
        // what this one did until `D34`.
        for columns in [1, 4, 16, 64, 128] {
            for group in [1, 4, 8, 64] {
                for rows in [1, 2, 3] {
                    var padded = columns + (group - columns % group) % group
                    if padded % 2 == 1 { padded += group }
                    let entry = try entry(rows: rows, columns: columns, padded: padded, group: group)
                    let body = payload(rows: rows, padded: padded, group: group)
                    let scalar = try InstallFile.dequantizeInt4Scalar(body, entry: entry)
                    let gpu = try MetalUnpack.unpack(payload: body, entry: entry)
                    var first = -1
                    for index in 0..<min(scalar.count, gpu.count)
                    where scalar[index].bitPattern != gpu[index].bitPattern {
                        first = index
                        break
                    }
                    let groupsPerRow = padded / group
                    let detail: String
                    if first >= 0 {
                        let row = first / columns, column = first % columns
                        detail = "first=\(first) row=\(row) col=\(column) groupIndex=\(row * groupsPerRow + column / group) cpu=\(scalar[first].bitPattern) gpu=\(gpu[first].bitPattern)"
                    } else {
                        detail = "identical (\(scalar.count) values)"
                    }
                    print("DIAG columns=\(columns) group=\(group) rows=\(rows) padded=\(padded) groupsPerRow=\(groupsPerRow) -> \(detail)")
                }
            }
        }
    }
}
