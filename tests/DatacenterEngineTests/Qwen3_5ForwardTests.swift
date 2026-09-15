import XCTest
import DatacenterIR

@testable import DatacenterEngine

/// The whole `qwen3_5` tower in Swift, against the contract's golden bit patterns.
///
/// The checkpoint is tiny on purpose (157 KB rather than 5 GB) but it is a *real* one: the
/// real naming, the real nesting in its configuration file, three Gated DeltaNet layers and
/// one full-attention layer, a partial RoPE and a tied head. So this exercises the reader,
/// the importer, the per-layer loading and the head — the wiring — and asserts bits, not
/// closeness, for every captured tensor.
final class Qwen3_5ForwardTests: XCTestCase {
    struct Vector: Decodable {
        var shape: [Int]
        var bits: [UInt32]
        var floats: [Float] { bits.map { Float(bitPattern: $0) } }
    }

    struct Golden: Decodable {
        var tokens: [Int]
        var layer_types: [String]
        var tensors: [String: Vector]
        var argmax: [Int]
    }

    private func checkpoint() throws -> URL {
        try XCTUnwrap(
            Bundle.module.url(forResource: "tiny-qwen35", withExtension: nil, subdirectory: "Fixtures")
        )
    }

    private func golden() throws -> Golden {
        let url = try checkpoint().appendingPathComponent("golden.json")
        return try JSONDecoder().decode(Golden.self, from: Data(contentsOf: url))
    }

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

    func testTheWholeTowerMatchesTheContractBitForBit() throws {
        let golden = try golden()
        let forward = try Qwen3_5Forward(snapshot: try checkpoint())
        let captured = try forward.forward(tokens: golden.tokens)

        var byName: [String: [Float]] = [:]
        for tensor in captured { byName[tensor.name] = tensor.values }

        for (name, expected) in golden.tensors.sorted(by: { $0.key < $1.key }) {
            guard let actual = byName[name] else {
                XCTFail("the engine did not capture \(name)")
                continue
            }
            assertSameBits(actual, expected, name)
        }
        XCTAssertEqual(captured.count, golden.tensors.count, "same set of captured tensors")
    }

    func testTheDiscreteDecisionAgreesExactly() throws {
        let golden = try golden()
        let forward = try Qwen3_5Forward(snapshot: try checkpoint())
        let captured = try forward.forward(tokens: golden.tokens)
        let logits = try XCTUnwrap(captured.first { $0.name == "logits" }).values

        var argmax: [Int] = []
        for position in 0..<golden.tokens.count {
            let offset = position * forward.config.vocabSize
            argmax.append(Greedy.argmax(logits, offset: offset, width: forward.config.vocabSize))
        }
        // I3: the discrete decisions are asserted as an index set, separately from any
        // numeric comparison.
        XCTAssertEqual(argmax, golden.argmax)
    }

    func testTheConfigurationComesFromTheCheckpointNotFromDefaults() throws {
        let forward = try Qwen3_5Forward(snapshot: try checkpoint())
        let config = forward.config
        XCTAssertEqual(config.numLayers, 4)
        XCTAssertEqual(config.fullAttentionInterval, 4)
        XCTAssertTrue(config.attnOutputGate)
        // The v5 config keeps rope settings in a nested block; reading a top-level key would
        // have left these at their defaults.
        XCTAssertEqual(config.ropeTheta, 1e7)
        XCTAssertEqual(config.partialRotaryFactor, 0.5)
        XCTAssertEqual(config.linearKeyDim, 16)
        XCTAssertEqual(config.linearValueDim, 16)
        XCTAssertEqual(config.linearConvKernelDim, 4)
    }

    func testTheRopeTablesArePartial() throws {
        let forward = try Qwen3_5Forward(snapshot: try checkpoint())
        let tables = forward.ropeTables(positions: [0, 1, 2])
        // head_dim 16 at partial_rotary_factor 0.5 leaves 8 channels rotating.
        XCTAssertEqual(tables.cos.count, 3 * 8)
        XCTAssertEqual(tables.sin.count, 3 * 8)
        for position in 0..<3 {
            for index in 0..<4 {
                XCTAssertEqual(tables.cos[position * 8 + index], tables.cos[position * 8 + index + 4])
            }
        }
    }

    /// The layer kinds must follow the checkpoint's own tensors; the tests above would pass
    /// on a model whose layers were all one kind only if the contract agreed, so this states
    /// which kinds the fixture actually has.
    func testTheFixtureExercisesBothLayerKinds() throws {
        let golden = try golden()
        XCTAssertEqual(golden.layer_types, ["linear_attention", "linear_attention", "linear_attention", "full_attention"])
    }


    /// The int4 install, decoded by the engine and compared bit for bit with the contract's
    /// own dequantization. Integer codes, integer zero points and fp32 scales: there is no
    /// rounding here to disagree about, so a tolerance would only hide a wrong layout.
    func testTheInt4InstallMatchesTheContractBitForBit() throws {
        let install = try checkpoint().appendingPathComponent("install")
        let goldenURL = try checkpoint().appendingPathComponent("golden-install.json")
        struct InstallGolden: Decodable {
            var tokens: [Int]
            var quantized: Int
            var kept: Int
            var tensors: [String: Vector]
        }
        let golden = try JSONDecoder().decode(InstallGolden.self, from: Data(contentsOf: goldenURL))
        XCTAssertGreaterThan(golden.quantized, 0, "the fixture must actually quantize something")

        let forward = try Qwen3_5Forward(install: install)
        let captured = try forward.forward(tokens: golden.tokens)
        var byName: [String: [Float]] = [:]
        for tensor in captured { byName[tensor.name] = tensor.values }
        for (name, expected) in golden.tensors.sorted(by: { $0.key < $1.key }) {
            guard let actual = byName[name] else {
                XCTFail("the engine did not capture \(name)")
                continue
            }
            assertSameBits(actual, expected, name)
        }
    }

    /// The install carries its own spec, so the engine needs nothing beside it — and the
    /// configuration comes back out of that spec, which is the check that the spec is
    /// sufficient to run the model rather than a description of one.
    func testTheInstallIsSelfDescribing() throws {
        let install = try checkpoint().appendingPathComponent("install")
        let file = try InstallFile(url: install)
        XCTAssertEqual(file.manifest.family, "qwen3_5")
        XCTAssertEqual(file.manifest.passes, ["quantize-group-affine-int4"])
        XCTAssertEqual(file.manifest.policy_files, ["tools/quant_policy.json"])
        XCTAssertEqual(file.manifest.spec.tensors.count, 55)
        let forward = try Qwen3_5Forward(install: install)
        XCTAssertEqual(forward.config.partialRotaryFactor, 0.5)
        XCTAssertEqual(forward.config.ropeTheta, 1e7)
    }

    /// The layout is the thing a reader can get wrong while every shape stays right: the
    /// codes are signed and the low nibble comes first.
    func testTheFourBitCodesAreSignedAndLowNibbleFirst() throws {
        let install = try checkpoint().appendingPathComponent("install")
        let file = try InstallFile(url: install)
        // A quantized weight, checked against the same arithmetic written out longhand.
        let name = try XCTUnwrap(file.manifest.tensors.first { $0.dtype == "int4" }?.name)
        let entry = try file.entry(name)
        let payload = try Data(contentsOf: install.appendingPathComponent("data.bin"))[
            entry.offset..<(entry.offset + entry.nbytes)
        ]
        let decoded = try file.tensor(named: name)
        let rows = entry.shape[0]
        let columns = entry.shape[1]
        let padded = entry.padded_columns
        let groups = padded / entry.group
        let codeBytes = rows * padded / 2
        var expected = [Float](repeating: 0, count: rows * columns)
        payload.withUnsafeBytes { raw in
            let base = raw.baseAddress!
            for row in 0..<rows {
                for index in 0..<columns {
                    let byte = base.loadUnaligned(fromByteOffset: row * (padded / 2) + index / 2, as: UInt8.self)
                    let nibble = index % 2 == 0 ? (byte & 0x0F) : (byte >> 4)
                    let code = Int(nibble) >= 8 ? Int(nibble) - 16 : Int(nibble)
                    let groupIndex = row * groups + index / entry.group
                    let scaleWord = base.loadUnaligned(fromByteOffset: codeBytes + groupIndex * 4, as: UInt32.self)
                    let zeroByte = base.loadUnaligned(fromByteOffset: codeBytes + groups * rows * 4 + groupIndex, as: UInt8.self)
                    let zero = Int(zeroByte) >= 128 ? Int(zeroByte) - 256 : Int(zeroByte)
                    expected[row * columns + index] = Float(code - zero) * Float(bitPattern: UInt32(littleEndian: scaleWord))
                }
            }
        }
        XCTAssertEqual(decoded, expected, "the reader and the format's own arithmetic must agree exactly")
    }

    /// Row-range reads are what let a 2 B model run on an 8 GB node, so they are checked
    /// against the whole-tensor read rather than assumed.
    func testRowReadsMatchTheWholeTensorRead() throws {
        let url = try XCTUnwrap(
            Bundle.module.url(forResource: "tiny", withExtension: "safetensors", subdirectory: "Fixtures")
        )
        let file = try SafetensorsFile(url: url)
        let whole = try file.float32("bf16.tensor")
        let rows = try file.float32("bf16.tensor", rows: 1..<2)
        XCTAssertEqual(rows, Array(whole[3..<6]), "row 1 of a [2, 3] tensor is elements 3..<6")
        let first = try file.float32("bf16.tensor", rows: 0..<1)
        XCTAssertEqual(first, Array(whole[0..<3]))
        XCTAssertThrowsError(try file.float32("bf16.tensor", rows: 2..<3), "reading past the end must fail")
    }
}
