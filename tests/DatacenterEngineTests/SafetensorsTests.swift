import XCTest

@testable import DatacenterEngine

/// The reader against a tiny fixture, so the header cases are covered without a 5 GB
/// checkpoint. Every one of these was a real failure while writing the reader against a
/// real checkpoint: the `__metadata__` block is not a tensor, dtype names are upper case,
/// and the payload is little-endian.
final class SafetensorsTests: XCTestCase {
    private func fixture() throws -> SafetensorsFile {
        let url = try XCTUnwrap(
            Bundle.module.url(forResource: "tiny", withExtension: "safetensors", subdirectory: "Fixtures")
        )
        return try SafetensorsFile(url: url)
    }

    func testMetadataBlockIsNotMistakenForATensor() throws {
        let file = try fixture()
        XCTAssertEqual(file.names, ["bf16.tensor", "f16.tensor", "f32.tensor"])
        XCTAssertEqual(file.metadata["purpose"], "reader-test")
        XCTAssertEqual(file.metadata["format"], "pt")
    }

    func testUpperCaseDtypesAreRead() throws {
        let file = try fixture()
        XCTAssertEqual(try file.info("bf16.tensor").dtype, "BF16")
        XCTAssertEqual(try file.info("f32.tensor").dtype, "F32")
        XCTAssertEqual(try file.info("f16.tensor").dtype, "F16")
    }

    func testShapesAndValuesSurviveTheRoundTrip() throws {
        let file = try fixture()
        XCTAssertEqual(try file.info("bf16.tensor").shape, [2, 3])
        XCTAssertEqual(try file.info("f32.tensor").shape, [4])

        // Values chosen to be exactly representable, so this tests the decoding rather
        // than the rounding.
        XCTAssertEqual(try file.float32("f32.tensor"), [1.5, 2.5, -3.5, 4.5])
        XCTAssertEqual(try file.float32("f16.tensor"), [0.25, -0.75])
        XCTAssertEqual(try file.float32("bf16.tensor"), [1.0, -2.5, 0.5, 0.0, 3.25, -0.125])
    }

    func testUnknownTensorIsRefused() throws {
        let file = try fixture()
        XCTAssertThrowsError(try file.float32("model.layers.0.nonexistent.weight"))
    }

    /// bf16 → fp32 is a shift, not a rounded conversion, so it cannot lose information.
    /// The test states that rather than trusting the implementation comment.
    func testBfloat16WideningIsExact() throws {
        let words: [UInt16] = [0x3F80, 0xBF80, 0x0001, 0x7F7F, 0x8000]
        for word in words {
            let widened = Float(bitPattern: UInt32(word) << 16)
            let roundTripped = widened.bitPattern >> 16
            XCTAssertEqual(UInt16(roundTripped), word, "bf16 0x\(String(word, radix: 16)) did not round-trip")
        }
    }
}
