import CryptoKit
import Foundation
import XCTest

@testable import DatacenterEngine

/// The read path that the development node's panics made into a stability requirement.
///
/// Two properties are tested, and they are the two that can be wrong at this layer: an uncached
/// read returns **the same bytes** as a mapped one — the point is to skip the buffer cache, not to
/// change the data — and an install's integrity is checked without reading the whole payload,
/// because reading the whole payload is what filled the page cache and drove an 8 GB node's free
/// disk from 17 GB to 2.96 GB in half a minute.
final class UncachedFileTests: XCTestCase {
    private var fixture: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().appendingPathComponent("Fixtures/tiny-qwen36")
    }

    private var installURL: URL { fixture.appendingPathComponent("install") }

    func testUncachedReadsReturnTheSameBytesAsMappedOnes() throws {
        let url = installURL.appendingPathComponent("data.bin")
        let mapped = try Data(contentsOf: url)
        let file = try UncachedFile(url: url)

        XCTAssertEqual(file.byteCount, mapped.count)
        XCTAssertTrue(file.isUncached)
        let whole = try file.read(offset: 0, byteCount: file.byteCount)
        XCTAssertEqual(Data(whole), mapped, "skipping the buffer cache must not change the bytes")
    }

    func testWindowedReadsMatchTheSameWindowOfTheWholeFile() throws {
        let url = installURL.appendingPathComponent("data.bin")
        let mapped = try Data(contentsOf: url)
        let file = try UncachedFile(url: url)
        // Deliberately unaligned, and deliberately not at zero: a real reader asks for a
        // quantized payload at an arbitrary offset.
        let offset = 8192 + 37
        let count = 5000
        let window = try file.read(offset: offset, byteCount: count)
        XCTAssertEqual(Data(window), mapped.subdata(in: offset..<(offset + count)))
    }

    func testReadingPastTheEndIsAnErrorRatherThanZeros() throws {
        let url = installURL.appendingPathComponent("data.bin")
        let file = try UncachedFile(url: url)
        XCTAssertThrowsError(try file.read(offset: file.byteCount - 4, byteCount: 8)) { error in
            guard case UncachedFile.Error.tooShort = error else {
                return XCTFail("expected a bounds error, got \(error)")
            }
        }
        XCTAssertThrowsError(try file.read(offset: -1, byteCount: 4))
    }

    func testTheStreamingDigestMatchesTheOneShotDigest() throws {
        let url = installURL.appendingPathComponent("data.bin")
        let mapped = try Data(contentsOf: url)
        let file = try UncachedFile(url: url)
        // A window deliberately smaller than the file, so the loop actually iterates.
        let streamed = try file.digest(window: 7000)
        XCTAssertEqual(streamed, Array(SHA256.hash(data: mapped)))
    }

    func testTheCachedModeIsAvailableForFilesWhosePagesAreWorthKeeping() throws {
        let url = installURL.appendingPathComponent("install.json")
        let file = try UncachedFile(url: url, uncached: false)
        XCTAssertFalse(file.isUncached)
        let text = String(decoding: try file.read(offset: 0, byteCount: file.byteCount), as: UTF8.self)
        XCTAssertTrue(text.contains("tensors"))
    }

    func testOpeningAMissingFileThrows() {
        let missing = installURL.appendingPathComponent("no-such-file")
        XCTAssertThrowsError(try UncachedFile(url: missing)) { error in
            guard case UncachedFile.Error.openFailed = error else {
                return XCTFail("expected an open failure, got \(error)")
            }
        }
    }
}

/// Integrity, after moving the check from the whole payload to the payload being read.
final class InstallVerificationTests: XCTestCase {
    private var fixture: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().appendingPathComponent("Fixtures/tiny-qwen36")
    }

    /// A copy of the fixture install with one byte flipped inside the first entry's payload.
    private func tamperedCopy() throws -> (url: URL, first: String, second: String) {
        let source = fixture.appendingPathComponent("install")
        let directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("install-tamper-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        for name in ["install.json", "data.bin"] {
            try FileManager.default.copyItem(
                at: source.appendingPathComponent(name), to: directory.appendingPathComponent(name)
            )
        }
        let manifest = try JSONSerialization.jsonObject(
            with: Data(contentsOf: directory.appendingPathComponent("install.json"))
        ) as! [String: Any]
        let tensors = manifest["tensors"] as! [[String: Any]]
        let first = tensors[0]
        let second = tensors[1]
        var bytes = try Data(contentsOf: directory.appendingPathComponent("data.bin"))
        let at = (first["offset"] as! Int) + 3
        bytes[at] = bytes[at] ^ 0xFF
        try bytes.write(to: directory.appendingPathComponent("data.bin"))
        return (directory, first["name"] as! String, second["name"] as! String)
    }

    func testATamperedTensorThrowsWhenReadAndNotWhenTheInstallIsOpened() throws {
        let (url, first, second) = try tamperedCopy()
        defer { try? FileManager.default.removeItem(at: url) }

        // Opening does not read the payload, which is the change that keeps a 20 GB install from
        // being a 20 GB read on every start.
        let install = try InstallFile(url: url)

        // The tensor that was not touched still reads.
        _ = try install.tensor(named: second)

        // The one that was touched fails its own digest, before any decoding could turn it into
        // plausible numbers.
        XCTAssertThrowsError(try install.tensor(named: first)) { error in
            guard case InstallFile.Error.digestMismatch(let name) = error else {
                return XCTFail("expected a digest mismatch, got \(error)")
            }
            XCTAssertEqual(name, first)
        }
    }

    func testAskingForVerificationOnOpenChecksEverything() throws {
        let (url, _, _) = try tamperedCopy()
        defer { try? FileManager.default.removeItem(at: url) }
        XCTAssertThrowsError(try InstallFile(url: url, verify: true)) { error in
            guard case InstallFile.Error.digestMismatch = error else {
                return XCTFail("expected a digest mismatch, got \(error)")
            }
        }
    }

    func testVerifyAllOnAnIntactInstallSucceeds() throws {
        let install = try InstallFile(url: fixture.appendingPathComponent("install"))
        try install.verifyAll()
    }

    func testAnIntactInstallStillReadsAfterAVerifiedEntryIsReadTwice() throws {
        // The memo means the second read does not hash again; it must still return the tensor.
        let install = try InstallFile(url: fixture.appendingPathComponent("install"))
        let once = try install.rows(named: "lm_head.weight", range: 0..<4)
        let twice = try install.rows(named: "lm_head.weight", range: 0..<4)
        XCTAssertEqual(once, twice)
    }
}
