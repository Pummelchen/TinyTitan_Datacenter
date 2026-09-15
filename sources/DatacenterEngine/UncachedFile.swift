import CryptoKit
import Foundation

#if canImport(Darwin)
import Darwin
#endif

/// A file whose reads deliberately bypass the unified buffer cache.
///
/// The brief's runtime I/O rule for expert slabs is `F_NOCACHE` / `O_DIRECT`, and measuring it
/// here turned that rule from a throughput optimisation into a stability requirement. Verifying
/// the 20 GB install — a sequential read plus a sha256 over every entry — took free disk from
/// 17 GB to 2.96 GB in about thirty seconds on an 8 GB node: the read filled the page cache, the
/// page cache filled memory, memory pressure made macOS grow swap, and swap is disk. That is the
/// same loop that panicked the development machine twice, and none of it is visible in the
/// arithmetic.
///
/// So a payload larger than the machine is read through this type rather than through `mmap` or
/// `Data(contentsOf:)`. The cost is real — every read is a syscall and the pages are not reused —
/// and it is the right cost for a slab that would otherwise evict everything useful.
///
/// `F_NOCACHE` cannot be read back from the descriptor, so this type cannot prove to a caller
/// that the kernel honoured it. What it can do, and does, is make the intent explicit at the one
/// place that opens a payload file, and it is tested for the property that matters: uncached and
/// mapped reads return the same bytes.
public final class UncachedFile {
    public enum Error: Swift.Error, Equatable {
        case openFailed(String, Int32)
        case tooShort(expected: Int, got: Int)
        case readFailed(offset: Int, errno: Int32)
    }

    /// Whether to ask the kernel to skip the buffer cache. Exposed so the decision is visible at
    /// the call site and assertable in a test; `false` is only for small files whose pages are
    /// worth keeping, such as a manifest.
    public let isUncached: Bool
    public let byteCount: Int
    private let descriptor: Int32

    public init(url: URL, uncached: Bool = true) throws {
        let descriptor = open(url.path, O_RDONLY)
        guard descriptor >= 0 else { throw Error.openFailed(url.path, errno) }
        if uncached {
            // Advisory and best-effort: if the kernel declines, the reads are still correct.
            _ = fcntl(descriptor, F_NOCACHE, 1)
        }
        var status = stat()
        guard fstat(descriptor, &status) == 0 else {
            close(descriptor)
            throw Error.openFailed(url.path, errno)
        }
        self.descriptor = descriptor
        self.isUncached = uncached
        self.byteCount = Int(status.st_size)
    }

    deinit {
        close(descriptor)
    }

    /// Read exactly `byteCount` bytes from `offset`.
    ///
    /// `pread` rather than a seek plus a read, so two readers of the same file cannot move each
    /// other's offset, and looped because a short read is legal.
    public func read(offset: Int, byteCount: Int) throws -> [UInt8] {
        guard offset >= 0, byteCount >= 0, offset + byteCount <= self.byteCount else {
            throw Error.tooShort(expected: offset + byteCount, got: self.byteCount)
        }
        var buffer = [UInt8](repeating: 0, count: byteCount)
        var filled = 0
        while filled < byteCount {
            let got = buffer.withUnsafeMutableBytes { raw -> Int in
                pread(descriptor, raw.baseAddress!.advanced(by: filled), byteCount - filled, off_t(offset + filled))
            }
            if got < 0 {
                if errno == EINTR { continue }
                throw Error.readFailed(offset: offset + filled, errno: errno)
            }
            if got == 0 { throw Error.tooShort(expected: byteCount, got: filled) }
            filled += got
        }
        return buffer
    }

    /// The same bytes, as `Data`, for the decoders that take one.
    public func readData(offset: Int, byteCount: Int) throws -> Data {
        Data(try read(offset: offset, byteCount: byteCount))
    }

    /// A digest over the whole file, read in bounded windows so the caller's memory does not
    /// depend on the file's size.
    public func digest(window: Int = 4 * 1024 * 1024) throws -> [UInt8] {
        var hasher = SHA256()
        var offset = 0
        while offset < byteCount {
            let count = min(window, byteCount - offset)
            hasher.update(data: try readData(offset: offset, byteCount: count))
            offset += count
        }
        return Array(hasher.finalize())
    }
}
