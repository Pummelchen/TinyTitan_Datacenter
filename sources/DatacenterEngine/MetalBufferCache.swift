import Foundation
import Metal

/// A one-slot cache of GPU buffers, shared by the kernels so neither allocates per call.
///
/// The unpack learned this the expensive way: its first version allocated a fresh output buffer for every
/// fetch — 8 MB for one expert, 2,218 fetches in a five-token trace — and measured *slower* than the scalar
/// path it was replacing (`D58`, `D59`). The head has the same shape: 31 calls with a 64 MB block of
/// vocabulary rows, which a per-call allocation would dominate.
///
/// Reuse is safe because the lock is held across the dispatch **and** its completion, so a buffer cannot be
/// overwritten while a kernel is still reading it, and it costs nothing because work on one GPU is
/// serialised in any case. That is what `@unchecked Sendable` stands on here: the class has mutable state
/// and no compiler-checkable proof, and the proof is the lock.
final class MetalBufferCache: @unchecked Sendable {
    private let lock = NSLock()
    private var cached: [any MTLBuffer] = []

    static func makeBuffer(_ device: any MTLDevice, _ length: Int) throws -> any MTLBuffer {
        // A zero-length buffer is not a thing Metal will make, and an absent section is normal.
        guard let buffer = device.makeBuffer(length: max(length, 1), options: .storageModeShared) else {
            throw MetalBufferCacheError.allocationFailed(length)
        }
        return buffer
    }

    /// Runs `body` with one buffer per requested size, each at least that long, growing as needed.
    func withBuffers<T>(
        _ sizes: [Int], device: any MTLDevice, _ body: ([any MTLBuffer]) throws -> T
    ) throws -> T {
        lock.lock()
        defer { lock.unlock() }
        if cached.count != sizes.count || zip(cached, sizes).contains(where: { $0.length < max($1, 1) }) {
            cached = try sizes.map { try Self.makeBuffer(device, $0) }
        }
        return try body(cached)
    }
}

enum MetalBufferCacheError: Error {
    case allocationFailed(Int)
}
