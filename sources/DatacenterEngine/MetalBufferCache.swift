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
    ///
    /// **Per slot, and grow-only** (`D109`). The first version replaced *every* buffer the moment one slot
    /// was too small, and the expert path alternates a 1.2 MB gate-up slab with a 0.6 MB down slab: slot 0
    /// grew and shrank on alternate fetches, so the whole set — including the 8 MB fp32 output buffer —
    /// was reallocated with it, ~320 times in a decode step. A slot now keeps the largest size it has ever
    /// been asked for, which is what makes this a cache rather than an allocator. Reuse is still safe for
    /// the same reason it always was: the lock is held across the dispatch and its completion.
    func withBuffers<T>(
        _ sizes: [Int], device: any MTLDevice, _ body: ([any MTLBuffer]) throws -> T
    ) throws -> T {
        lock.lock()
        defer { lock.unlock() }
        if cached.count != sizes.count {
            cached = try sizes.map { try Self.makeBuffer(device, $0) }
        } else {
            for (index, size) in sizes.enumerated() where cached[index].length < max(size, 1) {
                cached[index] = try Self.makeBuffer(device, size)
            }
        }
        return try body(cached)
    }
}

enum MetalBufferCacheError: Error {
    case allocationFailed(Int)
}
