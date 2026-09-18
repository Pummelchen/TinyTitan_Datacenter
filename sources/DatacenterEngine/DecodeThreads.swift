import Foundation

/// How many threads a CPU-bound decode pass may use.
///
/// One knob for the dequantiser, the raw decoder and the matmul, because they are all the same decision: this
/// machine has eight cores and the engine used one. `D94` introduced it for the int4 unpack and measured
/// **2.4x** on the phase it touched; `D99` extends it to the other two single-threaded passes and this type
/// exists so the three cannot drift apart into three different knobs.
///
/// `SHARD_DECODE_THREADS=1` restores the single-threaded behaviour, which is not a debugging convenience — it
/// is how the two are compared **on one binary** rather than across builds, which is `D62`'s lesson and the
/// reason the `D94` A/B could attribute its win at all.
public enum DecodeThreads {
    /// The thread count, from `SHARD_DECODE_THREADS` when it is sane and the machine's core count otherwise.
    public static let count: Int = {
        if let raw = ProcessInfo.processInfo.environment["SHARD_DECODE_THREADS"],
           let requested = Int(raw), requested >= 1 {
            return requested
        }
        return max(1, ProcessInfo.processInfo.activeProcessorCount)
    }()

    /// Whether a piece of work is worth splitting at all.
    ///
    /// A dispatch is not free, and this project has already paid for learning that: `D59`'s first GPU unpack
    /// was *slower* than the scalar path it replaced, and `DC-093`'s minimum-work rule exists for the same
    /// reason. The figure is deliberately well above a dispatch's cost — a million multiply-adds is about a
    /// millisecond of this machine's single-core rate — so a small matmul in a fixture stays sequential and
    /// only real work is fanned out.
    public static let minimumWork = 1 << 20

    /// Whether to split work of this size.
    public static func wantsParallelism(work: Int) -> Bool {
        count > 1 && work >= minimumWork
    }
}
