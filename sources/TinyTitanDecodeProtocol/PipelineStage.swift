import Foundation
import Metal

/// **The ring's core: what a pipeline stage does with the two ends it owns.**
///
/// `RealForwardRunner` has the hooks - `onHidden` runs after a stage publishes, `nextHidden` before a stage consumes
/// - and `PipelineLink` carries a frame. What was missing is the four lines between them, and this is those lines,
/// written as two pure functions so the composition is testable **without a model**: a `MTLBuffer` needs a device
/// and every Mac has one, so none of this requires 19 GB of weights to verify.
///
///   * `frame(from:rowWidth:position:layer:)` reads a published buffer into a frame;
///   * `store(_:into:)` writes a received frame into a buffer the runner owns, which is what `nextHidden` returns.
///
/// **`rowWidth` is a parameter rather than something inferred from `buffer.length`, and that is deliberate.** A
/// buffer's length says how many half floats it holds; it does not say how they divide into rows. Guessing the
/// divisor would be right for the one-row decode handoff and silently wrong for a chunk, which is exactly the class
/// of error `D335` cost three rounds to find.
///
/// **Both directions read and write shared storage.** `hiddenOut` and `hiddenProbe` are `storageModeShared`, so
/// `contents()` is valid for them - which `D324` established the hard way, after six rounds of a SIGSEGV that
/// presented as a network fault. A buffer with private storage cannot be read this way.
public enum PipelineStage {
    public enum StageError: Error, Equatable {
        /// The frame carries more values than the destination buffer holds.
        case frameTooLarge(values: Int, capacity: Int)
    }

    /// Read one or more rows of half-precision values from a published buffer into a frame.
    ///
    /// `rows: 1` is the per-token decode handoff; a larger value is the `t x D` prefill handoff. The same code
    /// serves both, which is what `D337` settled on.
    public static func frame(from buffer: MTLBuffer,
                             rowWidth: Int,
                             rows: Int = 1,
                             position: Int,
                             layer: Int) -> PipelineFrame {
        let count = rows * rowWidth
        let values = buffer.contents().bindMemory(to: Float16.self, capacity: count)
        return PipelineFrame(token: position, layer: layer,
                             hidden: Array(UnsafeBufferPointer(start: values, count: count)))
    }

    /// Write a received frame into a buffer the runner owns, and answer how many values landed.
    ///
    /// A frame longer than the destination is refused rather than truncated: a short write leaves the residual
    /// partly seeded, and an unseeded residual computes garbage while looking like it worked (`D335`).
    @discardableResult
    public static func store(_ frame: PipelineFrame, into buffer: MTLBuffer) throws -> Int {
        let capacity = buffer.length / MemoryLayout<Float16>.stride
        guard frame.hidden.count <= capacity else {
            throw StageError.frameTooLarge(values: frame.hidden.count, capacity: capacity)
        }
        frame.hidden.withUnsafeBytes { source in
            buffer.contents().copyMemory(from: source.baseAddress!, byteCount: source.count)
        }
        return frame.hidden.count
    }
}
