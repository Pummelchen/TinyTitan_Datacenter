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
        /// `install` was given a stage with no `hiddenIn`, so a received frame would have nowhere to land.
        case noLandingBuffer
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

/// **The four things a stage needs from a runner, and nothing else.**
///
/// Drawn as a protocol rather than taking `RealForwardRunner` directly, for the same reason `PipelineStage`'s two
/// functions are pure: the wiring is what could be wrong, and **a protocol lets the wiring be tested against a fake
/// in milliseconds instead of against a 19 GB model.** `RealForwardRunner` already has all four properties, so the
/// conformance below is empty and the engine is untouched.
public protocol PipelineEndpoints: AnyObject {
    var hiddenIn: MTLBuffer? { get set }
    var hiddenOut: MTLBuffer? { get set }
    var onHidden: ((Int, MTLBuffer) -> Void)? { get set }
    var nextHidden: ((Int) -> MTLBuffer)? { get set }
}

extension PipelineStage {
    /// **Put this stage on the ring.** `output` carries what this stage publishes, `input` carries what the previous
    /// stage sent, and both are the same socket seen from its two ends - the ring's forward edge and its backward one
    /// are one connection.
    ///
    /// `rows` is the handoff width: **1 for the per-token decode handoff and the chunk width for a prefill**, which
    /// is the same code either way (`D337`). `exitLayer` is the layer this stage's output corresponds to, and it is
    /// what the receiver needs to know which state it is holding.
    ///
    /// **`hiddenIn` is the buffer `nextHidden` fills**, because a buffer has to come from somewhere and this stage
    /// already owns one sized for the handoff it expects. It is required rather than optional: without it a receive
    /// would have nowhere to land, and a stage that silently drops its input is worse than one that refuses to start.
    /// **`input` and `output` are both optional, because a ring has two ends that are not stages.** The head of the
    /// pipeline consumes nothing - it embeds - so it has no `input`; the tail publishes nothing - it runs the head
    /// and samples - so it has no `output`. Requiring both, as the first version did, made the head unable to
    /// install at all, and the test against a fake is what caught it rather than a four-node run.
    public static func install(on stage: some PipelineEndpoints,
                               input: FileHandle?,
                               output: FileHandle?,
                               rowWidth: Int,
                               rows: Int = 1,
                               exitLayer: Int) throws {
        if let output {
            stage.onHidden = { position, buffer in
                let outgoing = frame(from: buffer, rowWidth: rowWidth, rows: rows,
                                     position: position, layer: exitLayer)
                // COUNTING, NOT GUESSING (D350). The decode handoff desynchronised and the cheapest way to find out
                // how is to print what each side thinks it is doing: every position this stage publishes, and every
                // position the peer says it is receiving for. A pair of lists answers the question that three
                // hypotheses did not.
                FileHandle.standardError.write(Data(
                    "[wire] send pos=\(position) layer=\(exitLayer) values=\(outgoing.hidden.count)\n".utf8))
                try? PipelineLink.send(outgoing, to: output)
            }
        }
        if let input {
            guard let landing = stage.hiddenIn else {
                throw StageError.noLandingBuffer
            }
            stage.nextHidden = { position in
                guard let received = try? PipelineLink.receive(from: input) else {
                    FileHandle.standardError.write(Data(
                        "[wire] recv FAILED for pos=\(position) - the peer sent nothing usable\n".utf8))
                    // POISON THE LANDING BUFFER RATHER THAN LEAVING IT AS IT WAS. `nextHidden` returns a
                    // non-optional buffer, so it cannot signal this failure by returning nothing - and the first
                    // version returned the buffer unchanged, which meant a stage that never received anything
                    // computed four tokens from whatever the buffer held and REPORTED SUCCESS. That is exactly the
                    // failure D335 cost three rounds to identify: an unseeded residual produces garbage while
                    // looking like it worked. NaN is unmistakable where plausible noise is not, so a broken
                    // handoff now produces NaN logits rather than believable ones.
                    let words = landing.length / MemoryLayout<Float16>.stride
                    let out = landing.contents().bindMemory(to: Float16.self, capacity: words)
                    for i in 0..<words { out[i] = .nan }
                    return landing
                }
                _ = try? store(received, into: landing)
                FileHandle.standardError.write(Data(
                    "[wire] recv pos=\(position) got token=\(received.token) layer=\(received.layer) values=\(received.hidden.count)\n".utf8))
                return landing
            }
        }
    }
}
