import Foundation
import Metal
import Testing

@testable import TinyTitanDecodeProtocol

/// The ring's composition, tested **without a model**. `PipelineStage` bridges the runner's hooks and
/// `PipelineLink`, and both halves of that bridge touch only an `MTLBuffer` - so a device is the whole requirement,
/// and the 19 GB install is not. Testing this against the real engine would test the engine, not the composition.
@Suite("PipelineStage composition")
struct PipelineStageTests {
    private func device() throws -> MTLDevice {
        try #require(MTLCreateSystemDefaultDevice(), "no Metal device on this machine")
    }

    private func buffer(_ device: MTLDevice, values: [Float16]) -> MTLBuffer {
        let b = device.makeBuffer(length: values.count * MemoryLayout<Float16>.stride,
                                  options: .storageModeShared)!
        values.withUnsafeBytes { b.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
        return b
    }

    @Test("a published buffer becomes a frame of the right rows")
    func frameFromBuffer() throws {
        let d = try device()
        let values = (0..<16).map { Float16($0) }
        let frame = PipelineStage.frame(from: buffer(d, values: values), rowWidth: 4, rows: 4,
                                        position: 2, layer: 30)
        #expect(frame.token == 2)
        #expect(frame.layer == 30)
        #expect(frame.hidden == values, "rowWidth x rows values should all be carried")
    }

    @Test("a received frame lands in the buffer unchanged")
    func storeIntoBuffer() throws {
        let d = try device()
        let frame = PipelineFrame(token: 1, layer: 20, hidden: [1.5, -2.25, 0, 4])
        let target = d.makeBuffer(length: 4 * MemoryLayout<Float16>.stride, options: .storageModeShared)!
        let wrote = try PipelineStage.store(frame, into: target)
        #expect(wrote == 4)
        #expect(PipelineStage.frame(from: target, rowWidth: 4, position: 1, layer: 20) == frame,
                "store followed by frame should be the identity")
    }

    @Test("a frame larger than the destination is refused, not truncated")
    func oversizeRefused() throws {
        let d = try device()
        let frame = PipelineFrame(token: 0, layer: 0, hidden: [1, 2, 3, 4, 5])
        let target = d.makeBuffer(length: 4 * MemoryLayout<Float16>.stride, options: .storageModeShared)!
        #expect(throws: PipelineStage.StageError.frameTooLarge(values: 5, capacity: 4)) {
            _ = try PipelineStage.store(frame, into: target)
        }
    }

    @Test("rowWidth is a parameter, so a chunk is not mistaken for one token")
    func rowWidthMatters() throws {
        let d = try device()
        let values = (0..<12).map { Float16($0) }
        let b = buffer(d, values: values)
        #expect(PipelineStage.frame(from: b, rowWidth: 4, rows: 1, position: 0, layer: 0).hidden.count == 4)
        #expect(PipelineStage.frame(from: b, rowWidth: 4, rows: 3, position: 0, layer: 0).hidden.count == 12)
    }
}
