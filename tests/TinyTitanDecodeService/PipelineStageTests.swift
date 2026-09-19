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

/// A stand-in for a runner, so the WIRING can be tested without one. `install(on:)` writes two closures and reads
/// one buffer, and those are the whole of its contract - so a fake with four properties exercises everything a real
/// runner would, in milliseconds, and without 19 GB of weights.
private final class FakeEndpoint: PipelineEndpoints {
    var hiddenIn: MTLBuffer?
    var hiddenOut: MTLBuffer?
    var onHidden: ((Int, MTLBuffer) -> Void)?
    var nextHidden: ((Int) -> MTLBuffer)?
}

@Suite("PipelineStage wiring", .serialized)
struct PipelineStageWiringTests {
    static let port: UInt16 = 47_700

    private func device() throws -> MTLDevice {
        try #require(MTLCreateSystemDefaultDevice(), "no Metal device")
    }

    private func connectWhenListening(_ port: UInt16, retries: Int = 150) throws
        -> (input: FileHandle, output: FileHandle) {
        var last: Error = PipelineStage.StageError.noLandingBuffer
        for _ in 0..<retries {
            do { return try DecodeTCPSocket.connect(host: "127.0.0.1", port: port) }
            catch { last = error; usleep(20_000) }
        }
        throw last
    }

    /// The property that matters: a stage that publishes hands the SAME values to a stage that consumes. That is
    /// the ring, exercised on a real socket, with no runner and no model in it.
    @Test("a stage's published state reaches a consuming stage unchanged")
    func ringRoundTrip() throws {
        let d = try device()
        let values = (0..<8).map { Float16($0) - 2 }
        let published = d.makeBuffer(length: values.count * MemoryLayout<Float16>.stride, options: .storageModeShared)!
        values.withUnsafeBytes { published.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }

        var consumer: FakeEndpoint?
        let done = DispatchSemaphore(value: 0)
        // The consumer listens; the producer connects and installs itself with `input` and `output` being the two
        // ends of ONE connection, which is what a ring edge is.
        DispatchQueue.global().async {
            defer { done.signal() }
            guard let pair = try? DecodeTCPSocket.listenAndAccept(host: "127.0.0.1", port: Self.port) else { return }
            let fake = FakeEndpoint()
            fake.hiddenIn = d.makeBuffer(length: 8 * MemoryLayout<Float16>.stride, options: .storageModeShared)!
            try? PipelineStage.install(on: fake, input: pair.input, output: pair.output,
                                       rowWidth: 8, rows: 1, exitLayer: 20)
            if let consume = fake.nextHidden { _ = consume(0) }
            consumer = fake
        }

        let pair = try connectWhenListening(Self.port)
        let producer = FakeEndpoint()
        // A PRODUCER: it publishes and consumes nothing, so it installs with no input and needs no landing buffer.
        try PipelineStage.install(on: producer, input: nil, output: pair.output,
                                  rowWidth: 8, rows: 1, exitLayer: 20)
        producer.onHidden?(0, published)

        #expect(done.wait(timeout: .now() + 15) == .success, "the peer did not finish")
        let landed = consumer!.hiddenIn!.contents().bindMemory(to: Float16.self, capacity: 8)
        #expect(Array(UnsafeBufferPointer(start: landed, count: 8)) == values,
                "the consumed residual is not the published one")
    }

    @Test("a stage with nowhere to land refuses to install")
    func noLandingBuffer() throws {
        let fake = FakeEndpoint()
        #expect(throws: PipelineStage.StageError.noLandingBuffer) {
            try PipelineStage.install(on: fake, input: FileHandle.nullDevice, output: nil,
                                      rowWidth: 8, rows: 1, exitLayer: 10)
        }
    }
}
