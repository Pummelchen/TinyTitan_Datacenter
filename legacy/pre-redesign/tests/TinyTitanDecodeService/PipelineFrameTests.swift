import Foundation
import Testing
@testable import TinyTitanDecodeProtocol

@Suite("PipelineFrame")
struct PipelineFrameTests {
    @Test("round trips a hidden state exactly")
    func roundTrip() throws {
        let hidden = (0..<2048).map { Float16($0 % 97) }
        let frame = PipelineFrame(token: 42, layer: 9, hidden: hidden)
        let back = try PipelineFrame.decode(frame.encode())
        #expect(back == frame)
        #expect(back.hidden.count == 2048)
    }

    @Test("a frame is 16 bytes of header plus the payload")
    func sizeIsWhatTheLawAssumes() {
        // D = 2048 fp16 is 4 KB, and the projection rests on a frame being kilobytes, not megabytes.
        let frame = PipelineFrame(token: 0, layer: 0, hidden: [Float16](repeating: 1, count: 2048))
        #expect(frame.encode().count == PipelineFrame.headerBytes + 4096)
    }

    @Test("a truncated frame is refused, not silently padded")
    func refusesShortInput() {
        let full = PipelineFrame(token: 1, layer: 2, hidden: [Float16](repeating: 0, count: 64)).encode()
        #expect(throws: PipelineFrame.DecodeError.self) { try PipelineFrame.decode(full.prefix(8)) }
        #expect(throws: PipelineFrame.DecodeError.self) { try PipelineFrame.decode(full.dropLast(2)) }
    }
}
