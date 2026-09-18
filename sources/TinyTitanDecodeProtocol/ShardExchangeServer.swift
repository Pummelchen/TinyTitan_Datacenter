import Foundation

/// The serving half of the exchange: accept a peer, answer its requests, stop.
///
/// `D207` found that only the requesting half existed and that every test had used a fake peer. This is the real
/// one, with the **expert computation injected** rather than baked in, for the same reason the transport protocol
/// exists: the framing, the slot contract and the lifecycle are testable with a stand-in compute, and the runner
/// supplies the real one when the serving call site is wired. A server that could only be tested by running a 20 GB
/// model on four machines would not be tested.
///
/// **It answers in the requested order.** `ShardExchange.Reply` carries its slots back even though the requester
/// already knows them, and the reason is here: the peer is not required to know or preserve slot identity, only to
/// answer the experts it was asked about in sequence. A server that reordered its answers would land contributions
/// on the wrong slots, and `D154` makes that a wrong number rather than an error.
/// unchecked-invariant: `boundPort` is written once by the serving thread and read afterwards, and a server
/// serves one connection at a time; the same contract `ShardPeerSet` and `PreadExpertStreamer` carry.
public final class ShardExchangeServer: @unchecked Sendable {
    /// Run the named experts over the activation and return one row of `dimensions` per expert, in the order
    /// asked. Injected so this type is testable without a model.
    public typealias Compute = (_ layer: Int, _ experts: [Int], _ activation: [Float]) throws -> [Float]

    public enum Error: Swift.Error, Equatable {
        case computeReturnedWrongWidth(expected: Int, got: Int)
    }

    public let port: UInt16
    private let compute: Compute
    private let ready: DispatchSemaphore

    public init(port: UInt16, compute: @escaping Compute) {
        self.port = port
        self.compute = compute
        self.ready = DispatchSemaphore(value: 0)
    }

    /// The port the listener actually bound, once `serve` has started. Useful when `port` is 0 in a test.
    public private(set) var boundPort: UInt16 = 0

    /// Serve `connections` peers, answering every request each one sends until it closes.
    ///
    /// Blocking, and meant to be run off the cooperative pool - `D197`'s lesson from the test suite, where a
    /// blocking accept inside a `Task` starved the very task that had to connect to it.
    public func serve(connections: Int) throws {
        for _ in 0..<connections {
            let accepted = try DecodeTCPSocket.listenAndAccept(host: "0.0.0.0", port: port)
            boundPort = (try? DecodeTCPSocket.boundPort(of: accepted.input.fileDescriptor)) ?? port
            do {
                try answer(accepted)
            } catch {
                // A REQUEST THAT CANNOT BE ANSWERED MUST NOT TAKE THE SERVER WITH IT. Before this catch, one
                // unanswerable request ended the node's willingness to answer any request - `answer` throwing
                // propagated out of the accept loop - and its peers, which had not yet connected, were then refused.
                // That made the failure systematic and order-dependent: whichever node refused a request first died
                // first, and who survived depended on who connected when (D262).
                //
                // The protocol has no error frame, so the honest thing is to close this connection and keep
                // serving. The requester sees a closed channel and falls back to single-node for that layer, which
                // is the same behaviour a missing peer gets.
                refusedRequests += 1
                lastRefusal = "\(error)"
                accepted.input.closeFile()
                if accepted.output.fileDescriptor != accepted.input.fileDescriptor {
                    accepted.output.closeFile()
                }
            }
        }
    }

    /// Requests this server could not answer. Exposed so a run can report it rather than merely being slower.
    public private(set) var refusedRequests: Int = 0
    /// The most recent refusal, for a startup line that says why a peer is falling back.
    public private(set) var lastRefusal: String = ""

    /// Refuse a request the server cannot answer, and say so, without ending the accept loop.
    public func recordRefusal(_ error: Swift.Error) {
        refusedRequests += 1
        lastRefusal = "\(error)"
    }

    /// Answer one connection until its peer closes. Each request is decoded, computed and replied to in order.
    public func answer(_ handles: (input: FileHandle, output: FileHandle)) throws {
        let channel = ShardPeerChannel(input: handles.input, output: handles.output)
        defer { channel.close() }
        while true {
            let frame: Data
            do {
                frame = try channel.receive()
            } catch ShardPeerChannel.Error.peerClosed {
                return
            }
            let request = try ShardExchange.decodeRequest(from: frame)
            let dimensions = request.activation.count
            let values = try compute(request.layer, request.experts, request.activation)
            let expected = request.experts.count * dimensions
            guard values.count == expected else {
                // Refused rather than padded: a short reply would be read as a shorter row and land on the wrong
                // slots, which is a wrong number rather than a failure.
                throw Error.computeReturnedWrongWidth(expected: expected, got: values.count)
            }
            try channel.send(try ShardExchange.encode(ShardExchange.Reply(
                layer: request.layer, slots: request.slots,
                dimensions: dimensions, values: values)))
        }
    }
}
