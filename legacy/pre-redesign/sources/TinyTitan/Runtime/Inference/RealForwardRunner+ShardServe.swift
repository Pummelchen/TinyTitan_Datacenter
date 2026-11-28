import Foundation
import Metal

/// The serving half of the shard exchange: evaluate the experts a peer asks for, on the activation it supplies, and
/// return one row of `dims` per expert **in the order asked**.
///
/// `ShardExchangeServer` takes this as its injected `Compute` (`ShardExchangeServer.Compute`), which is why the
/// server could be built and tested before anything could answer a request. This is the thing that answers.
///
/// **Synchronous on purpose.** `fetchRoutedExperts` is `async`, and a synchronous `Compute` that blocked on it would
/// be `D197`'s failure - a blocking call inside a `Task` that starves the task which has to make progress. The MoE
/// path already reaches expert buffers synchronously through `planRoutedExperts` and `routedExpertBuffers(for:)`, so
/// this uses those and needs no change to the server, its signature or its tests.
///
/// **One expert at a time, because phase 2 reduces.** `encodeRoutedPersistentPhase2Reduce` applies the routing weight
/// and reduces across slots, so there is no per-expert output. Running it with `topK: 1`, a unit weight and a zero
/// residual makes the reduce a no-op and yields that expert's down output - which is what a peer owes the requester.
/// The weight is deliberately 1.0: the requester owns the router's decision and applies the weight, so a peer that
/// applied it too would be counted twice, silently (`D168`).
extension RealForwardRunner {
    public enum ShardServeError: Swift.Error, Equatable {
        case noPlan(layer: Int)
        case argumentBufferUnavailable
        case commandBufferUnavailable
        case wrongActivationWidth(expected: Int, got: Int)
    }

    public func remoteExpertValues(layer: Int,
                                   experts: [Int],
                                   activation: [Float],
                                   dims: Int) async throws -> [Float] {
        guard activation.count == dims else {
            throw ShardServeError.wrongActivationWidth(expected: dims, got: activation.count)
        }
        // Shares the node's expert cache: these are the entry points the request path itself uses.
        // THE STREAMING READ, not the planning one. planRoutedExperts -> routedExpertBuffers describes where an
        // expert can be placed in THIS node's bank, which fails with "8 experts do not fit in 40 cache slots" when
        // a peer asks for an expert this node never routes to (D263, D264). fetchRoutedExperts is the path the MoE
        // takes on a miss - and it is async, which is why the whole serving path is.
        let views = try await model.fetchRoutedExperts(layer: layer, experts: experts)
        let offsets = try model.routedExpertOffsets(layer: layer)

        // The activation, widened from fp32 to the fp16 the kernels load (`...U16Load`).
        let dst = remoteActivation.contents().bindMemory(to: Float16.self, capacity: dims)
        for index in 0..<dims { dst[index] = Float16(activation[index]) }

        // FmoE is a decode-path local, not a property, but `remoteActs` was allocated from it in `init` - so its
        // length gives it back and the two cannot drift.
        let f = UInt32(remoteActs.length / MemoryLayout<Float16>.stride / 8)

        var out = [Float](repeating: 0, count: experts.count * dims)

        // ONE COMMAND BUFFER FOR THE WHOLE REQUEST. This loop committed and awaited once per expert, so eight
        // experts meant eight commits and eight blocking waits - and at these shapes the kernel is microseconds of
        // the call, so the wait was the cost. D114 made exactly this change single-node (640 dispatches to 80,
        // 179 -> 84 ms on the phase). The argument buffers are held in an array because they must outlive the
        // commit: released early, the failure is wrong values rather than an error, which is the shape of D269.
        guard let cb = ctx.queue.makeCommandBuffer() else {
            throw ShardServeError.commandBufferUnavailable
        }
        var argumentBuffers: [MTLBuffer] = []
        argumentBuffers.reserveCapacity(experts.count)
        let slotBytes = Int(f) * MemoryLayout<Float16>.stride
        for (index, _) in experts.enumerated() {
            let blobs = [views[index].buffer]
            let blobOffsets = [Int(views[index].offset)]
            guard let argBuf = serveMoE.makeRoutedArgumentBuffer(
                routedBlobs: blobs, topK: 1, routedBufferOffsets: blobOffsets)
            else { throw ShardServeError.argumentBufferUnavailable }
            argumentBuffers.append(argBuf)
            try serveMoE.encodeRoutedPersistentPhase1U16Load(
                commandBuffer: cb, routedArgBuffer: argBuf, routedBlobs: blobs, routedOffsets: offsets,
                x: remoteActivation, acts: remoteActs, actsOffset: index * slotBytes,
                d: UInt32(dims), f: f, topK: 1)
            try serveMoE.encodeRoutedPersistentPhase2Reduce(
                commandBuffer: cb, routedArgBuffer: argBuf, routedBlobs: blobs, routedOffsets: offsets,
                acts: remoteActs, actsOffset: index * slotBytes, routingWeights: remoteWeight,
                residual: remoteResidual, y: remoteY[index], d: UInt32(dims), f: f, topK: 1)
        }
        cb.commit()
        // `await completed()`, NOT `waitUntilCompleted()`: Swift marks the blocking form unavailable from an
        // asynchronous context, which is the compiler saying a cooperative-pool thread must not be parked.
        await cb.completed()
        for (index, _) in experts.enumerated() {
            let src = remoteY[index].contents().bindMemory(to: Float16.self, capacity: dims)
            for d in 0..<dims { out[index * dims + d] = Float(src[d]) }
        }
        return out
    }
}
