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
        for (index, _) in experts.enumerated() {
            // EIGHT SLOTS, ALWAYS. `validate(routedBlobs:topK:)` preconditions `topK == maxStreamedExperts`, so a
            // request cannot ask for one expert - the kernels address a fixed eight-slot bank with no sub-batch.
            // The same expert is placed in every slot and `remoteWeight` is [1,0,0,0,0,0,0,0], so the phase-2
            // reduce sums eight terms of which seven are multiplied by zero. That yields one expert's down output
            // without a new kernel. D260 has the crash report that established the constraint.
            let slots = 8
            let blob = views[index].buffer
            let blobs = [MTLBuffer](repeating: blob, count: slots)
            let blobOffsets = [Int](repeating: Int(views[index].offset), count: slots)
            guard let argBuf = moe.makeRoutedArgumentBuffer(
                routedBlobs: blobs, topK: UInt32(slots), routedBufferOffsets: blobOffsets)
            else { throw ShardServeError.argumentBufferUnavailable }
            guard let cb = ctx.queue.makeCommandBuffer() else {
                throw ShardServeError.commandBufferUnavailable
            }
            try moe.encodeRoutedPersistentPhase1U16Load(
                commandBuffer: cb, routedArgBuffer: argBuf, routedBlobs: blobs, routedOffsets: offsets,
                x: remoteActivation, acts: remoteActs, d: UInt32(dims), f: f, topK: UInt32(slots))
            try moe.encodeRoutedPersistentPhase2Reduce(
                commandBuffer: cb, routedArgBuffer: argBuf, routedBlobs: blobs, routedOffsets: offsets,
                acts: remoteActs, routingWeights: remoteWeight, residual: remoteResidual,
                y: remoteY, d: UInt32(dims), f: f, topK: UInt32(slots))
            cb.commit()
            // `await completed()`, NOT `waitUntilCompleted()`: Swift marks the blocking form unavailable from an
            // asynchronous context, which is the compiler saying a cooperative-pool thread must not be parked.
            await cb.completed()
            let src = remoteY.contents().bindMemory(to: Float.self, capacity: dims)
            for d in 0..<dims { out[index * dims + d] = src[d] }
        }
        return out
    }
}
