import Foundation

/// Serializes forward steps through one `RealForwardRunner`.
///
/// The runner's scratch buffers, decode cursors and statistic counters are
/// owned exclusively -- `RealForwardRunner` documents that as its whole safety
/// argument -- so two sequences must never be inside a step at the same time,
/// even though each uses its own KV and GDN slot. A batched scheduler therefore
/// holds this gate around each `produce`/`prefillChunked` call: the slots are
/// independent, the step is not.
///
/// Actor-isolated, so acquisition and release are both `await`ed. Waiting is
/// FIFO and cancellable; a cancelled waiter is removed and resumed with the
/// error so it cannot be handed the gate later.
public actor ForwardStepGate {
    private struct Waiter {
        let id: UUID
        let continuation: CheckedContinuation<Void, Error>
    }

    private var busy = false
    private var waiters: [Waiter] = []

    public init() {}

    /// Wait until no step is in flight, then take the gate.
    public func acquire() async throws {
        try Task.checkCancellation()
        if !busy {
            busy = true
            return
        }
        let id = UUID()
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                waiters.append(Waiter(id: id, continuation: continuation))
            }
        } onCancel: {
            Task { await self.cancelWaiter(id) }
        }
        if Task.isCancelled {
            release()
            throw CancellationError()
        }
    }

    /// Release the gate, handing it to the oldest waiter if there is one.
    public func release() {
        if waiters.isEmpty {
            busy = false
        } else {
            waiters.removeFirst().continuation.resume()
        }
    }

    private func cancelWaiter(_ id: UUID) {
        guard let index = waiters.firstIndex(where: { $0.id == id }) else { return }
        waiters.remove(at: index).continuation.resume(throwing: CancellationError())
    }
}
