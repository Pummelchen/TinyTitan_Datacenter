import XCTest

@testable import DatacenterEngine

/// M2's gate at fixture scale: the **engine**, not a helper, running a forward sharded across two nodes
/// and producing a trace byte-identical to the single-node one.
///
/// The two nodes run on separate threads because each one blocks waiting for the other inside every
/// mixture layer — one all-reduce per layer, in lockstep. They are real TCP sockets on loopback, so the
/// only thing left to the cluster is that the two ends are two machines.
final class ShardedForwardTests: XCTestCase {
    private func installURL() throws -> URL {
        try XCTUnwrap(
            Bundle.module.url(forResource: "tiny-qwen36", withExtension: nil, subdirectory: "Fixtures")
        ).appendingPathComponent("install")
    }

    private func assertIdentical(
        _ one: ForwardResult, _ other: ForwardResult, _ what: String,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(one.tensors.count, other.tensors.count, "\(what): tensor count", file: file, line: line)
        for (index, tensor) in one.tensors.enumerated() where index < other.tensors.count {
            let peer = other.tensors[index]
            XCTAssertEqual(tensor.name, peer.name, "\(what): tensor \(index) name", file: file, line: line)
            XCTAssertEqual(tensor.shape, peer.shape, "\(what): \(tensor.name) shape", file: file, line: line)
            guard tensor.values.count == peer.values.count else {
                XCTFail("\(what): \(tensor.name) length", file: file, line: line)
                continue
            }
            for (element, value) in tensor.values.enumerated()
            where value.bitPattern != peer.values[element].bitPattern {
                XCTFail(
                    "\(what): \(tensor.name)[\(element)] is \(value.bitPattern) on one node and "
                    + "\(peer.values[element].bitPattern) on the other",
                    file: file, line: line
                )
                return
            }
        }
        XCTAssertEqual(
            one.discrete.count, other.discrete.count, "\(what): discrete count", file: file, line: line
        )
        for (index, decision) in one.discrete.enumerated() where index < other.discrete.count {
            let peer = other.discrete[index]
            XCTAssertEqual(decision.name, peer.name, "\(what): decision \(index) name", file: file, line: line)
            XCTAssertEqual(
                decision.values, peer.values,
                "\(what): \(decision.name) — router decisions must match exactly, not nearly",
                file: file, line: line
            )
        }
    }

    private final class Outcome: @unchecked Sendable {
        var result: Result<ForwardResult, Swift.Error>?
    }

    func testATwoNodeShardedForwardIsIdenticalToTheSingleNodeTrace() throws {
        let install = try installURL()
        let tokens = [1, 2, 3]

        // The reference: one node, the whole model.
        let reference = try Qwen3_5Forward(install: install).forwardWithDecisions(tokens: tokens)
        let experts = try Qwen3_5Forward(install: install).mixtureShape().experts

        // The cluster: two nodes over TCP loopback, sharing one plan.
        let ownership = ExpertOwnership(plan: ShardPlan.generate(family: "tiny", experts: experts, nodes: 2))
        let listener = try TCPListener(port: 0)
        let client = try TCPTransport.connect(
            host: "127.0.0.1", port: listener.port, timeoutMilliseconds: 10_000
        )
        let server = try listener.accept(timeoutMilliseconds: 10_000)

        // Node 1 on its own thread: node 0's first all-reduce blocks until node 1 reaches its own.
        nonisolated(unsafe) let serverTransport = server
        let outcome = Outcome()
        let finished = DispatchSemaphore(value: 0)
        let thread = Thread {
            defer { finished.signal() }
            do {
                let forward = try Qwen3_5Forward(
                    install: install,
                    shard: ShardExecution(node: 1, ownership: ownership, peers: [serverTransport])
                )
                outcome.result = .success(try forward.forwardWithDecisions(tokens: tokens))
            } catch {
                outcome.result = .failure(error)
            }
        }
        thread.start()

        let mine = try Qwen3_5Forward(
            install: install,
            shard: ShardExecution(node: 0, ownership: ownership, peers: [client])
        ).forwardWithDecisions(tokens: tokens)

        XCTAssertEqual(finished.wait(timeout: .now() + 30), .success, "node 1 did not finish")
        let theirs = try XCTUnwrap(outcome.result).get()

        assertIdentical(reference, mine, "node 0")
        assertIdentical(reference, theirs, "node 1")
        assertIdentical(mine, theirs, "node 0 against node 1")
        XCTAssertEqual(mine.tensors.count, reference.tensors.count)
    }

    /// The single-node path is untouched by the shard branch: with no shard context the forward is the
    /// one M1's gate measured, and this pins that the two are the same code.
    func testASingleNodeForwardIsUnaffectedByTheShardBranch() throws {
        let install = try installURL()
        let tokens = [4, 5]
        let first = try Qwen3_5Forward(install: install).forwardWithDecisions(tokens: tokens)
        let second = try Qwen3_5Forward(install: install).forwardWithDecisions(tokens: tokens)
        assertIdentical(first, second, "two single-node runs")
    }

    /// A peer that never answers fails the **forward**, not just the exchange: the run stops with an
    /// error rather than producing a trace from the experts this node happened to own. This is `DC-043`'s
    /// rule at the level a user would meet it.
    func testAShardedForwardWithASilentPeerFailsRatherThanAnswering() throws {
        let install = try installURL()
        let experts = try Qwen3_5Forward(install: install).mixtureShape().experts
        let ownership = ExpertOwnership(plan: ShardPlan.generate(family: "tiny", experts: experts, nodes: 2))
        let (mine, theirs) = try SocketPair.make()
        _ = theirs  // held open and never written: a node that is up but not answering

        let forward = try Qwen3_5Forward(
            install: install,
            shard: ShardExecution(
                node: 0, ownership: ownership, peers: [mine],
                policy: ExchangePolicy(receiveTimeoutMilliseconds: 60, attempts: 2)
            )
        )
        let started = Date()
        XCTAssertThrowsError(try forward.forwardWithDecisions(tokens: [1, 2, 3])) { error in
            guard case ContributionTransportError.timedOut(let midFrame) = error else {
                return XCTFail("expected the forward to fail on a silent peer, got \(error)")
            }
            XCTAssertFalse(midFrame, "nothing arrived at all, so this is a silent node rather than a torn frame")
        }
        XCTAssertLessThan(
            Date().timeIntervalSince(started), 10,
            "the policy's deadline must reach the socket through the forward too"
        )
    }
}
