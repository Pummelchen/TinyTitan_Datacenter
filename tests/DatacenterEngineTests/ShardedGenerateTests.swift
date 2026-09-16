import XCTest

@testable import DatacenterEngine

/// `DC-109`: generation across a shard, in both decode modes.
///
/// The cached decode path called the mixture directly, so a sharded `--cached` run would have computed
/// **only its own experts**, all-reduced nothing, and produced a plausible token sequence that no test
/// would have questioned. That is the failure mode this project exists to eliminate, and the reason the
/// cached path now goes through the same `mixtureOutput` the sequence path uses.
final class ShardedGenerateTests: XCTestCase {
    private func installURL() throws -> URL {
        try XCTUnwrap(
            Bundle.module.url(forResource: "tiny-qwen36", withExtension: nil, subdirectory: "Fixtures")
        ).appendingPathComponent("install")
    }

    private final class Outcome: @unchecked Sendable {
        var generated: [Int]?
        var error: Swift.Error?
    }

    private func sharded(
        install: URL, node: Int, ownership: ExpertOwnership, peers: [any ContributionTransport],
        prompt: [Int], steps: Int, cached: Bool
    ) throws -> [Int] {
        let forward = try Qwen3_5Forward(
            install: install,
            shard: ShardExecution(
                node: node, ownership: ownership, peers: peers,
                policy: ExchangePolicy(receiveTimeoutMilliseconds: 10_000, attempts: 3)
            )
        )
        let generation = cached
            ? try forward.generateCached(prompt: prompt, maxNewTokens: steps)
            : try forward.generate(prompt: prompt, maxNewTokens: steps)
        return generation.generated
    }

    /// Run the two nodes on two threads, because each blocks on the other inside every mixture layer.
    private func runBothNodes(
        install: URL, ownership: ExpertOwnership, left: any ContributionTransport,
        right: any ContributionTransport, prompt: [Int], steps: Int, cached: Bool
    ) throws -> ([Int], [Int]) {
        nonisolated(unsafe) let rightTransport = right
        let outcome = Outcome()
        let finished = DispatchSemaphore(value: 0)
        let thread = Thread {
            defer { finished.signal() }
            do {
                outcome.generated = try self.sharded(
                    install: install, node: 1, ownership: ownership, peers: [rightTransport],
                    prompt: prompt, steps: steps, cached: cached
                )
            } catch {
                outcome.error = error
            }
        }
        thread.start()
        let mine = try sharded(
            install: install, node: 0, ownership: ownership, peers: [left],
            prompt: prompt, steps: steps, cached: cached
        )
        XCTAssertEqual(finished.wait(timeout: .now() + 60), .success, "node 1 did not finish")
        if let error = outcome.error { throw error }
        return (mine, try XCTUnwrap(outcome.generated))
    }

    func testShardedGenerationMatchesSingleNodeInBothDecodeModes() throws {
        let install = try installURL()
        let prompt = [1, 2, 3]
        let steps = 3
        let single = try Qwen3_5Forward(install: install).generate(prompt: prompt, maxNewTokens: steps)
        let singleCached = try Qwen3_5Forward(install: install)
            .generateCached(prompt: prompt, maxNewTokens: steps)
        XCTAssertEqual(
            single.generated, singleCached.generated,
            "the two single-node modes must agree before a sharded comparison means anything"
        )

        let experts = try Qwen3_5Forward(install: install).mixtureShape().experts
        let ownership = ExpertOwnership(
            plan: ShardPlan.generate(family: "tiny", experts: experts, nodes: 2)
        )

        for cached in [true, false] {
            let (left, right) = try SocketPair.make()
            let (mine, theirs) = try runBothNodes(
                install: install, ownership: ownership, left: left, right: right,
                prompt: prompt, steps: steps, cached: cached
            )
            let mode = cached ? "cached decode" : "full sequence"
            XCTAssertEqual(mine, single.generated, "\(mode): node 0's tokens")
            XCTAssertEqual(theirs, single.generated, "\(mode): node 1's tokens")
        }
        XCTAssertFalse(single.generated.isEmpty)
    }

    /// Three nodes, joined the way the cluster actually joins: real sockets, a cluster config and the
    /// production mesh rule, rather than a triangle of socket pairs wired by hand in the test. The first
    /// version of this test built the triangle itself and deadlocked, which is a fine reason to stop
    /// testing the engine through wiring the engine never uses.
    func testShardedGenerationMatchesSingleNodeAtThreeNodes() throws {
        let install = try installURL()
        let prompt = [4, 5]
        let steps = 1
        let single = try Qwen3_5Forward(install: install)
            .generateCached(prompt: prompt, maxNewTokens: steps)
        let experts = try Qwen3_5Forward(install: install).mixtureShape().experts
        let ownership = ExpertOwnership(
            plan: ShardPlan.generate(family: "tiny", experts: experts, nodes: 3)
        )

        // Three endpoints on loopback: the same bind-lower/accept-higher rule the farm runs. The ports
        // have to be known before anybody joins, because the join is what binds them — so each is found
        // by binding `0` in a **function**, whose return releases the probe, and the join then binds the
        // port it reported. (Holding three probe listeners and joining on their ports fails with
        // "already in use"; retrying does not help either, because node 0's join binds successfully and
        // then blocks accepting peers that only the rest of the test would start.)
        func freePort() throws -> Int {
            try TCPListener(port: 0).port
        }
        let ports = try (0..<3).map { _ in try freePort() }
        let joinedConfig = ClusterConfig(
            endpoints: ports.map { NodeEndpoint(host: "127.0.0.1", port: $0) }
        )
        let node0Joined: (transports: [any ContributionTransport], listener: TCPListener)
        let outcomes = (0..<3).map { _ in Outcome() }
        let finished = DispatchSemaphore(value: 0)

        func run(node: Int, joined: (transports: [any ContributionTransport], listener: TCPListener)) {
            do {
                let forward = try Qwen3_5Forward(
                    install: install,
                    shard: ShardExecution(
                        node: node, ownership: ownership, peers: joined.transports,
                        policy: ExchangePolicy(receiveTimeoutMilliseconds: 10_000, attempts: 3)
                    )
                )
                outcomes[node].generated = try forward
                    .generateCached(prompt: prompt, maxNewTokens: steps).generated
            } catch {
                outcomes[node].error = error
            }
        }

        // Node 0 joined above; nodes 1 and 2 run on their own threads, because every node blocks on its
        // peers inside every mixture layer.
        let threads = (1..<3).map { node -> Thread in
            Thread {
                defer { finished.signal() }
                do {
                    let joined = try ClusterJoin.mesh(
                        config: joinedConfig, node: node, nodes: 3, timeoutMilliseconds: 10_000
                    )
                    run(node: node, joined: joined)
                } catch {
                    outcomes[node].error = error
                    finished.signal()  // signalled twice at worst; the wait only needs two arrivals
                }
            }
        }
        threads.forEach { $0.start() }
        // Node 0 joins after the others are starting: its accept waits for them.
        let node0 = try ClusterJoin.mesh(
            config: joinedConfig, node: 0, nodes: 3, timeoutMilliseconds: 10_000
        )
        run(node: 0, joined: node0)

        XCTAssertEqual(finished.wait(timeout: .now() + 60), .success, "the peers did not finish")
        for node in 0..<3 {
            if let error = outcomes[node].error { throw error }
            XCTAssertEqual(outcomes[node].generated, single.generated, "node \(node)'s tokens")
        }
        XCTAssertFalse(single.generated.isEmpty)
    }
}
