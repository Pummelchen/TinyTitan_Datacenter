import DatacenterEngine
import DatacenterIR
import Foundation

/// `datacenter-node <install> <out-dir> <token,ids> <plan.json> --node N`
/// `                  (--listen [host:]port | --connect host:port) [--revision SHA] [--timeout-ms N]`
///
/// One node of a sharded run, as its own process.
///
/// This exists so the last claim before the cluster can be made on one machine: two **processes**, each
/// with its own address space and its own install reader, joining over TCP, handshaking, and producing a
/// trace that `trace_diff` compares with the single-node one byte for byte. Two threads in one process
/// share a heap; this does not, and the difference is exactly the class of bug the whole phase is about.
///
/// `--listen` prints `PORT <n>` on stdout once it is bound, so a harness can ask for port 0 and read back
/// what it got rather than guessing a free port and racing for it.
func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data(("datacenter-node: " + message + "\n").utf8))
    exit(2)
}

// `--version` answers before anything else: §1.3 wants the identity of an artifact to be observable from
// the artifact itself, and a version that exists only in an archive's filename is not observable from the
// binary inside it.
if CommandLine.arguments.contains("--version") {
    print(TinyTitanVersion.string)
    exit(0)
}

var arguments = Array(CommandLine.arguments.dropFirst())
var node = -1
var listen: String?
var connect: String?
var config: String?
var revision = "local"
var timeoutMilliseconds = 30_000

var index = 0
while index < arguments.count {
    switch arguments[index] {
    case "--node":
        guard index + 1 < arguments.count, let value = Int(arguments[index + 1]) else { fail("--node needs a number") }
        node = value
        arguments.removeSubrange(index...(index + 1))
    case "--listen":
        guard index + 1 < arguments.count else { fail("--listen needs [host:]port") }
        listen = arguments[index + 1]
        arguments.removeSubrange(index...(index + 1))
    case "--connect":
        guard index + 1 < arguments.count else { fail("--connect needs host:port") }
        connect = arguments[index + 1]
        arguments.removeSubrange(index...(index + 1))
    case "--config":
        guard index + 1 < arguments.count else { fail("--config needs a cluster config path") }
        config = arguments[index + 1]
        arguments.removeSubrange(index...(index + 1))
    case "--revision":
        guard index + 1 < arguments.count else { fail("--revision needs a value") }
        revision = arguments[index + 1]
        arguments.removeSubrange(index...(index + 1))
    case "--timeout-ms":
        guard index + 1 < arguments.count, let value = Int(arguments[index + 1]) else {
            fail("--timeout-ms needs a number")
        }
        timeoutMilliseconds = value
        arguments.removeSubrange(index...(index + 1))
    default:
        index += 1
    }
}

guard arguments.count == 4 else {
    fail(
        "usage: datacenter-node <install> <out-dir> <token,ids> <plan.json> --node N "
            + "(--listen [host:]port | --connect host:port)"
    )
}
guard node >= 0 else { fail("--node is required") }
if config == nil {
    guard (listen == nil) != (connect == nil) else { fail("give --config, or exactly one of --listen / --connect") }
} else {
    guard listen == nil, connect == nil else { fail("--config replaces --listen and --connect") }
}

let install = URL(fileURLWithPath: arguments[0])
let output = URL(fileURLWithPath: arguments[1])
let tokens = arguments[2].split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
guard !tokens.isEmpty else { fail("no token ids given") }
let planURL = URL(fileURLWithPath: arguments[3])

// The plan is the agreement, so it is loaded and validated before anything else — including before the
// socket, because a node that has already bound a port is a node that has already half-joined.
let plan = try ShardPlan.load(from: planURL)
guard node < plan.nodes else { fail("node \(node) is outside the plan's \(plan.nodes) nodes") }

let forward: Qwen3_5Forward
do {
    forward = try Qwen3_5Forward(install: install)
} catch {
    fail("could not load \(install.path): \(error)")
}
let shape: MixtureShape
do {
    shape = try forward.mixtureShape()
} catch {
    fail("could not read the mixture geometry: \(error)")
}
do {
    try plan.validate(forFamily: forward.spec.family, experts: shape.experts)
} catch {
    // Reported, not trapped: an uncaught top-level `try` in Swift becomes a SIGTRAP and a stack line,
    // which is a poor way to learn that two files disagree about which model this is.
    fail("the plan does not fit this model: \(error)")
}
let ownership = ExpertOwnership(plan: plan)

// Bring-up: anyone may connect, but the declaration must agree with what this node actually holds.
//
// Two topologies, one exchange. `--listen`/`--connect` is the two-node case a gate needs; `--config`
// is the **full mesh** that N nodes need, because the all-reduce is pairwise (`D17`) — a star would
// leave a leaf holding only the coordinator's terms, and the completeness check would refuse the run
// rather than let it sum less. The mesh rule is deterministic so no pair connects twice: a node
// connects to every peer with a **lower** id and accepts from every peer with a **higher** one.
var peers: [any ContributionTransport] = []
var listener: TCPListener?
if let config {
    let addresses: ClusterConfig
    do {
        addresses = try ClusterConfig.load(from: URL(fileURLWithPath: config), thisNode: node)
    } catch {
        fail("the cluster config does not fit this node: \(error)")
    }
    do {
        let joined = try ClusterJoin.mesh(
            config: addresses, node: node, nodes: plan.nodes, timeoutMilliseconds: timeoutMilliseconds
        )
        peers = joined.transports
        listener = joined.listener
        print("PORT \(joined.listener.port)")
        fflush(stdout)
    } catch {
        fail("\(error)")
    }
} else if let listen {
    let parts = listen.split(separator: ":")
    let host = parts.count > 1 ? String(parts[0]) : "127.0.0.1"
    let port = Int(parts.count > 1 ? parts[1] : parts[0]) ?? 0
    let bound: TCPListener
    do {
        bound = try TCPListener(host: host, port: port)
    } catch {
        fail("could not listen on \(listen): \(error)")
    }
    listener = bound
    print("PORT \(bound.port)")
    fflush(stdout)
    do {
        peers.append(try bound.accept(timeoutMilliseconds: timeoutMilliseconds))
    } catch {
        fail("no peer connected: \(error)")
    }
} else if let connect {
    let parts = connect.split(separator: ":")
    guard parts.count == 2, let port = Int(parts[1]) else { fail("--connect needs host:port") }
    do {
        peers.append(
            try TCPTransport.connect(
                host: String(parts[0]), port: port, timeoutMilliseconds: timeoutMilliseconds
            )
        )
    } catch {
        fail("could not connect to \(connect): \(error)")
    }
} else {
    fail("unreachable")
}
_ = listener  // held for the life of the run so the port stays bound

let identity: ClusterIdentity
do {
    identity = ClusterIdentity(
        family: forward.spec.family, revision: revision, experts: shape.experts,
        hiddenSize: shape.hiddenSize, topK: shape.topK, planDigest: try plan.canonicalDigest()
    )
} catch {
    fail("could not digest the plan: \(error)")
}
do {
    _ = try ClusterHandshake.perform(
        ours: NodeDeclaration(identity: identity, node: node, nodes: plan.nodes),
        peers: peers,
        policy: ExchangePolicy(receiveTimeoutMilliseconds: timeoutMilliseconds, attempts: 1)
    )
    print("BRINGUP ok: node \(node) of \(plan.nodes), plan \(try plan.canonicalDigest().prefix(16))…")
    fflush(stdout)
} catch {
    fail("bring-up refused: \(error)")
}

let started = Date()
let bytesBeforeForward = forward.sourceBytesRead
/// The shard's exchange counters, named so the metrics after the forward can report them (`DC-081`).
var exchangeLedger: ExchangeLedger?
let shardedForward: Qwen3_5Forward
let result: ForwardResult
do {
    let shard = ShardExecution(
        node: node, ownership: ownership, peers: peers,
        policy: ExchangePolicy(receiveTimeoutMilliseconds: timeoutMilliseconds, attempts: 3)
    )
    exchangeLedger = shard.ledger
    shardedForward = try Qwen3_5Forward(install: install, shard: shard)
    result = try shardedForward.forwardWithDecisions(tokens: tokens)
} catch {
    fail("forward failed: \(error)")
}

var writer = TraceWriter(
    producer: "datacenter-engine-swift-sharded",
    model: ["id": install.lastPathComponent, "revision": revision, "compute": "fp32", "contract": "ordered_reference.py"]
)
writer.discrete = result.discrete
if !result.expertMetrics.isEmpty {
    // Measured, not derived: `elementsRead * 2` was removed from the single-node trace in the first round
    // of this work because it assumes bf16 on disk and overstated a 4-bit install by ~3.5x, and it does
    // not belong back in the sharded one wearing a different name.
    let timing = shardedForward.sourceTiming
    let exchange = exchangeLedger?.metrics ?? ExchangeMetrics()
    let payload = forward.payloadCacheMetrics
    let metrics: [String: Any] = [
        "expert_requests": result.expertMetrics.reduce(0) { $0 + $1.requests },
        "expert_elements_read": result.expertElementsRead,
        "node": node,
        "nodes": plan.nodes,
        "plan_digest": (try? plan.canonicalDigest()) ?? "",
        "install_bytes_read_this_forward": max(0, timing.bytes - bytesBeforeForward),
        "install_verified_bytes": timing.verifiedBytes,
        "install_read_seconds": timing.readSeconds,
        "install_unpack_seconds": timing.unpackSeconds,
        // `DC-081`: what the cluster cost this node, and what the dense backbone's residency saved it.
        "exchange_reduces": exchange.reduces,
        "exchange_terms_sent": exchange.termsSent,
        "exchange_terms_received": exchange.termsReceived,
        "exchange_bytes_sent": exchange.bytesSent,
        "exchange_bytes_received": exchange.bytesReceived,
        "exchange_seconds": exchange.seconds,
        "dense_payload_bytes_read": payload.bytesRead,
        "dense_payload_cache_hits": payload.hits,
        "dense_payload_bytes_held": payload.bytesHeld,
    ]
    try? FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
    if let data = try? JSONSerialization.data(withJSONObject: metrics, options: [.prettyPrinted, .sortedKeys]) {
        try? data.write(to: output.appendingPathComponent("metrics.json"))
    }
}
do {
    let manifest = try writer.write(
        to: output, tensors: result.tensors,
        prompt: ["tokens": tokens.map(String.init).joined(separator: ",")]
    )
    print(
        "wrote \(output.path): \(result.tensors.count) tensors, \(result.discrete.count) discrete, "
            + "digest \(manifest.digest.prefix(16))…, "
            + String(format: "%.1f s", Date().timeIntervalSince(started))
    )
} catch {
    fail("could not write the trace: \(error)")
}
