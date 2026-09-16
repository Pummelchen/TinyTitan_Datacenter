import DatacenterEngine
import DatacenterIR
import Foundation

/// `datacenter-generate <snapshot> <out-dir> <prompt,ids> <max-new-tokens> [--cached] [--model ID] [--revision SHA]`
///
/// `--cached` decodes one position per token against a per-layer state (the KV cache and the
/// Gated DeltaNet recurrence) instead of re-running the whole sequence each step. It is a second
/// numeric path by `D8`, so the two modes are expected to produce the same *tokens* rather than
/// the same bytes, and the tool reports which mode it used so a measurement can never be
/// attributed to the wrong one.
///
/// Greedy generation with the whole sequence re-run at every step (M0 has no KV cache).
/// Writes a trace of the final forward pass, with the generated tokens recorded as a
/// **discrete decision** so the differ compares them as an index set rather than as
/// numbers — which is what I3 asks for and what a tolerance cannot express.

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data(("datacenter-generate: " + message + "\n").utf8))
    exit(2)
}

var arguments = Array(CommandLine.arguments.dropFirst())
var modelID = ""
var revision = ""
// Sharding options. `--plan`/`--config`/`--node` join a mesh exactly as `datacenter-node` does, so a
// generation run can be spread over the cluster and measured — which is what M3's gate needs. The
// cached decode path goes through the same mixture entry point as the sequence path, so a sharded
// `--cached` run reduces rather than quietly computing a fraction of the experts.
var planPath: String?
var configPath: String?
var shardNode = -1
var shardTimeout = 30_000

// `--cached` is a flag rather than a positional argument, so it is removed from the list before
// the arity check — leaving it in made every cached invocation print the usage line and exit.
let cached = arguments.contains("--cached")
arguments.removeAll { $0 == "--cached" }

var index = 0
while index < arguments.count {
    switch arguments[index] {
    case "--plan", "--config", "--node", "--timeout-ms":
        let flag = arguments[index]
        guard index + 1 < arguments.count else { fail("\(flag) needs a value") }
        let value = arguments[index + 1]
        switch flag {
        case "--plan": planPath = value
        case "--config": configPath = value
        case "--node":
            guard let parsed = Int(value) else { fail("--node needs a number") }
            shardNode = parsed
        default:
            guard let parsed = Int(value) else { fail("--timeout-ms needs a number") }
            shardTimeout = parsed
        }
        arguments.removeSubrange(index...(index + 1))
    default:
        index += 1
    }
}

for flag in ["--model", "--revision"] {
    if let index = arguments.firstIndex(of: flag) {
        guard index + 1 < arguments.count else { fail("\(flag) needs a value") }
        if flag == "--model" { modelID = arguments[index + 1] } else { revision = arguments[index + 1] }
        arguments.removeSubrange(index...(index + 1))
    }
}
guard arguments.count == 4 else {
    fail("usage: datacenter-generate <snapshot> <out-dir> <prompt,ids> <max-new-tokens> [--cached] [--model ID] [--revision SHA]")
}

let snapshot = URL(fileURLWithPath: arguments[0])
let output = URL(fileURLWithPath: arguments[1])
let prompt = arguments[2].split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
let maxNewTokens = Int(arguments[3]) ?? -1
guard !prompt.isEmpty else { fail("the prompt must contain at least one token") }
guard maxNewTokens >= 0 else { fail("max-new-tokens must be a non-negative integer") }

/// The shard context, when this is one node of a cluster run.
struct Sharded {
    let forward: any ForwardPass
    let listener: TCPListener
    let node: Int
    let nodes: Int
}

let sharded: Sharded?
if let planPath {
    guard let configPath, shardNode >= 0 else { fail("--plan needs --config and --node") }
    do {
        let plan = try ShardPlan.load(from: URL(fileURLWithPath: planPath))
        let addresses = try ClusterConfig.load(
            from: URL(fileURLWithPath: configPath), thisNode: shardNode
        )
        // The geometry and the family come from the model this node actually opened, and the plan is
        // checked against them (`D20`) — a plan for another model must be refused, not run.
        let probe = try Qwen3_5Forward(install: snapshot)
        let shape = try probe.mixtureShape()
        try plan.validate(forFamily: probe.spec.family, experts: shape.experts)
        let joined = try ClusterJoin.mesh(
            config: addresses, node: shardNode, nodes: plan.nodes, timeoutMilliseconds: shardTimeout
        )
        print("PORT \(joined.listener.port)")
        fflush(stdout)
        let identity = ClusterIdentity(
            family: probe.spec.family, revision: revision, experts: shape.experts,
            hiddenSize: shape.hiddenSize, topK: shape.topK, planDigest: try plan.canonicalDigest()
        )
        _ = try ClusterHandshake.perform(
            ours: NodeDeclaration(identity: identity, node: shardNode, nodes: plan.nodes),
            peers: joined.transports,
            policy: ExchangePolicy(receiveTimeoutMilliseconds: shardTimeout, attempts: 1)
        )
        let shard = ShardExecution(
            node: shardNode, ownership: ExpertOwnership(plan: plan), peers: joined.transports,
            policy: ExchangePolicy(receiveTimeoutMilliseconds: shardTimeout, attempts: 3)
        )
        sharded = Sharded(
            forward: try Qwen3_5Forward(install: snapshot, shard: shard), listener: joined.listener,
            node: shardNode, nodes: plan.nodes
        )
    } catch {
        fail("could not join the cluster: \(error)")
    }
} else {
    sharded = nil
}

let forward: any ForwardPass
if let sharded {
    forward = sharded.forward
} else {
    do {
        forward = try ModelLoader.open(snapshot: snapshot)
    } catch {
        fail("could not load \(snapshot.path): \(error)")
    }
}
_ = sharded  // held for the life of the run so the port stays bound

let generation: Generation
do {
    generation = cached
        ? try (forward as? Qwen3_5Forward)?.generateCached(prompt: prompt, maxNewTokens: maxNewTokens)
            ?? forward.generate(prompt: prompt, maxNewTokens: maxNewTokens)
        : try forward.generate(prompt: prompt, maxNewTokens: maxNewTokens)
} catch {
    fail("generation failed: \(error)")
}

var writer = TraceWriter(
    producer: "datacenter-engine-swift",
    model: ["id": modelID, "revision": revision, "compute": "fp32", "contract": "ordered_reference.py"]
)
writer.discrete = [
    TraceWriter.Discrete(
        name: "generated.tokens", shape: [generation.generated.count], values: generation.generated
    )
]

do {
    let manifest = try writer.write(
        to: output, tensors: generation.captured,
        prompt: ["tokens": generation.tokens.map(String.init).joined(separator: ",")]
    )
    let total = generation.secondsPerStep.reduce(0, +)
    let slowest = generation.secondsPerStep.max() ?? 0
    print("generated: \(generation.generated.map(String.init).joined(separator: ","))")
    if let sharded {
        print("sharded: node \(sharded.node) of \(sharded.nodes), one all-reduce per mixture layer")
    }
    // `DC-106`: the dense backbone's residency, as a **count** — whole-tensor payload bytes that had to
    // come from disk. A generation is several forwards over the same layers, so with residency on this
    // should be near one forward's worth rather than one per token.
    let cache = forward.payloadCacheMetrics
    print(
        "dense payload: \(cache.bytesRead) B read from disk, \(cache.hits) read(s) served from "
            + "\(cache.bytesHeld) B resident"
    )
    print("mode: \(cached ? "cached decode" : "full sequence each step")")
    // The margins, step by step: a marginal flip is not a defect and a large-margin disagreement
    // is, and the token ids alone cannot tell them apart.
    let margins = generation.margins.map { String(format: "%.4f", $0) }.joined(separator: ", ")
    print("top-2 margins: \(margins)")
    print(
        "wrote \(output.path): \(generation.captured.count) tensors, digest \(manifest.digest.prefix(16))…, "
            + String(format: "%.1f s over %d step(s), slowest %.1f s", total, generation.secondsPerStep.count, slowest)
    )
} catch {
    fail("could not write the trace: \(error)")
}
