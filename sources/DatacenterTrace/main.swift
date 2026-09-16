import DatacenterEngine
import DatacenterIR
import Foundation

/// `datacenter-trace <snapshot> <out-dir> <token,ids> [--model ID] [--revision SHA]`
///
/// The engine's half of the M0 comparison: run the forward on a real checkpoint and write
/// a trace in the project's container format, so `tools/trace_diff.py` can put it next to
/// the contract's trace. Before this existed the engine's numerics had only been checked
/// op by op, which cannot catch a wrong wiring — the op-level tests would all still pass.

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data(("datacenter-trace: " + message + "\n").utf8))
    exit(2)
}

var arguments = Array(CommandLine.arguments.dropFirst())
var emitSpecPath: String?
if let index = arguments.firstIndex(of: "--emit-spec") {
    guard index + 1 < arguments.count else { fail("--emit-spec needs a path") }
    emitSpecPath = arguments[index + 1]
    arguments.removeSubrange(index...(index + 1))
}
var modelID = ""
var revision = ""
if let index = arguments.firstIndex(of: "--model") {
    guard index + 1 < arguments.count else { fail("--model needs a value") }
    modelID = arguments[index + 1]
    arguments.removeSubrange(index...(index + 1))
}
if let index = arguments.firstIndex(of: "--revision") {
    guard index + 1 < arguments.count else { fail("--revision needs a value") }
    revision = arguments[index + 1]
    arguments.removeSubrange(index...(index + 1))
}
// `--emit-spec <path> <snapshot>` writes the IR spec and stops: the spec is data (L1),
// and handing it to another implementation is how that implementation can stay ignorant of
// every tensor name.
if let specPath = emitSpecPath {
    guard !arguments.isEmpty else { fail("--emit-spec needs a snapshot") }
    let forward: any ForwardPass
    do {
        forward = try ModelLoader.open(snapshot: URL(fileURLWithPath: arguments[0]))
    } catch {
        fail("could not load \(arguments[0]): \(error)")
    }
    do {
        try forward.spec.encodeJSON().write(to: URL(fileURLWithPath: specPath))
    } catch {
        fail("could not write the spec: \(error)")
    }
    print("wrote \(specPath): family \(forward.spec.family), \(forward.spec.tensors.count) tensors")
    exit(0)
}

guard arguments.count == 3 else {
    fail("usage: datacenter-trace <snapshot> <out-dir> <token,ids> [--model ID] [--revision SHA]")
}

let snapshot = URL(fileURLWithPath: arguments[0])
let output = URL(fileURLWithPath: arguments[1])
let tokens = arguments[2].split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
guard !tokens.isEmpty else { fail("no token ids given") }

let started = Date()
let forward: any ForwardPass
do {
    forward = try ModelLoader.open(snapshot: snapshot)
} catch {
    fail("could not load \(snapshot.path): \(error)")
}

// Measured, not derived: the install counts payload bytes as it reads them, so the traffic a
// forward caused is the difference across it. Zero would mean this family does not count.
let bytesBefore = forward.sourceBytesRead
let result: ForwardResult
do {
    result = try forward.forwardWithDecisions(tokens: tokens)
} catch {
    fail("forward failed: \(error)")
}
let bytesThisForward = forward.sourceBytesRead - bytesBefore
let captured = result.tensors

var writer = TraceWriter(
    producer: "datacenter-engine-swift",
    model: ["id": modelID, "revision": revision, "compute": "fp32", "contract": "ordered_reference.py"]
)
writer.discrete = result.discrete
// The expert traffic is written *beside* the trace rather than into its manifest: the manifest
// carries the digest, and a counter that changes between two runs of the same prompt would
// change the digest and make I1's comparison impossible. Evidence and identity are different
// things and belong in different files.
if !result.expertMetrics.isEmpty {
    var metrics: [String: Any] = [:]
    metrics["expert_requests"] = result.expertMetrics.reduce(0) { $0 + $1.requests }
    metrics["expert_hits"] = result.expertMetrics.reduce(0) { $0 + $1.hits }
    metrics["expert_misses"] = result.expertMetrics.reduce(0) { $0 + $1.misses }
    metrics["expert_elements_read"] = result.expertElementsRead
    // The brief's currency is **bytes read from the SSD**, and this is the measured figure: the
    // install counts the payload bytes it hands out, and a forward's share is the difference
    // across it. It replaces `expertElementsRead * 2`, which assumed bf16 on disk and so
    // overstated a 4-bit install by ~3.5x — this model stores experts at 0.578 bytes per weight
    // (4-bit codes with group-64 scales and zeros), where the estimate gave 2.0.
    metrics["install_bytes_read_this_forward"] = bytesThisForward
    metrics["install_bytes_read_total"] = forward.sourceBytesRead
    if forward.sourceTiming.counted {
        // Where the fetch's seconds went. `mix.read` was 65% of a real forward while the disk
        // measures ~1 GB/s, so read-versus-unpack is the difference between an I/O problem and a
        // kernel problem — and only the reader can say which.
        metrics["install_read_seconds"] = forward.sourceTiming.readSeconds
        metrics["install_digest_seconds"] = forward.sourceTiming.digestSeconds
        metrics["install_unpack_seconds"] = forward.sourceTiming.unpackSeconds
    }
    metrics["expert_bytes_in_memory"] = result.expertElementsRead * 4
    metrics["expert_hit_rate"] = result.expertHitRate
    metrics["expert_distinct"] = Set(result.discrete.flatMap { $0.values }).count
    metrics["layers"] = result.expertMetrics.count
    if let profile = result.profile {
        // `SHARD_PROFILE=1` asked for these. Seconds per phase, accumulated over every layer, so a
        // share is a division by the sum rather than a claim about where time goes.
        metrics["profile_seconds"] = profile.seconds
        metrics["profile_layers"] = profile.layers
    }
    let directory = output
    try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    if let data = try? JSONSerialization.data(withJSONObject: metrics, options: [.prettyPrinted, .sortedKeys]) {
        try? data.write(to: directory.appendingPathComponent("metrics.json"))
    }
}
do {
    let manifest = try writer.write(
        to: output,
        tensors: captured,
        prompt: ["tokens": tokens.map(String.init).joined(separator: ",")]
    )
    let elapsed = Date().timeIntervalSince(started)
    print(
        "wrote \(output.path): \(captured.count) tensors, "
            + "\(result.discrete.count) discrete, digest \(manifest.digest.prefix(16))…, "
            + String(format: "%.1f s", elapsed)
    )
} catch {
    fail("could not write the trace: \(error)")
}
