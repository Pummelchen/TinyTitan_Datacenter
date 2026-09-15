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

let captured: [TraceWriter.Tensor]
do {
    captured = try forward.forward(tokens: tokens)
} catch {
    fail("forward failed: \(error)")
}

let writer = TraceWriter(
    producer: "datacenter-engine-swift",
    model: ["id": modelID, "revision": revision, "compute": "fp32", "contract": "ordered_reference.py"]
)
do {
    let manifest = try writer.write(
        to: output,
        tensors: captured,
        prompt: ["tokens": tokens.map(String.init).joined(separator: ",")]
    )
    let elapsed = Date().timeIntervalSince(started)
    print(
        "wrote \(output.path): \(captured.count) tensors, digest \(manifest.digest.prefix(16))…, "
            + String(format: "%.1f s", elapsed)
    )
} catch {
    fail("could not write the trace: \(error)")
}
