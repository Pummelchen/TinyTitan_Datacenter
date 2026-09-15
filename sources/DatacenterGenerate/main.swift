import DatacenterEngine
import DatacenterIR
import Foundation

/// `datacenter-generate <snapshot> <out-dir> <prompt,ids> <max-new-tokens> [--model ID] [--revision SHA]`
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
for flag in ["--model", "--revision"] {
    if let index = arguments.firstIndex(of: flag) {
        guard index + 1 < arguments.count else { fail("\(flag) needs a value") }
        if flag == "--model" { modelID = arguments[index + 1] } else { revision = arguments[index + 1] }
        arguments.removeSubrange(index...(index + 1))
    }
}
guard arguments.count == 4 else {
    fail("usage: datacenter-generate <snapshot> <out-dir> <prompt,ids> <max-new-tokens> [--model ID] [--revision SHA]")
}

let snapshot = URL(fileURLWithPath: arguments[0])
let output = URL(fileURLWithPath: arguments[1])
let prompt = arguments[2].split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
let maxNewTokens = Int(arguments[3]) ?? -1
guard !prompt.isEmpty else { fail("the prompt must contain at least one token") }
guard maxNewTokens >= 0 else { fail("max-new-tokens must be a non-negative integer") }

let forward: any ForwardPass
do {
    forward = try ModelLoader.open(snapshot: snapshot)
} catch {
    fail("could not load \(snapshot.path): \(error)")
}

let generation: Generation
do {
    generation = try forward.generate(prompt: prompt, maxNewTokens: maxNewTokens)
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
    print(
        "wrote \(output.path): \(generation.captured.count) tensors, digest \(manifest.digest.prefix(16))…, "
            + String(format: "%.1f s over %d step(s), slowest %.1f s", total, generation.secondsPerStep.count, slowest)
    )
} catch {
    fail("could not write the trace: \(error)")
}
