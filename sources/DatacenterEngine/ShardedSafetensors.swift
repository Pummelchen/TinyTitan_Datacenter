import Foundation

/// A checkpoint spread over shards, read as if it were one file.
///
/// The 35 B model ships as 26 shards of about 2.6 GB each with a `model.safetensors.index.json`
/// mapping every tensor to its shard, and that index is the only thing that knows the layout.
/// Reading only the first shard — which is what a single-file loader does — gives a checkpoint
/// that looks complete and is missing five sixth of its layers, so this is not a convenience:
/// M1 cannot run at all without it.
///
/// Shards are opened **lazily and kept**: an mmap costs no resident memory until a page is
/// touched, so holding 26 of them is holding 26 mappings, not 67 GB. Re-opening per read would
/// be a syscall storm on the path where a token fetches eight experts from eight places.
public final class ShardedSafetensors: WeightSource {
    private let directory: URL
    private var opened: [String: SafetensorsFile] = [:]
    private let owner: [String: String]
    /// Every tensor in the checkpoint, with the shard that holds it.
    public let tensors: [String: SafetensorsFile.Info]

    public struct Layout {
        public let shard: String
        public let info: SafetensorsFile.Info
    }

    /// Open through the index, or return nil when this checkpoint is not sharded — the caller
    /// then reads the single file it already knows about.
    public init?(snapshot: URL) {
        let indexURL = snapshot.appendingPathComponent("model.safetensors.index.json")
        guard let data = try? Data(contentsOf: indexURL),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let weightMap = object["weight_map"] as? [String: String]
        else { return nil }

        self.directory = snapshot
        self.owner = weightMap
        var collected: [String: SafetensorsFile.Info] = [:]
        var shards: [String: SafetensorsFile] = [:]
        for (name, shard) in weightMap {
            guard let file = shards[shard] ?? (try? SafetensorsFile(url: snapshot.appendingPathComponent(shard)))
            else { return nil }
            shards[shard] = file
            guard let info = file.tensors[name] else { return nil }
            collected[name] = info
        }
        self.opened = shards
        self.tensors = collected
    }

    public var names: [String] { tensors.keys.sorted() }
    public var shardCount: Int { opened.count }

    public func info(_ name: String) throws -> SafetensorsFile.Info {
        guard let info = tensors[name] else {
            throw SafetensorsFile.Error.malformedHeader("no tensor named '\(name)' in the index")
        }
        return info
    }

    private func file(for name: String) throws -> SafetensorsFile {
        guard let shard = owner[name], let file = opened[shard] else {
            throw SafetensorsFile.Error.malformedHeader("no shard holds '\(name)'")
        }
        return file
    }

    public func tensor(named name: String) throws -> [Float] {
        try file(for: name).float32(name)
    }

    public func rows(named name: String, range: Range<Int>) throws -> [Float] {
        try file(for: name).float32(name, rows: range)
    }
}

/// The inventory a spec is built from, whatever the checkpoint's layout.
///
/// The importer sees `(name, shape)` pairs and nothing else, which is what keeps L2 free of the
/// file format: a sharded checkpoint and a single one present the same list.
public protocol TensorInventory {
    var inventory: [(name: String, shape: [Int])] { get }
}

extension SafetensorsFile: TensorInventory {
    public var inventory: [(name: String, shape: [Int])] { names.map { (name: $0, shape: tensors[$0]!.shape) } }
}

extension ShardedSafetensors: TensorInventory {
    public var inventory: [(name: String, shape: [Int])] { names.map { (name: $0, shape: tensors[$0]!.shape) } }
}

public enum SnapshotWeights {
    /// Open a checkpoint directory as a weight source and an inventory, whichever layout it has.
    public static func open(_ snapshot: URL) throws -> (source: any WeightSource, inventory: any TensorInventory) {
        if let sharded = ShardedSafetensors(snapshot: snapshot) {
            return (sharded, sharded)
        }
        let weights = snapshot.appendingPathComponent("model.safetensors")
        if FileManager.default.fileExists(atPath: weights.path) {
            let file = try SafetensorsFile(url: weights)
            return (file, file)
        }
        // A sharded checkpoint whose index is missing still names its shards predictably; that is
        // a broken download rather than an unsupported layout, so say so instead of reading one
        // shard and calling the model complete.
        let entries = try FileManager.default.contentsOfDirectory(atPath: snapshot.path)
        if entries.contains(where: { $0.hasPrefix("model.safetensors-") }) {
            throw SafetensorsFile.Error.malformedHeader(
                "\(snapshot.lastPathComponent) has shards but no model.safetensors.index.json"
            )
        }
        throw SafetensorsFile.Error.truncatedFile(0)
    }
}
