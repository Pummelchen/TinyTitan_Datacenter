import CryptoKit
import Foundation

/// How a plan spreads experts over nodes.
public enum ShardDistribution: String, Sendable, Codable, CaseIterable {
    /// Blocks of adjacent experts. The default: a node's experts are contiguous in the stacked tensor,
    /// so its reads walk forward through the payload instead of striding across it.
    case contiguous
    /// Expert *i* to node *i* mod N. Perfectly balanced, and it interleaves the read pattern.
    case roundRobin = "round-robin"
}

/// Which node owns which expert — the one artifact `D17` says must be pinned.
///
/// The format is chosen so that the property the reduction needs is **structural rather than
/// validated**: `owners` is a flat array indexed by expert id, so an expert cannot be owned twice and
/// cannot be quietly unowned. A per-node list of expert ids, the obvious alternative, makes both
/// mistakes representable and leaves a validator to catch them after the fact.
///
/// The plan is data, not code (I4), because it is the thing every node must agree on before a run, and
/// an agreement that lives in a compiled binary cannot be inspected, diffed or frozen. Two nodes
/// compare `canonicalDigest` during bring-up (`DC-042`) and refuse to start if they disagree — the
/// digest is over the canonical JSON, so it changes if any owner moves.
public struct ShardPlan: Sendable, Equatable, Codable {
    public static let schema = 1

    /// The model family this plan is for, so a plan cannot be applied to a model it was not made for.
    public let family: String
    /// How many experts the model has.
    public let experts: Int
    /// How many nodes the plan spreads them over.
    public let nodes: Int
    /// How `owners` was produced. Informational, and checked: a plan whose owners do not match its
    /// label is refused rather than trusted.
    public let distribution: ShardDistribution
    /// Expert id → owning node. Length is `experts`, every value is in `0..<nodes`.
    public let owners: [Int]

    /// The schema version this build writes.
    public let schema: Int

    public init(
        family: String, experts: Int, nodes: Int,
        distribution: ShardDistribution, owners: [Int], schema: Int = ShardPlan.schema
    ) {
        self.family = family
        self.experts = experts
        self.nodes = nodes
        self.distribution = distribution
        self.owners = owners
        self.schema = schema
    }

    // MARK: - generating

    /// Assign `experts` experts to `nodes`, deterministically.
    public static func generate(
        family: String, experts: Int, nodes: Int, distribution: ShardDistribution = .contiguous
    ) -> ShardPlan {
        precondition(experts > 0, "a plan for \(experts) experts owns nothing")
        precondition(nodes >= 1, "a plan for \(nodes) nodes is not a cluster")
        let owners: [Int]
        switch distribution {
        case .contiguous:
            // Blocks differ by at most one, so the counts stay as even as the division allows.
            let base = experts / nodes
            let extra = experts % nodes
            var result: [Int] = []
            result.reserveCapacity(experts)
            for node in 0..<nodes {
                result.append(contentsOf: Array(repeating: node, count: base + (node < extra ? 1 : 0)))
            }
            owners = result
        case .roundRobin:
            owners = (0..<experts).map { $0 % nodes }
        }
        return ShardPlan(
            family: family, experts: experts, nodes: nodes, distribution: distribution, owners: owners
        )
    }

    // MARK: - validating

    public enum Error: Swift.Error, CustomStringConvertible, Equatable {
        case unsupportedSchema(Int)
        case expertCountMismatch(declared: Int, found: Int)
        case unknownNode(expert: Int, node: Int, nodes: Int)
        case distributionMismatch(declared: ShardDistribution, actual: ShardDistribution)
        case noExperts
        case modelMismatch(plan: String, model: String, experts: Int)

        public var description: String {
            switch self {
            case .unsupportedSchema(let found):
                return "plan schema \(found) is not schema \(ShardPlan.schema)"
            case .expertCountMismatch(let declared, let found):
                return "plan declares \(declared) experts and assigns \(found) of them: an unowned expert "
                    + "is a term no node can produce, and the reduction would fail at the first token"
            case .unknownNode(let expert, let node, let nodes):
                return "expert \(expert) is assigned to node \(node), which is outside 0..<\(nodes)"
            case .distributionMismatch(let declared, let actual):
                return "plan is labelled \(declared.rawValue) and its owners are \(actual.rawValue)"
            case .noExperts:
                return "a plan with no experts assigns nothing"
            case .modelMismatch(let plan, let model, let experts):
                return "plan is for \(plan) with \(experts) experts and the model is \(model)"
            }
        }
    }

    /// Everything the plan must be true of, checked before a run rather than discovered during one.
    public func validate() throws {
        guard schema == Self.schema else { throw Error.unsupportedSchema(schema) }
        guard experts > 0 else { throw Error.noExperts }
        guard owners.count == experts else {
            throw Error.expertCountMismatch(declared: experts, found: owners.count)
        }
        for (expert, node) in owners.enumerated() where node < 0 || node >= nodes {
            throw Error.unknownNode(expert: expert, node: node, nodes: nodes)
        }
        let actual: ShardDistribution = owners == (0..<experts).map({ $0 % nodes }) ? .roundRobin : .contiguous
        guard actual == distribution || distribution == .contiguous else {
            throw Error.distributionMismatch(declared: distribution, actual: actual)
        }
        if distribution == .contiguous {
            // Contiguous is a claim about the shape, not just a label, so check it: each node's
            // experts must be adjacent and the blocks must run in node order.
            var seen: [Int] = []
            for node in owners where seen.last != node { seen.append(node) }
            guard seen == Array(0..<nodes) else {
                throw Error.distributionMismatch(declared: .contiguous, actual: .roundRobin)
            }
        }
        // There is deliberately no "node owns nothing" check. A missing node always fails the
        // contiguous-shape check above first, and when a model has fewer experts than the cluster has
        // nodes an idle node is a legitimate plan, not an error — it sends an empty frame and the
        // reduction still completes. The check existed until a test could not reach it.
    }

    /// Check the plan against the model a node actually has loaded.
    public func validate(forFamily model: String, experts modelExperts: Int) throws {
        try validate()
        guard family == model, experts == modelExperts else {
            throw Error.modelMismatch(plan: family, model: model, experts: modelExperts)
        }
    }

    // MARK: - identity

    /// Canonical JSON: sorted keys, no whitespace, so two nodes that agree produce the same bytes.
    public func canonicalJSON() throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return try encoder.encode(self)
    }

    /// A digest over the canonical JSON. Compare this before a run starts: two nodes that disagree about
    /// who owns what would disagree about the answer, and this is the cheapest place to find out.
    public func canonicalDigest() throws -> String {
        SHA256.hash(data: try canonicalJSON()).map { String(format: "%02x", $0) }.joined()
    }

    public func write(to url: URL) throws {
        try canonicalJSON().write(to: url)
    }

    public static func load(from url: URL) throws -> ShardPlan {
        let plan = try JSONDecoder().decode(ShardPlan.self, from: Data(contentsOf: url))
        try plan.validate()
        return plan
    }

    // MARK: - queries

    public func owner(of expert: Int) -> Int { owners[expert] }

    public func experts(ownedBy node: Int) -> [Int] {
        (0..<experts).filter { owners[$0] == node }
    }

    public func count(ownedBy node: Int) -> Int { owners.filter { $0 == node }.count }
}
