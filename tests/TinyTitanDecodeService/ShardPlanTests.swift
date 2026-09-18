import Foundation
import Testing

@testable import TinyTitanDecodeProtocol

/// The shard plan, tested where it can fail: the properties that make it safe to use as an agreement between
/// nodes, and the ones that make two nodes able to tell that they disagree.
///
/// These matter more than ordinary unit tests because the plan is **data that must be identical on every
/// node**. A plan that round-trips differently, or digests differently for the same content, or accepts an
/// incomplete owner list, would not fail loudly at bring-up — it would fail as a wrong answer at the first
/// token, which is the failure mode `D154`'s reduction and `D158`'s exchange budget are both built to avoid.
@Suite("Shard plan")
struct ShardPlanTests {
    /// The real shape: 256 experts over 4 nodes, contiguous, which is the plan `D160` records.
    @Test("a contiguous 4-node plan gives every node an equal block")
    func contiguousQuarters() throws {
        let plan = ShardPlan.generate(family: "qwen3_5_moe", experts: 256, nodes: 4, distribution: .contiguous)
        try plan.validate()
        for node in 0..<4 {
            #expect(plan.count(ownedBy: node) == 64)
        }
        // Contiguous means adjacent and in node order: node 0 owns 0..<64, node 1 owns 64..<128, and so on.
        #expect(plan.owner(of: 0) == 0)
        #expect(plan.owner(of: 63) == 0)
        #expect(plan.owner(of: 64) == 1)
        #expect(plan.owner(of: 255) == 3)
    }

    /// The property the flat `owners` array exists to make structural rather than validated: every expert has
    /// exactly one owner, so the array's length is the whole check and nothing can be unowned or doubled.
    @Test("every expert is owned exactly once, and the digest changes when one moves")
    func ownershipIsTotalAndDigestIsSensitive() throws {
        let plan = ShardPlan.generate(family: "qwen3_5_moe", experts: 256, nodes: 4, distribution: .contiguous)
        #expect(plan.owners.count == plan.experts)
        #expect(Set(0..<plan.experts).allSatisfy { plan.owners[$0] >= 0 && plan.owners[$0] < plan.nodes })

        // Moving a single expert to another node must change the digest. If it did not, two nodes could
        // disagree about who owns expert 200 and both pass the bring-up comparison.
        var moved = plan.owners
        moved[200] = moved[200] == 0 ? 1 : 0
        let altered = ShardPlan(
            family: plan.family, experts: plan.experts, nodes: plan.nodes,
            distribution: .roundRobin, owners: moved
        )
        #expect(try plan.canonicalDigest() != altered.canonicalDigest())
    }

    /// Two nodes agree by comparing digests, so the same content must produce the same digest across a
    /// JSON round trip — otherwise a plan that survives the wire would look like a disagreement.
    @Test("a plan survives a JSON round trip with its digest intact")
    func roundTripPreservesDigest() throws {
        let plan = ShardPlan.generate(family: "qwen3_5_moe", experts: 256, nodes: 4, distribution: .contiguous)
        let decoded = try JSONDecoder().decode(ShardPlan.self, from: try plan.canonicalJSON())
        #expect(decoded == plan)
        #expect(try decoded.canonicalDigest() == plan.canonicalDigest())
    }

    /// An incomplete owner list is the failure the flat array is shaped to prevent, so validation must refuse
    /// it rather than let a node run with an expert nobody produces.
    @Test("an incomplete owner list is refused")
    func incompleteOwnersRefused() throws {
        let plan = ShardPlan(
            family: "qwen3_5_moe", experts: 256, nodes: 4,
            distribution: .contiguous, owners: Array(repeating: 0, count: 255)
        )
        #expect(throws: ShardPlan.Error.self) { try plan.validate() }
    }

    /// A plan carries the family it was made for, so it cannot be silently applied to a different model. That
    /// check only has teeth when it is given the model's real expert count.
    @Test("a plan is refused against a model it was not made for")
    func familyMismatchRefused() throws {
        let plan = ShardPlan.generate(family: "qwen3_5_moe", experts: 256, nodes: 4)
        #expect(throws: ShardPlan.Error.self) { try plan.validate(forFamily: "other_model", experts: 256) }
        #expect(throws: ShardPlan.Error.self) { try plan.validate(forFamily: "qwen3_5_moe", experts: 128) }
        try plan.validate(forFamily: "qwen3_5_moe", experts: 256)
    }

    /// Fewer experts than nodes is a legitimate plan, not an error: the idle node sends an empty contribution
    /// set and the reduction still completes. This is why there is no "node owns nothing" check, and it is
    /// worth pinning so nobody adds one back.
    @Test("a node may legitimately own nothing")
    func anIdleNodeIsNotAnError() throws {
        let plan = ShardPlan.generate(family: "qwen3_5_moe", experts: 2, nodes: 4, distribution: .contiguous)
        try plan.validate()
        #expect(plan.count(ownedBy: 0) == 1)
        #expect(plan.count(ownedBy: 1) == 1)
        #expect(plan.count(ownedBy: 2) == 0)
        #expect(plan.count(ownedBy: 3) == 0)
    }
}
