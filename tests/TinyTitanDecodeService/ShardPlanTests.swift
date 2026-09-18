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

/// How the plan is used by the engine, as opposed to what it is. `D164` located the seam at the routed expert
/// list, filtered by ownership before planning.
@Suite("Shard plan ownership")
struct ShardPlanOwnershipTests {
    /// The 4-node, 256-expert plan, and a routed set that straddles the boundary between nodes 0 and 1.
    private static func plan() -> ShardPlan {
        ShardPlan.generate(family: "qwen3_5_moe", experts: 256, nodes: 4, distribution: .contiguous)
    }

    @Test("a node keeps only its own experts, and keeps their slot positions")
    func ownedSlotsPreservePosition() throws {
        let plan = Self.plan()
        // Node 0 owns 0..<64. Slots 0,2,4 are its own; 1 and 3 belong to node 1.
        let routed = [10, 100, 20, 200, 30]
        let owned = plan.ownedSlots(amongRouted: routed, by: 0)
        #expect(owned.map(\.slot) == [0, 2, 4])
        #expect(owned.map(\.expert) == [10, 20, 30])

        // The positions matter more than the values: the reduce is slot-ordered and zero-pads, so renumbering
        // these to 0,1,2 would put node 0's contributions in the wrong slots and still look plausible.
        #expect(owned.first?.slot == 0, "the first owned expert occupies slot 0, not a renumbered one")
        #expect(owned.last?.slot == 4, "the last owned expert keeps slot 4, not a compacted index")
    }

    @Test("the complement is grouped by the node that owns each expert")
    func remoteSlotsGroupByOwner() throws {
        let plan = Self.plan()
        // The boundaries are the thing to get right, and the first version of this test got them wrong:
        // contiguous over 256 experts and 4 nodes is 0..<64, 64..<128, 128..<192, 192..<256 — so expert 100 is
        // node 1's and expert **200 is node 3's**, not node 1's. Writing the expectation from the boundary
        // arithmetic rather than from intuition is the whole point of asserting the grouped keys.
        let routed = [10, 100, 20, 200, 30]
        let remote = plan.remoteSlots(amongRouted: routed, by: 0)
        #expect(remote[1]?.map(\.slot) == [1])
        #expect(remote[1]?.map(\.expert) == [100])
        #expect(remote[3]?.map(\.slot) == [3])
        #expect(remote[3]?.map(\.expert) == [200])
        #expect(remote[2] == nil, "no expert in this routed set belongs to node 2")
        // Every routed expert is owned by exactly one node, so the two halves must reconstruct the whole set.
        let ownedCount = plan.ownedSlots(amongRouted: routed, by: 0).count
        let remoteCount = remote.values.reduce(0) { $0 + $1.count }
        #expect(ownedCount + remoteCount == routed.count)
    }

    @Test("a node that owns everything sends nothing")
    func aNodeOwningEverythingHasNoPeers() throws {
        let plan = Self.plan()
        // A routed set drawn entirely from node 3's block.
        let routed = [200, 210, 220]
        let remote = plan.remoteSlots(amongRouted: routed, by: 3)
        #expect(remote.isEmpty)
        #expect(plan.ownedSlots(amongRouted: routed, by: 3).map(\.expert) == [200, 210, 220])
    }

    /// An expert id outside the plan would trap on `owners[expert]`. A wrong plan must surface as an error, not
    /// as a crash — the same reasoning as the recoverable `expertCacheUnplaceable` in the reference's own
    /// planner, which a trap had once aborted the process on.
    @Test("an out-of-range expert is skipped rather than trapping")
    func outOfRangeExpertIsSkipped() throws {
        let plan = Self.plan()
        let routed = [0, 9999, 1]
        #expect(plan.ownedSlots(amongRouted: routed, by: 0).map(\.expert) == [0, 1])
        #expect(plan.remoteSlots(amongRouted: routed, by: 0).isEmpty)
        #expect(plan.contains(expert: 9999) == false)
    }
}

/// Expert replication: the lever `D166` identified for reaching the throughput target without breaking
/// bit-exactness. What must hold is that a replicated expert is **local everywhere** and therefore never on the
/// wire, and that adding the field did not move the digest of a plan that does not use it.
@Suite("Shard plan replication")
struct ShardPlanReplicationTests {
    private static func partitioned() -> ShardPlan {
        ShardPlan.generate(family: "qwen3_5_moe", experts: 256, nodes: 4, distribution: .contiguous)
    }

    @Test("a replicated expert is local to every node and never queued for a peer")
    func replicatedExpertNeverCrossesTheWire() throws {
        let routed = [10, 100, 20, 200]     // nodes 0, 1, 0, 3
        // Replicate 100 — node 1's expert — so nobody has to ask node 1 for it.
        let plan = ShardPlan(
            family: "qwen3_5_moe", experts: 256, nodes: 4, distribution: .contiguous,
            owners: Self.partitioned().owners, replicated: [100]
        )
        try plan.validate()

        for node in 0..<4 {
            #expect(plan.isLocal(expert: 100, to: node), "a replicated expert is local on every node")
        }
        // Node 0 now computes slot 1 itself, so only the experts it neither owns nor replicates are remote.
        #expect(plan.ownedSlots(amongRouted: routed, by: 0).map(\.slot) == [0, 1, 2])
        #expect(plan.remoteSlots(amongRouted: routed, by: 0).keys.sorted() == [3])
    }

    /// The property that makes the field safe to add to a published format: a plan that does not use it must
    /// encode exactly as it did before, or every existing plan would look like a disagreement at bring-up.
    @Test("an unreplicated plan keeps the digest it had before the field existed")
    func emptyReplicationDoesNotMoveTheDigest() throws {
        let plan = Self.partitioned()
        let json = try plan.canonicalJSON()
        #expect(!String(decoding: json, as: UTF8.self).contains("replicated"),
                "the key is omitted when empty, so the bytes — and the digest — are unchanged")
        // And the decoding path still accepts a document that has no such key at all, which is what a plan
        // written by the previous version of this type is.
        let legacy = #"{"distribution":"contiguous","experts":256,"family":"qwen3_5_moe","nodes":4,"owners":\#(plan.owners),"schema":1}"#
        let decoded = try JSONDecoder().decode(ShardPlan.self, from: Data(legacy.utf8))
        #expect(decoded.replicated.isEmpty)
        #expect(try decoded.canonicalDigest() == plan.canonicalDigest())
    }

    /// A replicated set is a *set*: two nodes listing the same experts in a different order must agree, or the
    /// digest would report a disagreement that is not one.
    @Test("the replicated set is canonicalised, so order does not change the digest")
    func replicationIsOrderInsensitive() throws {
        let owners = Self.partitioned().owners
        let a = ShardPlan(family: "qwen3_5_moe", experts: 256, nodes: 4, distribution: .contiguous,
                          owners: owners, replicated: [100, 50, 75])
        let b = ShardPlan(family: "qwen3_5_moe", experts: 256, nodes: 4, distribution: .contiguous,
                          owners: owners, replicated: [75, 100, 50])
        #expect(a.replicated == [50, 75, 100])
        #expect(try a.canonicalDigest() == b.canonicalDigest())
    }

    @Test("a replicated expert outside the model is refused")
    func replicatedExpertMustExist() throws {
        let plan = ShardPlan(family: "qwen3_5_moe", experts: 256, nodes: 4, distribution: .contiguous,
                             owners: Self.partitioned().owners, replicated: [256])
        #expect(throws: ShardPlan.Error.self) { try plan.validate() }
    }
}
