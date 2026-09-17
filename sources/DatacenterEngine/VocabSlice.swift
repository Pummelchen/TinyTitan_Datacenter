import Foundation

/// Which part of the output head a node computes.
///
/// The head is **1.05 s/step on every node** (`D88`), the largest piece of replicated work left, and the
/// cluster's ratio is `(replicated + experts) / (replicated + experts/n + exchange)` — so the head is what
/// raises the ceiling, not the expert plan. Splitting it by vocabulary row makes each node read and multiply
/// only its own rows.
///
/// **The split is derived, not declared.** It comes from the node count, so there is no plan field to disagree
/// with and no way for two nodes to hold different ideas of it. `ownership.nodes` is already the authority on
/// how many nodes there are.
///
/// **Why a row range is safe for bit-exactness.** Each output row of the head is one dot product over the
/// hidden width, independent of every other row: which rows a node computes cannot change what any row's
/// value is. The head loop reads and multiplies in `headBlockRows` blocks, and a slice is applied *inside*
/// that loop, so a node's rows are computed by exactly the same multiplications in exactly the same order as
/// they were before sharding.
public struct VocabSlice: Sendable, Equatable {
    public let node: Int
    public let nodes: Int
    public let vocabSize: Int

    public init(node: Int, nodes: Int, vocabSize: Int) {
        precondition(node >= 0 && node < nodes, "node \(node) is outside a \(nodes)-node split")
        precondition(vocabSize >= 0, "a head of \(vocabSize) rows is not a head")
        self.node = node
        self.nodes = nodes
        self.vocabSize = vocabSize
    }

    /// The half-open range of vocabulary rows this node owns.
    ///
    /// Blocks differ by at most one row, the same rule the shard plan uses for experts, so a vocabulary that
    /// does not divide evenly does not leave the last node with the remainder.
    public var range: Range<Int> {
        let base = vocabSize / nodes
        let extra = vocabSize % nodes
        let start = node * base + min(node, extra)
        return start..<(start + base + (node < extra ? 1 : 0))
    }

    /// Whether this node owns every row — a one-node split, where the gather has nothing to do.
    public var isWholeHead: Bool { range.lowerBound == 0 && range.upperBound == vocabSize }
}
