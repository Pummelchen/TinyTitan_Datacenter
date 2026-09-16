import Foundation

/// Joining a mesh, written once for every CLI that needs it.
///
/// A mesh rather than a star, because the all-reduce is pairwise (`D17`): a leaf in a star would hold
/// its own terms and the coordinator's and nothing else, and the completeness check would refuse the run
/// rather than let it sum less. Joining needs no ordering agreement — node *i* **connects** to every
/// lower id and **accepts** from every higher one, so no pair connects twice and every node joins in a
/// single phase, which is why every node has to be starting at roughly the same time.
public enum ClusterJoin {
    public enum Error: Swift.Error, CustomStringConvertible {
        case nodeOutsideCluster(Int, nodes: Int)
        case nodeCountMismatch(config: Int, nodes: Int)
        case cannotJoin(String)

        public var description: String {
            switch self {
            case .nodeOutsideCluster(let node, let nodes):
                return "node \(node) is outside a cluster of \(nodes)"
            case .nodeCountMismatch(let config, let nodes):
                return "the config describes \(config) nodes and the run has \(nodes)"
            case .cannotJoin(let why):
                return "could not join the mesh: \(why)"
            }
        }
    }

    /// Bind this node's endpoint and connect the mesh. The listener is returned so the caller keeps the
    /// port bound for the life of the run.
    public static func mesh(
        config: ClusterConfig, node: Int, nodes: Int, timeoutMilliseconds: Int
    ) throws -> (transports: [any ContributionTransport], listener: TCPListener) {
        guard node >= 0, node < nodes else { throw Error.nodeOutsideCluster(node, nodes: nodes) }
        guard config.nodes == nodes else { throw Error.nodeCountMismatch(config: config.nodes, nodes: nodes) }
        let mine = config.endpoints[node]
        let listener: TCPListener
        do {
            listener = try TCPListener(host: mine.host, port: mine.port, backlog: Int32(max(1, nodes)))
        } catch {
            throw Error.cannotJoin("could not bind \(mine.host):\(mine.port): \(error)")
        }
        var transports: [any ContributionTransport] = []
        do {
            for peer in 0..<node {
                transports.append(
                    try TCPTransport.connect(
                        host: config.endpoints[peer].host, port: config.endpoints[peer].port,
                        timeoutMilliseconds: timeoutMilliseconds
                    )
                )
            }
            for _ in (node + 1)..<nodes {
                transports.append(try listener.accept(timeoutMilliseconds: timeoutMilliseconds))
            }
        } catch {
            throw Error.cannotJoin("\(error)")
        }
        return (transports, listener)
    }
}
