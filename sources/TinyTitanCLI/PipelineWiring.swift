import Foundation
import TinyTitan
import TinyTitanDecodeProtocol

/// The runner already has the four properties a pipeline stage needs, so putting it on a ring adds no state.
///
/// **It lives in the CLI rather than beside the runner because of the module graph**, and getting that wrong is not
/// cosmetic: `TinyTitan` does not declare a dependency on `TinyTitanDecodeProtocol`, and SwiftPM's dependency scan
/// reports the violation the moment the conformance is written there. Adding the dependency would have changed the
/// module graph **to satisfy a test rather than the design** (`D348`). The CLI imports both, so the conformance
/// belongs here.
extension RealForwardRunner: PipelineEndpoints {}

/// Put this process on a pipeline ring, if the environment says to.
///
/// The ring is a directed cycle of stages, and **each stage has two sockets rather than one**: it accepts from its
/// predecessor and connects to its successor. One connection per edge, one direction per connection - which is what
/// makes a stage's `input` and `output` separate handles here even though `D348`'s test used the two ends of a
/// single socket. Both are expressible because `install` takes them independently.
///
/// Configuration is by environment, matching the A5 probe's seam: `TINYTITAN_STAGE_LISTEN` is the port this stage
/// accepts its predecessor on, and `TINYTITAN_STAGE_CONNECT` is the `host:port` of its successor. **Either may be
/// absent**, because the ring's ends are not stages - the first stage has no predecessor to accept from and the last
/// has no successor to connect to (`D348`).
///
/// **`rowWidth` comes from the runner rather than from configuration**: a stage cannot know its chunk width before it
/// has tokens, and a wrong width here would silently misread a residual as several rows of garbage, which is the
/// failure `D335` cost three rounds to find. It is derived from `hiddenStateBytes`, the public accessor, because the
/// residual width itself is internal to the runtime - and reaching into the runtime for a number the runtime already
/// publishes would be the wrong direction of dependency.
public enum PipelineWiring {
    public enum WiringError: Error, Equatable {
        case malformedConnect(String)
        case badPort(String)
    }

    /// **The reverse edge, read and written.** `D358` established that a ring needs the chosen token to travel back
    /// to the stage that embeds it: without it a first stage generates from its own tokens, which for a partial
    /// forward are meaningless, and every hidden state it publishes after the first is the state of the wrong token.
    ///
    /// **It is a separate socket from the forward edge, not the other end of it**, because the two legs have
    /// different lifetimes: a stage binds the forward edge on a port its predecessor connects to, and it connects
    /// the reverse edge to a port its successor binds. One stage can therefore be a listener on one and a client on
    /// the other, which a single connection could not express.
    ///
    /// `TINYTITAN_STAGE_BACK_LISTEN` is for the stage that EMBEDS - it waits for a token to come back.
    /// `TINYTITAN_STAGE_BACK_CONNECT` is for the stage that SAMPLES - it sends the token it chose.
    public static func installReverseEdge(on runner: RealForwardRunner) throws -> Bool {
        let env = ProcessInfo.processInfo.environment
        let listen = env["TINYTITAN_STAGE_BACK_LISTEN"]
        let connect = env["TINYTITAN_STAGE_BACK_CONNECT"]
        guard listen != nil || connect != nil else { return false }

        // WHICH END ORIGINATES IS INDEPENDENT OF WHICH END READS, and separating them is what makes this edge usable
        // on a node that cannot originate at all. D287 established that macOS Local Network Privacy lets a
        // harness-launched process receive and refuses to let it originate; D364 measured node1 refusing an outbound
        // connection to node3 while node3 reached node1 without trouble. So the stage that READS the token may also
        // be the one that CONNECTS: it opens the socket, the sampling stage accepts, and the bytes flow the other
        // way along it. `TINYTITAN_STAGE_BACK_ROLE` names which end of the DATA this stage is - `source` or `sink` -
        // while LISTEN and CONNECT name which end of the SOCKET, and the two need not agree.
        let role = env["TINYTITAN_STAGE_BACK_ROLE"] ?? (listen != nil ? "source" : "sink")
        let isSource = role == "source"
        // A MIDDLE STAGE IS BOTH, AND THAT IS THE WHOLE FOUR-STAGE EXTENSION (D417, D418). With two stages a stage
        // either consumes the returned token or publishes it, so one predicate sufficed and `end` could be a single
        // handle. A stage with a predecessor AND a successor does both at once: it reads the token its successor
        // chose and writes its own to its predecessor. So there are two predicates and two endpoints, and the
        // listening socket carries the source while the connecting one carries the sink - which is the same
        // LISTEN/CONNECT pair the two-stage ring uses, read in a combination it never tried.
        let isBoth = role == "both"
        let wantsSource = isSource || isBoth
        let wantsSink = !wantsSource || isBoth

        // WHICH END DIALS IS A DEPLOYMENT CHOICE, NOT A CONSEQUENCE OF THE ROLE (D435). The comment above already
        // says the two are independent, and the default wiring reads from the LISTEN end and writes to the CONNECT
        // end - which forces the successor to dial this stage. On a machine whose engine cannot originate at all
        // (node4, refused by macOS Local Network Privacy) that is fatal, and it need not be: with
        // TINYTITAN_STAGE_BACK_SWAP the stage READS from the connection it dialled and WRITES to the one it
        // accepted, so a pure-listener head is expressible and the whole chain works without that machine ever
        // opening an outbound socket.
        let swap = env["TINYTITAN_STAGE_BACK_SWAP"] != nil
        var sourceEnd: FileHandle?
        var sinkEnd: FileHandle?
        if let listen {
            guard let port = UInt16(listen) else { throw WiringError.badPort(listen) }
            let pair = try DecodeTCPSocket.listenAndAccept(host: "0.0.0.0", port: port)
            if swap {
                if wantsSink { sinkEnd = pair.output }
            } else {
                if wantsSource { sourceEnd = pair.input }
                if wantsSink { sinkEnd = pair.output }
            }
        }
        if let connect {
            let parts = connect.split(separator: ":")
            guard parts.count == 2, let port = UInt16(parts[1]) else {
                throw WiringError.malformedConnect(connect)
            }
            var last: Error = WiringError.malformedConnect(connect)
            var opened: FileHandle?
            // REPORT BEFORE ACTING. Every network-level explanation for this connect has been excluded by direct
            // test - the address (D363), the routing (D364), the listening side (D367) and the port's reachability
            // from this node (D368) - so what is left is this path, and a path that fails silently cannot be told
            // apart from one that was never entered.
            FileHandle.standardError.write(Data(
                "[back] connecting to \(parts[0]):\(port) as \(role)\n".utf8))
            for attempt in 0..<connectRetries {
                do {
                    let pair = try DecodeTCPSocket.connect(host: String(parts[0]), port: port)
                    if swap {
                        if wantsSource { sourceEnd = pair.input }
                    } else {
                        if wantsSink { sinkEnd = pair.output }
                        if wantsSource && sourceEnd == nil { sourceEnd = pair.input }
                    }
                    opened = pair.output
                    break
                } catch {
                    // The FIRST failure is reported, not the last: a retry loop that only reports its final error
                    // hides whether the target was ever reachable at all, and the first attempt is the one that
                    // says what the network thought.
                    if attempt == 0 {
                        FileHandle.standardError.write(Data(
                            "[back] first connect failed: \(error)\n".utf8))
                    }
                    last = error
                    usleep(200_000)
                }
            }
            guard opened != nil else {
                FileHandle.standardError.write(Data(
                    "[back] gave up after \(connectRetries) attempts: \(last)\n".utf8))
                throw last
            }
            FileHandle.standardError.write(Data("[back] connected\n".utf8))
        }
        guard sourceEnd != nil || sinkEnd != nil else { return false }

        if wantsSource, let end = sourceEnd {
            // -1 is the sentinel for "nothing has come back yet"; 0 is a legitimate token id, so the two must be
            // distinguishable (D361).
            runner.nextTokenSource = { position in
                guard let token = try? PipelineStage.receiveToken(from: end) else { return -1 }
                // WHAT THIS STAGE WAS TOLD, and for which position. D389 narrowed the remaining fault to a one-step
                // alignment and this is the instrument for it: the sequence of tokens consumed here, against the
                // sequence published by the peer, is the whole question - and printing positions alone (D350's
                // instrument) cannot answer it because the positions are already known to line up.
                FileHandle.standardError.write(Data(
                    "[tok] told pos=\(position) token=\(token)\n".utf8))
                return Int32(token)
            }
        }
        if wantsSink, let end = sinkEnd {
            runner.nextTokenSink = { token, layer in
                FileHandle.standardError.write(Data(
                    "[tok] chose pos=\(layer) token=\(token)\n".utf8))
                try? PipelineStage.sendToken(Int(token), layer: layer, to: end)
            }
        }
        return true
    }

    /// Rows a handoff buffer can carry - enough for every prompt this engine has been run against, and 16 MB at
    /// D=2048, which is nothing beside the 19 GB install.
    static let chunkRows = 4096

    /// How many times to retry the connect to a successor, at 200 ms each - about 90 seconds, which is comfortably
    /// past a cold 19 GB model load on this hardware and still finite.
    static let connectRetries = 450

    /// Returns true when this process was put on a ring, so the caller can report it.
    @discardableResult
    public static func installIfConfigured(on runner: RealForwardRunner, exitLayer: Int) throws -> Bool {
        let env = ProcessInfo.processInfo.environment
        let listen = env["TINYTITAN_STAGE_LISTEN"]
        let connect = env["TINYTITAN_STAGE_CONNECT"]
        guard listen != nil || connect != nil else { return false }

        // A stage that accepts must have somewhere for the received frame to land, and `hiddenIn` is that buffer.
        // `install` refuses without one rather than dropping the input silently.
        // SIZED FOR A CHUNK, not for one token. A prefill publishes and consumes `t` rows at once (D337), and a
        // one-row buffer silently truncates that to one - which is how the ring produced whitespace where a single
        // node produces ' Paris, a city'. Oversizing is only SAFE because the row count now travels with the
        // publish (`D354`): without that, the receiver would read the whole 16 MB allocation.
        if listen != nil, runner.hiddenIn == nil {
            runner.hiddenIn = runner.makeHiddenStateBuffer(rows: Self.chunkRows)
        }
        // AND `hiddenOut` FOR A PUBLISHER, which the first version forgot. Both the post-loop publish and the
        // decode hook are gated on `hiddenOut` being non-nil, so a stage that connects but is never given one
        // publishes NOTHING - and the peer blocks on a frame that will never arrive. The instrumented run said so
        // in one line: the publisher printed no `[wire] send` at all while its consumer reported
        // `recv FAILED for pos=0`. Counting the two sides took one run where three hypotheses had taken two.
        if connect != nil, runner.hiddenOut == nil {
            runner.hiddenOut = runner.makeHiddenStateBuffer(rows: Self.chunkRows)
        }

        var input: FileHandle?
        if let listen {
            guard let port = UInt16(listen) else { throw WiringError.badPort(listen) }
            // Bind before this stage is asked to compute, so a predecessor connecting early waits in the backlog
            // rather than being refused.
            // BIND ON EVERY INTERFACE, not loopback. The first cross-machine run used 127.0.0.1 here, which is
            // correct for a one-machine test and unreachable from a peer: the producer on node3 was refused while
            // this stage sat happily listening on an address only it could see. `DecodeTCPSocket` takes a literal
            // address rather than a name by design (`D18`), and `0.0.0.0` is the literal that means "any".
            input = try DecodeTCPSocket.listenAndAccept(host: "0.0.0.0", port: port).input
        }

        var output: FileHandle?
        if let connect {
            let parts = connect.split(separator: ":")
            guard parts.count == 2, let port = UInt16(parts[1]) else {
                throw WiringError.malformedConnect(connect)
            }
            // RETRY, because a successor binds only after it has loaded the model - which takes far longer than
            // this process takes to start. A single connect is therefore refused in the normal case rather than the
            // exceptional one, and the first cross-machine run failed exactly this way. The shard path solved the
            // same race with a bounded retry (`connectRetrySeconds`, 60 s), and this mirrors it rather than
            // inventing a second answer.
            var last: Error = WiringError.malformedConnect(connect)
            for _ in 0..<connectRetries {
                do {
                    output = try DecodeTCPSocket.connect(host: String(parts[0]), port: port).output
                    break
                } catch { last = error; usleep(200_000) }
            }
            guard output != nil else { throw last }
        }

        try PipelineStage.install(on: runner,
                                  input: input,
                                  output: output,
                                  rowWidth: runner.hiddenStateBytes / MemoryLayout<Float16>.stride,
                                  rows: 1,
                                  exitLayer: exitLayer)
        return true
    }
}
