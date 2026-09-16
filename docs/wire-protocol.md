# The all-reduce wire protocol

What two nodes say to each other, and what happens when one of them stops saying it. This is the contract
`DC-009` asked for as an ADR; the decisions behind it are `D17` (the reduction order), `D18` (framing and
dtype), `D19` (failure semantics) and `D30`/`DC-083` (what a mismatched peer can do), and this document is
where they are readable in one place. What runs over the link — Thunderbolt, LAN, SFP/QSFP — is `DC-008`'s
question and does not change anything below.

## What is on the wire

Every message is a length-framed contribution frame. The transport owns the length prefix; the frame carries
the numbers; both are refused on the number **before** anything is allocated.

| Field | Width | Note |
| --- | --- | --- |
| length prefix | u32, little-endian | owned by the transport, not the frame; `SocketContributionTransport` writes it |
| magic | 4 bytes | `TTDC` |
| version | u16 | `1` |
| tokens | u32 | the batch this frame is about |
| hiddenSize | u32 | the width every term must have |
| count | u32 | how many contributions follow |
| per contribution: token | u32 | which token |
| per contribution: expert | u32 | which expert |
| per contribution: width | u32 | must equal `hiddenSize` |
| per contribution: scale | u32 | an IEEE-754 bit pattern, not a decimal |
| per contribution: values | width × u32 | IEEE-754 bit patterns |

`headerBytes` is 18. **Floats travel as bit patterns**, in both directions: a value that is decoded and
re-encoded must be the same value, and `-0.0`, `NaN` and the denormal range are all things a decimal or a
JSON encoding would quietly change (`D18`).

**Refusals happen on the declared numbers.** `maximumDimension` and `maximumTerms` are each `1 << 20`, and
the transport refuses a frame larger than `1 << 26` bytes, so a corrupt or hostile length cannot ask this
node to allocate what it says. A term whose declared width disagrees with the frame's is refused before it
reaches the reduction (`D30`).

**A frame validates itself; the geometry is checked against the node that received it.** Those are different
checks, and conflating them is how a crash arrives from the network: `OrderedReduction` *preconditions* on
the width, so a peer that declares a geometry this node does not have must be refused by the caller rather
than by a precondition. `decode` therefore returns a `DecodedFrame` — what the frame *said* as well as what
it carried — and the exchange compares it with the geometry it asked for, failing with
`ContributionWireError.geometryMismatch` (`DC-083`).

## The reduction, and why a retry cannot move it

All-reduce here is not a sum of per-node partials. Each frame carries **per-`(token, expert)` contributions**,
and they are summed in `(token, expert)` ascending order — `D17`. That ordering is the whole reason a retry is
safe: the same terms arriving twice, or arriving in a different sequence, are summed in the same order, so the
result is bit-identical rather than merely close. `D17` records the measured counterexample that makes this
non-negotiable — `(a+b)+c = 20000008` against `a+(b+c) = 20000006` for the same three numbers.

What has to be pinned is therefore the **ownership map** — which node holds which experts — and `D20` pins it
as data, with a digest, validated against the model actually opened.

## Failure semantics

Three ways a cluster run ends badly, and they must not look alike (`D19`). The rules are implemented in
`ShardExchange` and held by `ShardExchangeTests`, whose transport counts attempts, so "was this retried?" is an
exact assertion rather than a stopwatch reading.

| Failure | Rule | Held by |
| --- | --- | --- |
| a peer says nothing | **retry**, up to `ExchangePolicy.attempts` | `testASilentPeerIsRetriedAndThenFailsTheRun` |
| a peer stops mid-frame | **fail, no retry** — the stream is desynchronised | `testAPeerThatStopsMidFrameIsNotRetried` |
| a peer is gone | **fail** — a closed connection, not a deadline to wait out | `testAClosedPeerFailsTheRun` |
| a term never arrives | **fail the run** — never sum what is there | `testAMissingTermFailsTheRunInsteadOfShrinkingTheSum` |
| a resend arrives | **merge**, because the bits are identical | `testAResendMergesBecauseTheBitsAreIdentical` |
| a duplicate with different bits | **refuse** — two nodes disagree about a term | `testADuplicateWithDifferentBitsIsRefused` |

`ExchangePolicy` is two numbers: `receiveTimeoutMilliseconds` (default 30 000) and `attempts` (default 3;
one means no retry). A **clean** timeout is retryable; anything else is not, because only a clean timeout
leaves the stream at a frame boundary.

**The deadline is a value that travels.** It is passed to the transport through `applyTimeout`, and a
decorator that wraps a transport must forward it: the first version of this code had an `ExchangePolicy`
timeout that never reached the transport, so a test configured 60 ms, waited 30 s per attempt, and **passed** —
three attempts, ninety seconds, green, because it asserted behaviour and not duration (`D19`).

**The last four rows are asserted over a real socket too** (`TCPTransportTests`), because a peer that is a
Swift object can be told to misbehave and a descriptor cannot: that is how the mid-frame path was found to
**hang** rather than fail — `FileHandle.read(upToCount:)` blocks until it has every byte, so after `poll`
reported three the read waited for sixty-one. The transport now reads with `read(2)`, which returns what is
there, and the mid-frame test completes in its 60 ms deadline (`D42`).

## What the exchange reports

`ExchangeLedger` counts what actually happened — terms and bytes sent and received, and reduces performed —
**including every retry**, because a transport that retried is a fact about the run and an average that hides it
is not. Each node writes its ledger into its `metrics.json`, which is what makes the per-node table in a cluster
run possible (`D37`), and what lets a failure test assert the *count* of attempts rather than a duration.
