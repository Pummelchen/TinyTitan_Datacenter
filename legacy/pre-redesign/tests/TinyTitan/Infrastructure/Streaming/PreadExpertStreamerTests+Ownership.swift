import Testing

@testable import TinyTitan

/// `D164`: ownership narrows the routed set **before** the cache plan is made.
///
/// The distinction is not cosmetic. Marking a peer's expert as a *miss* would still give it a
/// slot, because the plan is sized from the routed set — so a distributed node would evict an
/// expert it actually holds in order to cache one it does not, and the exchange would then
/// deliver the answer it had just thrown away. The filter has to happen at the top, and these
/// tests pin that it does, that it preserves order, and that having no filter at all is
/// bit-for-bit the single-node path.
extension PreadExpertStreamerTests {
  @Test func noOwnershipFilterLeavesTheRoutedSetUntouched() {
    let routed = [3, 1, 2]
    #expect(PreadExpertStreamer.ownedExperts(routed: routed, owns: nil) == routed)
  }

  @Test func ownershipNarrowsTheRoutedSetBeforePlanning() {
    let routed = [0, 1, 2, 3, 4, 5]
    // A node owning the even experts plans only those; the odd ones arrive over the exchange.
    let owned = PreadExpertStreamer.ownedExperts(routed: routed) { $0.isMultiple(of: 2) }
    #expect(owned == [0, 2, 4])
  }

  @Test func ownershipKeepsTheRoutedOrderBecauseSlotsArePositional() {
    let routed = [5, 0, 3, 2]
    // Both 5 and 0 are excluded, so the survivors stay in the routed order and are *not*
    // ascending: [2, 3] would be equal as a set and would move which expert lands in which slot.
    let owned = PreadExpertStreamer.ownedExperts(routed: routed) { $0 > 0 && $0 < 4 }
    #expect(owned == [3, 2])
  }

  @Test func aNodeOwningNothingPlansNothingAndReadsNothing() {
    let routed = [0, 1, 2]
    #expect(PreadExpertStreamer.ownedExperts(routed: routed) { _ in false }.isEmpty)
  }
}
