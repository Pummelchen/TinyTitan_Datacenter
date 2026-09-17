import XCTest
@testable import DatacenterEngine

/// The profiler is the instrument the throughput work turns on, so what it reports has to be pinned: the marks
/// land in the phases they name, the fields carry every phase, and **an absent profile contributes nothing**.
///
/// That last one is the rule `ForwardResult.profile` is optional for. Zero is a measurement — it says a phase
/// took no measurable time. Nil says the instrument was not run, and writing zeroes for it would turn "we did
/// not look" into "there was nothing to see".
final class ProfileTests: XCTestCase {
    func testMarksAccumulateIntoThePhaseTheyName() {
        let profiler = Profiler()
        profiler.mark("attn.core")
        profiler.mark("ff")
        profiler.mark("attn.core")

        let report = profiler.report(layers: 40)
        XCTAssertEqual(Set(report.seconds.keys), ["attn.core", "ff"])
        XCTAssertEqual(report.layers, 40)
        // The clock is monotonic, so a phase that was marked twice accumulated a positive time.
        XCTAssertGreaterThan(report.seconds["attn.core"] ?? 0, 0)
    }

    func testTheFieldsCarryEveryPhaseAndTheLayerCount() {
        let report = ProfileReport(seconds: ["attn.core": 1.5, "ff": 2.25], layers: 40)
        let fields = ProfileMetrics.fields(report)

        XCTAssertEqual(fields["profile_layers"] as? Int, 40)
        let seconds = fields["profile_seconds"] as? [String: Double]
        XCTAssertEqual(seconds, ["attn.core": 1.5, "ff": 2.25])
    }

    func testAnAbsentProfileContributesNoFieldsRatherThanZeroes() {
        XCTAssertTrue(ProfileMetrics.fields(nil).isEmpty)
        // And a forward that was not asked to profile does not carry one.
        XCTAssertNil(ForwardResult(tensors: []).profile)
    }

    func testCombiningAddsPhasesAndTakesTheLayerCount() {
        let combined = ProfileReport.combined([
            ProfileReport(seconds: ["mix.read": 1.0, "head": 0.5], layers: 2),
            ProfileReport(seconds: ["mix.read": 0.25, "attn.core": 2.0], layers: 2),
        ])
        XCTAssertEqual(combined?.seconds["mix.read"], 1.25)
        XCTAssertEqual(combined?.seconds["head"], 0.5)
        XCTAssertEqual(combined?.seconds["attn.core"], 2.0)
        XCTAssertEqual(combined?.layers, 2)
    }

    func testCombiningNothingIsNilRatherThanZeroes() {
        // A generation of zero steps profiled nothing; it did not measure every phase at 0.000 s.
        XCTAssertNil(ProfileReport.combined([]))
    }

    func testAReportWithNoMarksIsStillAReport() {
        // A forward that marks nothing is not the same as one that was not profiled: the first was measured
        // and found nothing, the second was not measured. Both are representable, and they differ.
        let empty = Profiler().report(layers: 0)
        XCTAssertEqual(ProfileMetrics.fields(empty)["profile_layers"] as? Int, 0)
    }
}
