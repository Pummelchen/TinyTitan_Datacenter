import Foundation
import Testing

@testable import TinyTitanAppCore

/// Changing the model from inside the app.
///
/// The app has always *read* a model selector (`TURBO_FIELDFARE_MODEL`, else the
/// persisted `defaults write TinyTitan model <selector>` preference, else Ornith
/// 1.5 8-bit) and nothing ever wrote one, so a user with eleven installs could
/// not point the app at any of them: every launch showed "Model required" for
/// the one build it had picked, with no way to choose another. These tests pin
/// the table that makes the relation run both ways, and the write that records a
/// choice.
@Suite struct AppModelSelectionTests {

  /// A build the menu could not name is a build the user cannot switch to, so
  /// every recognizable checkpoint has to have a row — and no row may name
  /// something the app does not recognize.
  @Test func everyRecognizableBuildIsSelectable() {
    let selectable = AppModelInstallDescriptor.selectable.map(\.descriptor)
    for descriptor in AppModelInstallDescriptor.all {
      #expect(selectable.contains(descriptor),
              Comment(rawValue: descriptor.installDirectoryName))
    }
    #expect(AppModelInstallDescriptor.selectable.count
            == AppModelInstallDescriptor.all.count)
  }

  /// The two directions of one relation: a row's selector must resolve to that
  /// row's build, or the menu would check the wrong box after a switch.
  @Test func everyRowResolvesToItsOwnDescriptor() {
    for row in AppModelInstallDescriptor.selectable {
      #expect(AppModelInstallDescriptor.selectedDescriptor(for: row.selector)
              == row.descriptor,
              Comment(rawValue: row.selector))
      #expect(AppModelInstallDescriptor.selector(for: row.descriptor)
              == row.selector,
              Comment(rawValue: row.selector))
    }
  }

  /// A selector a person can actually copy out of `models/` also works. The
  /// directory names are what is in front of them; silently selecting Ornith
  /// for one of those is the "changing the model does nothing" symptom again.
  @Test func aModelDirectoryNameIsAcceptedAsASelector() {
    #expect(AppModelInstallDescriptor.selectedDescriptor(
      for: "qwen3.8-flash-next_125B_A6B_4Bit") == .qwen38)
    #expect(AppModelInstallDescriptor.selectedDescriptor(
      for: "qwen-agentworld_35B_A3B_4Bit") == .agentworld)
    #expect(AppModelInstallDescriptor.selectedDescriptor(
      for: "kat-coder-v2.5_35B_A3B_8Bit") == .katcoder8bit)
  }

  /// The dense Qwen 3.5 installs (2B/4B/9B, either engine) are a gap, not a
  /// selector spelling: the app carries no descriptor for that family, so it
  /// cannot name, load or list them — a `models/` directory name for one falls
  /// through to the default. Recorded here so the limitation is a known one
  /// rather than a surprise for whoever exports those builds.
  @Test func aDenseInstallIsNotSelectable() {
    for directory in ["qwen3.5_2B_4Bit", "qwen3.5_4B_8Bit", "qwen3.5_9B_4Bit"] {
      #expect(AppModelInstallDescriptor.selectedDescriptor(for: directory)
              .installDirectoryName == "ornith-1.5_35B_A3B_8Bit",
              Comment(rawValue: directory))
      #expect(AppModelInstallDescriptor.selector(
        for: AppModelInstallDescriptor.selectedDescriptor(for: directory)) == "ornith15-8bit")
    }
  }

  /// The aliases people and the docs type keep working, and a selector nobody
  /// recognizes still lands on the default rather than failing — pinned here so
  /// that behaviour is a decision rather than an accident.
  @Test func aliasesResolveAndAnUnknownSelectorFallsBack() {
    #expect(AppModelInstallDescriptor.selectedDescriptor(for: "ornith")
            == .ornith15Converted)
    #expect(AppModelInstallDescriptor.selectedDescriptor(for: "qwen3.8")
            == .qwen38)
    #expect(AppModelInstallDescriptor.selectedDescriptor(for: "kat")
            == .katcoder)
    #expect(AppModelInstallDescriptor.selectedDescriptor(for: "Qwen3.8")
            .installDirectoryName == "ornith-1.5_35B_A3B_8Bit")
    #expect(AppModelInstallDescriptor.selectedDescriptor(for: nil)
            .installDirectoryName == "ornith-1.5_35B_A3B_8Bit")
  }

  /// One key, one suite, both directions. The domain is injected so a test does
  /// not write into the operator's real preferences.
  @Test func thePersistedSelectorRoundTrips() throws {
    let suite = "TinyTitanTests-\(UUID().uuidString)"
    let defaults = try #require(UserDefaults(suiteName: suite))
    defer { defaults.removePersistentDomain(forName: suite) }

    #expect(AppModelSelection.persistedSelector(in: defaults) == nil)
    AppModelSelection.setPersistedSelector("agentworld", in: defaults)
    #expect(AppModelSelection.persistedSelector(in: defaults) == "agentworld")
    #expect(AppModelInstallDescriptor.selectedDescriptor(
      for: AppModelSelection.persistedSelector(in: defaults))
      == .agentworld)
  }

  @MainActor
  @Test func choosingAModelPersistsItAndSaysToReopen() throws {
    let suite = "TinyTitanTests-\(UUID().uuidString)"
    let defaults = try #require(UserDefaults(suiteName: suite))
    defer { defaults.removePersistentDomain(forName: suite) }
    let model = AppModel(
      modelDirectory: temporarySelectionPath(),
      client: MockLifecycleInferenceClient(),
      installer: MockModelInstallerClient())

    #expect(model.canChangeModel)
    model.selectModel(.agentworld, selector: "agentworld", defaults: defaults)

    #expect(AppModelSelection.persistedSelector(in: defaults) == "agentworld")
    // The switch cannot be applied under a running window, so the app says what
    // will happen rather than appearing to switch.
    let notice = try #require(model.modelSwitchNotice)
    #expect(notice.contains("Qwen-AgentWorld 35B-A3B 4-bit"))
    #expect(notice.contains("Quit and reopen"))
    model.dismissModelSwitchNotice()
    #expect(model.modelSwitchNotice == nil)
  }

  @MainActor
  @Test func aRunningGenerationRefusesAModelChange() throws {
    let suite = "TinyTitanTests-\(UUID().uuidString)"
    let defaults = try #require(UserDefaults(suiteName: suite))
    defer { defaults.removePersistentDomain(forName: suite) }
    let model = AppModel(
      modelDirectory: temporarySelectionPath(),
      client: MockLifecycleInferenceClient(),
      installer: MockModelInstallerClient())
    model.runState = .running

    #expect(!model.canChangeModel)
    model.selectModel(.agentworld, selector: "agentworld", defaults: defaults)
    #expect(AppModelSelection.persistedSelector(in: defaults) == nil)
    #expect(model.modelSwitchNotice == nil)
  }
}

private func temporarySelectionPath() -> URL {
  FileManager.default.temporaryDirectory
    .appendingPathComponent("app-model-selection-\(UUID().uuidString)",
                            isDirectory: true)
}
