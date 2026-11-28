import AppKit
import TinyTitanAppCore
import SwiftUI

// Run as a regular foreground app even when launched as a bare SwiftPM
// executable (no .app bundle): Dock icon, click-to-activate, full main menu
// with Quit (Cmd+Q).
private final class ForegroundAppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        if let iconURL = Bundle.module.url(
            forResource: "tinytitan-app-icon",
            withExtension: "png"
        ), let icon = NSImage(contentsOf: iconURL) {
            NSApp.applicationIconImage = icon
            NSApp.dockTile.display()
        }
        NSApp.activate()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }
}

@main
struct TinyTitanMacApp: App {
    @NSApplicationDelegateAdaptor private var appDelegate: ForegroundAppDelegate
    @State private var model: AppModel

    init() {
        let client: any AppInferenceClient
        var launchError: AppInferenceError?
        do {
            client = try DecodeServiceInferenceClient()
        } catch {
            // D4: never crash at launch. Fall back to a client that fails
            // every operation with the recorded error and surface it in the
            // UI so the user can recover.
            let error = AppInferenceError.unknown(
                "Cannot create decode service client: \(error)")
            launchError = error
            client = UnavailableInferenceClient(failure: error)
        }
        let model = AppModel(client: client, settingsPersistenceEnabled: true)
        if let launchError {
            model.error = launchError
        }
        _model = State(initialValue: model)
    }

    var body: some Scene {
        Window("TinyTitan", id: "main") {
            RootView(model: model)
                .frame(minWidth: 1040, minHeight: 560)
        }
        .windowStyle(.hiddenTitleBar)
        .defaultSize(width: 1040, height: 720)
        .windowResizability(.contentMinSize)
        .commands {
            CommandMenu("Generation") {
                Button("Cancel Generation") { model.cancel() }
                    .keyboardShortcut(".", modifiers: .command)
                    .disabled(!model.canCancel)
                Button("Cancel Model Installation") { model.cancelInstall() }
                    .disabled(!model.canCancelInstall)
            }
            CommandMenu("Model") {
                Button("Load Model", action: model.loadModel)
                    .disabled(!model.canLoadModel)
                Button("Reload Model", action: model.reloadModel)
                    .disabled(!model.canReloadModel)
                Button("Unload Model", action: model.unloadModel)
                    .disabled(!model.canUnloadModel)
                Divider()
                Menu("Change Model") {
                    // One row per build, checked against the one in use. The app
                    // reads this preference at launch and had no way to write
                    // it, so the model could not be changed from the UI at all.
                    ForEach(AppModelInstallDescriptor.selectable) { row in
                        Toggle(row.displayName, isOn: Binding(
                            get: { model.installDescriptor == row.descriptor },
                            set: { _ in
                                model.selectModel(row.descriptor,
                                                  selector: row.selector)
                            }))
                    }
                }
                .disabled(!model.canChangeModel)
            }
            CommandMenu("Settings") {
                Picker("Send Message With", selection: newlineShortcutBinding) {
                    ForEach(AppNewlineShortcut.sendMessageOptions) { shortcut in
                        Text(shortcut.sendMessageLabel).tag(shortcut)
                    }
                }
                Picker("Prompt Examples", selection: showPromptExamplesBinding) {
                    Text("Show").tag(true)
                    Text("Hide").tag(false)
                }
            }
        }
    }

    private var newlineShortcutBinding: Binding<AppNewlineShortcut> {
        Binding {
            model.newlineShortcut
        } set: { shortcut in
            model.setNewlineShortcut(shortcut)
        }
    }

    private var showPromptExamplesBinding: Binding<Bool> {
        Binding {
            model.showPromptExamples
        } set: { show in
            model.setShowPromptExamples(show)
        }
    }
}
