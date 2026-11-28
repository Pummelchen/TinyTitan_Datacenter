import TinyTitanAppCore
import TinyTitanMacPresentation
import SwiftUI

/// Says which model the next launch will use, after a pick in the Model menu.
///
/// Switching cannot be applied under a running window: the model directory is
/// resolved when the app starts, and the descriptor, the per-model settings
/// file and the decode-service process are all bound to it. So the pick is
/// persisted and this says so, rather than appearing to switch while the runner
/// still holds the previous model.
struct ModelSwitchBanner: View {
    @Bindable var model: AppModel

    var body: some View {
        if let notice = model.modelSwitchNotice {
            HStack(spacing: 10) {
                Image(systemName: "arrow.triangle.2.circlepath")
                    .foregroundStyle(TinyTitanMacTheme.accentColor)
                    .accessibilityHidden(true)
                Text(notice)
                    .font(.callout)
                    .lineLimit(2)
                Spacer(minLength: 8)
                Button("Dismiss", action: model.dismissModelSwitchNotice)
                    .buttonStyle(.bordered)
                    .controlSize(.small)
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 8)
            .background {
                RoundedRectangle(cornerRadius: 14)
                    .fill(Color(nsColor: .controlBackgroundColor))
                    .overlay {
                        RoundedRectangle(cornerRadius: 14)
                            .stroke(TinyTitanMacTheme.accentColor.opacity(0.5),
                                    lineWidth: 1)
                    }
            }
            .accessibilityElement(children: .contain)
            .transition(.move(edge: .bottom).combined(with: .opacity))
        }
    }
}
