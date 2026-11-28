import Foundation
import Testing

@testable import TinyTitan

/// A chat template reaches the tokenizer folder by one of two routes: the sidecar
/// `chat_template.jinja`, or the `chat_template` field inside `tokenizer_config.json`.
/// They are alternatives, not requirements.
///
/// The server treated the sidecar as mandatory and threw `missingToolTemplate` without it, so it
/// refused to load every install this project produces - whose `tokenizer_config.json` carries the
/// template - and reported that as a model-install fault ("reinstall the model") that was not one.
/// The prompt cache also hashed the sidecar file into its `runtimeIdentity`, so the config route
/// had no identity at all. These cover both sources, the precedence between them, and the absent
/// case that must still be refused.
@Suite("Chat template resolution")
struct ChatTemplateResolutionTests {

    /// A throwaway tokenizer folder holding exactly the files a case asks for.
    ///
    /// A plain temporary directory per case, because the point is which files are present: the
    /// fixtures under `Fixtures/` all ship a sidecar and so cannot express the config-only install
    /// this guard exists for.
    private func makeFolder(sidecar: String? = nil,
                            config: [String: Any]? = nil) throws -> URL {
        let folder = FileManager.default.temporaryDirectory
            .appendingPathComponent("ttd-chat-template-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        if let sidecar {
            try sidecar.write(to: folder.appendingPathComponent("chat_template.jinja"),
                              atomically: true, encoding: .utf8)
        }
        if let config {
            let data = try JSONSerialization.data(withJSONObject: config)
            try data.write(to: folder.appendingPathComponent("tokenizer_config.json"))
        }
        return folder
    }

    private func templateText(_ folder: URL) throws -> String {
        String(decoding: try GFTokenizer.chatTemplateData(in: folder), as: UTF8.self)
    }

    // MARK: - Both sources are valid

    @Test("A sidecar chat_template.jinja is a template")
    func sidecarIsATemplate() throws {
        let folder = try makeFolder(sidecar: "{{ sidecar }}")
        #expect(GFTokenizer.hasChatTemplate(in: folder))
        #expect(try templateText(folder) == "{{ sidecar }}")
    }

    @Test("A chat_template field in tokenizer_config.json is a template")
    func configFieldIsATemplate() throws {
        let folder = try makeFolder(config: ["tokenizer_class": "Qwen2Tokenizer",
                                            "chat_template": "{{ from-config }}"])
        #expect(GFTokenizer.hasChatTemplate(in: folder),
                "an install carrying its template in the config must be loadable")
        #expect(try templateText(folder) == "{{ from-config }}")
    }

    @Test("The sidecar takes precedence when both are present")
    func sidecarWins() throws {
        let folder = try makeFolder(sidecar: "{{ sidecar }}",
                                    config: ["chat_template": "{{ from-config }}"])
        #expect(try templateText(folder) == "{{ sidecar }}")
    }

    // MARK: - What must still be refused

    @Test("A folder with neither source has no template")
    func neitherSourceHasNoTemplate() throws {
        let folder = try makeFolder(config: ["tokenizer_class": "Qwen2Tokenizer"])
        #expect(!GFTokenizer.hasChatTemplate(in: folder))
        #expect(throws: GFTokenizerError.self) {
            _ = try GFTokenizer.chatTemplateData(in: folder)
        }
    }

    @Test("An empty or blank config template is not a template")
    func blankConfigTemplateIsNotATemplate() throws {
        for blank in ["", "   \n\t "] {
            let folder = try makeFolder(config: ["chat_template": blank])
            #expect(!GFTokenizer.hasChatTemplate(in: folder),
                    "a blank template cannot render a prompt")
        }
    }

    @Test("A config file that is not JSON has no template")
    func malformedConfigHasNoTemplate() throws {
        let folder = FileManager.default.temporaryDirectory
            .appendingPathComponent("ttd-chat-template-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        try Data("not json".utf8).write(to: folder.appendingPathComponent("tokenizer_config.json"))
        #expect(!GFTokenizer.hasChatTemplate(in: folder))
    }
}
