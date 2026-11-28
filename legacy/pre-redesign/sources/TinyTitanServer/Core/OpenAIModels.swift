import Foundation
import TinyTitan

public struct OpenAIErrorEnvelope: Codable, Equatable, Sendable {
    public struct Detail: Codable, Equatable, Sendable {
        public let message: String
        public let type: String
        public let param: String?
        public let code: String
    }

    public let error: Detail

    public init(message: String, param: String? = nil, code: String, type: String = "invalid_request_error") {
        error = Detail(message: message,
                       type: type,
                       param: param,
                       code: code)
    }
}

public struct OpenAITextPart: Codable, Equatable, Sendable {
    public let type: String
    public let text: String?
}

public enum OpenAIMessageContent: Codable, Equatable, Sendable {
    case text(String)
    case parts([OpenAITextPart])

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let text = try? container.decode(String.self) {
            self = .text(text)
        } else {
            self = .parts(try container.decode([OpenAITextPart].self))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .text(let text): try container.encode(text)
        case .parts(let parts): try container.encode(parts)
        }
    }

    func textValue() throws -> String {
        switch self {
        case .text(let text):
            return text
        case .parts(let parts):
            guard parts.allSatisfy({ $0.type == "text" && $0.text != nil }) else {
                throw ServerRequestError.invalid(
                    message: "only text content parts are supported",
                    param: "messages",
                    code: "unsupported_content")
            }
            return parts.compactMap(\.text).joined()
        }
    }
}

public struct OpenAIFunctionCall: Codable, Equatable, Sendable {
    public let name: String
    public let arguments: String
}

public struct OpenAIToolCall: Codable, Equatable, Sendable {
    public let id: String
    public let type: String
    public let function: OpenAIFunctionCall
}

public struct OpenAIChatMessage: Codable, Equatable, Sendable {
    public let role: String
    public let content: OpenAIMessageContent?
    public let toolCalls: [OpenAIToolCall]?
    public let toolCallID: String?
    public let name: String?

    enum CodingKeys: String, CodingKey {
        case role, content, name
        case toolCalls = "tool_calls"
        case toolCallID = "tool_call_id"
    }
}

public struct OpenAIFunctionDefinition: Codable, Equatable, Sendable {
    public let name: String
    public let description: String?
    public let parameters: JSONValue
}

public struct OpenAITool: Codable, Equatable, Sendable {
    public let type: String
    public let function: OpenAIFunctionDefinition
}

public enum OpenAIStop: Codable, Equatable, Sendable {
    case one(String)
    case many([String])

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let one = try? container.decode(String.self) {
            self = .one(one)
        } else {
            self = .many(try container.decode([String].self))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .one(let value): try container.encode(value)
        case .many(let value): try container.encode(value)
        }
    }

    var values: [String] {
        switch self {
        case .one(let value): [value]
        case .many(let value): value
        }
    }
}

public struct OpenAIStreamOptions: Codable, Equatable, Sendable {
    public let includeUsage: Bool?

    enum CodingKeys: String, CodingKey {
        case includeUsage = "include_usage"
    }
}

/// The thinking controls as Qwen's chat templates take them.
///
/// llama.cpp, vLLM and TabbyAPI clients send these inside a
/// `chat_template_kwargs` object rather than as the top-level
/// `reasoning_effort`, and `enable_thinking: false` there is the only way those
/// clients turn thinking off for one request. A server that read only the
/// top-level field honoured the *level* beside this object and dropped the
/// switch, which is the case that matters: a client that forces thinking off
/// for its summarization calls (so the model's own thinking cannot eat the
/// output cap and truncate the summary) had the fix silently lost.
public struct OpenAIChatTemplateKwargs: Codable, Equatable, Sendable {
    public let enableThinking: Bool?
    public let reasoningEffort: String?

    enum CodingKeys: String, CodingKey {
        case enableThinking = "enable_thinking"
        case reasoningEffort = "reasoning_effort"
    }

    public init(enableThinking: Bool? = nil, reasoningEffort: String? = nil) {
        self.enableThinking = enableThinking
        self.reasoningEffort = reasoningEffort
    }
}

public struct OpenAIChatRequest: Codable, Equatable, Sendable {
    public let model: String
    public let messages: [OpenAIChatMessage]
    public let stream: Bool?
    public let streamOptions: OpenAIStreamOptions?
    public let temperature: Float?
    public let topP: Float?
    public let maxTokens: Int?
    public let maxCompletionTokens: Int?
    public let stop: OpenAIStop?
    public let seed: UInt64?
    public let tools: [OpenAITool]?
    public let toolChoice: JSONValue?
    public let parallelToolCalls: Bool?
    public let topK: Int?
    public let repetitionPenalty: Float?
    public let n: Int?
    public let logprobs: Bool?
    public let presencePenalty: Float?
    public let frequencyPenalty: Float?
    /// Requested reasoning-effort level. Validated against the served model
    /// family's chat template and the server's load-time profile.
    public let reasoningEffort: String?
    /// The same controls in the template-kwargs dialect (see above).
    public let chatTemplateKwargs: OpenAIChatTemplateKwargs?
    /// llama.cpp's hard per-request thinking-token budget.
    ///
    /// Decoded, never refused, and not enforced: this runtime bounds thinking by
    /// the level a template renders rather than by a token count, and a client
    /// that sends this also sends the level it wants. Refusing a field the
    /// runtime does not implement would break that client for the rest of the
    /// session, which is the failure the effort mapping already exists to avoid.
    public let reasoningBudgetTokens: Int?
    /// The requested output format. Decoded so a structured-output request can
    /// be *refused* rather than silently answered as prose: a client that asks
    /// for JSON and gets unconstrained text is worse off than one told no.
    /// `{"type": "text"}`, the API's own default, is accepted.
    public let responseFormat: JSONValue?

    /// Explicit, with the two thinking-control extras defaulted, so the protocol
    /// mappers that build a chat request from their own shapes keep compiling
    /// unchanged. A `let` with an inline default would have been skipped by the
    /// synthesised decoder, which is how the budget silently decoded to nil.
    public init(model: String,
                messages: [OpenAIChatMessage],
                stream: Bool? = nil,
                streamOptions: OpenAIStreamOptions? = nil,
                temperature: Float? = nil,
                topP: Float? = nil,
                maxTokens: Int? = nil,
                maxCompletionTokens: Int? = nil,
                stop: OpenAIStop? = nil,
                seed: UInt64? = nil,
                tools: [OpenAITool]? = nil,
                toolChoice: JSONValue? = nil,
                parallelToolCalls: Bool? = nil,
                topK: Int? = nil,
                repetitionPenalty: Float? = nil,
                n: Int? = nil,
                logprobs: Bool? = nil,
                presencePenalty: Float? = nil,
                frequencyPenalty: Float? = nil,
                reasoningEffort: String? = nil,
                chatTemplateKwargs: OpenAIChatTemplateKwargs? = nil,
                reasoningBudgetTokens: Int? = nil,
                responseFormat: JSONValue? = nil) {
        self.model = model
        self.messages = messages
        self.stream = stream
        self.streamOptions = streamOptions
        self.temperature = temperature
        self.topP = topP
        self.maxTokens = maxTokens
        self.maxCompletionTokens = maxCompletionTokens
        self.stop = stop
        self.seed = seed
        self.tools = tools
        self.toolChoice = toolChoice
        self.parallelToolCalls = parallelToolCalls
        self.topK = topK
        self.repetitionPenalty = repetitionPenalty
        self.n = n
        self.logprobs = logprobs
        self.presencePenalty = presencePenalty
        self.frequencyPenalty = frequencyPenalty
        self.reasoningEffort = reasoningEffort
        self.chatTemplateKwargs = chatTemplateKwargs
        self.reasoningBudgetTokens = reasoningBudgetTokens
        self.responseFormat = responseFormat
    }

    enum CodingKeys: String, CodingKey {
        case model, messages, stream, temperature, stop, seed, tools, n, logprobs
        case streamOptions = "stream_options"
        case topP = "top_p"
        case maxTokens = "max_tokens"
        case maxCompletionTokens = "max_completion_tokens"
        case toolChoice = "tool_choice"
        case parallelToolCalls = "parallel_tool_calls"
        case topK = "top_k"
        case repetitionPenalty = "repetition_penalty"
        case presencePenalty = "presence_penalty"
        case frequencyPenalty = "frequency_penalty"
        case reasoningEffort = "reasoning_effort"
        case chatTemplateKwargs = "chat_template_kwargs"
        case reasoningBudgetTokens = "reasoning_budget_tokens"
        case responseFormat = "response_format"
    }
}

/// The load-time reasoning configuration the HTTP layer validates requests
/// against: the served family's template capability plus the flags the model
/// was loaded with. Effort is a load-time control because it changes the
/// rendered prompt, so a request may only confirm the active level, never
/// switch it.
public struct ServerReasoningProfile: Sendable, Equatable {
    public let family: ModelFamily
    public let thinkingMode: ModelThinkingMode
    public let reasoningEffort: ModelReasoningEffort?

    public init(family: ModelFamily,
                thinkingMode: ModelThinkingMode,
                reasoningEffort: ModelReasoningEffort?) {
        self.family = family
        self.thinkingMode = thinkingMode
        self.reasoningEffort = reasoningEffort
    }

    /// The compatible Qwen3.5-MoE baseline: binary thinking, off.
    public static let `default` = ServerReasoningProfile(
        family: .qwen36, thinkingMode: .off, reasoningEffort: nil)

    /// The effort the template actually applies under this profile; nil for
    /// binary families and while thinking is off.
    public var effectiveEffort: ModelReasoningEffort? {
        family.effectiveReasoningEffort(thinkingMode: thinkingMode,
                                        effort: reasoningEffort)
    }
}

public struct OpenAIUsage: Codable, Equatable, Sendable {
    public struct PromptTokensDetails: Codable, Equatable, Sendable {
        public let cachedTokens: Int

        enum CodingKeys: String, CodingKey {
            case cachedTokens = "cached_tokens"
        }

        public init(cachedTokens: Int) {
            self.cachedTokens = cachedTokens
        }
    }

    /// How many of `completion_tokens` the model spent thinking.
    ///
    /// The clients that route reasoning separately (llama.cpp's patched servers,
    /// the coding harnesses built on them) read this to bill and to budget; the
    /// runtime already knows the split, because the decoder puts every token in
    /// one channel or the other. Kept as its own object rather than folded into
    /// `completion_tokens` so a client can see both.
    public struct CompletionTokensDetails: Codable, Equatable, Sendable {
        public let reasoningTokens: Int

        enum CodingKeys: String, CodingKey {
            case reasoningTokens = "reasoning_tokens"
        }

        public init(reasoningTokens: Int) {
            self.reasoningTokens = reasoningTokens
        }
    }

    public let promptTokens: Int
    public let completionTokens: Int
    public let totalTokens: Int
    public let promptTokensDetails: PromptTokensDetails
    public let completionTokensDetails: CompletionTokensDetails

    enum CodingKeys: String, CodingKey {
        case promptTokens = "prompt_tokens"
        case completionTokens = "completion_tokens"
        case totalTokens = "total_tokens"
        case promptTokensDetails = "prompt_tokens_details"
        case completionTokensDetails = "completion_tokens_details"
    }

    public init(promptTokens: Int,
                completionTokens: Int,
                totalTokens: Int,
                cachedTokens: Int = 0,
                reasoningTokens: Int = 0) {
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
        self.totalTokens = totalTokens
        self.promptTokensDetails = PromptTokensDetails(cachedTokens: cachedTokens)
        self.completionTokensDetails = CompletionTokensDetails(
            reasoningTokens: reasoningTokens)
    }
}

public struct OpenAIModelList: Codable, Equatable, Sendable {
    public struct Model: Codable, Equatable, Sendable {
        public let id: String
        public let object: String
        /// Model creation time. Omitted when unknown rather than lying with a
        /// fabricated epoch (S30).
        public let created: Int?
        public let ownedBy: String

        enum CodingKeys: String, CodingKey {
            case id, object, created
            case ownedBy = "owned_by"
        }

        public init(id: String, object: String, created: Int?, ownedBy: String) {
            self.id = id
            self.object = object
            self.created = created
            self.ownedBy = ownedBy
        }
    }

    public let object: String
    public let data: [Model]
}

public enum ServerRequestError: Error, Equatable, Sendable {
    case invalid(message: String, param: String?, code: String)
    case unknownModel
    case queueFull
    /// A well-formed request for something this backend cannot do at all
    /// (a token count without a tokenizer, for instance): 501, not 400.
    case unsupportedOperation(String)
    /// A stored response named by `previous_response_id` or a path that
    /// does not exist: 404.
    case notFound(message: String, param: String?)

    public var envelope: OpenAIErrorEnvelope {
        switch self {
        case .invalid(let message, let param, let code):
            OpenAIErrorEnvelope(message: message, param: param, code: code)
        case .unknownModel:
            OpenAIErrorEnvelope(message: "requested model is not available",
                                param: "model", code: "model_not_found")
        case .queueFull:
            OpenAIErrorEnvelope(message: "generation queue is full",
                                code: "queue_full",
                                type: "rate_limit_error")
        case .unsupportedOperation(let operation):
            OpenAIErrorEnvelope(message: "\(operation) is not supported by this backend",
                                code: "unsupported_operation",
                                type: "server_error")
        case .notFound(let message, let param):
            OpenAIErrorEnvelope(message: message, param: param, code: "not_found")
        }
    }

    /// The HTTP status each error maps to, shared by every API surface.
    public var httpStatus: Int {
        switch self {
        case .invalid: 400
        case .unknownModel, .notFound: 404
        case .queueFull: 429
        case .unsupportedOperation: 501
        }
    }
}

public struct ValidatedChatRequest: Sendable {
    public let messages: [GFTokenizer.Message]
    public let tools: [GFTokenizer.FunctionDefinition]
    public let stream: Bool
    public let includeUsage: Bool
    public let generationConfig: GenerationConfig
    public let maximumCompletionTokens: Int
    /// Set when the request named the "<model>-fast" alias: the CLI-strip
    /// heuristic runs for this request regardless of TINYTITAN_STRIP_CLI_PROMPT.
    public let stripCLIPrompt: Bool
    /// Memory workspace named by the X-TinyTitan-Workspace header, when the
    /// server allows a request to choose one. Nil means the workspace the
    /// server was launched with.
    public let workspace: String?
    /// True for a generation the engine asked for itself -- memory
    /// consolidation is the only one today. Watchdogs do not police these
    /// (B6): their prompts are repetitive by construction and their answers
    /// are meant to be terse, which is the shape the detectors hunt, and no
    /// person is waiting on the result.
    public let isEngineInternal: Bool
    /// The catalog id the request was validated against. The routing backend
    /// loads it; a single-model backend serves what it has and ignores it.
    /// Nil for the engine's own requests, which run on whatever is resident.
    public let model: String?
    /// Request fields that asked for something the served model cannot do and
    /// were answered by the nearest thing it can. Empty on the common path.
    ///
    /// These are never errors: a coding agent that names a reasoning level
    /// this project never defined keeps working, and the server says what it
    /// applied in its log. Additive and defaulted so every existing caller is
    /// unchanged.
    public let reasoningNotes: [String]
    /// The thinking mode and effort this request should actually render at.
    ///
    /// `nil` means "whatever the model was loaded with", which is the common
    /// path and keeps the session's own tokenizer. A value that differs from
    /// the loaded one is a mid-session switch: the session resolves a
    /// tokenizer for it instead of reusing its own, so a coding agent can
    /// turn thinking off (or change effort) inside a live session.
    ///
    /// Carried on the request rather than passed alongside it because the
    /// prompt cache keys on this value: two levels render different prompts,
    /// and a cached KV range from one must never be spliced onto the other.
    public let reasoning: RequestReasoning?
    /// The compiled JSON schema this request must produce, when it asked for
    /// structured output. Nil is free text -- the only value every caller but
    /// the three JSON spellings passes.
    ///
    /// The schema is compiled here, during validation, so an unsupported
    /// keyword is a 400 before a model is touched rather than a failure in the
    /// middle of a generation.
    public let jsonSchema: JSONSchemaNode?

    public init(messages: [GFTokenizer.Message],
                tools: [GFTokenizer.FunctionDefinition],
                stream: Bool,
                includeUsage: Bool,
                generationConfig: GenerationConfig,
                maximumCompletionTokens: Int,
                stripCLIPrompt: Bool = false,
                workspace: String? = nil,
                isEngineInternal: Bool = false,
                model: String? = nil,
                reasoningNotes: [String] = [],
                reasoning: RequestReasoning? = nil,
                jsonSchema: JSONSchemaNode? = nil) {
        self.messages = messages
        self.tools = tools
        self.stream = stream
        self.includeUsage = includeUsage
        self.generationConfig = generationConfig
        self.maximumCompletionTokens = maximumCompletionTokens
        self.stripCLIPrompt = stripCLIPrompt
        self.workspace = workspace
        self.isEngineInternal = isEngineInternal
        self.model = model
        self.reasoningNotes = reasoningNotes
        self.reasoning = reasoning
        self.jsonSchema = jsonSchema
    }

    /// Every derived request is built through here.
    ///
    /// The three public helpers below used to rebuild the struct field by
    /// field, which silently dropped any field added later -- `jsonSchema` was
    /// lost that way the moment it existed, so a request that asked for
    /// structured output validated, then generated free text. One builder that
    /// carries every unmentioned field makes that impossible to repeat.
    private func copy(messages: [GFTokenizer.Message]? = nil,
                      tools: [GFTokenizer.FunctionDefinition]? = nil,
                      stripCLIPrompt: Bool? = nil,
                      workspace: String?? = nil,
                      isEngineInternal: Bool? = nil,
                      model: String?? = nil) -> ValidatedChatRequest {
        ValidatedChatRequest(
            messages: messages ?? self.messages,
            tools: tools ?? self.tools,
            stream: stream,
            includeUsage: includeUsage,
            generationConfig: generationConfig,
            maximumCompletionTokens: maximumCompletionTokens,
            stripCLIPrompt: stripCLIPrompt ?? self.stripCLIPrompt,
            workspace: workspace ?? self.workspace,
            isEngineInternal: isEngineInternal ?? self.isEngineInternal,
            model: model ?? self.model,
            reasoningNotes: reasoningNotes,
            reasoning: reasoning,
            jsonSchema: jsonSchema)
    }

    /// The post-strip view of this request: the same request carrying the
    /// messages and tools that were actually encoded into the prompt.
    ///
    /// The prompt cache must key on this view, not the raw request. Its
    /// entries describe a KV range that was prefilled from the filtered
    /// messages, and its continuation paths re-render the tail with the same
    /// template -- so matching on the raw messages would splice an unfiltered
    /// tail onto a filtered prefix (see `ServerPromptCache`).
    public func replacingMessages(
        _ messages: [GFTokenizer.Message],
        tools: [GFTokenizer.FunctionDefinition]
    ) -> ValidatedChatRequest {
        copy(messages: messages, tools: tools)
    }

    /// The memory workspace this request names, from the X-TinyTitan-Workspace
    /// header. Nil takes the server's launch-time workspace, which is the
    /// usual case: one server, one checkout.
    public func withWorkspace(_ workspace: String?) -> ValidatedChatRequest {
        copy(workspace: .some(workspace))
    }

    /// The same request, bound to the catalog model it was validated for.
    public func withModel(_ model: String) -> ValidatedChatRequest {
        copy(model: .some(model))
    }
}

public enum OpenAIRequestValidator {
    /// lint:allow-long a straight-line validation cascade: each guard
    /// rejects one malformed field with its own error. Grouping them into
    /// sub-validators would add indirection without removing a single check.
    public static func validate(_ request: OpenAIChatRequest,
                                modelID: String,
                                maxContext: Int = RuntimeConfiguration
                                    .supportedContextTokens.max() ?? 262_144,
                                reasoningProfile: ServerReasoningProfile = .default,
                                // Filled in for a request that omits the value.
                                // Defaults to the house settings so callers that
                                // do not know the family keep today's behaviour.
                                sampling: GenerationDefaults.Sampling
                                    = GenerationDefaults.house) throws -> ValidatedChatRequest {
        // The "<model>-fast" alias selects the same weights as the base model
        // but enables the CLI-strip heuristic per request (chat-only speed),
        // so tool-using clients keep the base model and chat users opt in.
        let fastModelID = modelID + "-fast"
        let stripCLIPrompt = request.model == fastModelID
        guard request.model == modelID || stripCLIPrompt else { throw ServerRequestError.unknownModel }
        guard request.n == nil || request.n == 1 else {
            throw invalid("only n=1 is supported", "n", "unsupported_value")
        }
        guard request.logprobs != true else {
            throw invalid("logprobs are not supported", "logprobs", "unsupported_value")
        }
        guard request.presencePenalty == nil || request.presencePenalty == 0 else {
            throw invalid("presence_penalty must be zero", "presence_penalty", "unsupported_value")
        }
        guard request.frequencyPenalty == nil || request.frequencyPenalty == 0 else {
            throw invalid("frequency_penalty must be zero", "frequency_penalty", "unsupported_value")
        }
        // Reasoning effort is defined per family and fixed at model load
        // because it changes the rendered prompt. A request may not be able
        // to switch it, but it must never be refused for asking: coding
        // agents send vocabularies this project never defined (`xhigh` on a
        // model with no effort levels, `ultra`, `none`, `extra-high`), and
        // failing those breaks the agent for the rest of the session. So the
        // request is mapped to the nearest level the served model renders,
        // and the difference is recorded below rather than turned into an
        // error.
        var reasoningNotes: [String] = []
        var reasoning = RequestReasoning(
            thinkingMode: reasoningProfile.thinkingMode,
            effort: reasoningProfile.effectiveEffort)
        // The controls arrive in two dialects. `reasoning_effort` is this
        // project's own spelling; llama.cpp, vLLM and TabbyAPI clients put the
        // same controls in a `chat_template_kwargs` object, where
        // `enable_thinking: false` is the only way they turn thinking off.
        // Reading only the top-level field honoured the level beside that
        // object and dropped the switch. Precedence is explicit: an explicit
        // top-level effort wins, then the template-kwargs effort, then the
        // template switch (true -> on, false -> off).
        let requestedEffortRaw = request.reasoningEffort
            ?? request.chatTemplateKwargs?.reasoningEffort
            ?? request.chatTemplateKwargs?.enableThinking.map { $0 ? "on" : "off" }
        if request.reasoningBudgetTokens != nil {
            reasoningNotes.append(
                "reasoning_budget_tokens is accepted but not enforced; this runtime "
                + "bounds thinking by the requested level, not by a token count")
        }
        if let effortRaw = requestedEffortRaw {
            let control = reasoningProfile.family.reasoningControl
            let supported = control.supportedLevels
            if let requested = ReasoningLevel.requested(effortRaw) {
                // The same mapping the server-wide level goes through, so a
                // request and `--reasoning` cannot disagree about what a
                // model does with a level it lacks.
                let efforts = supported.filter { $0 != .off && $0 != .on }
                let applied = ReasoningFallback.effectiveLevel(
                    requested, supported: supported,
                    whenOn: efforts.last)
                // This is what makes a mid-session switch real: the level is
                // carried into generation, which resolves a tokenizer for it
                // rather than re-rendering with the loaded one.
                reasoning = RequestReasoning(
                    thinkingMode: applied == .off ? .off : .on,
                    effort: applied == .off
                        ? nil
                        : ModelReasoningEffort(rawValue: applied.rawValue))
                if applied != requested {
                    reasoningNotes.append(
                        "reasoning level '\(effortRaw)' is not supported by this model; "
                        + "applied \(applied.displayName) instead (supports: "
                        + supported.map(\.displayName).joined(separator: ", ") + ")")
                }
            } else {
                // Unintelligible, not impossible: keep what the model was
                // loaded with rather than guessing at a level.
                reasoningNotes.append(
                    "reasoning level '\(effortRaw)' was not recognised; "
                    + "the model's own default applies (supports: "
                    + supported.map(\.displayName).joined(separator: ", ") + ")")
            }
        }
        // Structured output. The grammar constrains *every* token, so a
        // thought cannot be generated beside the document -- the template's
        // think block would have to be written as part of the JSON. Thinking
        // is therefore off for a request that names a format, and the note
        // says so rather than letting a client wonder why its level was
        // ignored.
        let jsonSchema = try structuredOutputSchema(request.responseFormat)
        if jsonSchema != nil {
            reasoning = RequestReasoning(thinkingMode: .off, effort: nil)
            reasoningNotes.append(
                "a JSON response format constrains every token, so thinking is off "
                + "for this request")
        }
        // `parallel_tool_calls` is accepted and not enforced, on either value.
        // The decoder emits the calls the model produces, so the server cannot
        // promise one at a time; refusing the field would fail every client
        // that sends it defensively (the OpenAI SDKs default it, Codex sends
        // `false` on every turn) for a preference it cannot verify anyway. The
        // Responses object echoes what it was given; Chat Completions has no
        // field to echo into, which is why this is documented instead.
        // S17: include_usage is a streaming option; silently ignoring it on a
        // non-stream request hides a client bug.
        if request.streamOptions?.includeUsage == true, request.stream != true {
            throw invalid("stream_options.include_usage requires stream=true",
                          "stream_options", "invalid_value")
        }
        // S16: OpenAI forbids setting both bounds in one request.
        guard request.maxCompletionTokens == nil || request.maxTokens == nil else {
            throw invalid("max_tokens and max_completion_tokens cannot both be set",
                          "max_tokens", "invalid_value")
        }

        let temperature = request.temperature ?? sampling.temperature
        guard temperature >= 0, temperature <= 2 else {
            throw invalid("temperature must be between 0 and 2",
                          "temperature", "invalid_value")
        }
        let topP = request.topP ?? sampling.topP
        guard topP > 0, topP <= 1 else {
            throw invalid("top_p must be greater than 0 and at most 1",
                          "top_p", "invalid_value")
        }
        let topK = request.topK ?? sampling.topK
        guard (1...256).contains(topK) else {
            throw invalid("top_k must be between 1 and 256", "top_k", "invalid_value")
        }
        let repetitionPenalty = request.repetitionPenalty ?? 1
        guard repetitionPenalty > 0 else {
            throw invalid("repetition_penalty must be positive",
                          "repetition_penalty", "invalid_value")
        }
        // No artificial output cap: when the client omits max_tokens /
        // max_completion_tokens, generation is bounded only by the session's
        // configured context window (further clamped to the available context
        // at inference time), so the model replies until it is done.
        let maximum = request.maxCompletionTokens ?? request.maxTokens ?? maxContext
        guard maximum > 0 else {
            throw invalid("maximum completion tokens must be positive",
                          request.maxCompletionTokens != nil ? "max_completion_tokens" : "max_tokens",
                          "invalid_value")
        }
        // S11: validate against the session's configured context window, not
        // the hard architectural ceiling.
        let cappedMaximum = min(maximum, maxContext)
        guard cappedMaximum == maximum else {
            throw invalid("maximum completion tokens exceeds the configured context window (\(maxContext))",
                          request.maxCompletionTokens != nil ? "max_completion_tokens" : "max_tokens",
                          "value_too_large")
        }

        // S18: stop strings must be non-empty, unique, and bounded.
        let stopValues = request.stop?.values ?? []
        var stopStrings: [String] = []
        if !stopValues.isEmpty {
            guard stopValues.allSatisfy({ !$0.isEmpty }) else {
                throw invalid("stop strings must not be empty", "stop", "invalid_value")
            }
            guard stopValues.count <= 4 else {
                throw invalid("at most 4 stop strings are supported", "stop", "value_too_large")
            }
            let totalLength = stopValues.reduce(0) { $0 + $1.utf8.count }
            guard totalLength <= 256 else {
                throw invalid("stop strings must total at most 256 bytes", "stop", "value_too_large")
            }
            var seen: Set<String> = []
            stopStrings = stopValues.filter { seen.insert($0).inserted }
        }

        let includeTools: Bool
        switch request.toolChoice {
        case nil, .some(.string("auto")):
            includeTools = true
        case .some(.string("none")):
            includeTools = false
        case .some(.string("required")):
            throw invalid("tool_choice=required is not supported",
                          "tool_choice", "unsupported_value")
        case .some(.bool(true)):
            // Legacy boolean form of "auto" (S31).
            includeTools = true
        case .some(.bool(false)):
            // Legacy boolean form of "none" (S31).
            includeTools = false
        default:
            throw invalid("named tool choices are not supported",
                          "tool_choice", "unsupported_value")
        }

        let tools = try (includeTools ? request.tools ?? [] : []).map {
            try validateTool($0)
        }
        let messages = try validateMessages(request.messages)
        // A client-supplied seed makes sampling deterministic.
        let config = GenerationConfig(maxNewTokens: maximum,
                                      temperature: temperature,
                                      topK: topK,
                                      topP: topP,
                                      presencePenalty: request.presencePenalty
                                          ?? GenerationDefaults.presencePenalty,
                                      repetitionPenalty: repetitionPenalty,
                                      seed: request.seed,
                                      stopStrings: stopStrings)
        return ValidatedChatRequest(messages: messages,
                                    tools: tools,
                                    stream: request.stream ?? false,
                                    includeUsage: request.streamOptions?.includeUsage ?? false,
                                    generationConfig: config,
                                    maximumCompletionTokens: maximum,
                                    stripCLIPrompt: stripCLIPrompt,
                                    reasoningNotes: reasoningNotes,
                                    reasoning: reasoning,
                                    jsonSchema: jsonSchema)
    }

    /// The compiled schema a `response_format` asks for, or nil for plain text.
    ///
    /// Chat Completions' own spelling is the one parsed. The Responses surface
    /// and the Messages API reshape theirs into it before validation, so the
    /// rule lives in exactly one place: `{"type": "text"}` (the API's own
    /// default) and an unrecognized *shape* stay accepted, a `json_object`
    /// means an object at the top level, and a `json_schema` is compiled by
    /// `JSONSchemaNode` -- which refuses, by name, every keyword a byte-level
    /// grammar cannot promise.
    static func structuredOutputSchema(_ format: JSONValue?) throws -> JSONSchemaNode? {
        guard let format, case .object(let dict) = format,
              case .string(let type)? = dict["type"] else {
            return nil
        }
        switch type {
        case "text":
            return nil
        case "json_object":
            return .object(properties: [:], required: [], additional: true)
        case "json_schema":
            guard case .object(let wrapper)? = dict["json_schema"],
                  let schema = wrapper["schema"] else {
                throw invalid("json_schema requires json_schema.schema",
                              "response_format.json_schema.schema", "invalid_value")
            }
            do {
                return try JSONSchemaNode.compile(schema)
            } catch let error as JSONSchemaCompileError {
                throw invalid(error.description, "response_format.json_schema.schema",
                              "unsupported_value")
            }
        default:
            throw invalid(
                "response_format \(type) is not supported; use text, json_object or json_schema",
                "response_format", "unsupported_value")
        }
    }

    private static func validateTool(_ tool: OpenAITool) throws -> GFTokenizer.FunctionDefinition {
        guard tool.type == "function" else {
            throw invalid("only function tools are supported", "tools", "unsupported_tool")
        }
        let name = tool.function.name
        guard name.range(of: #"^[A-Za-z0-9_-]{1,64}$"#, options: .regularExpression) != nil else {
            throw invalid("tool name must match [A-Za-z0-9_-]{1,64}",
                          "tools", "invalid_tool_name")
        }
        guard tool.function.parameters.objectValue != nil else {
            throw invalid("tool parameters must be an object schema",
                          "tools", "invalid_tool_schema")
        }
        try validateSchemaKeys(tool.function.parameters)
        let parameters = tool.function.parameters
        guard (try? parameters.jinjaSendableValue()) != nil else {
            throw invalid("tool schema contains a number that cannot be represented exactly",
                          "tools", "invalid_tool_schema")
        }
        return GFTokenizer.FunctionDefinition(name: name,
                                              description: tool.function.description ?? "",
                                              parameters: parameters)
    }

    private static func validateSchemaKeys(_ schema: JSONValue) throws {
        switch schema {
        case .object(let object):
            for (schemaKey, value) in object {
                if schemaKey == "properties" {
                    guard case .object(let definitions) = value else {
                        throw invalid("tool schema properties must be an object",
                                      "tools", "invalid_tool_schema")
                    }
                    for (_, definition) in definitions {
                        // ChatML tool-call parameter names are free-form;
                        // only the schema structure itself is validated.
                        try validateSchemaKeys(definition)
                    }
                } else {
                    try validateSchemaKeys(value)
                }
            }
        case .array(let values):
            for value in values {
                try validateSchemaKeys(value)
            }
        default:
            break
        }
    }

    private static func validateMessages(_ input: [OpenAIChatMessage]) throws -> [GFTokenizer.Message] {
        guard !input.isEmpty else {
            throw invalid("messages must not be empty", "messages", "invalid_message")
        }
        guard input.count <= 1000 else {
            throw invalid("message count exceeds maximum of 1000",
                          "messages", "value_too_large")
        }
        var knownCalls: [String: (name: String, resolved: Bool)] = [:]
        var result: [GFTokenizer.Message] = []
        var sawConversationMessage = false
        for message in input {
            guard let role = GFTokenizer.Role(rawValue: message.role) else {
                throw invalid("unsupported message role \(message.role)",
                              "messages", "invalid_message")
            }
            if role == .system || role == .developer {
                guard !sawConversationMessage else {
                    throw invalid("system or developer guidance must precede the conversation",
                                  "messages", "invalid_message")
                }
            } else {
                sawConversationMessage = true
            }
            let content = try message.content?.textValue()
            let calls: [GFTokenizer.HistoricalToolCall] = try (message.toolCalls ?? []).map { call in
                guard role == .assistant, call.type == "function",
                      !call.id.isEmpty, knownCalls[call.id] == nil,
                      call.function.name.range(
                        of: #"^[A-Za-z0-9_-]{1,64}$"#,
                        options: .regularExpression) != nil else {
                    throw invalid("invalid or duplicate historical tool call",
                                  "messages", "invalid_tool_call")
                }
                let data = Data(call.function.arguments.utf8)
                let arguments = try JSONDecoder().decode(JSONValue.self, from: data)
                guard arguments.objectValue != nil else {
                    throw invalid("historical tool arguments must be a JSON object",
                                  "messages", "invalid_tool_arguments")
                }
                guard (try? arguments.jinjaSendableValue()) != nil else {
                    throw invalid(
                        "historical tool arguments cannot be represented exactly",
                        "messages",
                        "invalid_tool_arguments")
                }
                knownCalls[call.id] = (call.function.name, false)
                return GFTokenizer.HistoricalToolCall(
                    id: call.id, name: call.function.name, arguments: arguments)
            }
            if role == .tool {
                guard let id = message.toolCallID,
                      let call = knownCalls[id], !call.resolved else {
                    throw invalid("tool result must reference one unresolved call",
                                  "messages", "invalid_tool_result")
                }
                knownCalls[id] = (call.name, true)
                guard content != nil else {
                    throw invalid("tool result content is required",
                                  "messages", "invalid_tool_result")
                }
            } else if content == nil && calls.isEmpty {
                throw invalid("message content is required",
                              "messages", "invalid_message")
            }
            result.append(GFTokenizer.Message(role: role,
                                              content: content,
                                              toolCalls: calls,
                                              toolCallID: message.toolCallID,
                                              name: message.name))
        }
        // S19: a conversation that ends with an assistant tool call that is
        // never answered by a tool result would resume from an unanswerable
        // state; reject it instead of generating tool-response markup.
        if knownCalls.contains(where: { !$0.value.resolved }) {
            throw invalid("conversation ends with an unresolved tool call",
                          "messages", "invalid_tool_call")
        }
        return result
    }

    private static func invalid(_ message: String,
                                _ param: String?,
                                _ code: String) -> ServerRequestError {
        .invalid(message: message, param: param, code: code)
    }
}
