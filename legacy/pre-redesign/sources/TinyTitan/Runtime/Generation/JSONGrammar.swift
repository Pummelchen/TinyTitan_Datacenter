import Foundation

/// A byte-level grammar for JSON: which bytes may follow which.
///
/// This is the whole of structured output's correctness argument. A token may
/// only be chosen from the set the grammar still accepts, and the grammar only
/// accepts byte strings that are a prefix of some value matching the schema, so
/// whatever the model does the document it produces is well formed. Nothing
/// about *which* allowed token is chosen is decided here -- that stays the
/// sampler's job, which is why constraining output does not change the
/// unconstrained distribution except by removing what the schema forbids.
///
/// The state is a value: it can be copied, tested against a byte string, and
/// used as a cache key for the allowed-token set. `JSONConstraint` keeps one
/// live copy per generation and caches the set per position.
///
/// Schema awareness is carried by `JSONSchemaNode` on the value being parsed.
/// A grammar built with `.any` is the JSON document grammar alone; a schema
/// only ever *removes* byte sequences from that set, never adds one, so a
/// schema can never make the grammar accept something invalid JSON.
public struct JSONGrammar: Hashable, Sendable {
    /// Whether the string being read is an object key or a value.
    enum StringRole: Hashable, Sendable { case key, value }

    enum NumberState: Hashable, Sendable {
        case minus
        case zero
        case integer
        case fraction
        /// The dot has been read but no digit yet: not a valid place to stop.
        case fractionStart
        case exponent
        case exponentSign
        case exponentDigits
    }

    enum LiteralWord: Hashable, Sendable {
        case trueWord, falseWord, nullWord

        var bytes: [UInt8] {
            switch self {
            case .trueWord: return [0x74, 0x72, 0x75, 0x65]
            case .falseWord: return [0x66, 0x61, 0x6C, 0x73, 0x65]
            case .nullWord: return [0x6E, 0x75, 0x6C, 0x6C]
            }
        }
    }

    /// A literal set being matched byte for byte. `viable` indexes `literals`;
    /// `matched` is how many bytes every viable candidate has consumed.
    struct Enumeration: Hashable, Sendable {
        var literals: [String]
        var viable: [Int]
        var matched: Int
        var role: StringRole
        /// At least one viable literal is fully matched, so the value may end
        /// here -- but a longer candidate may still be alive (`1` and `12`).
        var complete: Bool

        /// The viable candidate that is exactly `matched` bytes long, when
        /// there is one.
        var fullLiteralIndex: Int? {
            viable.first { literals[$0].utf8.count == matched }
        }

        /// Whether a viable candidate is longer than what has been matched.
        var hasLongerCandidate: Bool {
            viable.contains { literals[$0].utf8.count > matched }
        }
    }

    enum State: Hashable, Sendable {
        /// A value is required: start of the document, after `:`, after `[`,
        /// or after a `,` inside an array.
        case value
        /// After `[`: a value or `]`.
        case arrayStart
        /// After `{`: a key string or `}`.
        case objectStart
        /// After a `,` inside an object: a key string.
        case objectKey
        /// After a key: `:`.
        case objectColon
        /// After a complete value: `,`, the matching close, or whitespace.
        case afterValue
        case string(StringRole)
        case escape(StringRole)
        /// `\u` plus this many hex digits still to come, in `role`.
        case unicode(Int, StringRole)
        /// A `true` / `false` / `null` progress, with this many bytes matched.
        case literal(LiteralWord, Int)
        case number(NumberState)
        case enumeration(Enumeration)
        case complete
    }

    struct ObjectFrame: Hashable, Sendable {
        var properties: [String: JSONSchemaNode]
        var required: [String]
        var additional: Bool
        var seen: Set<String>
    }

    enum Frame: Hashable, Sendable {
        case array(item: JSONSchemaNode)
        case object(ObjectFrame)
    }

    private(set) var state: State
    private var stack: [Frame]
    /// The schema position the next value must match.
    private var node: JSONSchemaNode
    /// Decoded bytes of the key currently being read, so a key can be looked up
    /// in `properties` when it completes.
    private var keyBytes: [UInt8]
    /// Hex digits of an in-progress `\uXXXX` inside a key.
    private var unicodeDigits: [UInt8]

    /// A grammar for any JSON value.
    public init() {
        self.init(node: .any)
    }

    /// A grammar for the values `node` describes.
    public init(node: JSONSchemaNode) {
        self.state = .value
        self.stack = []
        self.node = node
        self.keyBytes = []
        self.unicodeDigits = []
    }

    /// Whether what has been read is a complete document (trailing whitespace
    /// aside).
    ///
    /// A number or an enum value is finished by the byte that *cannot* extend
    /// it, so at the end of a generation there is no such byte to consume: `1`
    /// leaves the grammar inside the number rule, not in `.complete`. What
    /// makes it a document is that no container is open and the position is one
    /// the number rule may stop at -- which is also what tells the mask that a
    /// stop token is legal here.
    public var isComplete: Bool {
        if case .complete = state { return true }
        guard stack.isEmpty else { return false }
        switch state {
        case .number(.zero), .number(.integer), .number(.fraction),
             .number(.exponentDigits):
            return true
        case .enumeration(let enumeration):
            return enumeration.role == .value && enumeration.complete
        default:
            return false
        }
    }

    /// Whether some continuation of this position reaches a complete document.
    ///
    /// The mask uses this to rule out tokens that are *grammatically* legal but
    /// lead nowhere: after `{"a":1` in an object whose only property is `a`, a
    /// comma is a well-formed byte, but the key position it moves to has no key
    /// left to write and no `}` to close with -- a dead end. Masking it out is
    /// what keeps the model from walking into a state with no legal token,
    /// which would otherwise surface as a stalled decode.
    ///
    /// Every value node this compiler produces is satisfiable (an unsatisfiable
    /// schema is refused when it is compiled), so the only dead ends are the key
    /// positions of an object that has nothing left to offer.
    public var canComplete: Bool {
        switch state {
        case .complete:
            return true
        case .objectKey:
            return canWriteSomeKey
        case .objectStart:
            return canCloseCurrentObject || canWriteSomeKey
        case .afterValue:
            switch stack.last {
            case .none, .array:
                return true
            case .object(let frame):
                return frame.required.allSatisfy { frame.seen.contains($0) } || canWriteSomeKey
            }
        case .value:
            return node.canProduceValue
        default:
            // Strings, literals, numbers, escapes and enumerations can always
            // be finished from wherever they are.
            return true
        }
    }

    private var canWriteSomeKey: Bool {
        guard case .object(let frame) = stack.last else { return false }
        if frame.additional { return true }
        return frame.properties.keys.contains { !frame.seen.contains($0) }
    }

    private var canCloseCurrentObject: Bool {
        guard case .object(let frame) = stack.last else { return false }
        return frame.required.allSatisfy { frame.seen.contains($0) }
    }

    /// Whether every byte of `bytes` is accepted from here, leaving a grammar
    /// that may or may not be complete. The grammar is not advanced.
    public func accepts<S: Sequence>(bytes: S) -> Bool where S.Element == UInt8 {
        var probe = self
        for byte in bytes where !probe.consume(byte) { return false }
        return true
    }

    /// Whether every byte of `bytes` is accepted *and* the result is a complete
    /// document.
    public func acceptsDocument<S: Sequence>(bytes: S) -> Bool where S.Element == UInt8 {
        var probe = self
        for byte in bytes where !probe.consume(byte) { return false }
        return probe.isComplete
    }

    static func isWhitespace(_ byte: UInt8) -> Bool {
        byte == 0x20 || byte == 0x09 || byte == 0x0A || byte == 0x0D
    }

    static func isDigit(_ byte: UInt8) -> Bool { byte >= 0x30 && byte <= 0x39 }

    static func isHexDigit(_ byte: UInt8) -> Bool {
        isDigit(byte) || (byte >= 0x41 && byte <= 0x46) || (byte >= 0x61 && byte <= 0x66)
    }

    /// Consume one byte. Returns false when the byte cannot appear here, in
    /// which case the grammar is left where it was.
    public mutating func consume(_ byte: UInt8) -> Bool {
        var candidate = self
        guard candidate.step(byte) else { return false }
        self = candidate
        return true
    }

    private mutating func step(_ byte: UInt8) -> Bool {
        if JSONGrammar.isWhitespace(byte), allowsWhitespace { return true }
        switch state {
        case .value:
            return startValue(byte)
        case .arrayStart:
            if byte == 0x5D { return closeArray() }
            return startValue(byte)
        case .objectStart:
            if byte == 0x7D { return closeObject() }
            return startKey(byte)
        case .objectKey:
            return startKey(byte)
        case .objectColon:
            guard byte == 0x3A else { return false }
            state = .value
            return true
        case .afterValue:
            switch byte {
            case 0x2C:
                guard let top = stack.last else { return false }
                switch top {
                case .object: state = .objectKey
                case .array(let item):
                    node = item
                    state = .value
                }
                return true
            case 0x7D: return closeObject()
            case 0x5D: return closeArray()
            default: return false
            }
        case .string(let role):
            return continueString(byte, role: role)
        case .escape(let role):
            return continueEscape(byte, role: role)
        case .unicode(let remaining, let role):
            return continueUnicode(byte, remaining: remaining, role: role)
        case .literal(let word, let matched):
            let bytes = word.bytes
            guard matched < bytes.count, bytes[matched] == byte else { return false }
            if matched + 1 == bytes.count { return finishValue() }
            state = .literal(word, matched + 1)
            return true
        case .number(let number):
            return continueNumber(byte, number: number)
        case .enumeration(var enumeration):
            return continueEnumeration(byte, enumeration: &enumeration)
        case .complete:
            return false
        }
    }

    /// Whitespace may separate any two tokens, but may not appear inside one.
    private var allowsWhitespace: Bool {
        switch state {
        case .value, .arrayStart, .objectStart, .objectKey, .objectColon,
             .afterValue, .complete:
            return true
        default:
            return false
        }
    }

    // MARK: - Values

    private mutating func startValue(_ byte: UInt8) -> Bool {
        switch node {
        case .enumeration(let literals):
            return beginEnumeration(literals, role: .value, first: byte)
        case .object(let properties, let required, let additional):
            guard byte == 0x7B else { return false }
            stack.append(.object(ObjectFrame(properties: properties, required: required,
                                             additional: additional, seen: [])))
            state = .objectStart
            return true
        case .array(let items):
            guard byte == 0x5B else { return false }
            node = items ?? .any
            stack.append(.array(item: items ?? .any))
            state = .arrayStart
            return true
        case .scalar(let kinds):
            return startScalar(byte, kinds: kinds)
        case .any:
            switch byte {
            case 0x7B:
                stack.append(.object(ObjectFrame(properties: [:], required: [],
                                                 additional: true, seen: [])))
                state = .objectStart
                return true
            case 0x5B:
                node = .any
                stack.append(.array(item: .any))
                state = .arrayStart
                return true
            case 0x22:
                state = .string(.value)
                return true
            case 0x74, 0x66, 0x6E:
                return startLiteral(byte)
            case 0x2D, 0x30...0x39:
                return startNumber(byte)
            default:
                return false
            }
        }
    }

    private mutating func startScalar(_ byte: UInt8, kinds: Set<JSONScalarKind>) -> Bool {
        if kinds.contains(.string), byte == 0x22 {
            state = .string(.value)
            return true
        }
        if kinds.contains(.number) || kinds.contains(.integer),
           byte == 0x2D || JSONGrammar.isDigit(byte) {
            return startNumber(byte)
        }
        if kinds.contains(.boolean), byte == 0x74 || byte == 0x66 {
            return startLiteral(byte)
        }
        if kinds.contains(.null), byte == 0x6E {
            return startLiteral(byte)
        }
        return false
    }

    private mutating func startLiteral(_ byte: UInt8) -> Bool {
        let word: LiteralWord
        switch byte {
        case 0x74: word = .trueWord
        case 0x66: word = .falseWord
        default: word = .nullWord
        }
        guard word.bytes.count > 1 else { return false }
        state = .literal(word, 1)
        return true
    }

    private mutating func startNumber(_ byte: UInt8) -> Bool {
        switch byte {
        case 0x2D:
            state = .number(.minus)
        case 0x30:
            state = .number(.zero)
        default:
            guard JSONGrammar.isDigit(byte) else { return false }
            state = .number(.integer)
        }
        return true
    }

    private mutating func continueNumber(_ byte: UInt8, number: NumberState) -> Bool {
        let full = !node.forbidsFraction
        switch number {
        case .minus:
            guard JSONGrammar.isDigit(byte) else { return false }
            state = .number(byte == 0x30 ? .zero : .integer)
            return true
        case .zero:
            // A leading zero may not be followed by another digit.
            if byte == 0x2E, full { state = .number(.fractionStart); return true }
            if byte == 0x65 || byte == 0x45, full { state = .number(.exponent); return true }
            return finishValue(consuming: byte)
        case .integer:
            if JSONGrammar.isDigit(byte) { return true }
            if byte == 0x2E, full { state = .number(.fractionStart); return true }
            if byte == 0x65 || byte == 0x45, full { state = .number(.exponent); return true }
            return finishValue(consuming: byte)
        case .fractionStart:
            guard JSONGrammar.isDigit(byte) else { return false }
            state = .number(.fraction)
            return true
        case .fraction:
            if JSONGrammar.isDigit(byte) { return true }
            if byte == 0x65 || byte == 0x45 { state = .number(.exponent); return true }
            return finishValue(consuming: byte)
        case .exponent:
            if byte == 0x2B || byte == 0x2D { state = .number(.exponentSign); return true }
            if JSONGrammar.isDigit(byte) { state = .number(.exponentDigits); return true }
            return false
        case .exponentSign:
            guard JSONGrammar.isDigit(byte) else { return false }
            state = .number(.exponentDigits)
            return true
        case .exponentDigits:
            if JSONGrammar.isDigit(byte) { return true }
            return finishValue(consuming: byte)
        }
    }

    // MARK: - Strings

    private mutating func startKey(_ byte: UInt8) -> Bool {
        if case .object(let frame) = stack.last, !frame.additional {
            let remaining = frame.properties.keys.filter { !frame.seen.contains($0) }.sorted()
            guard !remaining.isEmpty else { return false }
            return beginEnumeration(remaining.map { "\"\($0)\"" }, role: .key, first: byte)
        }
        guard byte == 0x22 else { return false }
        keyBytes.removeAll(keepingCapacity: true)
        state = .string(.key)
        return true
    }

    private mutating func continueString(_ byte: UInt8, role: StringRole) -> Bool {
        if byte == 0x5C { state = .escape(role); return true }
        // A raw control character is not legal inside a JSON string.
        guard byte >= 0x20 else { return false }
        if byte == 0x22 {
            guard role == .key else { return finishValue() }
            return finishKey(String(decoding: keyBytes, as: UTF8.self))
        }
        if role == .key { keyBytes.append(byte) }
        return true
    }

    private mutating func continueEscape(_ byte: UInt8, role: StringRole) -> Bool {
        if byte == 0x75 {
            unicodeDigits.removeAll(keepingCapacity: true)
            state = .unicode(4, role)
            return true
        }
        let decoded: UInt8
        switch byte {
        case 0x22: decoded = 0x22
        case 0x5C: decoded = 0x5C
        case 0x2F: decoded = 0x2F
        case 0x62: decoded = 0x08
        case 0x66: decoded = 0x0C
        case 0x6E: decoded = 0x0A
        case 0x72: decoded = 0x0D
        case 0x74: decoded = 0x09
        default: return false
        }
        if role == .key { keyBytes.append(decoded) }
        state = .string(role)
        return true
    }

    private mutating func continueUnicode(_ byte: UInt8, remaining: Int, role: StringRole) -> Bool {
        guard JSONGrammar.isHexDigit(byte) else { return false }
        unicodeDigits.append(byte)
        if remaining > 1 {
            state = .unicode(remaining - 1, role)
            return true
        }
        if role == .key, let scalar = UInt32(String(decoding: unicodeDigits, as: UTF8.self), radix: 16),
           let unicode = Unicode.Scalar(scalar) {
            keyBytes.append(contentsOf: Array(String(Character(unicode)).utf8))
        }
        state = .string(role)
        return true
    }

    // MARK: - Enumerations

    private mutating func beginEnumeration(_ literals: [String], role: StringRole,
                                           first byte: UInt8) -> Bool {
        var enumeration = Enumeration(literals: literals, viable: Array(literals.indices),
                                      matched: 0, role: role, complete: false)
        state = .enumeration(enumeration)
        return continueEnumeration(byte, enumeration: &enumeration)
    }

    private mutating func continueEnumeration(_ byte: UInt8,
                                              enumeration: inout Enumeration) -> Bool {
        let extendable = enumeration.viable.contains { index in
            let bytes = Array(enumeration.literals[index].utf8)
            return enumeration.matched < bytes.count && bytes[enumeration.matched] == byte
        }
        if enumeration.complete, !extendable {
            // The value is already a legal literal and this byte cannot extend
            // it, so the byte belongs to whatever follows a value. A completed
            // key cannot reach this branch: a completed literal only stays
            // alive with a longer candidate, and quoted keys are never a
            // prefix of one another.
            state = .afterValue
            return step(byte)
        }
        guard extendable else { return false }
        enumeration.viable = enumeration.viable.filter { index in
            let bytes = Array(enumeration.literals[index].utf8)
            return enumeration.matched < bytes.count && bytes[enumeration.matched] == byte
        }
        enumeration.matched += 1
        enumeration.complete = enumeration.viable.contains {
            enumeration.literals[$0].utf8.count == enumeration.matched
        }
        state = .enumeration(enumeration)
        if enumeration.role == .key, enumeration.complete {
            guard let index = enumeration.fullLiteralIndex else { return false }
            // Property-name enumerations are matched as quoted literals; the
            // frame stores the bare name.
            return finishKey(String(enumeration.literals[index].dropFirst().dropLast()))
        }
        if enumeration.role == .value, enumeration.complete,
           !enumeration.hasLongerCandidate {
            return finishValue()
        }
        return true
    }

    // MARK: - Containers and completion

    /// A value is complete: the document ends here, or the enclosing container
    /// decides what may follow.
    private mutating func finishValue() -> Bool {
        state = stack.isEmpty ? .complete : .afterValue
        return true
    }

    /// Same, for a byte that belongs to whatever follows the value rather than
    /// to the value itself (a number or literal ended by `,` or `}`).
    private mutating func finishValue(consuming byte: UInt8) -> Bool {
        state = stack.isEmpty ? .complete : .afterValue
        return step(byte)
    }

    private mutating func finishKey(_ text: String) -> Bool {
        guard case .object(var frame) = stack.last else { return false }
        guard !frame.seen.contains(text) else { return false }
        if !frame.additional, frame.properties[text] == nil { return false }
        frame.seen.insert(text)
        stack[stack.count - 1] = .object(frame)
        node = frame.properties[text] ?? .any
        state = .objectColon
        return true
    }

    private mutating func closeObject() -> Bool {
        guard case .object(let frame) = stack.last else { return false }
        for name in frame.required where !frame.seen.contains(name) { return false }
        stack.removeLast()
        return finishValue()
    }

    private mutating func closeArray() -> Bool {
        guard case .array = stack.last else { return false }
        stack.removeLast()
        return finishValue()
    }

    /// The schema node governing the value being parsed, exposed for tests.
    public var valueNode: JSONSchemaNode { node }
}

extension JSONSchemaNode {
    /// `integer` forbids a fraction and an exponent, which the number grammar
    /// can enforce; every other node leaves the full JSON number grammar.
    var forbidsFraction: Bool { self == .scalar([.integer]) }
}
