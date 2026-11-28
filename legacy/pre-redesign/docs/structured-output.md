# Structured output

A request may ask TinyTitan for JSON, and get JSON. The server does not prompt for
it, or hope for it, or parse it out afterwards: it compiles the request into a
byte-level grammar and masks the sampler with it, so every token the model draws
is one the document can still contain. The result is a value the schema allows,
by construction, at the cost of the tokens the schema rules out.

This is what the three API surfaces' spellings mean:

| Surface | Field |
| --- | --- |
| Chat Completions | `response_format` |
| Responses | `text.format` |
| Messages | `output_config.format` |

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "qwen3.5-2b_4-Bit",
       "messages": [{"role": "user", "content": "The colour red, as JSON."}],
       "response_format": {"type": "json_object"}}'
```

All three are normalized into the Chat Completions spelling before one
validator parses them, so there is one rule in one place. `{"type": "text"}`
(the API's own default) and an unrecognized shape mean plain text, as before.

## What it guarantees, and what it does not

It guarantees the **shape**: the bytes that come back are a well-formed JSON
document, and (with `json_schema`) one whose keys, types and required properties
match the schema. It does not guarantee the **content** is true or useful — the
grammar picks nothing, it only removes what the schema forbids, and among the
remaining tokens the model's own distribution decides. A model that would have
written prose writes the least prose-shaped document it can instead.

Two further limits are worth stating plainly:

- **A truncated response is not valid JSON.** If `max_tokens` runs out
  mid-document the grammar is mid-document too, and what comes back will not
  parse. Enforced shape is not a promise about a response that was never
  finished.
- **Thinking is off for a constrained request.** The grammar constrains *every*
  token, so a `<think>` block would have to be written as part of the document.
  The validator turns the level off and records a reasoning note saying why,
  rather than silently ignoring the level the client asked for.

## The schema subset

`JSONSchemaNode` compiles a schema into a grammar. It is deliberately a subset:
a keyword that cannot be turned into a byte-level guarantee is refused at
request time, by name, in the API's own error shape. Accepting a constraint and
then not enforcing it is the one outcome worse than refusing it, because the
client validates against a promise the server never made.

**Supported**

| Keyword | Meaning here |
| --- | --- |
| `type` | `object`, `array`, `string`, `number`, `integer`, `boolean`, `null`, or a list of *scalar* types |
| `properties` | the object's allowed keys, each with its own schema |
| `required` | keys that must be present before the object may close |
| `additionalProperties` | `true` (the default) allows any other key; `false` restricts to `properties` |
| `items` | one schema for every array element |
| `enum`, `const` | exact literal spellings (strings without escapes, or integers) |
| `title`, `description`, `default`, `examples`, `$comment`, `$schema`, `$id`, `x-*` | annotations: accepted and ignored, because they constrain nothing |

**Refused, by name**

`$ref` and `$defs`, `allOf` / `anyOf` / `oneOf` / `not`, `if` / `then` / `else`,
`patternProperties`, `propertyNames`, `unevaluatedProperties`,
`dependencies` / `dependentSchemas` / `dependentRequired`, `pattern`, `format`,
`minimum` / `maximum` / `exclusiveMinimum` / `exclusiveMaximum` / `multipleOf`,
`minLength` / `maxLength`, `minItems` / `maxItems` / `uniqueItems`, `contains`,
`prefixItems` / `items` as a tuple, `minProperties` / `maxProperties`,
`additionalProperties` given as a schema, and `required` without `properties`.

A handful of shapes are refused as *unsatisfiable* rather than unsupported,
with the reason in the message: a `false` schema, a `required` name that is not
in `properties` while `additionalProperties` is `false`, enum values that are
prefixes of one another, a string enum value that would need escapes, and a
`properties`/`items` keyword that contradicts `type`.

Named exceptions are the point: `{"type": ["object", "null"]}` is refused
because the grammar would have to hold two shapes open at once, while
`{"type": ["string", "null"]}` is fine — the first byte decides which one it is.

## How it works

- **`JSONGrammar`** is a byte-level pushdown automaton for JSON with a schema
  node at the current value position. It answers two questions: does this byte
  string keep the document legal, and can the document still be finished from
  here. The second one matters — a comma after the last allowed key of an object
  is legal JSON but leads nowhere, and a mask that offered it would let the
  model walk into a state with no legal token.
- **`JSONTokenTable`** is the vocabulary as byte strings, built once per loaded
  model (248k tokenizer lookups, ~4 bytes each, stored flat). A token that
  carries no bytes — every special token, `<|im_end|>` included — is never
  allowed: emitting it would advance the document by nothing.
- **`JSONConstraint`** holds the document parsed so far and the token set that
  may extend it. The set is computed per *position* and cached under the whole
  position, so the second character of a string costs nothing and the thousandth
  costs nothing either. Only tokens whose first byte the grammar still accepts
  are tried, and the walk stops at the first byte that fails.
- **The mask is applied where the repetition penalty already is**: a host-side
  in-place pass over the shared logits buffer before the softcap+softmax
  front-end is encoded, on both engines. A disallowed logit is set to the most
  negative FP16 value, which the softcap folds to exactly `-softcap`, strictly
  below anything a finite logit can produce.
- **One whitespace-only token between two structural tokens is allowed, a
  second in a row is not.** A JSON grammar allows whitespace everywhere, which
  means it allows whitespace forever: with a `{"type": "boolean"}` schema and a
  prompt that did not prime JSON, the dense 2B spent 32 tokens on newlines and
  stopped on `length`. Pretty-printed JSON is unaffected (an indent is one
  token), and the stall is impossible.
- **Nothing is masked when no format is asked for.** `GenerationConfig`'s
  constraint is `nil`, the sampler skips the pass, and generation is
  byte-identical to what it was before this existed — which is what the golden
  baseline checks.

## Where it cannot be used

The MTP decode path drafts several tokens ahead of the sampler and never
consults a grammar, so a constrained request takes the ordinary decode path
instead; the fused greedy head picks its token without writing the logits a mask
would edit, and the server already runs the logits head. Both are refused
explicitly rather than served unconstrained.

## Verifying it

`tests/TinyTitan/Runtime/Generation/JSONGrammarTests.swift` covers the grammar and
the schema compiler on bytes; `JSONConstraintTests.swift` covers the token sets;
`JSONConstraintLoopTests.swift` runs the real decode loop over a scripted
distribution that *prefers* a token the grammar forbids, and asserts the
document that comes out — with the same script and no constraint as the control,
which writes the forbidden token instead. `tests/TinyTitanServer/StructuredOutputTests.swift`
covers the three spellings and the thinking-off rule.

On a real install, verified 2026-09-14 with `models/qwen3.5_2B_4Bit` on the GPU
engine and `--cpu`:

| Schema | Response |
| --- | --- |
| `{"type": "boolean"}` | `false` |
| `{"enum": ["HELLO"]}` | `"HELLO"` |
| `{"type": "json_object"}` | `{"name": "red", "hex": "#FF0000"}` |
| object with `required` + `additionalProperties: false` | only the declared keys, required ones present |

and the same `json_object` contract answered identically through
`/v1/messages` (`output_config.format`) and `/v1/responses` (`text.format`).
