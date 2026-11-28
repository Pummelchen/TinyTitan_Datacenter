# Two asks for DeepSeek Harness, from running it against a local TinyTitan server

Date: 2026-09-14. Verified against the installed `@deepseek-ai/dsh` **0.1.5-rc.2**
and `@earendil-works/pi-ai` 0.85.1. Upstream (`deepseek-ai/deepseek-harness`) has
**issues disabled and discussions enabled**, so these are written to be posted as
two discussions; the patches are small enough to apply locally meanwhile.

Both are about the *seam*, not about any one provider: they change what every
route gets, and both already hold on the first-party DeepSeek adapter.

---

## 1. Honor `purpose` when a call names no reasoning level

**What happens now.** The harness marks its auxiliary model calls with a
`purpose`, and they name no `reasoningEffort`:

- `@deepseek-ai/dsh-compaction-basic` — `summarizeWithLlm`
  (`lib/index.js:292-302`) streams with `purpose: "compaction"` and no
  `reasoningEffort`;
- `@deepseek-ai/dsh-session-title-llm` (`lib/index.js:209-217`) streams with
  `purpose: "session-title"` and no `reasoningEffort`.

`@deepseek-ai/dsh-llm` then fills the gap from the *route*:
`resolveCallWithInfo` (`lib/index.js:2117-2128`) takes
`const effective = requested ?? reasoning.defaultEffort`. So on any route whose
profile default level is a thinking level, compaction and session titles think.

**Why that is wrong rather than merely expensive.**

- Those calls exist to produce a *bounded* amount of visible text — a checkpoint
  or a title — inside an output cap the caller chose (`maxTokens`). Thinking
  spends that cap first, which is the failure the old Qwen plugin's author
  described: a summariser whose thinking eats the cap truncates the checkpoint.
- On a local model the cost is wall-clock, not quota. Measured here on a base M3
  with 24 GB and Qwen3.8-Flash-Next 4-bit: **~5.8 tok/s** decode, so a few
  hundred thinking tokens is minutes per compaction and tens of seconds per new
  session's title.
- The harness already decided this for its own route:
  `@deepseek-ai/dsh-llm-deepseek`'s `resolveThinking`
  (`lib/index.js:32`) returns `{ thinking: "disabled" }` for
  `purpose === "session-title"`, and the compaction call carries
  `x-deepseek-harness-compact: 1` (`lib/index.js:1667`) for the server to act on.
  A pi-ai-backed route has no equivalent, so the same harness behaves two ways
  depending on which adapter serves the model.

**Suggested patch** (one place, no schema change): treat an auxiliary purpose as
naming `off` when the call names no level, in whichever layer resolves the
default — e.g. in `dsh-llm`'s `resolveCallWithInfo`:

```js
const AUXILIARY_PURPOSES = new Set(["compaction", "session-title"]);
// …
const requested = defaulted.reasoningEffort
  ?? (AUXILIARY_PURPOSES.has(defaulted.purpose) ? "off" : undefined);
```

Every reasoning model the harness can route lists `off` among its efforts, so the
existing `UNSUPPORTED_REASONING_EFFORT` check stays the guard it is today.
Alternatively a profile field (`auxiliaryReasoning`, defaulting to `off`) would
let a deployment opt out; the seam-level default seems truer to the intent.

**Acceptance.** With a thinking route, a session's first prompt issues a
`purpose: "session-title"` request with no thinking (no `reasoning_content`, no
reasoning tokens in usage), and a compaction call behaves the same, while an
ordinary turn still thinks at the route's default.

**Local mitigation meanwhile.** TinyTitan ships `plugins/dsh-tinytitan`, a thin bundle
that mounts a compaction backend forcing `off` for those calls; the route's own
`reasoning` default still decides ordinary turns.

---

## 2. Map pi-ai's reasoning usage into `reasoningTokens`

**What happens now.** pi-ai parses the provider's split
(`@earendil-works/pi-ai/dist/api/openai-completions.js:1201`):

```js
reasoning: rawUsage.completion_tokens_details?.reasoning_tokens || 0,
```

`@deepseek-ai/dsh-llm-pi-ai`'s `mapUsage` (`lib/index.js:1357-1365`) then maps
`input`, `output`, `totalTokens`, `cacheRead` and `cacheWrite` — and drops
`reasoning`:

```js
function mapUsage(usage) {
  return {
    inputTokens: usage.input,
    outputTokens: usage.output,
    totalTokens: usage.totalTokens,
    ...usage.cacheRead > 0 ? { cacheReadTokens: usage.cacheRead } : {},
    ...usage.cacheWrite > 0 ? { cacheWriteTokens: usage.cacheWrite } : {}
  };
}
```

**Consequence.** The harness's `TokenUsage.reasoningTokens` is populated on the
DeepSeek route (visible in a session transcript as
`{"inputTokens": 9707, "outputTokens": 153, "totalTokens": 11140, "cacheReadTokens": 1280, "reasoningTokens": 132}`)
and never on a pi-ai route, so the context meter and any per-turn accounting
cannot separate thinking from answer there — exactly where local models spend
most of their output tokens. TinyTitan does report the field; the adapter discards
it. The doc comment ("reasoning folded into output by pi-ai") describes pi-ai's
*output* bucket, not the separate count pi-ai also returns.

**Suggested patch** (one line, mirroring the cache fields' "only when non-zero"
rule):

```js
...usage.reasoning > 0 ? { reasoningTokens: usage.reasoning } : {},
```

**Acceptance.** On a pi-ai route, a thinking turn's `assistant/message` record
carries `reasoningTokens` matching the provider's
`completion_tokens_details.reasoning_tokens`; a non-thinking turn omits it, as
the cache fields do.

---

## Applying these locally

Both are small enough to carry as a patch against the installed package while
upstream decides — but an upgrade replaces `node_modules`, so re-apply after one.
The first is already worked around by `plugins/dsh-tinytitan` for compaction; the
second cannot be worked around from outside the adapter, which is why it is the
more valuable of the two.
