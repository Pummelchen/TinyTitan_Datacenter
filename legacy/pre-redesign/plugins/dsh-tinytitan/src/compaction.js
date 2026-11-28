/**
 * Force thinking off for the harness's auxiliary model calls.
 *
 * The harness marks compaction and session-title calls with a `purpose` and
 * names no reasoning level, so `dsh-llm` fills in the route's default — which on
 * a local thinking model means a summariser that thinks inside its own output
 * cap, and a title call that costs tens of seconds on every new session.
 *
 * A `dsh-llm-deepseek` route has the same behaviour built in
 * (`resolveThinking`: `purpose === "session-title"` returns
 * `{ thinking: "disabled" }`); this module is the equivalent for any route the
 * harness serves through pi-ai, and it does it without reimplementing the
 * compaction engine: `dsh-compaction-basic`'s `summarizeWithLlm` reads exactly
 * one thing off the context it is handed — `ctx.llm.stream` — so a context that
 * delegates everything and overrides that one member is enough, and the
 * instruction, the checkpoint envelope, the pricing and the retry policy all
 * stay the engine's.
 *
 * @module dsh-tinytitan/compaction
 */

/** Purposes whose calls exist to produce bounded visible text. */
export const AUXILIARY_PURPOSES = new Set(["compaction", "session-title"]);

/**
 * The same options, with thinking off when the call is an auxiliary one.
 * @param options - the harness's `llm.stream` options.
 * @returns the options to stream, possibly with `reasoningEffort: "off"`.
 */
export function auxiliaryThinkingOff(options) {
  if (!AUXILIARY_PURPOSES.has(options?.purpose)) return options;
  if (options.reasoningEffort === "off") return options;
  return { ...options, reasoningEffort: "off" };
}

/**
 * Build the compaction backend class on top of a base engine.
 *
 * A factory rather than a subclass literal so the behaviour can be tested
 * against a stub base, without the harness's packages installed.
 *
 * @param Base - the engine to extend (`dsh-compaction-basic`'s default export).
 * @returns a class whose summarisation never thinks.
 */
export function createAuxiliaryQuietCompaction(Base) {
  return class extends Base {
    /**
     * Summarise with the auxiliary call's thinking forced off.
     *
     * `this.ctx` is swapped for a delegate for the duration of the call: the
     * engine reads `ctx.llm.stream` through it, and the real context is restored
     * in `finally` so a failure cannot leave it replaced.
     *
     * @param input - the replayed conversation prefix to condense.
     * @param agent - supplies routed-model history and the session id.
     * @param signal - optional cancellation forwarded to the adapter.
     * @returns the base engine's summary result.
     */
    async summarize(input, agent, signal) {
      const ctx = this.ctx;
      const llm = Object.create(ctx.llm);
      llm.stream = (options, ...rest) =>
        ctx.llm.stream(auxiliaryThinkingOff(options), ...rest);
      const quiet = Object.create(ctx);
      quiet.llm = llm;
      this.ctx = quiet;
      try {
        return await super.summarize(input, agent, signal);
      } finally {
        this.ctx = ctx;
      }
    }
  };
}

export default createAuxiliaryQuietCompaction;
