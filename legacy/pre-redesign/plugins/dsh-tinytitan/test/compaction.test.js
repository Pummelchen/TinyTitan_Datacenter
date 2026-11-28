import assert from "node:assert/strict";
import test from "node:test";

import {
  AUXILIARY_PURPOSES,
  auxiliaryThinkingOff,
  createAuxiliaryQuietCompaction,
} from "../src/compaction.js";

/**
 * A base engine shaped like `dsh-compaction-basic`'s summariser: it reaches the
 * model only through `this.ctx.llm.stream`, which is the one member the plugin
 * replaces (`summarizeWithLlm`, `dsh-compaction-basic/lib/index.js:302`).
 */
class RecordingBase {
  constructor(ctx, { fail = false } = {}) {
    this.ctx = ctx;
    this.fail = fail;
    this.calls = [];
  }

  async summarize() {
    const call = async (options) => {
      const streamed = this.ctx.llm.stream(options);
      this.calls.push(streamed);
      if (this.fail) throw new Error("engine failed");
      return streamed;
    };
    await call({ purpose: "compaction" });
    await call({ purpose: "session-title" });
    await call({ purpose: undefined });
    await call({ purpose: "compaction", reasoningEffort: "high" });
    return this.calls;
  }
}

function harness(options = {}) {
  const seen = [];
  const ctx = {
    // A service whose methods live on a prototype, like a cordis service.
    llm: Object.assign(Object.create({
      resolveModelInfo: () => ({ context: { contextWindow: 262144 } }),
    }), {
      stream: (opts) => {
        seen.push(opts);
        return opts;
      },
    }),
    tokenMeter: { measure: () => ({ totalTokens: 0 }) },
  };
  const Backend = createAuxiliaryQuietCompaction(RecordingBase);
  const engine = new Backend(ctx, options);
  return { engine, ctx, seen };
}

test("the auxiliary purposes are the two the harness marks", () => {
  assert.deepEqual([...AUXILIARY_PURPOSES].sort(), ["compaction", "session-title"]);
});

test("auxiliaryThinkingOff only touches auxiliary calls", () => {
  assert.equal(auxiliaryThinkingOff({ purpose: "compaction" }).reasoningEffort, "off");
  assert.equal(auxiliaryThinkingOff({ purpose: "session-title" }).reasoningEffort, "off");
  const ordinary = { purpose: undefined, reasoningEffort: "medium" };
  assert.equal(auxiliaryThinkingOff(ordinary), ordinary);
  const already = { purpose: "compaction", reasoningEffort: "off" };
  assert.equal(auxiliaryThinkingOff(already), already);
  // An auxiliary call is forced off even when a level is named: the point is
  // that the checkpoint's own output cap survives, which is what
  // `dsh-llm-deepseek` does for session titles by itself.
  assert.equal(
    auxiliaryThinkingOff({ purpose: "compaction", reasoningEffort: "high" }).reasoningEffort,
    "off",
  );
});

test("a compaction call reaches the llm with thinking off", async () => {
  const { engine, seen } = harness();
  await engine.summarize("input", "agent", undefined);
  assert.equal(seen[0].purpose, "compaction");
  assert.equal(seen[0].reasoningEffort, "off");
  assert.equal(seen[1].purpose, "session-title");
  assert.equal(seen[1].reasoningEffort, "off");
  assert.equal(seen[2].purpose, undefined);
  assert.equal(seen[2].reasoningEffort, undefined);
  assert.equal(seen[3].reasoningEffort, "off");
});

test("the real context is restored after the call", async () => {
  const { engine, ctx } = harness();
  await engine.summarize("input", "agent", undefined);
  assert.equal(engine.ctx, ctx);
});

test("the real context is restored when the engine throws", async () => {
  const { engine, ctx } = harness({ fail: true });
  await assert.rejects(() => engine.summarize("input", "agent", undefined), /engine failed/);
  assert.equal(engine.ctx, ctx);
});

test("every other context member still resolves during the call", async () => {
  const { engine } = harness();
  let measured;
  class Inspecting extends createAuxiliaryQuietCompaction(RecordingBase) {
    async summarize(input, agent, signal) {
      measured = this.ctx.tokenMeter.measure();
      return super.summarize(input, agent, signal);
    }
  }
  const instance = new Inspecting(engine.ctx);
  await instance.summarize("input", "agent", undefined);
  assert.deepEqual(measured, { totalTokens: 0 });
});
