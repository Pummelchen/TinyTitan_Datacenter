/**
 * `dsh-tinytitan` — the DeepSeek Harness side of running models from a local TinyTitan
 * server.
 *
 * It does two things, both at boot, both idempotent:
 *
 * 1. **Keeps the route current.** The harness's own `llm-pi-ai` adapter serves
 *    these models — there is no adapter here — and this checkout's
 *    `tools/dsh_route.sh` is the one place that turns the installed models into
 *    that route's block. Running it means the model picker follows `models/`
 *    instead of a copy someone typed once. A catalogue install has no checkout
 *    to run, so `generate.js` produces the same block in-process from the
 *    server's catalog; the shell tool stays authoritative wherever it exists.
 * 2. **Mounts a compaction backend that does not think.** Compaction and session
 *    titles name no reasoning level, so they inherit the route's default; on a
 *    local thinking model that spends a summariser's own output cap on thinking
 *    and costs tens of seconds on every new session's title. The preset this
 *    plugin generates points that row at `dsh-tinytitan/backend`.
 *
 * Neither job patches the harness or replaces its adapter, so a harness upgrade
 * cannot desynchronise a copied protocol implementation — and when the two
 * upstream asks in this repository's `docs/dsh-upstream-asks.md` land, the second
 * job becomes unnecessary.
 *
 * @module dsh-tinytitan
 */
import { resolveConfig } from "./config.js";
import { registerRoute } from "./route.js";
import { ensureCompactionPreset } from "./setup.js";

/** Plugin name, as the harness registry shows it. */
export const name = "dsh-tinytitan";

export {
  DEFAULT_PRESET_ID,
  DEFAULT_PROVIDER,
  REPO_ROOT,
  findRepoRoot,
  resolveConfig,
} from "./config.js";
export { registerRoute, routeScript } from "./route.js";
export {
  applyRouteToSettings,
  catalogRows,
  findModelsDir,
  findServerBinary,
  generateBlock,
  generateRoute,
  writeRouteSettings,
} from "./generate.js";
export {
  COMPACTION_BACKEND,
  defaultPreset,
  ensureCompactionPreset,
  repointCompactionRow,
  setDefaultPreset,
  standardPresetPath,
} from "./setup.js";
export {
  AUXILIARY_PURPOSES,
  auxiliaryThinkingOff,
  createAuxiliaryQuietCompaction,
} from "./compaction.js";

/**
 * Run the plugin.
 * @param ctx - the harness context (used only for logging; nothing is injected).
 * @param config - the row config; see {@link resolveConfig}.
 */
export function apply(ctx, config = {}) {
  const resolved = resolveConfig(config);
  const log = resolved.log ?? ((message) => {
    if (typeof ctx?.logger?.info === "function") ctx.logger.info(message);
    else console.log(message);
  });
  // A read-only home, a missing checkout or a failed write must not take the
  // profile down: the harness still works, only this convenience does not.
  if (resolved.registerRoute) {
    try {
      registerRoute({ ...resolved, log });
    } catch (error) {
      log(`dsh-tinytitan: route registration threw: ${error instanceof Error ? error.message : error}`);
    }
  }
  if (resolved.writeCompactionPreset) {
    try {
      ensureCompactionPreset({ ...resolved, log });
    } catch (error) {
      log(`dsh-tinytitan: compaction preset threw: ${error instanceof Error ? error.message : error}`);
    }
  }
}

export default apply;
