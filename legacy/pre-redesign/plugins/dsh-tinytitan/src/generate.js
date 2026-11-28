/**
 * Generate the harness's TinyTitan route with no checkout to shell out to.
 *
 * `tools/dsh_route.sh` is the one place that turns the installed models into the
 * `llm-pi-ai` route block, and it stays the source of truth wherever a checkout
 * exists. A plugin installed from a catalogue is a plain package beside no
 * checkout, though, so it cannot run that script; this module is the fallback.
 * It mirrors the shell tool exactly — the three header comment lines, the three
 * switches that are easy to get wrong by hand, the effort ladder each chat
 * template renders, and the line-based settings surgery — and
 * `test/generate.test.js` pins the block against the shell tool's own `--print`
 * output on the real catalog.
 *
 * Dependency-free on purpose (node builtins only): a package that needed a
 * checkout to build could not be the fallback for a checkout being absent.
 *
 * @module dsh-tinytitan/generate
 */
import { execFileSync } from "node:child_process";
import { accessSync, constants, copyFileSync, existsSync, readFileSync, statSync, writeFileSync }
  from "node:fs";
import { homedir } from "node:os";
import { delimiter, join } from "node:path";

/** The block's defaults, copied from `tools/dsh_route.sh`'s own. */
export const ROUTE_DEFAULTS = Object.freeze({
  port: 8080,
  provider: "tinytitan",
  context: 262144,
  maxTokens: 32768,
  reasoning: "medium",
});

/** The binary a built checkout or app ships. */
export const SERVER_BINARY_NAME = "TinyTitanServer";

/**
 * The order the shell tool offers thinking levels in.
 *
 * The catalog reports the levels a template renders, but not necessarily in
 * pi-ai's order; the shell parser re-sorts them by this list (keeping anything
 * newer, in catalog order), so the generated block matches its output.
 */
const LEVEL_ORDER = ["off", "on", "minimal", "low", "medium", "high", "xhigh", "max"];

/**
 * The shell's `field()`: a value that can print on one YAML line.
 *
 * The parser that feeds the route skips a model whose id, name or path is
 * missing or multi-line, so a malformed install cannot split the block across
 * lines. `null` is rejected here where Python would print `None`; a catalog is
 * JSON and a null there is a bug, not a name.
 */
function textField(value) {
  const text = value === undefined || value === null ? "" : String(value);
  if (text === "" || /[\t\r\n]/.test(text)) throw new Error("empty or multi-line field");
  return text;
}

/**
 * The levels of one model, in pi-ai's order, defaulting like the shell parser.
 *
 * The parser does `model.get("thinking") or ["off"]` and then iterates it, so a
 * missing, empty or other-falsy list means `off`, while a *string* is iterated
 * character by character. A truthy non-iterable is a TypeError there and skips
 * the model. Mirroring that keeps the block byte-identical even on a catalog a
 * hand-edit produced.
 */
function catalogLevels(thinking) {
  const listed = [];
  if (Array.isArray(thinking)) {
    if (thinking.length === 0) listed.push("off");
    else for (const level of thinking) listed.push(String(level));
  } else if (typeof thinking === "string") {
    if (thinking === "") listed.push("off");
    else for (const character of thinking) listed.push(character);
  } else if (thinking !== null && typeof thinking === "object") {
    const keys = Object.keys(thinking);
    if (keys.length === 0) listed.push("off");
    else listed.push(...keys);
  } else if (thinking === undefined || thinking === null || thinking === 0 || thinking === false) {
    listed.push("off");
  } else {
    throw new Error("thinking is not iterable");
  }
  return [
    ...LEVEL_ORDER.filter((level) => listed.includes(level)),
    ...listed.filter((level) => !LEVEL_ORDER.includes(level)),
  ];
}

/**
 * Normalize a catalog's `models` array into the rows the block is built from.
 *
 * GPU installs first, then CPU ones, mirroring the shell parser: a server
 * serves one resident model at a time, so the route only orders the picker, and
 * GPU is what a launcher starts by default. A row missing a field the block
 * needs is skipped rather than failing the whole route.
 *
 * @param models - the catalog's `models` array.
 * @returns one `{id, name, path, backend, family, engines, levels}` per servable model.
 */
export function catalogRows(models) {
  if (!Array.isArray(models)) throw new Error('catalog: "models" is not a list');
  const gpu = [];
  const cpu = [];
  for (const model of models) {
    let row;
    try {
      if (model === null || typeof model !== "object") throw new Error("not an object");
      const backend = textField(model.backend);
      if (backend !== "gpu" && backend !== "cpu") throw new Error(`backend ${backend}`);
      const quant = model.quant;
      if (quant === undefined || quant === null || quant === "" || !Number.isFinite(Number(quant))) {
        throw new Error("quant");
      }
      // Not used by the block, but the shell parser formats it and skips the
      // model when it cannot; dropping the same rows keeps the two in step.
      const size = model.size_gb;
      if (size !== undefined && size !== null && Number.isNaN(Number(size))) {
        throw new Error("size_gb");
      }
      row = {
        id: textField(model.id),
        name: textField(model.name),
        path: textField(model.path),
        backend,
        family: textField(model.family || "-"),
        engines: textField(model.engines || backend),
        levels: catalogLevels(model.thinking),
      };
    } catch {
      // The shell's parser prints "catalog: skipping …" on stderr and carries
      // on; the route is still useful for the installs that did parse.
      continue;
    }
    (row.backend === "gpu" ? gpu : cpu).push(row);
  }
  return [...gpu, ...cpu];
}

/**
 * One model's `reasoningEfforts` map.
 *
 * pi-ai's level vocabulary has no `on`, while a binary-thinking template renders
 * exactly off|on. Such a family's thinking mode is therefore offered as
 * `medium` with the wire value `on`, so the picker shows one thinking choice and
 * TinyTitan reads `on`.
 */
function effortsLines(levels) {
  const lines = ["          reasoningEfforts:"];
  for (const raw of levels) {
    const level = raw.replace(/ /g, "");
    if (level === "") continue;
    if (level === "off") lines.push("            off:");
    else if (level === "on") lines.push("            medium: on");
    else lines.push(`            ${level}: ${level}`);
  }
  return lines;
}

/** Build the block from already-normalized rows. */
function buildBlock(rows, options) {
  const port = String(options.port ?? ROUTE_DEFAULTS.port);
  const provider = String(options.provider ?? ROUTE_DEFAULTS.provider);
  const context = String(options.context ?? ROUTE_DEFAULTS.context);
  const maxTokens = String(options.maxTokens ?? ROUTE_DEFAULTS.maxTokens);
  const reasoning = String(options.reasoning ?? ROUTE_DEFAULTS.reasoning);

  const lines = [
    `# DeepSeek Harness route to the TinyTitan server on port ${port}.`,
    "# Generated by tools/dsh_route.sh from the installs under models/.",
    `# ${rows.length} served model(s); settings.yaml is hot-reloaded.`,
    "llm-pi-ai:",
    "  providers:",
    `    ${provider}:`,
    "      displayName: TinyTitan",
    "      api: openai-completions",
    `      baseURL: http://127.0.0.1:${port}/v1`,
    '      # pi-ai refuses a keyless route ("No API key for provider");',
    "      # TinyTitan has no authentication and ignores the header.",
    "      headers:",
    "        authorization: Bearer tinytitan-local",
    "      # Level for calls that name none: compaction and session titles.",
    "      # Use `off` to keep those unthinking, or install plugins/dsh-tinytitan,",
    "      # which forces it for them without turning chat off.",
    `      reasoning: ${reasoning}`,
    "      # TinyTitan emits nothing until the first token; pi-ai's own default",
    "      # abandons an idle stream after five minutes.",
    "      streamIdleTimeoutMs: 3600000",
    `      defaultContextWindow: ${context}`,
    `      defaultMaxTokens: ${maxTokens}`,
    "      models:",
  ];
  for (const row of rows) {
    lines.push(
      `        - id: ${row.id}`,
      `          name: ${row.name}`,
      `          contextWindow: ${context}`,
      `          maxTokens: ${maxTokens}`,
      ...effortsLines(row.levels),
      "          compat:",
      "            # The only place TinyTitan reads the thinking switch; pi-ai's",
      "            # `qwen` format sends it top-level, where TinyTitan ignores it.",
      "            thinkingFormat: chat-template",
      "            chatTemplateKwargs:",
      "              enable_thinking: { $var: thinking.enabled }",
      "              reasoning_effort: { $var: thinking.effort }",
      "            maxTokensField: max_tokens",
      "            supportsUsageInStreaming: true",
    );
  }
  return `${lines.join("\n")}\n`;
}

/**
 * The `llm-pi-ai` block for a catalog, byte-for-byte as the shell tool prints it.
 * @param models - the catalog's `models` array.
 * @param options - `port`, `provider`, `context`, `maxTokens`, `reasoning`.
 * @returns the block, ending in a newline.
 */
export function generateBlock(models, options = {}) {
  return buildBlock(catalogRows(models), options);
}

/**
 * Whether a comment line belongs to a previous run's generated header.
 *
 * The header sits *above* `llm-pi-ai:`, so the section replacement below has to
 * remove it explicitly; without that, every refresh (the plugin runs at every
 * harness boot) would leave three more stale comment lines behind.
 */
export function generatedHeader(line) {
  const stripped = String(line).trim();
  if (stripped.startsWith("# DeepSeek Harness route to the ")) return true;
  if (stripped.startsWith("# Generated by tools/dsh_route.sh ")) return true;
  return /^# \d+ served model\(s\); settings\.yaml is hot-reloaded\.$/.test(stripped);
}

/** Split keeping line endings, the way Python's `splitlines(keepends=True)` does. */
function splitKeepingEnds(text) {
  if (text === "") return [];
  return text.match(/[^\n]*\n|[^\n]+$/g) ?? [];
}

/**
 * Replace the `llm-pi-ai` section in a settings file with the generated block.
 *
 * Line-based surgery, not a YAML round-trip: the file is the person's, with
 * their comments, and a parse-and-dump would rewrite all of it. A section ends
 * at the next line that starts in column 0 and is not a comment or blank. The
 * replacement starts at `llm-pi-ai:`, which is *below* the block's own three-line
 * header, so the previous refresh's header is removed explicitly rather than
 * left orphaned. When there is no section the block is appended.
 *
 * CRLF is normalized to LF first because the shell tool reads the file through
 * Python's universal newlines and rewrites it with LF, so a route refresh
 * normalizes the whole file; matching that keeps the two implementations
 * byte-identical on a file edited elsewhere.
 *
 * Pure: it returns the new text and writes nothing, so `writeRouteSettings` can
 * back the file up first and tests can compare a rewrite byte-for-byte.
 *
 * @param settingsText - the settings file.
 * @param block - the generated block.
 * @returns the settings text, with exactly one generated block.
 */
export function applyRouteToSettings(settingsText, block) {
  const body = String(block).endsWith("\n") ? String(block) : `${block}\n`;
  const lines = splitKeepingEnds(String(settingsText).replace(/\r\n|\r/g, "\n"));
  const out = [];
  let index = 0;
  let replaced = false;
  while (index < lines.length) {
    const line = lines[index];
    if (line.startsWith("llm-pi-ai:")) {
      index += 1;
      while (index < lines.length) {
        const following = lines[index];
        if (following.trim() === "" || following.startsWith(" ") || following.startsWith("\t")) {
          index += 1;
          continue;
        }
        break;
      }
      while (out.length > 0 && out[out.length - 1].trim() === "") out.pop();
      while (out.length > 0 && generatedHeader(out[out.length - 1])) {
        out.pop();
        while (out.length > 0 && out[out.length - 1].trim() === "") out.pop();
      }
      if (out.length > 0) out.push("\n");
      out.push(body);
      replaced = true;
      continue;
    }
    out.push(line);
    index += 1;
  }
  if (!replaced) {
    while (out.length > 0 && out[out.length - 1].trim() === "") out.pop();
    if (out.length > 0) out.push("\n");
    out.push(body);
  }
  return out.join("");
}

/**
 * Write the block into a settings file, after backing it up.
 *
 * A refresh that would change nothing writes nothing and makes no backup, so
 * the plugin's per-boot run does not pile up identical `.bak-*` files.
 *
 * @param options - `settingsPath`, `block`, `stamp`, `backup`.
 * @returns `{settingsPath, backup, changed}`.
 * @throws when the settings file does not exist — the shell tool refuses too,
 *   because creating a settings file from nothing would guess at the profile's
 *   other configuration.
 */
export function writeRouteSettings({
  settingsPath,
  block,
  stamp = new Date().toISOString().replace(/[:.]/g, "-"),
  backup = true,
} = {}) {
  if (!settingsPath || !existsSync(settingsPath)) {
    throw new Error(`no DSH settings file at ${settingsPath ?? "(no path)"}`);
  }
  const before = readFileSync(settingsPath, "utf8");
  const after = applyRouteToSettings(before, block);
  if (after === before) return { settingsPath, backup: null, changed: false };
  let backupPath = null;
  if (backup) {
    backupPath = `${settingsPath}.bak-${stamp}`;
    copyFileSync(settingsPath, backupPath);
  }
  writeFileSync(settingsPath, after);
  return { settingsPath, backup: backupPath, changed: true };
}

/** Whether a path is a file this process may execute. */
export function defaultIsExecutable(path) {
  try {
    if (!statSync(path).isFile()) return false;
    accessSync(path, constants.X_OK);
    return true;
  } catch {
    return false;
  }
}

/** Whether a path is a directory. */
export function defaultIsDirectory(path) {
  try {
    return statSync(path).isDirectory();
  } catch {
    return false;
  }
}

/** The first `name` on `env.PATH`, or null. Injectable so tests need no real PATH. */
export function pathLookup(name, env = process.env, isExecutable = defaultIsExecutable) {
  for (const directory of String(env.PATH ?? "").split(delimiter)) {
    if (directory === "") continue;
    const candidate = join(directory, name);
    if (isExecutable(candidate)) return candidate;
  }
  return null;
}

/**
 * Find a TinyTitan server binary.
 *
 * Explicit config wins, then `TINYTITAN_SERVER`, then PATH, then the app bundle a
 * built app ships, then the checkout's release build. The checkout is last on
 * purpose: an installed app is the newer, user-visible copy, and the catalogue
 * package this fallback exists for usually has no `.build/` at all.
 *
 * @param options - `explicit`, `env`, `repoRoot`, `home`, `isExecutable`.
 * @returns the path, or null when nothing is found.
 */
export function findServerBinary({
  explicit,
  env = process.env,
  repoRoot,
  home = homedir(),
  isExecutable = defaultIsExecutable,
} = {}) {
  const candidates = [];
  if (explicit) candidates.push(String(explicit));
  if (env.TINYTITAN_SERVER) candidates.push(String(env.TINYTITAN_SERVER));
  const onPath = pathLookup(SERVER_BINARY_NAME, env, isExecutable);
  if (onPath !== null) candidates.push(onPath);
  candidates.push(join(home, "Applications", "TinyTitan.app", "Contents", "MacOS", SERVER_BINARY_NAME));
  if (repoRoot) {
    candidates.push(join(repoRoot, ".build", "arm64-apple-macosx", "release", SERVER_BINARY_NAME));
    candidates.push(join(repoRoot, ".build", "release", SERVER_BINARY_NAME));
  }
  for (const candidate of candidates) {
    if (isExecutable(candidate)) return candidate;
  }
  return null;
}

/**
 * Find the directory of installed models.
 *
 * Explicit config wins, then `TINYTITAN_MODELS_DIR`, then `<repoRoot>/models`.
 * There is deliberately no app-bundle guess: the app does not ship models, so a
 * wrong directory would describe nothing rather than describe it wrongly.
 *
 * @param options - `explicit`, `env`, `repoRoot`, `isDirectory`.
 * @returns the path, or null when nothing is found.
 */
export function findModelsDir({
  explicit,
  env = process.env,
  repoRoot,
  isDirectory = defaultIsDirectory,
} = {}) {
  const candidates = [];
  if (explicit) candidates.push(String(explicit));
  if (env.TINYTITAN_MODELS_DIR) candidates.push(String(env.TINYTITAN_MODELS_DIR));
  if (repoRoot) candidates.push(join(repoRoot, "models"));
  for (const candidate of candidates) {
    if (isDirectory(candidate)) return candidate;
  }
  return null;
}

/**
 * Discover a server, read its catalog, and refresh the settings route.
 *
 * The catalog command loads no model: it only enumerates what is installed, so
 * this is safe to run at every harness boot.
 *
 * @param options - resolved config (`port`, `provider`, `repoRoot`, `dshHome`,
 *   optional `serverBinary`/`modelsDir`), plus injectable `env`, `run`, `log`,
 *   `home`, `isExecutable`, `isDirectory`, `stamp` and `backup` for tests.
 * @returns `{status, detail, …}` with status `written-self-contained`, `missing`
 *   or `failed`.
 */
export function generateRoute({
  port,
  provider,
  context,
  maxTokens,
  reasoning,
  repoRoot,
  serverBinary,
  modelsDir,
  dshHome,
  settingsPath,
  env = process.env,
  run = execFileSync,
  log = () => {},
  home = homedir(),
  isExecutable = defaultIsExecutable,
  isDirectory = defaultIsDirectory,
  stamp = new Date().toISOString().replace(/[:.]/g, "-"),
  backup = true,
} = {}) {
  const binary = findServerBinary({ explicit: serverBinary, env, repoRoot, home, isExecutable });
  if (binary === null) {
    const detail = "no TinyTitan server binary (set serverBinary or TINYTITAN_SERVER)";
    log(`dsh-tinytitan: ${detail}; leaving the llm-pi-ai route as it is`);
    return { status: "missing", detail, serverBinary: null, modelsDir: null };
  }
  const directory = findModelsDir({ explicit: modelsDir, env, repoRoot, isDirectory });
  if (directory === null) {
    const detail = "no models directory (set modelsDir or TINYTITAN_MODELS_DIR)";
    log(`dsh-tinytitan: ${detail}; leaving the llm-pi-ai route as it is`);
    return { status: "missing", detail, serverBinary: binary, modelsDir: null };
  }

  let catalog;
  try {
    const stdout = run(binary, ["--catalog", "--models-dir", directory],
      { encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] });
    catalog = JSON.parse(String(stdout));
  } catch (error) {
    const detail = String(error?.stderr ?? error?.message ?? error).trim().split("\n")[0]
      || "the server catalog could not be read";
    log(`dsh-tinytitan: the server catalog failed: ${detail}`);
    return { status: "failed", detail, serverBinary: binary, modelsDir: directory };
  }
  if (catalog === null || typeof catalog !== "object" || !Array.isArray(catalog.models)) {
    const detail = 'the server catalog carried no "models" list';
    log(`dsh-tinytitan: ${detail}; leaving the llm-pi-ai route as it is`);
    return { status: "failed", detail, serverBinary: binary, modelsDir: directory };
  }
  const rows = catalogRows(catalog.models);
  if (rows.length === 0) {
    const detail = `the catalog describes no servable install under ${directory}`;
    log(`dsh-tinytitan: ${detail}; leaving the llm-pi-ai route as it is`);
    return { status: "failed", detail, serverBinary: binary, modelsDir: directory };
  }

  const path = settingsPath ?? join(String(dshHome ?? ""), "settings.yaml");
  if (!existsSync(path)) {
    const detail = `no DSH settings file at ${path}`;
    log(`dsh-tinytitan: ${detail}; leaving the llm-pi-ai route as it is`);
    return { status: "missing", detail, serverBinary: binary, modelsDir: directory,
             settingsPath: path };
  }
  try {
    const block = buildBlock(rows, { port, provider, context, maxTokens, reasoning });
    const written = writeRouteSettings({ settingsPath: path, block, stamp, backup });
    const detail = written.changed ? "written" : "already current";
    log(`dsh-tinytitan: route refreshed with the built-in generator (${rows.length} model(s), ${detail})`);
    return {
      status: "written-self-contained",
      detail,
      serverBinary: binary,
      modelsDir: directory,
      settingsPath: path,
      models: rows.length,
      backup: written.backup,
    };
  } catch (error) {
    const detail = String(error?.message ?? error).trim().split("\n")[0];
    log(`dsh-tinytitan: route refresh failed: ${detail}`);
    return { status: "failed", detail, serverBinary: binary, modelsDir: directory,
             settingsPath: path };
  }
}
