/**
 * Give the harness a compaction preset that uses this plugin's backend.
 *
 * The compaction backend is mounted by an *agent preset*, not by a bundle row,
 * so a plugin that wants one has to write a preset. Two files are involved:
 *
 *   ~/.dsh/.agent-presets/<presetId>/agent.cordis.yml   generated from the
 *                                                       shipped `standard`
 *   ~/.dsh/settings.yaml                                `agent-presets.default`
 *
 * The person's own preset files are never rewritten wholesale: the only edit
 * this module makes to one is the single `compaction-basic` row, and only when
 * that row still names the stock engine. Everything is line-based surgery, so
 * comments survive; every file it changes is backed up first.
 *
 * @module dsh-tinytitan/setup
 */
import { createRequire } from "node:module";
import { copyFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";

/** The backend module a preset row names. */
export const COMPACTION_BACKEND = "dsh-tinytitan/backend";

/** Where the shipped `standard` preset's composition lives. */
export function standardPresetPath() {
  const require = createRequire(import.meta.url);
  const manifest = require.resolve("@deepseek-ai/dsh-agent-presets/package.json");
  return join(dirname(manifest), "presets", "standard", "agent.cordis.yml");
}

/**
 * Point the stock `compaction-basic` row at this plugin's backend.
 *
 * Idempotent: a row that already names this backend is returned unchanged, as
 * is a composition with no such row (nothing to mount into).
 *
 * @param text - the preset composition.
 * @param options - `maxTokens` for the summariser, and the backend module id.
 * @returns the composition, with at most that one row replaced.
 */
export function repointCompactionRow(
  text,
  { maxTokens = 32768, backend = COMPACTION_BACKEND } = {},
) {
  const lines = text.split("\n");
  const out = [];
  let index = 0;
  let changed = false;
  while (index < lines.length) {
    const header = /^(\s*)- id: compaction-basic\s*$/.exec(lines[index]);
    if (header === null) {
      out.push(lines[index]);
      index += 1;
      continue;
    }
    const indent = header[1];
    // The row's own body: blank lines and anything indented deeper than the
    // `- id:` key.
    let end = index + 1;
    while (end < lines.length) {
      const line = lines[end];
      if (line.trim() === "" || line.startsWith(`${indent}  `)) {
        end += 1;
        continue;
      }
      break;
    }
    const body = lines.slice(index, end).join("\n");
    if (body.includes(backend)) {
      out.push(...lines.slice(index, end));
      index = end;
      continue;
    }
    out.push(`${indent}# Mounted by dsh-tinytitan: thinking off for compaction calls.`);
    out.push(`${indent}- id: compaction-basic`);
    out.push(`${indent}  name: ${backend}`);
    out.push(`${indent}  config:`);
    out.push(`${indent}    maxTokens: ${maxTokens}`);
    index = end;
    changed = true;
  }
  return { text: out.join("\n"), changed };
}

/**
 * The preset `agent-presets.default` names, or null when the file names none.
 * @param settingsText - the settings file.
 * @returns the preset id, or null.
 */
export function defaultPreset(settingsText) {
  const lines = settingsText.split("\n");
  const start = lines.findIndex((line) => /^agent-presets:\s*$/.test(line));
  if (start === -1) return null;
  for (let index = start + 1; index < lines.length; index += 1) {
    const line = lines[index];
    if (line.trim() !== "" && !line.startsWith(" ") && !line.startsWith("\t")) break;
    const match = /^\s+default:\s*(\S+)\s*$/.exec(line);
    if (match !== null) return match[1];
  }
  return null;
}

/**
 * Set `agent-presets.default`, when the file does not already name one.
 * @param settingsText - the settings file.
 * @param presetId - the preset to name.
 * @returns the settings text, changed or unchanged.
 */
export function setDefaultPreset(settingsText, presetId) {
  if (defaultPreset(settingsText) !== null) {
    return { text: settingsText, changed: false };
  }
  const lines = settingsText.split("\n");
  const start = lines.findIndex((line) => /^agent-presets:\s*$/.test(line));
  if (start === -1) {
    const base = settingsText.endsWith("\n") || settingsText === "" ? settingsText : `${settingsText}\n`;
    return { text: `${base}\nagent-presets:\n  default: ${presetId}\n`, changed: true };
  }
  lines.splice(start + 1, 0, `  default: ${presetId}`);
  return { text: lines.join("\n"), changed: true };
}

function backup(path, stamp) {
  if (!existsSync(path)) return null;
  const target = `${path}.bak-${stamp}`;
  copyFileSync(path, target);
  return target;
}

/**
 * Generate this plugin's preset and put it — or the current default preset's
 * compaction row — on this backend.
 *
 * @param options - resolved config, plus injectable `standardPath`/`stamp` for tests.
 * @returns what happened, for the log and for tests.
 */
export function ensureCompactionPreset({
  dshHome,
  presetId,
  standardPath = standardPresetPath(),
  adopt = true,
  setDefault = true,
  maxTokens = 32768,
  log = () => {},
  stamp = new Date().toISOString().replace(/[:.]/g, "-"),
}) {
  const result = { preset: null, generated: false, adopted: null, defaultSet: false };
  if (!existsSync(standardPath)) {
    log(`dsh-tinytitan: no standard preset at ${standardPath}; not writing a compaction preset`);
    return result;
  }
  const presetPath = join(dshHome, ".agent-presets", presetId, "agent.cordis.yml");
  const generated = repointCompactionRow(readFileSync(standardPath, "utf8"), { maxTokens });
  if (!existsSync(presetPath) || readFileSync(presetPath, "utf8") !== generated.text) {
    mkdirSync(dirname(presetPath), { recursive: true });
    backup(presetPath, stamp);
    writeFileSync(presetPath, generated.text);
    result.generated = true;
    log(`dsh-tinytitan: wrote the ${presetId} agent preset (${presetPath})`);
  }
  result.preset = presetId;

  const settingsPath = join(dshHome, "settings.yaml");
  if (!existsSync(settingsPath)) {
    log(`dsh-tinytitan: no ${settingsPath}; select the ${presetId} preset yourself`);
    return result;
  }
  const settings = readFileSync(settingsPath, "utf8");
  const active = defaultPreset(settings);
  if (active === null && setDefault) {
    const merged = setDefaultPreset(settings, presetId);
    if (merged.changed) {
      backup(settingsPath, stamp);
      writeFileSync(settingsPath, merged.text);
      result.defaultSet = true;
      log(`dsh-tinytitan: set agent-presets.default to ${presetId}`);
    }
    return result;
  }
  if (active === presetId) return result;
  if (active === null) {
    log(`dsh-tinytitan: no default agent preset; select ${presetId} to mount the backend`);
    return result;
  }
  if (!adopt) {
    log(`dsh-tinytitan: the default agent preset is ${active}; ${presetId} has the quiet backend`);
    return result;
  }
  const activePath = join(dshHome, ".agent-presets", active, "agent.cordis.yml");
  if (!existsSync(activePath)) {
    log(`dsh-tinytitan: the default agent preset ${active} is not a user preset; select ${presetId} to mount the backend`);
    return result;
  }
  const activeText = readFileSync(activePath, "utf8");
  const repointed = repointCompactionRow(activeText, { maxTokens });
  if (!repointed.changed) {
    result.adopted = active;
    return result;
  }
  backup(activePath, stamp);
  writeFileSync(activePath, repointed.text);
  result.adopted = active;
  log(`dsh-tinytitan: pointed the ${active} preset's compaction row at the quiet backend`);
  return result;
}
