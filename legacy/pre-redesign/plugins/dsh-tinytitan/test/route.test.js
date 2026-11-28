import assert from "node:assert/strict";
import { mkdtempSync, writeFileSync, mkdirSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { findRepoRoot, REPO_ROOT, resolveConfig } from "../src/config.js";
import { registerRoute } from "../src/route.js";

function repo(withScript) {
  const root = mkdtempSync(join(tmpdir(), "dsh-tinytitan-route-"));
  mkdirSync(join(root, "tools"), { recursive: true });
  if (withScript) writeFileSync(join(root, "tools", "dsh_route.sh"), "#!/usr/bin/env bash\n");
  return root;
}

test("the route is refreshed by running the checkout's tool", () => {
  const calls = [];
  const messages = [];
  const result = registerRoute({
    repoRoot: repo(true),
    port: 8096,
    provider: "tinytitan",
    dshHome: "/tmp/dsh-home",
    run: (command, args, options) => {
      calls.push({ command, args, options });
      return "replaced\n";
    },
    log: (message) => messages.push(message),
  });
  assert.equal(result.status, "written");
  assert.equal(calls.length, 1);
  assert.equal(calls[0].command, "bash");
  assert.deepEqual(calls[0].args.slice(1), [
    "--write", "--port", "8096", "--provider", "tinytitan",
    "--settings", "/tmp/dsh-home/settings.yaml",
  ]);
  assert.ok(calls[0].args[0].endsWith("tools/dsh_route.sh"));
  assert.ok(messages.some((message) => message.includes("route refreshed")));
});

test("a checkout without the tool is reported, not fatal", () => {
  const messages = [];
  const result = registerRoute({ repoRoot: repo(false), port: 8080, provider: "tinytitan",
                                 dshHome: "/tmp/dsh-home", log: (m) => messages.push(m) });
  assert.equal(result.status, "missing");
  assert.ok(messages.some((message) => message.includes("leaving the llm-pi-ai route as it is")));
});

test("a failed refresh is reported with the tool's own words", () => {
  const messages = [];
  const failure = new Error("exit 2");
  failure.stderr = "dsh_route: no DSH settings file at /tmp/dsh-home/settings.yaml\n";
  const result = registerRoute({ repoRoot: repo(true), port: 8080, provider: "tinytitan",
                                 dshHome: "/tmp/dsh-home", run: () => { throw failure; },
                                 log: (m) => messages.push(m) });
  assert.equal(result.status, "failed");
  assert.equal(result.detail, "dsh_route: no DSH settings file at /tmp/dsh-home/settings.yaml");
  assert.ok(messages.some((message) => message.includes("route refresh failed")));
});

test("config defaults suit a local server and can be overridden", () => {
  const defaults = resolveConfig();
  assert.equal(defaults.port, 8080);
  assert.equal(defaults.provider, "tinytitan");
  assert.equal(defaults.presetId, "tinytitan");
  assert.equal(defaults.registerRoute, true);
  assert.equal(defaults.writeCompactionPreset, true);
  assert.ok(defaults.dshHome.endsWith(".dsh"));

  const configured = resolveConfig({ port: 8096, provider: "local", presetId: "tinytitan-thin",
                                     registerRoute: false, writeCompactionPreset: false,
                                     adoptDefaultPreset: false, setDefaultWhenUnset: false,
                                     repoRoot: "/repo", dshHome: "/home" });
  assert.equal(configured.port, 8096);
  assert.equal(configured.provider, "local");
  assert.equal(configured.presetId, "tinytitan-thin");
  assert.equal(configured.registerRoute, false);
  assert.equal(configured.writeCompactionPreset, false);
  assert.equal(configured.adoptDefaultPreset, false);
  assert.equal(configured.setDefaultWhenUnset, false);
  // An explicit root is a hint: it is used when it holds the tool, and a stale
  // one falls through to the checkout this test suite lives in.
  assert.equal(configured.repoRoot, REPO_ROOT);
  assert.equal(configured.repoFound, true);
  assert.equal(configured.dshHome, "/home");
});

test("config refuses what it cannot use", () => {
  assert.throws(() => resolveConfig({ port: 0 }), /port must be a port number/);
  assert.throws(() => resolveConfig({ port: "http" }), /port must be a port number/);
  assert.throws(() => resolveConfig({ provider: "  " }), /provider must not be empty/);
  assert.throws(() => resolveConfig({ presetId: "" }), /presetId must not be empty/);
});

test("the checkout is found from a hint, the environment, the module, or the profile", () => {
  const base = mkdtempSync(join(tmpdir(), "dsh-tinytitan-find-"));
  const checkout = join(base, "checkout");
  mkdirSync(join(checkout, "tools"), { recursive: true });
  writeFileSync(join(checkout, "tools", "dsh_route.sh"), "#!/usr/bin/env bash\n");

  // An explicit root that holds the tool.
  assert.deepEqual(findRepoRoot({ explicit: checkout, env: {}, moduleUrl: "file:///nowhere/x.js",
                                  cwd: base }), { root: checkout, found: true });
  // The environment variable.
  assert.deepEqual(findRepoRoot({ env: { TINYTITAN_REPO: checkout }, moduleUrl: "file:///nowhere/x.js",
                                  cwd: base }), { root: checkout, found: true });
  // Walking up from the module (running out of the checkout).
  const moduleUrl = pathToFileURL(join(checkout, "plugins", "dsh-tinytitan", "src", "config.js")).href;
  assert.deepEqual(findRepoRoot({ env: {}, moduleUrl, cwd: base }),
                   { root: checkout, found: true });

  // A profile that recorded where it installed the plugin from.
  const profile = join(base, "profile");
  const installed = join(profile, "node_modules", "dsh-tinytitan", "src");
  mkdirSync(installed, { recursive: true });
  writeFileSync(join(profile, "package.json"), JSON.stringify({
    dependencies: { "dsh-tinytitan": `file:${join(checkout, "plugins", "dsh-tinytitan")}` },
  }));
  assert.deepEqual(
    findRepoRoot({ env: {}, moduleUrl: pathToFileURL(join(installed, "config.js")).href, cwd: base }),
    { root: checkout, found: true });

  // Nothing anywhere: the first candidate is returned, and the caller says so.
  const empty = mkdtempSync(join(tmpdir(), "dsh-tinytitan-empty-"));
  const missing = findRepoRoot({ env: {}, moduleUrl: pathToFileURL(join(empty, "src", "config.js")).href,
                                 cwd: empty });
  assert.equal(missing.found, false);
});
