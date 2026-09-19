# dsh-tinytitan

A [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) bundle for
running models from a local **TinyTitan** server. It is deliberately thin: the
harness's own `llm-pi-ai` adapter serves these models, so this package adds
**configuration**, not a protocol implementation.

Two jobs, both at boot, both idempotent:

1. **Keeps the route current.** `tools/dsh_route.sh` in this checkout turns the
   installed models under `models/` into the `llm-pi-ai` route block — served
   ids, each template's thinking levels, and the three switches that are easy to
   get wrong by hand (`thinkingFormat: chat-template`, the keyless-route auth
   header, the long stream idle timeout). The plugin runs it, so the harness's
   model picker follows `models/` instead of a copy someone typed once.
   A plugin installed from a catalogue is a plain package beside no checkout, so
   there is no script to run: only then the same block is generated in-process
   from the server's own catalog (`generate.js`), and the log says so. Wherever
   `tools/dsh_route.sh` exists it stays the source of truth, so a checkout user
   has one implementation, not two.
2. **Mounts a compaction backend that does not think.** Compaction and session
   titles are marked `purpose: "compaction"` / `"session-title"` and name no
   reasoning level, so the harness fills in the route's default. On a local
   thinking model that spends a summariser's own output cap on thinking and
   costs tens of seconds on the title of every new session. The generated agent
   preset points that row at `dsh-tinytitan/backend`, which forces thinking off for
   those calls only — ordinary turns keep the route's level.

## Install

```sh
dsh plugin --profile web add file:/path/to/TinyTitan/plugins/dsh-tinytitan
```

Restart DSH (or start a new session) and the plugin logs what it did. Changes it
makes, each with a timestamped backup beside the original:

| File | Change |
|---|---|
| `~/.dsh/settings.yaml` | the `llm-pi-ai` route block, refreshed from the catalog |
| `~/.dsh/.agent-presets/tinytitan/agent.cordis.yml` | generated from the shipped `standard` preset, with the compaction row on this backend |
| `~/.dsh/settings.yaml` | `agent-presets.default: tinytitan` — **only** when the file names no default |
| the *current* default preset | its stock `compaction-basic` row, re-pointed (never a preset that is not a user file) |

The last one is `adoptDefaultPreset`: if you already chose a preset, that row is
adopted rather than your choice being overwritten. Set it to `false` to leave
every file you own untouched — then the plugin only writes its own `tinytitan`
preset, which you select on the Agent presets page.

A `file:` install is a copy, not a link: after editing this package, re-install
it (`dsh plugin --profile web remove dsh-tinytitan` then `add` again) or DSH keeps
running the copy it made.

## Configure

Every field is optional; these are the defaults the `cordis.patch.yml` row writes
out, and `TINYTITAN_PORT` / `TINYTITAN_REPO` / `TINYTITAN_SERVER` /
`TINYTITAN_MODELS_DIR` / `DSH_HOME` are the environment fallbacks.

| Field | Default | Meaning |
|---|---|---|
| `port` | `8080` | the port the TinyTitan server serves on |
| `provider` | `tinytitan` | the `llm-pi-ai` provider route name |
| `presetId` | `tinytitan` | the agent preset this plugin generates |
| `registerRoute` | `true` | refresh the route block from `tools/dsh_route.sh` |
| `selfContained` | `false` | use the built-in generator even where `tools/dsh_route.sh` exists |
| `serverBinary` | discovered | the `TinyTitanServer` the built-in generator runs (`$TINYTITAN_SERVER`) |
| `modelsDir` | `<repoRoot>/models` | the installs it describes (`$TINYTITAN_MODELS_DIR`) |
| `writeCompactionPreset` | `true` | generate the preset / adopt the default's row |
| `adoptDefaultPreset` | `true` | re-point the current default preset's stock row |
| `setDefaultWhenUnset` | `true` | set `agent-presets.default` only when absent |
| `repoRoot` | this checkout | where `tools/dsh_route.sh` lives |
| `dshHome` | `$DSH_HOME` or `~/.dsh` | settings and presets |

The built-in generator looks for the server at `serverBinary`, then
`TINYTITAN_SERVER`, then `TinyTitanServer` on `PATH`, then
`~/Applications/TinyTitan.app/Contents/MacOS/TinyTitanServer`, then the checkout's
release build; it looks for models at `modelsDir`, then `TINYTITAN_MODELS_DIR`,
then `<repoRoot>/models`. It refuses to write when the settings file does not
exist, and it makes no backup when the refresh would not change a byte.

## What it does not do

- **No adapter.** It registers no LLM provider: the harness's `llm-pi-ai` route
  does the work, so a harness upgrade cannot leave a copied protocol
  implementation behind.
- **No budgets, no vision, no dialects.** TinyTitan accepts `reasoning_budget_tokens`
  and does not enforce it; the models are text-only; llama.cpp/TabbyAPI are other
  servers with their own routes. None of that is here.
- **No session-title override.** Titles are issued by a host-plane plugin with
  its own context, which a bundle cannot wrap; that half is upstream ask 1 in
  `docs/dsh-upstream-asks.md`. Until it lands, the route's `reasoning: off` is
  the only way to keep titles unthinking — at the cost of chat starting
  unthinking too.

## Uninstall

```sh
dsh plugin --profile web remove dsh-tinytitan
rm -rf ~/.dsh/.agent-presets/tinytitan
```

Then re-point the preset you keep at `@deepseek-ai/dsh-compaction-basic` (a
`.bak-*` file beside it has the row as it was), and drop
`agent-presets.default: tinytitan` from `~/.dsh/settings.yaml` if this plugin set it.

## Test

```sh
cd plugins/dsh-tinytitan && node --test test/
```

Thirty-four tests, no harness packages required: the compaction seam is exercised
against a stub base, the preset and settings surgery against temporary homes, and
the route call against a stubbed runner. The built-in generator's block is
compared byte-for-byte with `tools/dsh_route.sh --print` — on the real catalog
when the checkout's server binary is built, and on a synthetic catalog (mixed
backends, re-sorted thinking levels) whenever the shell tool is present; both
comparisons skip with a clear message when the pieces are absent, so the suite
stays runnable elsewhere. The `dsh-tinytitan/backend` import itself is resolved
from the profile's own `node_modules` once installed.

## Licence

**MIT, deliberately.** The repository it lives in is MIT as well, and this package
states it on purpose: it is an independent work that talks to the server over its
public HTTP API and copies no code from the project's lineage, so it stays under
the licence its own author chose. Do not "align" it with the repository's
`LICENSE` — that would be a claim about provenance this package does not make.

## Publishing it

`docs/dsh-plugin-publication.md` in this repository is the research: how a DeepSeek
Harness plugin is distributed, the community catalogue that lists it, and exactly
what this package still needs before it can be submitted.
