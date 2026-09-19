# Publishing a DeepSeek Harness plugin

Research for `plugins/dsh-tinytitan/`, done 2026-09-14 against the installed
`@deepseek-ai/dsh` **0.1.5-rc.2**, the upstream repository
[`deepseek-ai/deepseek-harness`](https://github.com/deepseek-ai/deepseek-harness)
at `master`, the npm registry, and the community catalogue
[`awesome-dsh-plugin/awesome-dsh-plugin`](https://github.com/awesome-dsh-plugin/awesome-dsh-plugin).
Everything below is sourced; where something is a third party's convention rather
than an upstream guarantee, it says so.

## 1. Upstream has no plugin registry of its own

There is no "DeepSeek plugin store", no submission endpoint, and no catalogue
file in the upstream repository: a tree search of `deepseek-ai/deepseek-harness`
at `master` (11,905 entries) finds no catalogue, marketplace or plugin registry.
Distribution is npm and git, and **discovery is community-run** (§2).

What upstream does define is the packaging contract, in
[`docs/user/develop/basic/publish.md`](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/user/develop/basic/publish.md):

- **A plugin ships as a bundle**: an npm package whose `package.json` declares
  `dsh.bundle.patch` (a patch file that inserts or overrides plugin rows). A
  package that declares only `dsh.client` is *not* installable as a bundle.
- **A profile is what a user boots**: `$DSH_HOME/profiles/<name>` holds a
  `package.json` (its dependency list plus `dsh.profile.bundles`, ordered) and the
  user's own `cordis.patch.yml`. Bundles compose as layers; later layers win per
  row, and a patch replaces a row's whole `config` rather than deep-merging it.
- **`dsh plugin --profile <name> <args…>` forwards to pnpm** in the profile
  directory. Confirmed three ways: the launcher's own `--help` ("manage a
  profile's plugins by forwarding the remaining arguments to pnpm"), the shipped
  README (`README.md`: "forwards to pnpm in the profile directory"), and the
  command itself (`dsh plugin --profile web` → "plugin needs pnpm arguments to
  forward"). So the install channel is the npm ecosystem: registry, tarball, git
  or a local path.

The same guide names three ways to move code, with a real trade-off:

| Route | Command | Catch |
| --- | --- | --- |
| **npm registry** (recommended) | `npm publish`, then `dsh plugin --profile p add <pkg>` | none for the user — ships prebuilt code, no build permission needed |
| **Tarball** | `pnpm pack`, then `dsh plugin --profile p add ./x-0.1.0.tgz` | manual hand-off |
| **Git host** | `dsh plugin --profile p add github:owner/repo` | fetches **sources, not artifacts**: the author needs a self-contained `prepare` script, and pnpm ≥10 refuses to run it until the user allowlists the package in the profile's `pnpm-workspace.yaml` (`allowBuilds`) — which upstream describes as *permission to execute the package's code at install time*. Pin a commit (`#<sha>`). |

A package with no build step (plain ESM, like this one) has no reason to ask a
user for `allowBuilds`; publishing to npm is the clean route.

## 2. Where discovery happens — the community catalogue

The listing that storefronts and users actually read is
[`awesome-dsh-plugin/awesome-dsh-plugin`](https://github.com/awesome-dsh-plugin/awesome-dsh-plugin)
(15,655 stars, CC0-1.0, updated 2026-09-13; site `awesome-dsh-plugin.com`). It is
**third-party**, not operated by DeepSeek. Its convention, from
[`contributing.md`](https://github.com/awesome-dsh-plugin/awesome-dsh-plugin/blob/main/contributing.md):

- The `README`s are **generated** from `data/plugins/*.yml` — one file per plugin,
  which is the whole submission ("the one-file catalogue PR").
- Filename `<owner>__<repo>.yml`; for a monorepo subpackage,
  `<owner>__<repo>--<path>`, e.g. `Pummelchen__TinyTitan--plugins-dsh-tinytitan.yml`.
- At most **3 entries per PR**.
- Requirements: the repo declares `dsh.bundle` (root, or a
  `packages/`/`plugins/`/`apps/` subpackage — this repository's
  `plugins/dsh-tinytitan/` qualifies); real working code; the repo is **at least
  1 day old**; the **`dsh-plugin` GitHub topic** is set; the description states
  what the plugin does with no marketing language and is checked against the
  code.
- CI checks, in order: entry count, `dsh.bundle` fetched from the repo, repo age,
  then `awesome-lint` plus the site build. A maintainer still reads the repo
  before merging.
- **Screenshots** are optional and are declared in *your* repository — a
  `screenshots.json` beside the package's `package.json` listing 1–8 repo-relative
  image paths — so they can be updated without a PR elsewhere.
- **npm publishing is optional and does not affect listing.** If the package *is*
  published, its `repository` field must point back at the listed repository, or
  the two are not linked (deliberate, so a package cannot attach itself to a repo
  that never claimed it). The npm↔repo mapping is picked up automatically; you
  do not add an npm field to the entry.

Two other indexes exist and mirror the same ecosystem: the npm package
[`dsh-plugin-catalog`](https://www.npmjs.com/package/dsh-plugin-catalog)
("3632 entries", repo `awesome-dsh-plugin/awesome-dsh-plugin`), and the web
storefront `dsh-market` (`@linxin666/dsh-client-ui-community-plugins` publishes
the `community.json` that feeds it, alongside `dsh-plugin-shop`). Both are
community projects; neither is required to be installable.

Useful precedents when authoring: `@dsh-enhanced/hello` is a minimal MIT-licensed
bundle, and `create-dshx` scaffolds a plugin project.

## 3. This plugin against that checklist

| Requirement | State |
| --- | --- |
| `package.json` declares `dsh.bundle.patch` | **yes** (`./cordis.patch.yml`) |
| `cordis.patch.yml` beside `package.json` | **yes** |
| Lives in a subpackage the catalogue's CI looks at (`plugins/`) | **yes** |
| Real, working code | **yes** — 34 `node --test` tests, 0 skipped |
| Repository ≥ 1 day old | **yes** (created 2026-08-02) |
| `repository` field pointing at the listed repo | **yes**, with `directory: plugins/dsh-tinytitan` |
| `dsh-plugin` npm keyword | **yes** (added 2026-09-14, with `deepseek`) |
| `dsh-plugin` **GitHub topic on the repo** | **yes** — set 2026-09-14 (it was the only topic the repository had) |
| Submitted to the catalogue | **yes** — [PR #5094](https://github.com/awesome-dsh-plugin/awesome-dsh-plugin/pull/5094), one file, +6/−0, mergeable |
| Description, accurate, no superlatives | not written yet — §4 has a draft |
| Licence | **MIT, deliberately** — see the plugin README; the repository is Apache-2.0 and this package is an independent work that talks to the server over its HTTP API |
| npm name `dsh-tinytitan` | **unclaimed** (registry returns 404) |
| `screenshots.json` | absent (optional) |

## 4. The submission, ready to copy

`data/plugins/Pummelchen__TinyTitan--plugins-dsh-tinytitan.yml` in the catalogue
repository:

```yaml
url: https://github.com/Pummelchen/TinyTitan_Datacenter/tree/main/plugins/dsh-tinytitan
name: Pummelchen/TinyTitan_Datacenter#dsh-tinytitan
category: model
description:
  en: 'Keeps a local TinyTitan model server reachable from the harness: refreshes the llm-pi-ai route from the installed models and mounts a compaction backend that does not think.'
  zh: '把本机 TinyTitan 模型服务接入 Harness：按已安装模型刷新 llm-pi-ai 路由，并挂载一个不做思考的压缩后端。'
```

The description is quoted because it contains `: ` — unquoted, YAML reads that as
a nested key and the site build fails. `category: model` is the fit: the plugin
configures the model route, it is not UI, tooling or memory. Every claim in the
description is true of the code: the route refresh is `src/route.js` +
`src/generate.js`, and the quiet backend is `src/compaction.js` mounted by the
generated preset.

## 5. Steps, in order

1. **Publish to npm** (optional for listing, useful for storefront download
   counts): `npm publish` from `plugins/dsh-tinytitan`. The package is plain ESM
   with no build step, so what is published is what was tested. Needs an npm
   account with 2FA; the name is unclaimed. The `repository` field is already
   correct. **Not done — it is the operator's account.**
2. ~~**Add the `dsh-plugin` topic**~~ — **done 2026-09-14** on
   `Pummelchen/TinyTitan_Datacenter`.
3. ~~**Open the one-file PR**~~ — **done 2026-09-14**:
   [PR #5094](https://github.com/awesome-dsh-plugin/awesome-dsh-plugin/pull/5094),
   one file, +6/−0. Reviewed by that project's CI and a maintainer; nothing here
   can hurry the merge.
4. **Optionally add `screenshots.json`** to `plugins/dsh-tinytitan/` (1–8 images
   already in this repository) so the storefront shows the plugin rather than
   scraping the README.
5. Keep the entry's `url` correct if the package ever moves; the npm↔repo link
   needs no entry change.

### What was checked before submitting

Run against a fork kept level with the catalogue's `main`, using the catalogue's
own tooling:

- `validateEntries()` over all 3,633 entries — no problems for this entry, and
  the filename matches `slugFor(url)` (`Pummelchen__TinyTitan--plugins-dsh-tinytitan.yml`).
- `node scripts/generate-readme.mjs` — regenerates into both READMEs; the entry
  appears in `README.md` and `README.zh.md`, so locale parity holds.
- `npx awesome-lint` — **79 warnings, 0 errors**, byte-identical warning count to
  a pristine `main` checkout (all pre-existing; a first run in the fork clone
  showed an extra `awesome-github` error that came from that clone's remote
  layout, not the entry — it does not reproduce on a clean upstream clone).
- `node --test scripts/added-dates.test.mjs` — 3 passed.
- `SKIP_PUBLISH_CHECKS=1 node scripts/build-site.mjs` with the READMEs
  regenerated the way `pr-check.yml` does it — **3,633 rows × 2 locales**, with
  detail pages at `docs/p/Pummelchen/TinyTitan_Datacenter--plugins-dsh-tinytitan/` and
  sitemap entries in both locales.

One trap worth recording: `build-site.mjs` parses the **generated READMEs**, not
`data/plugins/*.yml`. A yml-only submission therefore builds a site without its
own entry locally unless the READMEs are regenerated first — which
`pr-check.yml` does for exactly that reason. A local build that reports one row
fewer than `readEntries()` is that, not a dropped entry.

## 6. What is a decision, not a task

- **Publishing to npm** puts a package on a public registry under someone's
  account, with 2FA and ownership consequences. Still open.
- **The catalogue PR** was a public contribution to a third-party repository,
  attributed to the account that opened it; it is opened and now belongs to that
  project's review.
- The two asks in [`dsh-upstream-asks.md`](dsh-upstream-asks.md) are upstream
  *Discussions* (issues are disabled there), and are separate from listing. Still
  unposted.

## Sources

- <https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/user/develop/basic/publish.md>
- `dsh --help`, `dsh plugin --profile web` (installed `@deepseek-ai/dsh` 0.1.5-rc.2)
- <https://github.com/awesome-dsh-plugin/awesome-dsh-plugin/blob/main/contributing.md>
- <https://www.npmjs.com/package/dsh-plugin-catalog> · <https://www.npmjs.com/package/dsh-plugin-shop> · <https://www.npmjs.com/package/@dsh-enhanced/hello> · <https://www.npmjs.com/package/create-dshx>
- npm registry metadata for `dsh-tinytitan` (404 = unclaimed)
