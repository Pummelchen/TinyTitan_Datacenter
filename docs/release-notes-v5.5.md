## TinyTitan 5.5 — the name, structured output, and a harness route

First release under the project's own name. Carries the two features that landed
after 5.4 — JSON the sampler is not allowed to leave, and a thinking switch that
belongs to the request instead of to load time — plus the launcher, client and
DeepSeek Harness work around them. Everything here was already on `main`.

### The project is TinyTitan

- One mechanical rename — `NVMAI_`→`TINYTITAN_`, `NVMAI`→`TinyTitan`, `nvmai`→`tinytitan` — across the package, all 28 SwiftPM targets and their 581 paths, the executables, env vars, launcher, benchmark scripts, DSH bundle, docs, wiki and repository. The old repository URL redirects.
- Binaries are now `TinyTitanServer`, `TinyTitanMac`, `TinyTitanCLI`, `TinyTitanRepack`, `TinyTitanDecodeService`, `TinyTitanBench`. A stale `.build/` may still hold the old `NVMAI*` executables; nothing updates them.
- Env vars are `TINYTITAN_*` (`TINYTITAN_PORT`, `TINYTITAN_MODELS_DIR`, `TINYTITAN_CLIENTS`, …).
- Installs as `TinyTitan.app`; the archive is `tinytitan-5.5-macos-arm64.tar.gz`. Earlier versions keep the name they shipped under.
- Fixed by hand after the pass: the wordmarks now read `Tiny` + `Titan` with a widened canvas, and the app icon is the brand image clipped to the macOS rounded square (`tools/make_app_icon.py`). One defect the pass introduced — a resource-bundle glob that matched nothing, because bundles are named after the *package* plus target — was found and fixed before release.

### Structured output is enforced by a grammar

- `response_format` (Chat Completions), `text.format` (Responses) and `output_config.format` (Messages) are no longer refused: a named JSON format compiles into a **byte-level grammar that masks the sampler on both engines**, so every token comes from the set that keeps the document inside the schema. The schema removes what is forbidden; among what remains the model's distribution still decides.
- Supported subset is explicit: `type`, `properties`, `required`, `additionalProperties`, `items`, `enum`, `const`. Everything outside it (`$ref`, `anyOf`, `pattern`, numeric bounds, tuple `items`, …) is refused by name at request time, and unsatisfiable shapes are refused with the reason. Special tokens carry no bytes and are never allowed.
- Thinking is off for a constrained request; MTP is skipped, because a draft ahead of the sampler never writes the logits a mask would edit.
- One whitespace-only token is allowed between structural tokens and a second in a row is not. A response truncated by `max_tokens` is a truncated document — content correctness is still the model's.
- Check: verified on a real install on both engines and through all three surfaces — [`docs/structured-output.md`](structured-output.md).

### Thinking belongs to the request, on all three surfaces

- `/v1/messages`: `thinking.disabled` is a real off, `adaptive` still means "you decide", and `thinking.enabled` maps Anthropic's `budget_tokens` onto the ladder the OpenAI path already uses (under 4k `low`, under 16k `medium`, else `xhigh`). Anthropic's own budget rules stay refusals.
- Chat Completions gained the same per-request control through `chat_template_kwargs.enable_thinking` and `reasoning_effort`, and reports reasoning tokens in usage.
- A server loaded with thinking on can be told to think less — or not at all — for one turn, without a restart.

### `developer` is the system turn, not an HTTP 500

- A leading `developer` message — the role that replaced `system` — was handed to templates that define only `system`/`user`/`assistant`/`tool`, so `raise_exception('Unexpected message role.')` surfaced as HTTP 500. Harnesses that switch to `developer` once a model reasons hit it.
- The role now renders as `system` in both prompt paths, and a leading `developer` message takes the effort instruction into itself.

### The launcher offers only what is installed, and warns above 40% of RAM

- The built-in fallback filters itself against the install directories (the same check applies to a stale `TINYTITAN_CATALOG_JSON`); families and widths left out are named in one line, an empty `models/` is a hard error carrying the install runbook, and an uninstalled width answers with the widths that are.
- The RAM recommendation moved from half to **40% of physical memory** (floored to whole GB), because the expert cache is wired and cannot be paged out.
- An explicit `--ram` above that line is warned about in bold red — swapping, instability, slower tokens — and used anyway, because it is the operator's call. The default path keeps the install's measured profile, still clamped to half of physical memory.

### One client catalogue for the launcher and the coder harness

- `TINYTITAN_CLIENTS` in `tools/tinytitan_models.sh` is now the single list; the launcher menu and `benchmark/coder_cli_benchmark.py` both build from it, so they cannot disagree.
- Four `coder` clients — Codex, Claude Code, Qwen Code, OpenCode — and Zed is an `editor`: `--clients zed` is refused and points at the new `--round clients`, which checks every client's wiring **without loading a model**.

### DeepSeek Harness: a generated route, and a thin bundle

- `tools/dsh_route.sh` generates the `llm-pi-ai` route block from the server's own catalog — served ids, each template's thinking ladder, and the three switches that are easy to get wrong by hand (`thinkingFormat: chat-template`, the keyless-route auth header, a stream idle timeout that outlives a cold prefill). `--write` replaces just that section of `~/.dsh/settings.yaml` after a timestamped backup, line-based so comments survive.
- `plugins/dsh-tinytitan/` refreshes the route at boot and generates a compaction preset whose backend forces thinking off for compaction and session titles only. It registers no adapter and copies no protocol implementation, so a harness upgrade cannot leave a stale copy behind. Twenty `node --test` tests.

### Also in this release

- **Six dense Qwen 3.5 installs have golden baselines of their own.** 2B/4B/9B at either width had no target and were declared exceptions; they have targets now and the gate checks them.
- **A route refresh no longer orphans its own header.** The generated block's three comment lines sit above `llm-pi-ai:`, so a section replacement left the previous header behind and every boot added three more. The writer removes its own header, a rewrite is byte-identical, and a regression test pins it.
- **CI scans the Swift runtime.** CodeQL covered only `actions`, `c-cpp` and `python`, and open alerts were zero — which is exactly why the gap was invisible. Swift is scanned by an advanced-setup workflow, weekly and on demand, building outside the checkout on `arm64` because these sources use `Float16`.
- **The coder harness can finish a round against a local model.** A cold prefill pays minutes before the first token and Codex abandoned an idle stream after five, retrying into another cold prefill; the harness sets the stream idle timeout and disables retries for Codex.
- **Every benchmark starts its server through `tools/server_launcher.sh`**, so a stored baseline and a live measurement cannot diverge through a different launch. `--round features` refuses the dense installs by name, because they have no routed experts.
- **The README is one benchmark table** with a reproducible GPU-versus-CPU column for dense Qwen 3.5, a names-only supported-model list, and no per-release callout.
- **The plugin package is publishable metadata-wise**: a `repository` field pointing at `plugins/dsh-tinytitan`, and peer ranges widened to `^0.1.5-rc.2 || ^0.1.6-rc.1`.

### Performance

No performance number was re-measured for this release, and the README table is
unchanged from 5.4's — this release renames, constrains the sampler, fixes
request handling and adds configuration. The grammar masks the logits buffer the
repetition penalty already made, and a request naming no format generates
byte-identically, which the golden baselines re-check rather than a benchmark.

### Verification

Cut from tag `v5.5` on the base M3 with 24 GB: macOS 26.6.2, Swift 6.3.3, Apple M3, 24 GB.

- **`tools/lint.sh`** — all four gates clean: force-cast, func-length (0 baselined, 0 new, 2059 functions scanned), unchecked-Sendable, and the converter expert-order probe.
- **`swift test --no-parallel`** — **1523 tests in 234 suites passed** (119.6 s).
- **Clean scratch release build** — warning-free, 107.9 s, staging the six executables and the `.bundle` resources the runtime loads its kernels from.
- **Golden baselines, byte-identical** — all ten targets installed here: `katcoder-4`, `katcoder-8`, `qwen38-4`, `qwen38-8`, `qwen35-2b-4`, `qwen35-2b-8`, `qwen35-4b-4`, `qwen35-4b-8`, `qwen35-9b-4` and `qwen35-9b-8`.
- **Not checked — no install under `models/`, and none may be fetched to fix that:** `ornith-4`, `ornith-8`, `qwen36-4`, `qwen36-8`, `agentworld-4` and `agentworld-8`. They are absent because the operator deleted those installs to save disk; nothing was downloaded, converted, repacked or re-installed to satisfy this gate.
- `models/` holds eleven installs. The eleventh, `qwen3.8-flash-next_125B_A6B_MTP_4Bit`, is the MTP draft head — a sidecar the covered `qwen38` targets already exercise — declared in `NON_GOLDEN_INSTALLS` with that reason rather than silently unchecked.

These results are from the dry run of this commit; `--publish` repeats every gate
from scratch and rebuilds the archive, which is why the digest and size below are
filled in only at publish time.

**Not re-measured for this release:** every performance number in the README,
including the dense GPU-versus-CPU rows, quoted from the wiki's
One Prompt, Every Model
page.

### Checksum

`tinytitan-5.5-macos-arm64.tar.gz` sha256: `SHA256_PENDING`
`tinytitan-5.5-macos-arm64.tar.gz` size: `ARCHIVE_BYTES_PENDING` bytes
