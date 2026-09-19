<p align="center">
  <img width="1254" height="1254" alt="TinyTitan Datacenter" src="TinyTitanDatacenter.png" />
</p>

# TinyTitan Datacenter

**TinyTitan Datacenter — short name `ttd` — is a dedicated repository.** It stands on its own: it
builds and runs without any other repository, and it neither links to nor depends on one. It is
built from scratch and is licensed under MIT — see [`LICENSE`](LICENSE).

[![Stars](https://img.shields.io/github/stars/Pummelchen/TinyTitan_Datacenter?style=flat-square&logo=github&label=Stars&color=e3b341)](https://github.com/Pummelchen/TinyTitan_Datacenter/stargazers)
[![Views (14d)](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Pummelchen/TinyTitan_Datacenter/main/.github/traffic.json)](https://github.com/Pummelchen/TinyTitan_Datacenter)
[![Last Commit](https://img.shields.io/github/last-commit/Pummelchen/TinyTitan_Datacenter?style=flat-square&logo=git&label=Last%20Commit&color=2ea44f)](https://github.com/Pummelchen/TinyTitan_Datacenter/commits/main)
[![Contact](https://img.shields.io/badge/Contact-0xa0b1%40gmail.com-blue?style=flat-square&logo=gmail&logoColor=white)](mailto:0xa0b1@gmail.com)

**TinyTitan Datacenter** runs large Mixture-of-Experts language models across a cluster of
Apple-silicon Macs, streaming routed experts from SSD on the
TinyTitan runtime.

What is new in each release lives in this repository's
[releases](https://github.com/Pummelchen/TinyTitan_Datacenter/releases) and in the wiki's
[News](https://github.com/Pummelchen/TinyTitan_Datacenter/wiki/News) page.

<br>


## Benchmarks

Peak decode on a base 8-core M3 MacBook Pro with 24 GB. 
`NA` means the CPU engine
does not serve that model: the MoE families stream their experts on the GPU + ANE path,
and only the dense Qwen 3.5 models run on either engine.

| Model | Quantization | GPU | CPU |
| --- | --- | ---: | ---: |
| Qwen 3.5 2B (dense) | 4-bit | **53.73 tok/s** | **15.42 tok/s** |
| Qwen 3.5 2B (dense) | 8-bit | **32.77 tok/s** | **15.83 tok/s** |
| Qwen 3.5 4B (dense) | 4-bit | **26.18 tok/s** | **7.71 tok/s** |
| Qwen-AgentWorld 35B-A3B | 4-bit | **21.74 tok/s** | NA |
| Ornith 1.5 35B-A3B | 4-bit | **21.65 tok/s** | NA |
| Qwen 3.6 35B-A3B | 4-bit | **21.41 tok/s** | NA |
| KAT-Coder-V2.5-Dev 35B-A3B | 4-bit | **17.86 tok/s** | NA |
| Qwen 3.5 4B (dense) | 8-bit | **16.14 tok/s** | **7.04 tok/s** |
| Qwen 3.5 9B (dense) | 4-bit | **14.93 tok/s** | **4.07 tok/s** |
| Qwen 3.6 35B-A3B | 8-bit | **12.37 tok/s** | NA |
| Qwen-AgentWorld 35B-A3B | 8-bit | **12.28 tok/s** | NA |
| Ornith 1.5 35B-A3B | 8-bit | **11.93 tok/s** | NA |
| Qwen 3.5 9B (dense) | 8-bit | **8.90 tok/s** | **4.51 tok/s** |
| KAT-Coder-V2.5-Dev 35B-A3B | 8-bit | **6.91 tok/s** | NA |
| Qwen3.8-Flash-Next 125B-A6B | 4-bit | **5.46 tok/s** | NA |
| Qwen3.8-Flash-Next 125B-A6B | 8-bit | **2.10 tok/s** | NA |



### Supported LLMs

Every model installs at **4-bit and 8-bit**:

- **Qwen3.8-Flash-Next 125B-A6B**
- **KAT-Coder-V2.5-Dev 35B-A3B**
- **Qwen-AgentWorld 35B-A3B**
- **Ornith 1.5 35B-A3B**
- **Qwen 3.6 35B-A3B**
- **Qwen 3.5 9B**
- **Qwen 3.5 4B**
- **Qwen 3.5 2B**



### Usage

- **Easiest install:** one command checks the Mac, builds TinyTitan, optionally
  downloads a model, and installs a double-clickable Mac app in
  `~/Applications`. Safe to re-run; it updates instead of cloning twice.
  ```bash
  curl -fsSL https://raw.githubusercontent.com/Pummelchen/TinyTitan_Datacenter/main/tools/install_tinytitan.sh | bash
  ```
  From a clone, `tools/install_tinytitan.sh` does the same. See
  [docs/site](docs/site/) for the plain-language article series, or
  `tools/install_tinytitan.sh --help` for its flags.
- **OpenAI-compatible server:** A loopback Chat Completions and Responses API
  for starting TinyTitan and connecting supported coding clients.
- **One server, one port, one launcher:** `tools/server_launcher.sh` starts the
  API on its own, or starts it and opens one of the supported clients — Codex,
  Claude Code, Qwen Code, OpenCode or the Zed editor — wiring that client's
  provider config to the model the server advertises. It asks what to launch
  from one list of every installed model and quantization (GPU and CPU), the
  thinking level that model supports, and an optional RAM limit for the expert
  cache (1/2/4/8/16/32 GB; anything over 40% of the Mac's physical memory is
  warned about in red and used anyway, and the default is the install's own
  measured profile, which the runtime holds to half of physical memory).
  It serves on `127.0.0.1:8080` (`TINYTITAN_PORT` overrides it), and every other
  installed model stays available by name through the API; the server switches
  on demand, keeping one model resident at a time.

```bash
tools/server_launcher.sh                                    # interactive
tools/server_launcher.sh --client codex --model ornith 4     # server + Codex
tools/server_launcher.sh --client zed --model qwen38 4 --ram 8
```

- **Persistent agent memory (optional):** With `TINYTITAN_MEMORY=1` the model gets
  memory that outlives a conversation, scoped per repository, with six memory
  tools the engine answers itself. It runs inside the server process, so there
  is no database to install and nothing to start. Off by default; see
  [docs/agent-memory.md](docs/agent-memory.md).
- **Three client protocols on one server:** OpenAI Chat Completions, the
  OpenAI Responses API (stored responses, `previous_response_id`, the full
  event grammar) and the Anthropic Messages API (`/v1/messages`,
  `count_tokens`, streaming), so Codex, Claude Code and the OpenAI and
  Anthropic SDKs all talk to the same model; see
  [docs/server-api.md](docs/server-api.md).
- **Enforced structured output:** a request may ask for JSON — `response_format`
  on Chat Completions, `text.format` on the Responses API,
  `output_config.format` on Messages — and the server compiles the schema into a
  byte-level grammar that masks the sampler on both engines, so the model can
  only emit a document the schema allows rather than being asked nicely for one.
  The supported schema subset is small and explicit, and everything outside it
  is refused by name; see [docs/structured-output.md](docs/structured-output.md).
- **Tested coding CLIs:** The launch workflow supports Codex, Claude Code, Qwen
  Code, OpenCode and the Zed editor against the local server; the coder benchmark
  scores the four that can be prompted (Claude Code through a loopback Anthropic
  shim) and checks every client's wiring without a model
  (`--round clients`); DeepSeek Harness reaches the server through its own
  `llm-pi-ai` provider route, which `tools/dsh_route.sh` generates from the
  installed models (and `plugins/dsh-tinytitan` keeps current inside the harness,
  adding a compaction backend that does not think) — see
  Connect a client.
- **Mac app and tools:** TinyTitan also provides a native Mac app, direct CLI
  generation, streaming responses, and client-authorized function-tool calls.


### Core Benefits

- TinyTitan streams LLM's faster than any other similar project.
- Run large MOE AI models on low RAM Apple Silicon Macs by keeping the AI model on SSD/NVMe. 
- A 125B model on 8 GB of RAM. TinyTitan streams experts straight from SSD, so model size is bounded by your disk space, not your memory.
- You set the RAM budget. TinyTitan stays inside it. Give it 4 GB or 8 GB — it holds the line, so your Mac stays responsive while the model runs.
- Apple Neural Engine acceleration for prompt processing - 2.3× faster than the GPU cores.
- Our own Metal kernels, our own engine. Purpose-built for Apple silicon and engineered to use your Mac at the physical limit.
- No MLX. No GGUF. TinyTitan ships its own high-speed model format and a converter that builds it straight from the original weights.
  

### Special Features

- **Bounded expert RAM:** The resident expert cache is sized per family from
  the model's own expert stride and clamped to half of physical memory, so a
  smaller Mac is not handed a budget tuned on a larger one. It is wired, so it
  cannot be paged out and everything else the Mac is running has to fit beside
  it: the launcher recommends **40% of physical memory** and warns in red above
  it — swapping, a less stable system and slower tokens — but a larger `--ram`
  is your call and is passed on, and the server's own `--ram-budget` takes
  exactly what it is given. Model state, KV cache, and runtime scratch use
  additional memory.
- **Long context:** Native RoPE supports up to 262K tokens, while optional YaRN
  extends the context to 512K or 1M tokens.
- **Compressed KV cache:** Live attention state can use 16-bit, 8-bit, or 4-bit
  storage independently of the installed model quantization.
- **Thinking mode:** Ornith and Qwen support truthful Off/On reasoning control;
  their chat templates do not define Low, Medium, or High effort levels.
- **MTP off by default:** Native speculative decoding remains experimental and
  disabled because measured Ornith runs showed no speed benefit and it
  currently requires greedy decoding, native RoPE, and prompt-cache reuse off.

### Performance Improvements

- **Tiled Top-K sampling:** Production sampling (Top-K 1–64) runs a
  three-stage tiled GPU reduction, cutting per-token sampling cost from
  15.5 ms to 1.4 ms with a token-for-token identical stream — the main
  source of the v4.6 decode gain.
- **ANE prefill:** `TINYTITAN_PREFILL_ANE=on` runs
  full-attention prefill blocks on the Neural Engine from a one-time
  exported Core ML sidecar, roughly halving long-prompt time to first
  token; short prompts and decode are untouched.
- **Follow-up cache:** Exact live and multi-prefix prompt-state reuse avoids
  repeating compatible prefill work across conversation turns.
- **Concise mode:** An optional terse system prompt reduces generated text for
  workloads that benefit from it; standard responses are the default because
  they generalized more reliably in the coding/tooling qualification.
- **Fast alias:** The chat-only `-fast` model alias strips coding-agent
  boilerplate before prefill for quicker direct answers, while the base alias
  preserves tools and agent loops.


## Core Links

- Getting started
- Features
- Local server and launchers
- Runtime controls
- Benchmarks
- Changelog
- [Repository layout](docs/repository-layout.md) — where everything lives, and
  the naming and file-size conventions

## Credits

TinyTitan is a focused fork of
[drumih/turbo-fieldfare](https://github.com/drumih/turbo-fieldfare), which
provides the bounded-memory runtime, installer, CLI, Mac app, and local server.
The Qwen 3.6 integration was created by
[NeelM0906](https://github.com/NeelM0906) in
[upstream PR #29](https://github.com/drumih/turbo-fieldfare/pull/29). Concise
mode is derived from the
[Nail-Qwen3.6-35B-A3B](https://huggingface.co/peculiar-ragdoll/Nail-Qwen3.6-35B-A3B-MLX)
chat template by [peculiar-ragdoll](https://huggingface.co/peculiar-ragdoll).

## License

MIT License — see [LICENSE](LICENSE). Copyright (c) 2026 André Borchert.

## Contact

Questions, bug reports and suggestions are always welcome. You can contact André Borchert by email at [0xa0b1@gmail.com](mailto:0xa0b1@gmail.com).
