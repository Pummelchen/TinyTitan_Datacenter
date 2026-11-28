# Internal speeds

One record per release, measured with `tools/internal-speeds.py`, so a speed
regression is a diff rather than a feeling. The numbers are TinyTitan's own —
kernel bandwidth, prefill and decode rates, time to first token, and a small
quality proxy on a fixed prompt — measured on the same machine and the same
model the golden baselines use.

```bash
tools/internal-speeds.py --record --label v5.6     # measure and write v5.6.json
tools/internal-speeds.py --record --label v5.6 --baseline benchmark/internal-speeds/v5.5.json
tools/internal-speeds.py --compare benchmark/internal-speeds/v5.5.json \
                                 benchmark/internal-speeds/v5.6.json
```

`--record` compares against the newest previous record **for the same model and
prompt** (or the explicit `--baseline`), prints the diff, and exits non-zero when
a performance metric regressed by more than 10% (`--threshold`). That non-zero
exit is the release gate; see `docs/release-process.md`.

## What is measured

| Field | What it is |
| --- | --- |
| `gpu.qkv_gemv_gbps` | the gated QKV GEMV (the dominant decode kernel), synthetic buffers |
| `gpu.routed_moe_gbps` | the routed-MoE decode kernels at the real shapes |
| `gpu.gdn_inproj_gbps` | the gated-DeltaNet in-projection GEMV |
| `cpu.best_gbps` | int8 affine GEMV on the CPU at its best thread width, with the full table |
| `generation.prefill_tokens_per_second` | prompt tokens / prefill seconds |
| `generation.decode_tokens_per_second` | the CLI's own decode rate |
| `generation.ttft_seconds` | time to first token = prefill seconds (the first token is sampled at the end of prefill) |
| `generation.effective_decode_gbps` | weight bytes × decode tokens / decode seconds — every token re-reads the weights, so this is the model-level bandwidth |
| `ane.prefill_tokens_per_second` | only when the model ships an `ane_prefill` sidecar; otherwise recorded as not applicable, with the model's family and whether the exporter could serve it |
| `ane.effective_prefill_gbps` | the model's declared bytes × chunks / prefill seconds. End-to-end, not ANE-only: the ANE attends and the routed experts still run on the GPU, and one wall time cannot separate them |
| `ane.chunks`, `ane.model_total_bytes` | the ingredients of that bandwidth; `model_total_bytes` is the manifest's own total, so a MoE's `packed_experts/` counts (AgentWorld 4-bit: 1.92 GB of 20.08 GB is `model_weights.bin`) |
| `quality.*` | the response, its SHA-256, keyword coverage and trigram repetition |

The prompt is fixed — *"difference swift vs c++ in detail"* — and generation is
greedy, so the response is deterministic: if `quality.response_sha256` changes,
the output changed and the comparison says so. Coverage is a proxy, not a grade;
the response text is stored so a human can read the one release where it drops.

## Which model

The default is the dense Qwen 3.5 4B (`models/qwen3.5_4B_4Bit`) because it is
the smallest install that exercises every path, and it is what the release gate
records. Records are per (model, prompt), so other models can be recorded beside
it without becoming its baseline:

```bash
tools/internal-speeds.py --record --label v5.6-4b
tools/internal-speeds.py --record --label v5.6-agentworld \
  --model models/qwen-agentworld_35B_A3B_4Bit
```

Without `--label`, the file is named from `git describe` — bare for the default
model, and suffixed with the model's directory name for any other, so a second
model can never overwrite the 4B's record.

**The ANE row needs a sidecar, and a sidecar needs a family the graph can
describe.** `tools/export_ane_prefill.py` reads the attention geometry from the
model's own manifest, so it serves the qwen36 MoE family, the dense Qwen 3.5
family and Qwen 3.8. The last of those selects keys with a QSA indexer rather
than computing different arithmetic, and the runtime folds that selection into
the additive mask the sidecar already takes as an input, so the graph is
unchanged — and it is still **not installed**, because it was measured: the ANE
runs 0.72× the GPU's prefill on that model, since the GPU already attends to only
the indexer's ~2,051 selected keys while the ANE graph is dense over the context.
The family left out entirely is the one-layer MTP draft, which is verified rather
than prefilled on the ANE. The record says which of the two it is rather than
quoting a number the model would never produce.

The ANE also needs a **4,096-token prefill chunk** and a prompt that fills one:
a shorter prompt is one partial chunk and deliberately stays on the GPU, and the
dense family is on 4,096 by default for exactly this reason. Export first, then
record:

```bash
~/.venvs/coreml-py311/bin/python tools/export_ane_prefill.py \
  --model models/qwen3.5_4B_4Bit --max-history 12288
tools/internal-speeds.py --record --label v5.6-4b \
  --model models/qwen3.5_4B_4Bit
```

The sidecar lives in the model directory (`<model>/ane_prefill/`) and is not in
the manifest, so it does not disturb the install receipt. Never download or
re-install a model to obtain a record — a target with no install is reported as
**not checked**.

## Reading a comparison

A metric moving by a few percent is noise (Metal timing varies run to run).
The gate fires at 10%: a drop in any `*_gbps` or `*_tokens_per_second`, or a
rise in `ttft_seconds`, `decode_seconds` or `total_seconds`. A changed response
hash is printed as a note rather than a failure, because a deliberate numerics
change legitimately changes it.

## Environment

A record is valid for one (machine, build, model) triple, like the golden
baselines. Record on the release machine, name the install in the `model`
field, and never compare a record from one Mac with one from another.
