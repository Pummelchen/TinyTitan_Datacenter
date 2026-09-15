# The M0 gate (DC-026)

**Status: passing.** Run recorded 2026-09-15 on `Qwen/Qwen3.5-2B` at revision
`15852e8c16360a2fea060d615a32b45270f8a8fc`, on one Mac mini M2 with 8 GB.

The gate is two claims, asserted separately because they can fail separately (I3):

1. **the engine against the contract, byte for byte** — two implementations of the same
   stated arithmetic, one in Swift and one in Python. A difference is a bug in one of them.
2. **the engine against the reference implementation, as decisions** — the numbers *cannot*
   match (bit-matching torch is impossible: D3, R13), but the **argmax of the logits must**,
   compared as an index set rather than through a tolerance.

## The frozen prompt set

`tools/m0_prompts.json`, sha256 `3455d25e282de818…`. Token ids are resolved once and
committed, so the gate does not depend on a tokenizer version to be reproducible. Changing
that file changes the gate, and has to be recorded as such.

## The result

| prompt | tokens | engine | contract | oracle | trace bytes | contract | discrete | worst relative | smallest top-1 margin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `capital` ("The capital of France is") | 5 | 17.7 s | 45.0 s | 29.8 s | 7,014,400 | **IDENTICAL** | **MATCH** | 2.03e-06 | 1.1554 |
| `fibonacci` ("1, 2, 3, 5, 8, 13,") | 18 | 36.5 s | 135.7 s | 34.1 s | 25,251,840 | **IDENTICAL** | **MATCH** | 2.05e-06 | 0.0862 |
| `boiling` ("Water boils at a temperature of") | 6 | 17.9 s | 48.4 s | 30.6 s | 8,417,280 | **IDENTICAL** | **MATCH** | 1.91e-06 | 0.0332 |

**40,683,520 bytes of trace data identical across the three prompts**, and every discrete
decision matching the reference implementation.

The margins are the interesting column. They vary by two orders of magnitude between
prompts — 1.1554 on one, 0.0332 on another — while the divergence stays near 2e-06. That is
the whole argument for asserting decisions separately: the safety of a prompt is a property
of the prompt, not of the implementation, and no tolerance on the numbers can tell you which
prompt you are looking at.

## Running it

```bash
.venv/bin/python tools/run_m0_gate.py \
    --snapshot .build/hf-cache/models--Qwen--Qwen3.5-2B/snapshots/<revision> \
    --model Qwen/Qwen3.5-2B --revision <sha>
```

It writes `.build/m0-gate/report.json` and exits non-zero on any failure. `--only <id>` runs
a single prompt.

## What this gate does not cover, and why that is deliberate

- **Long context.** M0's prompt set is short. The brief warns that the Gated DeltaNet's
  recurrence and any sparse-attention selection have failure modes invisible at short
  context; that is `M4`/`M5`'s gate at 128K, and the harness is ready for it.
- **4-bit.** `DC-031` (M0c) quantises, and this gate is the thing that will tell us what
  that costs.
- **Metal.** The engine's kernels are CPU Swift today; the brief's M0 asks for a forward
  pass that bit-matches, and the number of arithmetic steps is what matters, not the device.
  A Metal kernel has to reproduce the same order, and the contract tests are what will say
  whether it does.
