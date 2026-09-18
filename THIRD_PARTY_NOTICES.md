# Third-party software and model terms

This repository's own source is **MIT** (`LICENSE`, Copyright (c) 2026 André Borchert). This file records
the provenance review behind that, and what would change it.

## No third-party source is included

**Nothing in this repository is copied from another project.** The review on 2026-09-16 compared every
Swift and Python file here against the sister project [TinyTitan](https://github.com/Pummelchen/TinyTitan),
which is Apache-2.0 and carries its own `NOTICE`:

* its modules are `TinyTitan*` and its tools are `prepare_*.py` / `*_reference.py`; there is no shared
  module name and no shared file layout beyond directory conventions;
* of the **four** files whose basenames coincide, **none** shares more than **5 %** of its non-comment
  lines — the measured overlap is zero at the threshold that matters;
* the identifiers the two projects have in common are **format and model vocabulary** — `hiddenSize`,
  `numLayers`, `vocabSize`, `safetensors`, `quantized`, `dequantize`, `tieWordEmbeddings` — which is what
  two independent implementations of the same artifact necessarily share, and several of those names come
  from the model's own `config.json`;
* no file here carries a port marker, an upstream copyright line, or an attribution comment.

The positive evidence is stronger than the absence of matches: the install reader exists **twice in this
repository**, in Swift and in Python, written independently and checked against each other, and its layout
rules were established by reading the artifact rather than the source (`D29`: the `padded_columns = 0` rule
was learned by refusing a reader that demanded a padded width).

## What is shared, and why it carries no obligation

| Unit | Relationship | Obligation |
| --- | --- | --- |
| The **install container format** | An interface. This repository reads and verifies installs; the format is data, and the reader here was derived from the artifact (`D29`) | None. Interfaces and file formats are not the licensed expression, and no upstream code or documentation text is reproduced |
| **Repository conventions** (lower-case `sources/`, one directory per module, the language-feature register) | Ideas, deliberately kept (`docs/repository-layout.md`) | None. Ideas and conventions are not copyrightable subject matter |
| **Model weights** (`Qwen/Qwen3.6-35B-A3B`, Apache-2.0; the M4/M5 targets likewise) | Used, **never redistributed** — `.gitignore` excludes model payloads and the release rules forbid committing them | Attribution, kept in `docs/reference-qwen36-35b-a3b.md`. If weights were ever redistributed, Apache-2.0's attribution and NOTICE terms would apply to them |

## The two install formats are not interchangeable, re-verified 2026-09-17

`AGENTS.md` states that *"its int4 is **unsigned with a bias** while this repository's install container is its
own — signed codes, fp32 scales, int8 zero points — so a file from one is not readable by the other"*. That is a
claim about a third party's format, so it is worth recording what it rests on. Re-read in the sister checkout on
2026-09-17:

* **Its int4 is unsigned, with a bias, and the bias is BF16.** Its own reference says so in the first line of the
  function that consumes the format: `tools/qwen35_reference.py`, `dequantize()`, *"`bits`-wide unsigned lanes
  packed low-first inside each uint32, one BF16 scale and bias per group"* — the codes are masked, widened and
  used directly, and the value is `grouped * scales + biases`. Its Swift side names the same thing:
  `Sources/TinyTitan/Infrastructure/ModelIO/Quantization.swift` has `dequantizeInt4Affine`.
* **Ours is signed, with an int8 zero point, and fp32 scales.** `tools/install_reader.py` reads each four-bit
  code through `_signed()` — two's complement in four bits — and computes `(code - zero) * scale`, with the
  per-group zero stored as an `int8` and the scale as an `fp32` (`docs/m0c-quantization.md`, `D39`).
* **Its container family is `GTurbo*V1`**: `GTurboFormatV1`, `GTurboExpertV1`, `GTurboLayerV1`,
  `GTurboManifestArchV1` and `GTurboManifestFileV1` in the same checkout. Ours is `install.json` plus
  `data.bin`.

The conclusion is therefore not a matter of opinion but of arithmetic: the same bytes mean different numbers
under the two rules, and the bias types differ besides. **No defect is to be reported upstream**, because its
reader and its converter agree with each other — the finding is only that the formats are different, which is
what makes the two projects independent implementations rather than copies. This re-check confirms the review of
2026-09-16 and adds the file-level evidence it did not name.

## The NOTICE requirement, as it stands

Apache-2.0 §4(d) requires a distributor to reproduce a `NOTICE` file **when the distribution includes the
licensed material**. This repository includes none, so **no `NOTICE` is required here** — and the sister
project's own `NOTICE` (which names *turbo-fieldfare*, Apache-2.0) does not transfer, because that is
material they include rather than material anyone here does.

**What would change that answer**, recorded so the next contributor does not have to repeat the review:

1. if any file is ever ported from the sister project or another Apache-2.0 source, the obligations are
   attribution, a copy of the licence, their `NOTICE` material, and a statement of the changes made;
2. if model weights are ever redistributed, the model's own licence and attribution terms apply to them;
3. if a third-party dependency is ever vendored rather than resolved, its licence belongs in this file.

`tools/check_provenance.py` checks what can be checked offline: that this file exists, that it still names
the relationship it describes, and that no source file has acquired a third-party copyright header.

## The sister project's code is available to take — and what that obliges

On 2026-09-18 the operator authorised taking code from
[TinyTitan](https://github.com/Pummelchen/TinyTitan) rather than only reading it for approach. That permission
does not change this repository's licence, and it does not make the material free of obligations. TinyTitan is
**Apache-2.0**; this repository is **MIT**. Apache-2.0 permits use inside an MIT work, and §4 requires:

- a readable copy of the **NOTICE** it ships — `TinyTitan`, Copyright (c) 2026 André Borchert, together with the
  `turbo-fieldfare` notice it carries — which is why a root `NOTICE` file appears in the commit that first
  takes code, and not before it;
- the **licence text** of the work taken from;
- **prominent notices on modified files**, stating that they were changed.

**As of `D100` nothing has been copied yet**, so the statements above about this repository's own position still
hold as written. The commit that takes the first file lands all four things together — the code, the `NOTICE`,
this section's update, and the gate change in `tools/check_provenance.py`, which today *refuses* a third-party
copyright line and will have to *require* the attribution instead. A gate that forbids what the operator has
allowed is a gate that will be disabled in a hurry; `D100` records that it is changed deliberately, in the same
commit, with the reason.

