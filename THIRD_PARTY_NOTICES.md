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
