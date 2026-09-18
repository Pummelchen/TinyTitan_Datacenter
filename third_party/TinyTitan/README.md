# TinyTitan — third-party material

This directory holds the licence and notice material that **must** accompany any source taken from
[TinyTitan](https://github.com/Pummelchen/TinyTitan) into this repository. It is staged here in advance of the
first file being taken, on the operator's explicit authorisation (2026-09-18, `D100`), so that the obligation
cannot be met late.

- `LICENSE` — the Apache License, Version 2.0, as the reference distributes it.
- `NOTICE` — the reference's notice, including the `turbo-fieldfare` line it carries forward. Apache-2.0
  section 4(d) requires this notice to travel with any derivative work.

**Discipline for any file taken from the reference** (`D100`, `D123` stage 0):

1. A header naming the source, the licence and the fact that it was modified, e.g.
   `// Derived from TinyTitan (Apache-2.0) — see third_party/TinyTitan/. Modified for this repository.`
2. The file is added to the list in `THIRD_PARTY_NOTICES.md`.
3. `tools/check_provenance.py` is satisfied — it now **requires** this material and the mark, rather than
   forbidding third-party attribution.

Nothing here is a copy of any *source*; only licence and notice text, which is what Apache-2.0 requires to be
redistributed verbatim.
