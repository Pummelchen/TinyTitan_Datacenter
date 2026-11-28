# Pre-redesign Retirement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve the existing implementation unchanged under `legacy/pre-redesign/` while removing it as the active root package.

**Architecture:** Relocate the historical project as one unit, retaining its internal relative paths. Keep project-wide identity/legal files and the human-owned README at the root; place new retirement guidance in root-level agent instructions and redesign docs.

**Tech Stack:** Git, Markdown, Swift Package Manager manifest for structural verification only.

**Spec:** `docs/redesign/2026-11-28-pre-redesign-retirement.md`

## Global Constraints

- Do not edit the human-owned root `README.md`.
- Preserve every tracked historical source file byte-for-byte.
- Do not implement the redesign in this change.
- Do not modify the sister TinyTitan checkout.
- Publish to `main` and update the wiki tracker/status.

## Review Focus

- A missed historical path: compare the complete parent tree, not selected directories.
- An edited reference file: compare blob IDs between parent and archived paths.
- A still-active root package: assert root `Package.swift` is absent and archived manifest is present.
- Misleading current documentation: replace root agent guidance and wiki active-task status.
- Accidental unrelated changes: audit staged paths and both worktrees before publishing.

---

### Task 1: Archive and retirement guidance

**Files:**
- Move: historical implementation, engineering docs, CI, and support files to `legacy/pre-redesign/`
- Replace: `AGENTS.md`
- Create: `docs/redesign/2026-11-28-pre-redesign-retirement.md`
- Create: `docs/superpowers/plans/2026-11-28-pre-redesign-retirement.md`

- [ ] Record the set of root files retained for identity/legal reasons.
- [ ] Move every other tracked historical path beneath `legacy/pre-redesign/`, excluding the new redesign docs.
- [ ] Replace root agent guidance with reference-only and redesign-boundary rules.
- [ ] Audit every parent blob against its archived counterpart; expect zero missing or changed files.

### Task 2: Publish and synchronize

**Files:**
- Modify: wiki `Home.md`, `Roadmap.md`, `Architecture.md`, `Project-Tracker.md`, and `News.md` only as needed to make retirement current.

- [ ] Verify the reference integrity audit and inspect the staged diff summary.
- [ ] Commit the retirement and push `main`.
- [ ] Replace obsolete active milestones/tasks with a redesign tracker pointing to the archived evidence.
- [ ] Commit and push the wiki on `master`.
- [ ] Confirm remote main/wiki commits match local commits and both worktrees are clean.
