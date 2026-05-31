# SH17 Variant Docs Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Document the currently used notebook/config variants for `yolov9c` and `yolov9s` in separate files under `docs/variants`, showing how each variant differs from the baseline and from the previous variant.

**Architecture:** Keep the docs close to the current benchmark configs. Each model gets one Markdown file with the same structure so the variants are easy to compare across notebooks.

**Tech Stack:** Markdown

---

### Task 1: Create short plan and implement docs

**Files:**
- Create: `D:\DS-AI\SH17\docs\variants\yolov9c.md`
- Create: `D:\DS-AI\SH17\docs\variants\yolov9s.md`

- [ ] Read the current `yolov9c` and `yolov9s` configs.
- [ ] Write one Markdown file per model.
- [ ] For each variant, record:
  - purpose
  - exact differences vs baseline
  - exact differences vs the previous variant
- [ ] Review the wording against the current config values.
