# SH17 YOLOv9s Kaggle Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single-file Kaggle notebook for SH17 `yolov9s` training that embeds all helper logic and benchmark variants directly in the notebook.

**Architecture:** The Kaggle notebook is fully self-contained. It embeds helper functions, variant configuration, dataset manifest generation, oversampling, checkpoint handling, and leaderboard writing without importing any local repo modules.

**Tech Stack:** Jupyter Notebook, Python, Ultralytics YOLO, pandas, PyYAML

---

### Task 1: Add a notebook structure test

**Files:**
- Create: `D:\DS-AI\SH17\tests\test_sh17_yolov9s_kaggle_notebook.py`

- [ ] Write a failing test that checks the notebook is self-contained and uses Kaggle paths.
- [ ] Run the test and verify it fails before the notebook exists.

### Task 2: Create the self-contained Kaggle notebook

**Files:**
- Create: `D:\DS-AI\SH17\SH17_yolov9s_kaggle.ipynb`

- [ ] Add embedded helper functions directly in the notebook.
- [ ] Add the 4 `yolov9s` variants.
- [ ] Add Kaggle dataset/output paths.
- [ ] Add train/val disjoint assertion and oversampling logic.

### Task 3: Verify the notebook

**Files:**
- Modify: `D:\DS-AI\SH17\tests\test_sh17_yolov9s_kaggle_notebook.py`
- Modify: `D:\DS-AI\SH17\SH17_yolov9s_kaggle.ipynb`

- [ ] Run the notebook structure test until it passes.
- [ ] Validate the `.ipynb` file with `nbformat`.
- [ ] Review that the notebook does not depend on repo-local helper/config files.
