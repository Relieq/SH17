# SH17 YOLOv9s Shared Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a full SH17 `yolov9s` benchmark notebook that reuses the existing shared helper pipeline while keeping `yolov9s` weights, runs, and artifacts completely separate from `yolov9c`.

**Architecture:** Refactor the current helper just enough to support dynamic YOLOv9 weight names and per-model output paths, then add a `yolov9s` config and notebook that mirror the `yolov9c` experiment matrix from the SH17 paper summary. All checkpoint, manifest, oversampling, and leaderboard logic stays in the shared helper.

**Tech Stack:** Python 3.13, pytest, PyYAML, pandas, Ultralytics YOLO, Jupyter Notebook

---

## File Structure

- Modify: `D:\DS-AI\SH17\scripts\sh17_yolov9c_pipeline.py`
  Responsibility: shared SH17 helper for config loading, manifests, weight resolution, train/eval orchestration, checkpoint reuse, and result export.
- Modify: `D:\DS-AI\SH17\tests\test_sh17_yolov9c_pipeline.py`
  Responsibility: unit tests for shared helper behavior across `yolov9c` and `yolov9s`.
- Create: `D:\DS-AI\SH17\configs\sh17_yolov9s_experiments.yaml`
  Responsibility: `yolov9s`-specific paths, defaults, and experiment matrix.
- Create: `D:\DS-AI\SH17\SH17_yolov9s_benchmark.ipynb`
  Responsibility: user-facing benchmark notebook for `yolov9s`.

### Task 1: Generalize shared helper weight resolution for `yolov9s`

**Files:**
- Modify: `D:\DS-AI\SH17\tests\test_sh17_yolov9c_pipeline.py`
- Modify: `D:\DS-AI\SH17\scripts\sh17_yolov9c_pipeline.py`

- [ ] **Step 1: Write the failing tests**

```python
from scripts.sh17_yolov9c_pipeline import ensure_model_weights, expand_experiments


def test_ensure_model_weights_uses_requested_weight_name(tmp_path, monkeypatch):
    requested = []

    def fake_download(url, output_path):
        requested.append((url, output_path))
        Path(output_path).write_bytes(b"weights")

    monkeypatch.setattr("scripts.sh17_yolov9c_pipeline.urllib.request.urlretrieve", fake_download)

    path = ensure_model_weights(tmp_path, "yolov9s.pt")

    assert path.name == "yolov9s.pt"
    assert path.exists()
    assert requested[0][0].endswith("/yolov9s.pt")


def test_expand_experiments_keeps_model_specific_paths(tmp_path):
    config = {
        "paths": {
            "dataset_root": Path("E:/data/SH17"),
            "weights_dir": Path("E:/models/yolov9"),
            "artifact_root": Path("D:/DS-AI/SH17/artifacts/sh17_yolov9s"),
            "runs_root": Path("E:/models/sh17_yolov9s_runs"),
        },
        "defaults": {"epochs": 200},
        "experiments": [{"name": "yolov9s_baseline_640", "weights": "yolov9s.pt"}],
    }
    manifest = DatasetManifest(
        train_manifest=tmp_path / "train.txt",
        val_manifest=tmp_path / "val.txt",
        data_yaml=tmp_path / "sh17.yaml",
        train_count=1,
        val_count=1,
    )

    expanded = expand_experiments(config, manifest)

    assert expanded[0]["weights"] == "yolov9s.pt"
    assert expanded[0]["artifact_root"].endswith("sh17_yolov9s")
    assert expanded[0]["runs_root"].endswith("sh17_yolov9s_runs")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `D:\Anaconda\envs\SH17\python.exe -m pytest tests/test_sh17_yolov9c_pipeline.py -v --basetemp .pytest-tmp`
Expected: FAIL because `ensure_model_weights()` does not exist and the helper still hard-codes `yolov9c`.

- [ ] **Step 3: Implement the minimal shared helper refactor**

`scripts/sh17_yolov9c_pipeline.py`

```python
def ensure_model_weights(weights_dir: str | Path, weight_name: str) -> Path:
    weights_dir = Path(weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    weight_path = weights_dir / weight_name
    if not weight_path.exists():
        urllib.request.urlretrieve(
            f"https://github.com/ultralytics/assets/releases/download/v8.2.0/{weight_name}",
            weight_path,
        )
    return weight_path


def ensure_yolov9c_weights(weights_dir: str | Path) -> Path:
    return ensure_model_weights(weights_dir, "yolov9c.pt")
```

Inside `train_one_experiment()`:

```python
weights_name = spec["weights"]
...
        weights_path = ensure_model_weights(spec["weights_dir"], weights_name)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `D:\Anaconda\envs\SH17\python.exe -m pytest tests/test_sh17_yolov9c_pipeline.py -v --basetemp .pytest-tmp`
Expected: PASS

### Task 2: Add the `yolov9s` experiment config

**Files:**
- Create: `D:\DS-AI\SH17\configs\sh17_yolov9s_experiments.yaml`

- [ ] **Step 1: Create the config file**

```yaml
paths:
  dataset_root: E:/data/SH17
  models_root: E:/models
  weights_dir: E:/models/yolov9
  runs_root: E:/models/sh17_yolov9s_runs
  artifact_root: D:/DS-AI/SH17/artifacts/sh17_yolov9s

defaults:
  seed: 0
  epochs: 200
  imgsz: 640
  batch: -1
  workers: 8
  patience: 40
  device: 0
  pretrained: true
  deterministic: true
  save: true
  save_period: 10

experiments:
  - name: yolov9s_baseline_640
    weights: yolov9s.pt
    imgsz: 640

  - name: yolov9s_tuned_640
    weights: yolov9s.pt
    imgsz: 640
    optimizer: AdamW
    lr0: 0.001
    lrf: 0.01
    warmup_epochs: 5
    cos_lr: true
    close_mosaic: 10
    patience: 30

  - name: yolov9s_multiscale_960
    weights: yolov9s.pt
    imgsz: 960
    multi_scale: true
    optimizer: AdamW
    lr0: 0.001
    lrf: 0.01
    warmup_epochs: 5
    cos_lr: true
    close_mosaic: 10

  - name: yolov9s_oversample_minority_960
    weights: yolov9s.pt
    imgsz: 960
    multi_scale: true
    optimizer: AdamW
    lr0: 0.001
    lrf: 0.01
    warmup_epochs: 5
    cos_lr: true
    close_mosaic: 10
    use_oversampled_train_manifest: true
```

- [ ] **Step 2: Smoke-load the config**

Run: `D:\Anaconda\envs\SH17\python.exe -c "from scripts.sh17_yolov9c_pipeline import load_experiment_config; print(load_experiment_config('configs/sh17_yolov9s_experiments.yaml')['experiments'][0]['weights'])"`
Expected: `yolov9s.pt`

### Task 3: Create the `yolov9s` benchmark notebook

**Files:**
- Create: `D:\DS-AI\SH17\SH17_yolov9s_benchmark.ipynb`

- [ ] **Step 1: Create the notebook with the same thin structure as `yolov9c`**

Notebook cells:

```python
from pathlib import Path

import pandas as pd

from scripts.sh17_yolov9c_pipeline import (
    append_result,
    build_dataset_manifest,
    build_oversampled_train_manifest,
    ensure_model_weights,
    expand_experiments,
    load_experiment_config,
    train_one_experiment,
)
```

```python
CONFIG_PATH = Path("configs/sh17_yolov9s_experiments.yaml")
cfg = load_experiment_config(CONFIG_PATH)

dataset_manifest = build_dataset_manifest(
    cfg["paths"]["dataset_root"],
    Path(cfg["paths"]["artifact_root"]) / "dataset",
)
weights_path = ensure_model_weights(cfg["paths"]["weights_dir"], "yolov9s.pt")
experiments = expand_experiments(cfg, dataset_manifest)
```

```python
if any(exp.get("use_oversampled_train_manifest") for exp in experiments):
    oversampled_manifest = build_oversampled_train_manifest(
        train_manifest=dataset_manifest.train_manifest,
        labels_dir=Path(cfg["paths"]["dataset_root"]) / "labels",
        output_path=Path(cfg["paths"]["artifact_root"]) / "dataset" / "train_oversampled.txt",
        minority_class_ids={2, 4, 6, 10, 13, 16},
        repeat_factor=3,
    )
    for exp in experiments:
        if exp.get("use_oversampled_train_manifest"):
            exp["train_manifest_override"] = str(oversampled_manifest)
```

```python
ACTIVE_EXPERIMENTS = [
    "yolov9s_baseline_640",
    "yolov9s_tuned_640",
    "yolov9s_multiscale_960",
    "yolov9s_oversample_minority_960",
]
```

```python
records = []
for exp in experiments:
    if exp["name"] not in ACTIVE_EXPERIMENTS:
        continue
    record = train_one_experiment(exp, cfg["paths"]["runs_root"])
    append_result(record, cfg["paths"]["artifact_root"])
    records.append(record)
```

### Task 4: Verify notebook and `yolov9s` weight setup

**Files:**
- Modify: `D:\DS-AI\SH17\SH17_yolov9s_benchmark.ipynb`
- Modify: `D:\DS-AI\SH17\scripts\sh17_yolov9c_pipeline.py`

- [ ] **Step 1: Validate notebook structure and dataset prep**

Run: a short Python smoke script that:
- validates `SH17_yolov9s_benchmark.ipynb`
- loads `configs/sh17_yolov9s_experiments.yaml`
- builds the dataset manifest
- resolves `E:\models\yolov9\yolov9s.pt`

Expected:
- notebook valid
- train count `6479`
- val count `1620`
- `yolov9s.pt` exists after helper resolution

- [ ] **Step 2: Re-run shared tests**

Run: `D:\Anaconda\envs\SH17\python.exe -m pytest tests/test_sh17_yolov9c_pipeline.py -v --basetemp .pytest-tmp`
Expected: PASS

### Task 5: Summarize runtime guidance for `yolov9s`

**Files:**
- Modify: `D:\DS-AI\SH17\SH17_yolov9s_benchmark.ipynb`

- [ ] **Step 1: Set the run order inside the notebook comments or markdown**

Run order:
1. `yolov9s_baseline_640`
2. `yolov9s_tuned_640`
3. `yolov9s_multiscale_960`
4. `yolov9s_oversample_minority_960`

- [ ] **Step 2: Confirm isolation**

Expected:
- `yolov9s` writes only into `E:\models\sh17_yolov9s_runs`
- leaderboard and plots live only in `D:\DS-AI\SH17\artifacts\sh17_yolov9s`
- `yolov9c` notebook and outputs remain untouched
