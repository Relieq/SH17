# SH17 YOLOv9c Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a new, shorter SH17 training notebook and helper pipeline for `yolov9c` that uses `E:\data\SH17` as the dataset source, downloads model weights into `E:\models`, saves and resumes checkpoints reliably, and runs a focused set of improvement experiments aimed at beating the current `yolov9e` benchmark.

**Architecture:** Keep the notebook thin and move repeatable logic into one Python helper module plus one YAML experiment config. The notebook should only validate paths, launch baseline and improvement runs, compare metrics, and load the best checkpoint for evaluation or resume; all path handling, split generation, oversampling, checkpoint lookup, and metrics export live in the helper module.

**Tech Stack:** Python 3.10+, Ultralytics YOLO, PyYAML, pandas, pytest, Jupyter Notebook

---

## File Structure

- Create: `D:\DS-AI\SH17\SH17_yolov9c_benchmark.ipynb`
  Responsibility: user-facing notebook with a small number of cells for config, training, resume, evaluation, and leaderboard output.
- Create: `D:\DS-AI\SH17\configs\sh17_yolov9c_experiments.yaml`
  Responsibility: all paths, shared defaults, and named experiments for baseline and improvement runs.
- Create: `D:\DS-AI\SH17\scripts\sh17_yolov9c_pipeline.py`
  Responsibility: path validation, dataset split loading, YAML generation, class stats, oversampled manifests, weight download, checkpoint resolution, training/eval wrappers, and result export.
- Create: `D:\DS-AI\SH17\tests\test_sh17_yolov9c_pipeline.py`
  Responsibility: fast tests for config parsing, split resolution, oversampled manifest generation, and checkpoint selection.
- Create: `D:\DS-AI\SH17\artifacts\sh17_yolov9c\.gitkeep`
  Responsibility: keep a local artifact folder in the repo for small metadata outputs; large checkpoints stay under `E:\models`.

## Planned Experiment Set

- `yolov9c_baseline_640`
  Paper-like baseline for direct comparison.
- `yolov9c_tuned_640`
  Same image size, but with clearer optimizer and schedule choices: `AdamW`, `cos_lr`, warmup, `close_mosaic`, and longer patience discipline.
- `yolov9c_multiscale_960`
  Increase effective detail for small PPE objects with `imgsz=960` and `multi_scale=True`.
- `yolov9c_oversample_minority_960`
  Same as the high-resolution run, but train from an oversampled manifest that upweights images containing `face-guard`, `medical-suit`, `ear-mufs`, `safety-vest`, `helmet`, and `foot`.

The initial target is not to invent a complex two-stage detector. It is to beat `yolov9e` by exploiting the weaknesses identified in the summary: class imbalance, tiny objects, and under-specified training defaults.

### Task 1: Scaffold the focused pipeline files

**Files:**
- Create: `D:\DS-AI\SH17\configs\sh17_yolov9c_experiments.yaml`
- Create: `D:\DS-AI\SH17\scripts\sh17_yolov9c_pipeline.py`
- Create: `D:\DS-AI\SH17\tests\test_sh17_yolov9c_pipeline.py`
- Create: `D:\DS-AI\SH17\artifacts\sh17_yolov9c\.gitkeep`

- [ ] **Step 1: Write the failing test for config loading and default path wiring**

```python
from pathlib import Path

from scripts.sh17_yolov9c_pipeline import load_experiment_config


def test_load_experiment_config_reads_fixed_windows_paths(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
paths:
  dataset_root: E:/data/SH17
  models_root: E:/models
  runs_root: E:/models/sh17_yolov9c_runs
defaults:
  epochs: 200
  imgsz: 640
  batch: -1
experiments:
  - name: yolov9c_baseline_640
    weights: yolov9c.pt
""".strip(),
        encoding="utf-8",
    )

    cfg = load_experiment_config(config_path)

    assert cfg["paths"]["dataset_root"] == Path("E:/data/SH17")
    assert cfg["paths"]["models_root"] == Path("E:/models")
    assert cfg["paths"]["runs_root"] == Path("E:/models/sh17_yolov9c_runs")
    assert cfg["experiments"][0]["name"] == "yolov9c_baseline_640"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sh17_yolov9c_pipeline.py::test_load_experiment_config_reads_fixed_windows_paths -v`
Expected: `FAILED` with `ModuleNotFoundError` or `ImportError` because the helper module does not exist yet.

- [ ] **Step 3: Create the minimal config and loader implementation**

`configs/sh17_yolov9c_experiments.yaml`

```yaml
paths:
  dataset_root: E:/data/SH17
  models_root: E:/models
  weights_dir: E:/models/yolov9
  runs_root: E:/models/sh17_yolov9c_runs
  artifact_root: D:/DS-AI/SH17/artifacts/sh17_yolov9c

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
  - name: yolov9c_baseline_640
    weights: yolov9c.pt
```

`scripts/sh17_yolov9c_pipeline.py`

```python
from pathlib import Path
from typing import Any

import yaml


def _to_path_map(paths: dict[str, str]) -> dict[str, Path]:
    return {key: Path(value) for key, value in paths.items()}


def load_experiment_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["paths"] = _to_path_map(raw["paths"])
    return raw
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sh17_yolov9c_pipeline.py::test_load_experiment_config_reads_fixed_windows_paths -v`
Expected: `PASSED`

- [ ] **Step 5: Commit**

```bash
git add configs/sh17_yolov9c_experiments.yaml scripts/sh17_yolov9c_pipeline.py tests/test_sh17_yolov9c_pipeline.py artifacts/sh17_yolov9c/.gitkeep
git commit -m "feat: scaffold sh17 yolov9c benchmark pipeline"
```

### Task 2: Add dataset validation, official split resolution, and YOLO data YAML creation

**Files:**
- Modify: `D:\DS-AI\SH17\scripts\sh17_yolov9c_pipeline.py`
- Modify: `D:\DS-AI\SH17\tests\test_sh17_yolov9c_pipeline.py`

- [ ] **Step 1: Write the failing test for official split resolution**

```python
from pathlib import Path

from scripts.sh17_yolov9c_pipeline import build_dataset_manifest


def test_build_dataset_manifest_resolves_train_and_val_lists(tmp_path):
    dataset_root = tmp_path / "SH17"
    images = dataset_root / "images"
    labels = dataset_root / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)

    (images / "a.jpg").write_bytes(b"a")
    (images / "b.jpg").write_bytes(b"b")
    (labels / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (labels / "b.txt").write_text("10 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (dataset_root / "train_files.txt").write_text("a.jpg\n", encoding="utf-8")
    (dataset_root / "val_files.txt").write_text("b.jpg\n", encoding="utf-8")

    manifest = build_dataset_manifest(dataset_root, tmp_path / "out")

    assert manifest.train_count == 1
    assert manifest.val_count == 1
    assert manifest.data_yaml.exists()
    assert manifest.train_manifest.exists()
    assert manifest.val_manifest.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sh17_yolov9c_pipeline.py::test_build_dataset_manifest_resolves_train_and_val_lists -v`
Expected: `FAILED` because `build_dataset_manifest` is not defined yet.

- [ ] **Step 3: Implement dataset manifest creation**

`scripts/sh17_yolov9c_pipeline.py`

```python
from dataclasses import dataclass


SH17_NAMES = {
    0: "person",
    1: "ear",
    2: "ear-mufs",
    3: "face",
    4: "face-guard",
    5: "face-mask",
    6: "foot",
    7: "tool",
    8: "glasses",
    9: "gloves",
    10: "helmet",
    11: "hands",
    12: "head",
    13: "medical-suit",
    14: "shoes",
    15: "safety-suit",
    16: "safety-vest",
}


@dataclass
class DatasetManifest:
    train_manifest: Path
    val_manifest: Path
    data_yaml: Path
    train_count: int
    val_count: int


def _read_split(split_path: Path) -> list[str]:
    return [line.strip().replace("\\\\", "/") for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _resolve_image_paths(images_dir: Path, entries: list[str]) -> list[Path]:
    resolved = []
    for entry in entries:
        candidate = images_dir / Path(entry).name
        if not candidate.exists():
            raise FileNotFoundError(f"Missing image for split entry: {entry}")
        resolved.append(candidate.resolve())
    return resolved


def build_dataset_manifest(dataset_root: str | Path, output_dir: str | Path) -> DatasetManifest:
    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images_dir = dataset_root / "images"
    train_entries = _read_split(dataset_root / "train_files.txt")
    val_entries = _read_split(dataset_root / "val_files.txt")
    train_images = _resolve_image_paths(images_dir, train_entries)
    val_images = _resolve_image_paths(images_dir, val_entries)

    train_manifest = output_dir / "train_files_abs.txt"
    val_manifest = output_dir / "val_files_abs.txt"
    data_yaml = output_dir / "sh17.yaml"

    train_manifest.write_text("\\n".join(map(str, train_images)) + "\\n", encoding="utf-8")
    val_manifest.write_text("\\n".join(map(str, val_images)) + "\\n", encoding="utf-8")
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": "/",
                "train": str(train_manifest),
                "val": str(val_manifest),
                "test": str(val_manifest),
                "names": SH17_NAMES,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    return DatasetManifest(
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        data_yaml=data_yaml,
        train_count=len(train_images),
        val_count=len(val_images),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sh17_yolov9c_pipeline.py::test_build_dataset_manifest_resolves_train_and_val_lists -v`
Expected: `PASSED`

- [ ] **Step 5: Commit**

```bash
git add scripts/sh17_yolov9c_pipeline.py tests/test_sh17_yolov9c_pipeline.py
git commit -m "feat: add sh17 dataset manifest builder"
```

### Task 3: Add minority-class oversampling and experiment expansion

**Files:**
- Modify: `D:\DS-AI\SH17\configs\sh17_yolov9c_experiments.yaml`
- Modify: `D:\DS-AI\SH17\scripts\sh17_yolov9c_pipeline.py`
- Modify: `D:\DS-AI\SH17\tests\test_sh17_yolov9c_pipeline.py`

- [ ] **Step 1: Write the failing test for oversampled train manifest generation**

```python
from scripts.sh17_yolov9c_pipeline import build_oversampled_train_manifest


def test_build_oversampled_train_manifest_repeats_minority_samples(tmp_path):
    train_manifest = tmp_path / "train.txt"
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()

    img_a = tmp_path / "a.jpg"
    img_b = tmp_path / "b.jpg"
    img_a.write_bytes(b"a")
    img_b.write_bytes(b"b")
    train_manifest.write_text(f"{img_a}\n{img_b}\n", encoding="utf-8")

    (labels_dir / "a.txt").write_text("4 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (labels_dir / "b.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    oversampled = build_oversampled_train_manifest(
        train_manifest=train_manifest,
        labels_dir=labels_dir,
        output_path=tmp_path / "train_oversampled.txt",
        minority_class_ids={4},
        repeat_factor=3,
    )

    lines = oversampled.read_text(encoding="utf-8").splitlines()
    assert lines.count(str(img_a)) == 3
    assert lines.count(str(img_b)) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sh17_yolov9c_pipeline.py::test_build_oversampled_train_manifest_repeats_minority_samples -v`
Expected: `FAILED` because the oversampling helper does not exist yet.

- [ ] **Step 3: Implement oversampled manifests and define the immediate experiment matrix**

`scripts/sh17_yolov9c_pipeline.py`

```python
MINORITY_CLASS_IDS = {4, 6, 10, 13, 16, 2}


def build_oversampled_train_manifest(
    train_manifest: str | Path,
    labels_dir: str | Path,
    output_path: str | Path,
    minority_class_ids: set[int],
    repeat_factor: int = 3,
) -> Path:
    train_manifest = Path(train_manifest)
    labels_dir = Path(labels_dir)
    output_path = Path(output_path)

    expanded: list[str] = []
    for line in train_manifest.read_text(encoding="utf-8").splitlines():
        image_path = Path(line.strip())
        if not image_path:
            continue
        label_path = labels_dir / f"{image_path.stem}.txt"
        class_ids = {
            int(row.split()[0])
            for row in label_path.read_text(encoding="utf-8").splitlines()
            if row.strip()
        }
        repeats = repeat_factor if class_ids & minority_class_ids else 1
        expanded.extend([str(image_path)] * repeats)

    output_path.write_text("\\n".join(expanded) + "\\n", encoding="utf-8")
    return output_path


def expand_experiments(config: dict, dataset_manifest: DatasetManifest) -> list[dict]:
    experiments = []
    for exp in config["experiments"]:
        merged = {**config["defaults"], **exp}
        merged["data"] = str(dataset_manifest.data_yaml)
        merged["minority_class_ids"] = sorted(MINORITY_CLASS_IDS)
        experiments.append(merged)
    return experiments
```

`configs/sh17_yolov9c_experiments.yaml`

```yaml
experiments:
  - name: yolov9c_baseline_640
    weights: yolov9c.pt
    imgsz: 640

  - name: yolov9c_tuned_640
    weights: yolov9c.pt
    imgsz: 640
    optimizer: AdamW
    lr0: 0.001
    lrf: 0.01
    warmup_epochs: 5
    cos_lr: true
    close_mosaic: 10
    patience: 30

  - name: yolov9c_multiscale_960
    weights: yolov9c.pt
    imgsz: 960
    multi_scale: true
    optimizer: AdamW
    lr0: 0.001
    lrf: 0.01
    warmup_epochs: 5
    cos_lr: true
    close_mosaic: 10

  - name: yolov9c_oversample_minority_960
    weights: yolov9c.pt
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

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sh17_yolov9c_pipeline.py::test_build_oversampled_train_manifest_repeats_minority_samples -v`
Expected: `PASSED`

- [ ] **Step 5: Commit**

```bash
git add configs/sh17_yolov9c_experiments.yaml scripts/sh17_yolov9c_pipeline.py tests/test_sh17_yolov9c_pipeline.py
git commit -m "feat: add oversampling and experiment matrix"
```

### Task 4: Add YOLOv9 weight download, checkpoint save/resume, and evaluation helpers

**Files:**
- Modify: `D:\DS-AI\SH17\scripts\sh17_yolov9c_pipeline.py`
- Modify: `D:\DS-AI\SH17\tests\test_sh17_yolov9c_pipeline.py`

- [ ] **Step 1: Write the failing test for checkpoint selection**

```python
from scripts.sh17_yolov9c_pipeline import pick_checkpoint_for_resume


def test_pick_checkpoint_for_resume_prefers_last_then_best(tmp_path):
    run_dir = tmp_path / "train" / "yolov9c_baseline_640" / "weights"
    run_dir.mkdir(parents=True)
    best = run_dir / "best.pt"
    last = run_dir / "last.pt"
    best.write_bytes(b"best")
    last.write_bytes(b"last")

    selected = pick_checkpoint_for_resume(run_dir.parent)

    assert selected == last
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sh17_yolov9c_pipeline.py::test_pick_checkpoint_for_resume_prefers_last_then_best -v`
Expected: `FAILED` because the checkpoint selector does not exist yet.

- [ ] **Step 3: Implement weight download and resume-aware training helpers**

`scripts/sh17_yolov9c_pipeline.py`

```python
import json
import time
import urllib.request

import pandas as pd
from ultralytics import YOLO


YOLOV9C_URL = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov9c.pt"


def ensure_yolov9c_weights(weights_dir: str | Path) -> Path:
    weights_dir = Path(weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    weight_path = weights_dir / "yolov9c.pt"
    if not weight_path.exists():
        urllib.request.urlretrieve(YOLOV9C_URL, weight_path)
    return weight_path


def pick_checkpoint_for_resume(run_dir: str | Path) -> Path | None:
    run_dir = Path(run_dir)
    last_pt = run_dir / "weights" / "last.pt"
    best_pt = run_dir / "weights" / "best.pt"
    if last_pt.exists():
        return last_pt
    if best_pt.exists():
        return best_pt
    return None


def train_one_experiment(spec: dict, run_root: str | Path) -> dict:
    run_root = Path(run_root)
    run_dir = run_root / spec["name"]
    checkpoint = pick_checkpoint_for_resume(run_dir)
    weights_path = checkpoint or ensure_yolov9c_weights(spec["weights_dir"])

    model = YOLO(str(weights_path))
    train_kwargs = {
        "data": spec["data"],
        "epochs": spec["epochs"],
        "imgsz": spec["imgsz"],
        "batch": spec["batch"],
        "workers": spec["workers"],
        "device": spec["device"],
        "patience": spec["patience"],
        "project": str(run_root),
        "name": spec["name"],
        "exist_ok": True,
        "save": True,
        "save_period": spec["save_period"],
        "plots": True,
    }
    optional_keys = ["optimizer", "lr0", "lrf", "warmup_epochs", "cos_lr", "close_mosaic", "multi_scale"]
    for key in optional_keys:
        if key in spec:
            train_kwargs[key] = spec[key]

    if checkpoint is not None:
        train_kwargs["resume"] = True

    model.train(**train_kwargs)
    best_path = run_dir / "weights" / "best.pt"
    metrics = YOLO(str(best_path)).val(data=spec["data"], split="val", imgsz=spec["imgsz"], plots=True)

    record = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "name": spec["name"],
        "weights_path": str(best_path),
        "map50": float(metrics.box.map50),
        "map": float(metrics.box.map),
    }
    return record


def append_result(record: dict, artifact_root: str | Path) -> None:
    artifact_root = Path(artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    csv_path = artifact_root / "leaderboard.csv"
    jsonl_path = artifact_root / "leaderboard.jsonl"
    frame = pd.DataFrame([record])
    if csv_path.exists():
        frame = pd.concat([pd.read_csv(csv_path), frame], ignore_index=True)
    frame.to_csv(csv_path, index=False)
    with jsonl_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\\n")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sh17_yolov9c_pipeline.py::test_pick_checkpoint_for_resume_prefers_last_then_best -v`
Expected: `PASSED`

- [ ] **Step 5: Commit**

```bash
git add scripts/sh17_yolov9c_pipeline.py tests/test_sh17_yolov9c_pipeline.py
git commit -m "feat: add weight download and checkpoint resume helpers"
```

### Task 5: Build the short notebook that orchestrates baseline, improvement runs, and checkpoint loading

**Files:**
- Create: `D:\DS-AI\SH17\SH17_yolov9c_benchmark.ipynb`
- Modify: `D:\DS-AI\SH17\scripts\sh17_yolov9c_pipeline.py`

- [ ] **Step 1: Write the notebook skeleton with only the cells we actually need**

`SH17_yolov9c_benchmark.ipynb`

```python
# Cell 1
from pathlib import Path
import pandas as pd

from scripts.sh17_yolov9c_pipeline import (
    append_result,
    build_dataset_manifest,
    build_oversampled_train_manifest,
    ensure_yolov9c_weights,
    expand_experiments,
    load_experiment_config,
    train_one_experiment,
)
```

```python
# Cell 2
CONFIG_PATH = Path("configs/sh17_yolov9c_experiments.yaml")
cfg = load_experiment_config(CONFIG_PATH)
dataset_manifest = build_dataset_manifest(
    cfg["paths"]["dataset_root"],
    Path(cfg["paths"]["artifact_root"]) / "dataset",
)
weights_path = ensure_yolov9c_weights(cfg["paths"]["weights_dir"])
cfg["defaults"]["weights_dir"] = cfg["paths"]["weights_dir"]
experiments = expand_experiments(cfg, dataset_manifest)
weights_path
```

```python
# Cell 3
if any(exp.get("use_oversampled_train_manifest") for exp in experiments):
    oversampled_manifest = build_oversampled_train_manifest(
        train_manifest=dataset_manifest.train_manifest,
        labels_dir=cfg["paths"]["dataset_root"] / "labels",
        output_path=Path(cfg["paths"]["artifact_root"]) / "dataset" / "train_oversampled.txt",
        minority_class_ids={2, 4, 6, 10, 13, 16},
        repeat_factor=3,
    )
    for exp in experiments:
        if exp.get("use_oversampled_train_manifest"):
            exp["data"] = str(dataset_manifest.data_yaml)
            exp["train_manifest_override"] = str(oversampled_manifest)
```

```python
# Cell 4
records = []
for exp in experiments:
    record = train_one_experiment(exp, cfg["paths"]["runs_root"])
    append_result(record, cfg["paths"]["artifact_root"])
    records.append(record)

pd.DataFrame(records).sort_values(["map50", "map"], ascending=False)
```

```python
# Cell 5
leaderboard = pd.read_csv(Path(cfg["paths"]["artifact_root"]) / "leaderboard.csv")
leaderboard.sort_values(["map50", "map"], ascending=False).head(10)
```

- [ ] **Step 2: Implement the one missing hook for `train_manifest_override`**

`scripts/sh17_yolov9c_pipeline.py`

```python
def write_data_yaml_with_train_override(
    base_yaml: str | Path,
    train_manifest: str | Path,
    output_path: str | Path,
) -> Path:
    base_yaml = Path(base_yaml)
    output_path = Path(output_path)
    payload = yaml.safe_load(base_yaml.read_text(encoding="utf-8"))
    payload["train"] = str(train_manifest)
    output_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return output_path
```

Inside `train_one_experiment`:

```python
data_path = spec["data"]
if "train_manifest_override" in spec:
    data_path = write_data_yaml_with_train_override(
        base_yaml=spec["data"],
        train_manifest=spec["train_manifest_override"],
        output_path=Path(spec["artifact_root"]) / f"{spec['name']}_data.yaml",
    )
train_kwargs["data"] = str(data_path)
```

- [ ] **Step 3: Run the helper tests and one dry notebook smoke check**

Run: `pytest tests/test_sh17_yolov9c_pipeline.py -v`
Expected: all helper tests `PASSED`

Run: open `SH17_yolov9c_benchmark.ipynb` and execute Cells 1-2
Expected: the dataset manifest is created under `D:\DS-AI\SH17\artifacts\sh17_yolov9c\dataset`, and `E:\models\yolov9\yolov9c.pt` exists after download.

- [ ] **Step 4: Commit**

```bash
git add SH17_yolov9c_benchmark.ipynb scripts/sh17_yolov9c_pipeline.py
git commit -m "feat: add concise sh17 yolov9c benchmark notebook"
```

### Task 6: Verify the benchmark workflow and define the first execution order

**Files:**
- Modify: `D:\DS-AI\SH17\configs\sh17_yolov9c_experiments.yaml`
- Modify: `D:\DS-AI\SH17\SH17_yolov9c_benchmark.ipynb`

- [ ] **Step 1: Run the baseline only and confirm checkpoint save + reload**

Run: in the notebook, disable all experiments except `yolov9c_baseline_640`
Expected:
- training writes checkpoints to `E:\models\sh17_yolov9c_runs\yolov9c_baseline_640\weights`
- both `best.pt` and `last.pt` are present
- rerunning the same experiment resumes from `last.pt` instead of starting from scratch

- [ ] **Step 2: Run the three improvement experiments in this order**

Run order:
1. `yolov9c_tuned_640`
2. `yolov9c_multiscale_960`
3. `yolov9c_oversample_minority_960`

Expected:
- `yolov9c_tuned_640` tells us whether pure training-strategy tuning is enough
- `yolov9c_multiscale_960` tests whether small-object detail is the main bottleneck
- `yolov9c_oversample_minority_960` tests whether imbalance is holding back the rare PPE classes

- [ ] **Step 3: Compare the leaderboard against the paper target**

Run: review `D:\DS-AI\SH17\artifacts\sh17_yolov9c\leaderboard.csv`
Expected:
- if `map50 > 70.9`, we have beaten the paper's `yolov9e` reference on the main metric
- if `map50-95 > 48.7`, we have beaten the stronger localization metric too
- if not, inspect per-class gains first for `helmet`, `foot`, `tool`, `safety-vest`, `ear-mufs`, and `face-guard`

- [ ] **Step 4: Commit**

```bash
git add configs/sh17_yolov9c_experiments.yaml SH17_yolov9c_benchmark.ipynb
git commit -m "chore: verify sh17 yolov9c benchmark workflow"
```

## Self-Review

- Spec coverage: the plan covers the new notebook, direct use of `E:\data\SH17`, weight download into `E:\models`, checkpoint save and resume, a concise refactored pipeline, and the first set of practical improvement experiments for `yolov9c`.
- Placeholder scan: no `TODO`, `TBD`, or “implement later” markers remain.
- Type consistency: the same config shape is used across loader, manifest builder, experiment expansion, and training functions.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-30-sh17-yolov9c-benchmark-plan.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
