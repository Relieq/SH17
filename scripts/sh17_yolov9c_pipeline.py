import json
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


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

MINORITY_CLASS_IDS = {2, 4, 6, 10, 13, 16}
ULTRALYTICS_WEIGHT_BASE_URL = "https://github.com/ultralytics/assets/releases/download/v8.2.0"


@dataclass
class DatasetManifest:
    train_manifest: Path
    val_manifest: Path
    data_yaml: Path
    train_count: int
    val_count: int


def _to_path_map(paths: dict[str, str]) -> dict[str, Path]:
    return {key: Path(value) for key, value in paths.items()}


def _read_split(split_path: Path) -> list[str]:
    return [
        line.strip().replace("\\", "/")
        for line in split_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _resolve_image_paths(images_dir: Path, entries: list[str]) -> list[Path]:
    resolved = []
    for entry in entries:
        candidate = images_dir / Path(entry).name
        if not candidate.exists():
            raise FileNotFoundError(f"Missing image for split entry: {entry}")
        resolved.append(candidate.resolve())
    return resolved


def load_experiment_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["paths"] = _to_path_map(raw["paths"])
    return raw


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

    train_manifest.write_text("\n".join(map(str, train_images)) + "\n", encoding="utf-8")
    val_manifest.write_text("\n".join(map(str, val_images)) + "\n", encoding="utf-8")
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

    output_path.write_text("\n".join(expanded) + "\n", encoding="utf-8")
    return output_path


def expand_experiments(config: dict[str, Any], dataset_manifest: DatasetManifest) -> list[dict[str, Any]]:
    experiments = []
    for experiment in config["experiments"]:
        merged = {**config["defaults"], **experiment}
        merged["data"] = str(dataset_manifest.data_yaml)
        merged["dataset_root"] = str(config["paths"]["dataset_root"])
        merged["weights_dir"] = str(config["paths"]["weights_dir"])
        merged["artifact_root"] = str(config["paths"]["artifact_root"])
        merged["runs_root"] = str(config["paths"]["runs_root"])
        merged["minority_class_ids"] = sorted(MINORITY_CLASS_IDS)
        experiments.append(merged)
    return experiments


def ensure_model_weights(weights_dir: str | Path, weight_name: str) -> Path:
    weights_dir = Path(weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    weight_path = weights_dir / weight_name
    if not weight_path.exists():
        urllib.request.urlretrieve(f"{ULTRALYTICS_WEIGHT_BASE_URL}/{weight_name}", weight_path)
    return weight_path


def ensure_yolov9c_weights(weights_dir: str | Path) -> Path:
    return ensure_model_weights(weights_dir, "yolov9c.pt")


def pick_checkpoint_for_resume(run_dir: str | Path) -> Path | None:
    run_dir = Path(run_dir)
    epoch_checkpoints = sorted(
        (run_dir / "weights").glob("epoch*.pt"),
        key=lambda path: int(path.stem.replace("epoch", "")),
    )
    if epoch_checkpoints:
        return epoch_checkpoints[-1]

    last_pt = run_dir / "weights" / "last.pt"
    best_pt = run_dir / "weights" / "best.pt"
    if last_pt.exists():
        return last_pt
    if best_pt.exists():
        return best_pt
    return None


def pick_checkpoint_for_eval(run_dir: str | Path) -> Path | None:
    run_dir = Path(run_dir)
    best_pt = run_dir / "weights" / "best.pt"
    last_pt = run_dir / "weights" / "last.pt"
    if best_pt.exists():
        return best_pt
    if last_pt.exists():
        return last_pt
    return None


def count_completed_epochs(run_dir: str | Path) -> int:
    run_dir = Path(run_dir)
    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        return 0
    rows = results_csv.read_text(encoding="utf-8").splitlines()
    return max(len(rows) - 1, 0)


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


def train_one_experiment(spec: dict[str, Any], run_root: str | Path) -> dict[str, Any]:
    from ultralytics import YOLO

    run_root = Path(run_root)
    run_dir = run_root / spec["name"]
    resume_checkpoint = pick_checkpoint_for_resume(run_dir)
    eval_checkpoint = pick_checkpoint_for_eval(run_dir)
    completed_epochs = count_completed_epochs(run_dir)
    target_epochs = int(spec["epochs"])
    should_resume = resume_checkpoint is not None and completed_epochs < target_epochs
    should_reuse_complete_run = eval_checkpoint is not None and completed_epochs >= target_epochs

    if should_reuse_complete_run:
        weights_path = eval_checkpoint
    elif should_resume:
        weights_path = resume_checkpoint
    else:
        weights_path = ensure_model_weights(spec["weights_dir"], spec["weights"])

    data_path = spec["data"]
    if "train_manifest_override" in spec:
        data_path = write_data_yaml_with_train_override(
            base_yaml=spec["data"],
            train_manifest=spec["train_manifest_override"],
            output_path=Path(spec["artifact_root"]) / f"{spec['name']}_data.yaml",
        )

    if not should_reuse_complete_run:
        model = YOLO(str(weights_path))
        train_kwargs = {
            "data": str(data_path),
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
        for key in ("optimizer", "lr0", "lrf", "warmup_epochs", "cos_lr", "close_mosaic", "multi_scale"):
            if key in spec:
                train_kwargs[key] = spec[key]
        if should_resume:
            train_kwargs["resume"] = str(resume_checkpoint)

        model.train(**train_kwargs)

    best_path = pick_checkpoint_for_eval(run_dir)
    if best_path is None:
        raise FileNotFoundError(f"No evaluation checkpoint found in {run_dir}")

    metrics = YOLO(str(best_path)).val(
        data=str(data_path),
        split="val",
        imgsz=spec["imgsz"],
        plots=True,
        project=str(Path(spec["artifact_root"]) / "val"),
        name=f"{spec['name']}_best_val",
        exist_ok=True,
    )

    return {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "name": spec["name"],
        "weights_path": str(best_path),
        "map50": float(metrics.box.map50),
        "map": float(metrics.box.map),
    }


def append_result(record: dict[str, Any], artifact_root: str | Path) -> None:
    import pandas as pd

    artifact_root = Path(artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    csv_path = artifact_root / "leaderboard.csv"
    jsonl_path = artifact_root / "leaderboard.jsonl"

    frame = pd.DataFrame([record])
    if csv_path.exists():
        frame = pd.concat([pd.read_csv(csv_path), frame], ignore_index=True)
    frame.to_csv(csv_path, index=False)

    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
