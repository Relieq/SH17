from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


CLASS_NAMES = [
    "person",
    "ear",
    "ear-mufs",
    "face",
    "face-guard",
    "face-mask",
    "foot",
    "tool",
    "glasses",
    "gloves",
    "helmet",
    "hands",
    "head",
    "medical-suit",
    "shoes",
    "safety-suit",
    "safety-vest",
]

MINORITY_CLASS_IDS_ZERO_BASED = {2, 4, 6, 10, 13, 16}
MINORITY_REPEAT_FACTOR = 3
PICODET_L_PARAMS_M = 5.80
AUGMENTATION_PROFILES = ("official", "no_distort", "fast", "fixed")


@dataclass(frozen=True)
class PicodetExperiment:
    name: str
    input_size: int
    train_json_name: str
    batch_size: int
    base_lr: float
    purpose: str
    config_file_name: str
    augmentation_profile: str = "fast"
    resize_profile: str = "standard"

    @property
    def official_config_name(self) -> str:
        return f"picodet_l_{self.input_size}_coco_lcnet.yml"


def picodet_experiments(
    augmentation_profile: str = "fast",
    epoch_label: str = "20e",
) -> list[PicodetExperiment]:
    if augmentation_profile not in AUGMENTATION_PROFILES:
        raise ValueError(
            f"Unsupported augmentation_profile={augmentation_profile!r}. "
            f"Expected one of {AUGMENTATION_PROFILES}."
        )
    suffix = f"{augmentation_profile}_{epoch_label}"
    return [
        PicodetExperiment(
            name=f"picodet_l_320_baseline_{suffix}",
            input_size=320,
            train_json_name="instances_train.json",
            batch_size=24,
            base_lr=0.12,
            purpose=f"{augmentation_profile}-profile lightweight PP-PicoDet-L baseline at 320 input for pilot training",
            config_file_name=f"picodet_l_320_{augmentation_profile}_sh17.yml",
            augmentation_profile=augmentation_profile,
        ),
        PicodetExperiment(
            name=f"picodet_l_640_scale_{suffix}",
            input_size=640,
            train_json_name="instances_train.json",
            batch_size=12,
            base_lr=0.06,
            purpose=f"{augmentation_profile}-profile high-resolution PP-PicoDet-L scale variant for small PPE objects",
            config_file_name=f"picodet_l_640_{augmentation_profile}_sh17.yml",
            augmentation_profile=augmentation_profile,
        ),
        PicodetExperiment(
            name=f"picodet_l_640_oversample_{suffix}",
            input_size=640,
            train_json_name="instances_train_oversampled.json",
            batch_size=12,
            base_lr=0.06,
            purpose=f"{augmentation_profile}-profile high-resolution variant with minority-class oversampling",
            config_file_name=f"picodet_l_640_oversample_{augmentation_profile}_sh17.yml",
            augmentation_profile=augmentation_profile,
        ),
        PicodetExperiment(
            name=f"picodet_l_640_zoom_crop_{suffix}",
            input_size=640,
            train_json_name="instances_train.json",
            batch_size=12,
            base_lr=0.06,
            purpose=f"{augmentation_profile}-profile high-resolution variant with wider scale jitter inspired by the EffDet zoom-crop ablation",
            config_file_name=f"picodet_l_640_zoom_crop_{augmentation_profile}_sh17.yml",
            augmentation_profile=augmentation_profile,
            resize_profile="zoom_crop",
        ),
    ]


def picodet_speed_benchmark_experiments(
    input_size: int = 320,
    train_json_name: str = "instances_train_benchmark_960.json",
) -> list[PicodetExperiment]:
    return [
        PicodetExperiment(
            name=f"picodet_l_{input_size}_bench_{profile}",
            input_size=input_size,
            train_json_name=train_json_name,
            batch_size=24,
            base_lr=0.12,
            purpose=f"speed benchmark for {profile} augmentation profile",
            config_file_name=f"bench_picodet_l_{input_size}_{profile}.yml",
            augmentation_profile=profile,
        )
        for profile in AUGMENTATION_PROFILES
    ]


def _path_for_yaml(path: Path) -> str:
    return str(path).replace("\\", "/")


def read_manifest(manifest_path: Path, dataset_root: Path) -> list[Path]:
    image_paths: list[Path] = []
    for raw in manifest_path.read_text(encoding="utf-8").splitlines():
        item = raw.strip()
        if not item:
            continue
        candidate = Path(item)
        if not candidate.is_absolute():
            candidate = dataset_root / item
        if not candidate.exists():
            fallback = dataset_root / "images" / candidate.name
            if fallback.exists():
                candidate = fallback
        image_paths.append(candidate)
    return image_paths


def default_image_size_resolver(image_path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(image_path) as image:
        return image.size


def label_path_for_image(image_path: Path, dataset_root: Path) -> Path:
    candidates = [
        dataset_root / "labels" / f"{image_path.stem}.txt",
        image_path.with_suffix(".txt"),
        image_path.parent / "labels" / f"{image_path.stem}.txt",
        image_path.parent.parent / "labels" / f"{image_path.stem}.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _relative_file_name(image_path: Path, dataset_root: Path) -> str:
    try:
        return image_path.relative_to(dataset_root).as_posix()
    except ValueError:
        return image_path.name


def _validate_yolo_values(
    class_id: int,
    xc: float,
    yc: float,
    bw: float,
    bh: float,
    label_path: Path,
) -> None:
    if class_id < 0 or class_id >= len(CLASS_NAMES):
        raise ValueError(f"{label_path}: class id {class_id} outside [0, {len(CLASS_NAMES) - 1}]")
    values = {"xc": xc, "yc": yc, "bw": bw, "bh": bh}
    for name, value in values.items():
        if value < 0.0 or value > 1.0:
            raise ValueError(f"{label_path}: {name}={value} outside [0, 1]")
    if bw <= 0.0 or bh <= 0.0:
        raise ValueError(f"{label_path}: bbox width/height must be positive")


def yolo_label_rows(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    if not label_path.exists():
        raise FileNotFoundError(f"Missing label file: {label_path}")

    rows: list[tuple[int, float, float, float, float]] = []
    for line_number, raw in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            raise ValueError(f"{label_path}:{line_number}: expected 5 YOLO label values")
        class_id = int(float(parts[0]))
        xc, yc, bw, bh = [float(value) for value in parts[1:5]]
        _validate_yolo_values(class_id, xc, yc, bw, bh, label_path)
        rows.append((class_id, xc, yc, bw, bh))
    return rows


def convert_manifest_to_coco(
    manifest_path: Path,
    dataset_root: Path,
    output_json: Path,
    image_size_resolver: Callable[[Path], tuple[int, int]] | None = None,
) -> dict[str, int]:
    resolver = image_size_resolver or default_image_size_resolver
    image_paths = read_manifest(manifest_path, dataset_root)

    images = []
    annotations = []
    annotation_id = 1

    for image_id, image_path in enumerate(image_paths, start=1):
        width, height = resolver(image_path)
        images.append(
            {
                "id": image_id,
                "file_name": _relative_file_name(image_path, dataset_root),
                "width": int(width),
                "height": int(height),
            }
        )

        label_path = label_path_for_image(image_path, dataset_root)
        for class_id, xc, yc, bw, bh in yolo_label_rows(label_path):
            box_w = bw * width
            box_h = bh * height
            x_min = (xc - bw / 2.0) * width
            y_min = (yc - bh / 2.0) * height
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id + 1,
                    "bbox": [
                        round(x_min, 6),
                        round(y_min, 6),
                        round(box_w, 6),
                        round(box_h, 6),
                    ],
                    "area": round(box_w * box_h, 6),
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    payload = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": index + 1, "name": class_name, "supercategory": "ppe"}
            for index, class_name in enumerate(CLASS_NAMES)
        ],
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {
        "images": len(images),
        "instances": len(annotations),
        "categories": len(CLASS_NAMES),
    }


def build_oversampled_coco(
    train_json: Path,
    output_json: Path,
    minority_class_ids_zero_based: Iterable[int] = MINORITY_CLASS_IDS_ZERO_BASED,
    repeat_factor: int = MINORITY_REPEAT_FACTOR,
) -> dict[str, int]:
    if repeat_factor < 1:
        raise ValueError("repeat_factor must be >= 1")

    payload = json.loads(train_json.read_text(encoding="utf-8"))
    minority_category_ids = {class_id + 1 for class_id in minority_class_ids_zero_based}

    annotations_by_image: dict[int, list[dict]] = {}
    for annotation in payload["annotations"]:
        annotations_by_image.setdefault(annotation["image_id"], []).append(annotation)

    output_images = [copy.deepcopy(image) for image in payload["images"]]
    output_annotations = [copy.deepcopy(annotation) for annotation in payload["annotations"]]
    next_image_id = max((image["id"] for image in output_images), default=0) + 1
    next_annotation_id = max((annotation["id"] for annotation in output_annotations), default=0) + 1

    for image in payload["images"]:
        image_annotations = annotations_by_image.get(image["id"], [])
        has_minority = any(annotation["category_id"] in minority_category_ids for annotation in image_annotations)
        if not has_minority:
            continue
        for _ in range(repeat_factor - 1):
            duplicated_image = copy.deepcopy(image)
            duplicated_image["id"] = next_image_id
            output_images.append(duplicated_image)
            for annotation in image_annotations:
                duplicated_annotation = copy.deepcopy(annotation)
                duplicated_annotation["id"] = next_annotation_id
                duplicated_annotation["image_id"] = next_image_id
                output_annotations.append(duplicated_annotation)
                next_annotation_id += 1
            next_image_id += 1

    output_payload = {
        "images": output_images,
        "annotations": output_annotations,
        "categories": payload["categories"],
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
    return {
        "images": len(output_images),
        "instances": len(output_annotations),
        "categories": len(output_payload["categories"]),
    }


def build_benchmark_coco_subset(source_json: Path, output_json: Path, image_limit: int) -> dict[str, int]:
    if image_limit < 1:
        raise ValueError("image_limit must be >= 1")

    payload = json.loads(source_json.read_text(encoding="utf-8"))
    subset_images = copy.deepcopy(payload["images"][:image_limit])
    subset_image_ids = {image["id"] for image in subset_images}
    subset_annotations = [
        copy.deepcopy(annotation)
        for annotation in payload["annotations"]
        if annotation["image_id"] in subset_image_ids
    ]
    output_payload = {
        "images": subset_images,
        "annotations": subset_annotations,
        "categories": copy.deepcopy(payload["categories"]),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
    return {
        "images": len(subset_images),
        "instances": len(subset_annotations),
        "categories": len(output_payload["categories"]),
    }


def parse_picodet_speed_log(log_path: Path) -> dict[str, str | int]:
    pattern = re.compile(
        r"batch_cost:\s*([0-9.]+)\s+data_cost:\s*([0-9.]+)\s+ips:\s*([0-9.]+)"
    )
    memory_pattern = re.compile(r"max_mem_allocated:\s*([0-9]+)\s*MB")
    rows = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            memory_match = memory_pattern.search(line)
            rows.append(
                {
                    "batch_cost": float(match.group(1)),
                    "data_cost": float(match.group(2)),
                    "ips": float(match.group(3)),
                    "max_mem_allocated_mb": int(memory_match.group(1)) if memory_match else 0,
                }
            )

    if not rows:
        return {
            "profile_status": "pending",
            "batch_count": 0,
            "mean_batch_cost": "pending",
            "mean_data_cost": "pending",
            "mean_ips": "pending",
            "max_mem_allocated_mb": 0,
        }

    return {
        "profile_status": "real",
        "batch_count": len(rows),
        "mean_batch_cost": f"{sum(row['batch_cost'] for row in rows) / len(rows):.6f}",
        "mean_data_cost": f"{sum(row['data_cost'] for row in rows) / len(rows):.6f}",
        "mean_ips": f"{sum(row['ips'] for row in rows) / len(rows):.6f}",
        "max_mem_allocated_mb": max(row["max_mem_allocated_mb"] for row in rows),
    }


def build_picodet_config_text(
    experiment: PicodetExperiment,
    dataset_dir: Path,
    output_dir: Path,
    paddledet_root: Path,
    epochs: int = 50,
    snapshot_epoch: int = 5,
    worker_num: int = 6,
    use_shared_memory: bool = True,
    use_gpu: bool = True,
    annotations_dir: Path | None = None,
) -> str:
    official_config = paddledet_root / "configs" / "picodet" / experiment.official_config_name
    annotations_dir = annotations_dir or output_dir.parent / "dataset" / "annotations"
    use_gpu_value = "true" if use_gpu else "false"
    shared_memory_value = "true" if use_shared_memory else "false"
    reader_text = _picodet_reader_override(
        experiment=experiment,
        batch_size=experiment.batch_size,
        use_shared_memory=shared_memory_value,
    )
    return f"""_BASE_: ["{_path_for_yaml(official_config)}"]

# sh17_metadata:
# augmentation_profile: {experiment.augmentation_profile}
# resize_profile: {experiment.resize_profile}
metric: COCO
num_classes: 17
use_gpu: {use_gpu_value}
use_ema: true
epoch: {epochs}
snapshot_epoch: {snapshot_epoch}
worker_num: {worker_num}
save_dir: {_path_for_yaml(output_dir / experiment.name)}
pretrain_weights: https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x2_0_pretrained.pdparams
weights: {_path_for_yaml(output_dir / experiment.name / "best_model")}

TrainDataset:
  name: COCODataSet
  dataset_dir: {_path_for_yaml(dataset_dir)}
  image_dir: .
  anno_path: {_path_for_yaml(annotations_dir / experiment.train_json_name)}
  data_fields: ['image', 'gt_bbox', 'gt_class', 'is_crowd']

EvalDataset:
  name: COCODataSet
  dataset_dir: {_path_for_yaml(dataset_dir)}
  image_dir: .
  anno_path: {_path_for_yaml(annotations_dir / "instances_val.json")}
  allow_empty: true

TestDataset:
  name: ImageFolder
  dataset_dir: {_path_for_yaml(dataset_dir)}
  anno_path: {_path_for_yaml(annotations_dir / "instances_val.json")}

LearningRate:
  base_lr: {experiment.base_lr}
  schedulers:
  - name: CosineDecay
    max_epochs: {epochs}
  - name: LinearWarmup
    start_factor: 0.1
    steps: 300

{reader_text}

EvalReader:
  batch_size: 8
"""


def _picodet_reader_override(
    experiment: PicodetExperiment,
    batch_size: int,
    use_shared_memory: str,
) -> str:
    target_sizes = _batch_random_resize_targets(experiment)
    sample_transforms = _sample_transform_lines(experiment.augmentation_profile)
    return f"""TrainReader:
  sample_transforms:
{sample_transforms}
  batch_transforms:
  - BatchRandomResize: {{target_size: {target_sizes}, random_size: True, random_interp: True, keep_ratio: False}}
  - NormalizeImage: {{is_scale: true, mean: [0.485,0.456,0.406], std: [0.229, 0.224,0.225]}}
  - Permute: {{}}
  - PadGT: {{}}
  batch_size: {batch_size}
  shuffle: true
  drop_last: true
  use_shared_memory: {use_shared_memory}"""


def _sample_transform_lines(augmentation_profile: str) -> str:
    if augmentation_profile not in AUGMENTATION_PROFILES:
        raise ValueError(
            f"Unsupported augmentation_profile={augmentation_profile!r}. "
            f"Expected one of {AUGMENTATION_PROFILES}."
        )

    transforms = ["Decode"]
    if augmentation_profile == "official":
        transforms.extend(["RandomCrop", "RandomFlip", "RandomDistort"])
    elif augmentation_profile == "no_distort":
        transforms.extend(["RandomCrop", "RandomFlip"])
    else:
        transforms.append("RandomFlip")

    lines = []
    for transform in transforms:
        if transform == "RandomFlip":
            lines.append("  - RandomFlip: {prob: 0.5}")
        else:
            lines.append(f"  - {transform}: {{}}")
    return "\n".join(lines)


def _batch_random_resize_targets(experiment: PicodetExperiment) -> str:
    if experiment.augmentation_profile == "fixed":
        return f"[{experiment.input_size}]"
    if experiment.resize_profile == "zoom_crop":
        return "[512, 544, 576, 608, 640, 672, 704, 736, 768]"
    if experiment.input_size == 320:
        return "[256, 288, 320, 352, 384]"
    if experiment.input_size == 640:
        return "[576, 608, 640, 672, 704]"
    raise ValueError(f"Unsupported PP-PicoDet-L input size: {experiment.input_size}")


def write_picodet_configs(
    config_dir: Path,
    dataset_dir: Path,
    output_dir: Path,
    paddledet_root: Path,
    epochs: int = 50,
    snapshot_epoch: int = 5,
    worker_num: int = 6,
    use_shared_memory: bool = True,
    use_gpu: bool = True,
    experiments: Iterable[PicodetExperiment] | None = None,
    annotations_dir: Path | None = None,
) -> list[Path]:
    config_dir.mkdir(parents=True, exist_ok=True)
    config_paths = []
    for experiment in experiments or picodet_experiments():
        config_path = config_dir / experiment.config_file_name
        config_path.write_text(
            build_picodet_config_text(
                experiment=experiment,
                dataset_dir=dataset_dir,
                output_dir=output_dir,
                paddledet_root=paddledet_root,
                epochs=epochs,
                snapshot_epoch=snapshot_epoch,
                worker_num=worker_num,
                use_shared_memory=use_shared_memory,
                use_gpu=use_gpu,
                annotations_dir=annotations_dir,
            ),
            encoding="utf-8",
        )
        config_paths.append(config_path)
    return config_paths
