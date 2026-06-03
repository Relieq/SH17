# Kế Hoạch Triển Khai Notebook D-FINE-M 896 SH17 Nhiều Biến Thể

> **Dành cho agent triển khai:** SUB-SKILL BẮT BUỘC: dùng `superpowers:subagent-driven-development` (khuyến nghị) hoặc `superpowers:executing-plans` để triển khai kế hoạch này theo từng công việc. Các bước dùng checkbox (`- [ ]`) để theo dõi tiến độ.

**Mục tiêu:** Tạo một notebook chạy được trên Colab để fine-tune D-FINE-M ở kích thước 896px trên SH17 với một baseline và ba biến thể, sau đó xuất bảng so sánh tổng quan/theo từng class cùng kiểu với các bảng YOLOv9s trong slide hiện tại.

**Kiến trúc:** Notebook phụ trách toàn bộ quy trình: cài môi trường, đọc label YOLO của SH17, khai báo danh sách experiment, augmentation/oversampling theo từng biến thể, load D-FINE-M, smoke test trên TPU, training, evaluation và tạo bảng kết quả. Không tạo helper script, YAML config hoặc module repo phụ; artifact triển khai chỉ là một notebook duy nhất. Metric được tính từ prediction trên validation set bằng AP kiểu COCO, cộng với quy tắc khớp Precision/Recall cố định và được ghi rõ.

**Công nghệ sử dụng:** Google Colab TPU v5e-1, PyTorch/XLA, Hugging Face Transformers D-FINE, `ustc-community/dfine-medium-obj2coco`, Pillow, Albumentations, pycocotools, pandas, NumPy.

---

## Cấu Trúc File

Phần triển khai chỉ được tạo hoặc chỉnh sửa notebook này:

- Tạo: `/Users/tranquangtrong/SH17/SH17_dfine_m_896_benchmark.ipynb`

Notebook có thể ghi artifact runtime vào một thư mục output có thể cấu hình, thường là Google Drive:

- Thư mục output runtime gốc: `/content/drive/MyDrive/SH17_outputs/dfine_m_896_variants/`
- Output runtime cho từng biến thể: `checkpoints/last`, `checkpoints/last.bak`, `checkpoints/best`, metrics CSV, CSV theo từng class, prediction JSON, run manifest

Không tạo Python script riêng, config file riêng hoặc test file riêng. Các cell kiểm tra trong notebook thay thế test bên ngoài cho experiment này.

## Ràng Buộc Experiment Cố Định

- Model: `ustc-community/dfine-medium-obj2coco`
- Họ model: D-FINE-M, khoảng 19M parameters trong model zoo chính thức
- Kích thước input: 896 x 896
- Số class: 17 class SH17, giữ nguyên thứ tự id
- Train split: `train_files.txt`, kỳ vọng 6,479 ảnh
- Validation split: `val_files.txt`, kỳ vọng 1,620 ảnh
- Oversampling class thiểu số: lặp lại ảnh chứa class id `{2, 4, 6, 10, 13, 16}` với repeat factor 3
- Số train entries sau oversampling kỳ vọng: 8,483
- Chính sách batch cho TPU v5e-1: bắt đầu với physical batch size 1, gradient accumulation 8, bf16
- Mục tiêu chính cần vượt: YOLOv9s oversampled 960 hiện tại, mAP50 70.2, mAP50-95 47.5
- Mục tiêu phụ cần vượt: YOLOv9-e trong paper SH17, mAP50 70.9, mAP50-95 48.7

## Ma Trận Biến Thể D-FINE-M

Chạy các biến thể này trong cùng một notebook, theo đúng thứ tự dưới đây, để bảng cuối phản chiếu trực tiếp với slide YOLOv9s:

| Biến thể | Tên experiment trong notebook | Tương ứng biến thể YOLOv9s | Thay đổi chính | Dữ liệu train |
| --- | --- | --- | --- | --- |
| 0 | `dfine_m_baseline_896` | `yolov9s_baseline_640` | Baseline D-FINE-M ở kích thước cố định 896, chỉ horizontal flip tối thiểu, không dùng early stopping | 6,479 ảnh train gốc |
| 1 | `dfine_m_tuned_896_es` | `yolov9s_tuned_640` | Tinh chỉnh optimizer, cosine LR, warmup dài hơn, gradient clipping, EMA, augmentation màu nhẹ, early stopping theo mAP50-95 | 6,479 ảnh train gốc |
| 2 | `dfine_m_scale_jitter_896_es` | `yolov9s_multiscale_960` | Kế thừa Variant 1 và thêm fixed-output scale jitter / bbox-safe crop ở 896 để hỗ trợ PPE nhỏ mà không tạo shape động trên TPU | 6,479 ảnh train gốc |
| 3 | `dfine_m_oversample_minority_896_es` | `yolov9s_oversample_minority_960` | Kế thừa Variant 2 và oversample các class thiểu số/PPE với repeat factor 3 | 8,483 train entries sau oversampling |

Variant 1 cố ý được thêm early stopping vì đây là model non-baseline đầu tiên và sẽ trở thành recipe tuned ổn định cho các biến thể sau. Metric dừng là validation `mAP50-95`, kiểm tra mỗi 5 epoch, với patience tính theo số vòng evaluation.

## Đọc Nhanh: Baseline Và Variant Nằm Ở Đâu?

Nếu chỉ muốn biết baseline và variant 1/2/3 nằm ở đâu, đọc phần này trước:

| Cần tìm | Tên experiment | Nằm ở đâu trong notebook | Output sau khi chạy |
| --- | --- | --- | --- |
| Baseline | `dfine_m_baseline_896` | Section `6.1 Baseline (Variant 0) — dfine_m_baseline_896` | `OUTPUT_ROOT/dfine_m_baseline_896/` |
| Variant 1 | `dfine_m_tuned_896_es` | Section `6.2 Variant 1 — dfine_m_tuned_896_es` | `OUTPUT_ROOT/dfine_m_tuned_896_es/` |
| Variant 2 | `dfine_m_scale_jitter_896_es` | Section `6.3 Variant 2 — dfine_m_scale_jitter_896_es` | `OUTPUT_ROOT/dfine_m_scale_jitter_896_es/` |
| Variant 3 | `dfine_m_oversample_minority_896_es` | Section `6.4 Variant 3 — dfine_m_oversample_minority_896_es` | `OUTPUT_ROOT/dfine_m_oversample_minority_896_es/` |

Trong file plan này, các chỗ cần nhìn là:

- **Ma trận so sánh ngắn:** section `Ma Trận Biến Thể D-FINE-M`, ngay phía trên.
- **Config thật của baseline/variant:** `Công Việc 2`, `Bước 1`, biến `EXPERIMENTS`.
- **Markdown riêng trong notebook:** `Công Việc 1`, `Bước 2`, và code `VARIANT_MARKDOWN` trong `Công Việc 2`.
- **Thứ tự train thật:** `Công Việc 5`, `Bước 6`, biến `RUN_EXPERIMENT_NAMES`.
- **Kết quả/checkpoint:** mỗi experiment ghi vào thư mục `OUTPUT_ROOT/<tên_experiment>/`, ví dụ `OUTPUT_ROOT/dfine_m_tuned_896_es/checkpoints/last`.

Trong notebook sau khi tạo, phần `Runner Train Theo Biến Thể` phải nhìn được như sau:

```text
6. Runner Train Theo Biến Thể
   6.1 Baseline (Variant 0) — dfine_m_baseline_896
   6.2 Variant 1 — dfine_m_tuned_896_es
   6.3 Variant 2 — dfine_m_scale_jitter_896_es
   6.4 Variant 3 — dfine_m_oversample_minority_896_es
```

## Tài Liệu Tham Khảo Đã Dùng

- Repository/model zoo chính thức của D-FINE: https://github.com/Peterande/D-FINE
- Paper D-FINE: https://arxiv.org/abs/2410.13842
- Tài liệu Hugging Face D-FINE: https://huggingface.co/docs/transformers/main/model_doc/d_fine
- Thông số chính thức TPU v5e: https://docs.cloud.google.com/tpu/docs/v5e

---

### Công Việc 1: Tạo Khung Notebook Và Cell Cài Môi Trường

**File:**
- Tạo: `/Users/tranquangtrong/SH17/SH17_dfine_m_896_benchmark.ipynb`

- [ ] **Bước 1: Tạo notebook với các heading theo section**

Tạo các section notebook theo đúng thứ tự này:

```text
1. Cài Đặt Môi Trường
2. Config Experiment Và Registry Biến Thể
3. Kiểm Tra Dataset SH17
4. Dataset Và Collator
5. Model D-FINE-M Và TPU Smoke Test
6. Runner Train Theo Biến Thể
7. Metric Validation
8. Bảng So Sánh Cross-Variant
9. Phân Tích Lỗi
10. Export Cuối
```

- [ ] **Bước 2: Thêm markdown cell riêng cho từng biến thể**

Thêm bốn markdown cell này vào notebook, đặt trong section `Runner Train Theo Biến Thể` trước cell chạy `RUN_EXPERIMENT_NAMES`. Mỗi markdown cell phải đứng ngay trước lệnh chạy hoặc phần mô tả của biến thể tương ứng để sau này đọc notebook vẫn hiểu từng kết quả đến từ variant nào.

```text
### 6.1 Baseline (Variant 0) — dfine_m_baseline_896

Mục đích: tạo baseline D-FINE-M ở input 896 để có mốc so sánh ban đầu với YOLOv9s baseline.
Dữ liệu train: 6,479 ảnh gốc, không oversampling.
Training: horizontal flip tối thiểu, không EMA, không early stopping.
Output cần theo dõi: training_history.csv, best_overall_metrics.csv, best_per_class_metrics.csv, checkpoints/last, checkpoints/best.

### 6.2 Variant 1 — dfine_m_tuned_896_es

Mục đích: cải thiện hội tụ so với baseline bằng tuned optimizer và early stopping.
Dữ liệu train: 6,479 ảnh gốc, không oversampling.
Training: AdamW tách LR backbone/head, cosine LR, warmup dài hơn, gradient clipping, EMA, augmentation màu nhẹ, early stopping theo validation mAP50-95.
Output cần theo dõi: epoch hiện tại, train_loss, P, R, mAP50, mAP50-95, best_epoch, stopped_early.

### 6.3 Variant 2 — dfine_m_scale_jitter_896_es

Mục đích: mô phỏng hướng multiscale/high-resolution của YOLOv9s nhưng vẫn giữ output cố định 896 để tránh shape động trên TPU.
Dữ liệu train: 6,479 ảnh gốc, không oversampling.
Training: kế thừa Variant 1, thêm scale jitter và bbox-safe crop để cải thiện object nhỏ.
Output cần theo dõi: delta mAP50/mAP50-95 so với Variant 1 và YOLOv9s multiscale.

### 6.4 Variant 3 — dfine_m_oversample_minority_896_es

Mục đích: tăng recall/AP cho các class PPE thiểu số giống hướng YOLOv9s oversample minority.
Dữ liệu train: 8,483 entries sau oversampling; repeat factor 3 cho ảnh chứa class id {2, 4, 6, 10, 13, 16}.
Training: kế thừa Variant 2, bật oversampling và early stopping patience dài hơn.
Output cần theo dõi: per-class AP của helmet, safety-vest, ear-mufs, foot, medical-suit, face-guard.
```

- [ ] **Bước 3: Thêm cell install/import**

Dùng nội dung này làm code cell đầu tiên:

```python
!pip -q install "transformers>=4.56.0" "accelerate>=1.0.0" "albumentations>=1.4.0" "pycocotools>=2.0.7" "pandas>=2.0.0" "pillow>=10.0.0" "tqdm>=4.66.0"

import json
import math
import os
import random
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd
import torch
from IPython.display import Markdown, display
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, DFineForObjectDetection, get_cosine_schedule_with_warmup

try:
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
except Exception as exc:
    XLA_AVAILABLE = False
    XLA_IMPORT_ERROR = repr(exc)

print("torch", torch.__version__)
print("xla_available", XLA_AVAILABLE)
```

- [ ] **Bước 4: Chạy cell install/import**

Kỳ vọng:

```text
xla_available True
```

Nếu `xla_available` là false trên Colab TPU runtime, dừng run và chuyển Colab runtime sang TPU trước khi tiếp tục.

- [ ] **Bước 5: Commit khung notebook**

```bash
git add SH17_dfine_m_896_benchmark.ipynb
git commit -m "feat: add D-FINE-M 896 notebook shell"
```

---

### Công Việc 2: Thêm Config Experiment, Registry Biến Thể Và Kiểm Tra Dataset

**File:**
- Chỉnh sửa: `/Users/tranquangtrong/SH17/SH17_dfine_m_896_benchmark.ipynb`

- [ ] **Bước 1: Thêm cell config experiment**

```python
SEED = 42
MODEL_ID = "ustc-community/dfine-medium-obj2coco"
DEFAULT_IMAGE_SIZE = 896
NUM_CLASSES = 17
MINORITY_CLASS_IDS = {2, 4, 6, 10, 13, 16}
MINORITY_REPEAT_FACTOR = 3

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

ID2LABEL = {idx: name for idx, name in enumerate(CLASS_NAMES)}
LABEL2ID = {name: idx for idx, name in ID2LABEL.items()}

DATA_ROOT_CANDIDATES = [
    Path("/content/sh17_data"),
    Path("/content/drive/MyDrive/sh17_data"),
    Path("/content/drive/MyDrive/SH17/sh17_data"),
    Path("/Users/tranquangtrong/Desktop/SH17_trong_models/sh17_data"),
]

OUTPUT_ROOT = Path("/content/drive/MyDrive/SH17_outputs/dfine_m_896_variants")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    display_name: str
    mirrors: str
    image_size: int = DEFAULT_IMAGE_SIZE
    epochs: int = 120
    physical_batch_size: int = 1
    grad_accum_steps: int = 8
    num_workers: int = 2
    backbone_lr: float = 2e-5
    head_lr: float = 2e-4
    weight_decay: float = 1e-4
    warmup_ratio: float = 0.05
    eval_every_epochs: int = 5
    save_every_epochs: int = 10
    save_last_every_epoch: bool = True
    resume_from_checkpoint: bool = True
    max_grad_norm: float = 0.1
    augmentation_policy: str = "baseline"
    use_oversampling: bool = False
    use_ema: bool = False
    ema_decay: float = 0.999
    early_stopping: bool = False
    early_stopping_patience: int = 8
    early_stopping_min_delta: float = 0.10

EXPERIMENTS = [
    ExperimentConfig(
        name="dfine_m_baseline_896",
        display_name="D-FINE-M baseline 896",
        mirrors="yolov9s_baseline_640",
        epochs=100,
        backbone_lr=3e-5,
        head_lr=1e-4,
        augmentation_policy="baseline",
        use_oversampling=False,
        use_ema=False,
        early_stopping=False,
    ),
    ExperimentConfig(
        name="dfine_m_tuned_896_es",
        display_name="D-FINE-M tuned 896 + early stopping",
        mirrors="yolov9s_tuned_640",
        epochs=140,
        backbone_lr=2e-5,
        head_lr=2e-4,
        warmup_ratio=0.08,
        augmentation_policy="tuned",
        use_oversampling=False,
        use_ema=True,
        early_stopping=True,
        early_stopping_patience=8,
        early_stopping_min_delta=0.10,
    ),
    ExperimentConfig(
        name="dfine_m_scale_jitter_896_es",
        display_name="D-FINE-M scale-jitter 896 + early stopping",
        mirrors="yolov9s_multiscale_960",
        epochs=140,
        backbone_lr=2e-5,
        head_lr=2e-4,
        warmup_ratio=0.08,
        augmentation_policy="scale_jitter",
        use_oversampling=False,
        use_ema=True,
        early_stopping=True,
        early_stopping_patience=8,
        early_stopping_min_delta=0.10,
    ),
    ExperimentConfig(
        name="dfine_m_oversample_minority_896_es",
        display_name="D-FINE-M oversample minority 896 + early stopping",
        mirrors="yolov9s_oversample_minority_960",
        epochs=160,
        backbone_lr=2e-5,
        head_lr=2e-4,
        warmup_ratio=0.08,
        augmentation_policy="scale_jitter",
        use_oversampling=True,
        use_ema=True,
        early_stopping=True,
        early_stopping_patience=10,
        early_stopping_min_delta=0.10,
    ),
]

VARIANT_MARKDOWN = {
    "dfine_m_baseline_896": """
### 6.1 Baseline (Variant 0) — dfine_m_baseline_896

Mục đích: tạo baseline D-FINE-M ở input 896 để có mốc so sánh ban đầu với YOLOv9s baseline.
Dữ liệu train: 6,479 ảnh gốc, không oversampling.
Training: horizontal flip tối thiểu, không EMA, không early stopping.
Output cần theo dõi: training_history.csv, best_overall_metrics.csv, best_per_class_metrics.csv, checkpoints/last, checkpoints/best.
""",
    "dfine_m_tuned_896_es": """
### 6.2 Variant 1 — dfine_m_tuned_896_es

Mục đích: cải thiện hội tụ so với baseline bằng tuned optimizer và early stopping.
Dữ liệu train: 6,479 ảnh gốc, không oversampling.
Training: AdamW tách LR backbone/head, cosine LR, warmup dài hơn, gradient clipping, EMA, augmentation màu nhẹ, early stopping theo validation mAP50-95.
Output cần theo dõi: epoch hiện tại, train_loss, P, R, mAP50, mAP50-95, best_epoch, stopped_early.
""",
    "dfine_m_scale_jitter_896_es": """
### 6.3 Variant 2 — dfine_m_scale_jitter_896_es

Mục đích: mô phỏng hướng multiscale/high-resolution của YOLOv9s nhưng vẫn giữ output cố định 896 để tránh shape động trên TPU.
Dữ liệu train: 6,479 ảnh gốc, không oversampling.
Training: kế thừa Variant 1, thêm scale jitter và bbox-safe crop để cải thiện object nhỏ.
Output cần theo dõi: delta mAP50/mAP50-95 so với Variant 1 và YOLOv9s multiscale.
""",
    "dfine_m_oversample_minority_896_es": """
### 6.4 Variant 3 — dfine_m_oversample_minority_896_es

Mục đích: tăng recall/AP cho các class PPE thiểu số giống hướng YOLOv9s oversample minority.
Dữ liệu train: 8,483 entries sau oversampling; repeat factor 3 cho ảnh chứa class id {2, 4, 6, 10, 13, 16}.
Training: kế thừa Variant 2, bật oversampling và early stopping patience dài hơn.
Output cần theo dõi: per-class AP của helmet, safety-vest, ear-mufs, foot, medical-suit, face-guard.
""",
}

def display_variant_markdown(config):
    display(Markdown(VARIANT_MARKDOWN[config.name]))

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def resolve_data_root(candidates):
    for candidate in candidates:
        if (candidate / "images").exists() and (candidate / "labels").exists():
            return candidate
    checked = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Cannot find SH17 data root. Checked:\n{checked}")

def experiment_output_dir(config):
    path = OUTPUT_ROOT / config.name
    path.mkdir(parents=True, exist_ok=True)
    return path

DATA_ROOT = resolve_data_root(DATA_ROOT_CANDIDATES)
IMAGE_DIR = DATA_ROOT / "images"
LABEL_DIR = DATA_ROOT / "labels"
TRAIN_MANIFEST = DATA_ROOT / "train_files.txt"
VAL_MANIFEST = DATA_ROOT / "val_files.txt"

print("DATA_ROOT:", DATA_ROOT)
print("OUTPUT_ROOT:", OUTPUT_ROOT)
print(pd.DataFrame([config.__dict__ for config in EXPERIMENTS])[["name", "mirrors", "augmentation_policy", "use_oversampling", "early_stopping", "use_ema"]])
```

- [ ] **Bước 2: Thêm function đọc manifest và parse label YOLO**

```python
def read_manifest(path):
    rows = []
    for raw in Path(path).read_text().splitlines():
        raw = raw.strip()
        if not raw:
            continue
        image_path = Path(raw)
        if not image_path.is_absolute():
            image_path = DATA_ROOT / raw
        if not image_path.exists():
            image_path = IMAGE_DIR / image_path.name
        rows.append(image_path)
    return rows

def label_path_for_image(image_path):
    return LABEL_DIR / f"{Path(image_path).stem}.txt"

def parse_yolo_label(label_path, image_width, image_height):
    boxes_xyxy = []
    labels = []
    if not Path(label_path).exists():
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    for raw in Path(label_path).read_text().splitlines():
        raw = raw.strip()
        if not raw:
            continue
        cls, xc, yc, bw, bh = raw.split()[:5]
        cls = int(cls)
        xc, yc, bw, bh = map(float, (xc, yc, bw, bh))
        x1 = (xc - bw / 2.0) * image_width
        y1 = (yc - bh / 2.0) * image_height
        x2 = (xc + bw / 2.0) * image_width
        y2 = (yc + bh / 2.0) * image_height
        x1 = max(0.0, min(float(image_width), x1))
        y1 = max(0.0, min(float(image_height), y1))
        x2 = max(0.0, min(float(image_width), x2))
        y2 = max(0.0, min(float(image_height), y2))
        if x2 > x1 and y2 > y1:
            boxes_xyxy.append([x1, y1, x2, y2])
            labels.append(cls)

    return np.asarray(boxes_xyxy, dtype=np.float32), np.asarray(labels, dtype=np.int64)

def inspect_image(path):
    with Image.open(path) as image:
        return image.size
```

- [ ] **Bước 3: Thêm cell kiểm tra tính toàn vẹn dataset**

```python
train_images = read_manifest(TRAIN_MANIFEST)
val_images = read_manifest(VAL_MANIFEST)

def summarize_split(image_paths, split_name):
    class_counter = Counter()
    image_counter = Counter()
    missing_labels = 0
    for image_path in tqdm(image_paths, desc=f"checking {split_name}"):
        width, height = inspect_image(image_path)
        boxes, labels = parse_yolo_label(label_path_for_image(image_path), width, height)
        if not label_path_for_image(image_path).exists():
            missing_labels += 1
        class_counter.update(labels.tolist())
        image_counter.update(set(labels.tolist()))
    return {
        "split": split_name,
        "images": len(image_paths),
        "instances": sum(class_counter.values()),
        "missing_labels": missing_labels,
        "class_counter": class_counter,
        "image_counter": image_counter,
    }

train_summary = summarize_split(train_images, "train")
val_summary = summarize_split(val_images, "val")

print("train images", train_summary["images"])
print("val images", val_summary["images"])
print("train instances", train_summary["instances"])
print("val instances", val_summary["instances"])
print("total instances", train_summary["instances"] + val_summary["instances"])
print("missing labels", train_summary["missing_labels"] + val_summary["missing_labels"])
```

- [ ] **Bước 4: Chạy các cell kiểm tra dataset**

Kỳ vọng:

```text
train images 6479
val images 1620
total instances 75994
missing labels 0
```

- [ ] **Bước 5: Commit config và các kiểm tra dataset**

```bash
git add SH17_dfine_m_896_benchmark.ipynb
git commit -m "feat: add SH17 checks for D-FINE notebook"
```

---

### Công Việc 3: Thêm Dataset Theo Biến Thể, Augmentation Và Collator

**File:**
- Chỉnh sửa: `/Users/tranquangtrong/SH17/SH17_dfine_m_896_benchmark.ipynb`

- [ ] **Bước 1: Thêm helper oversampling và train-entry**

```python
def image_contains_minority_class(image_path):
    width, height = inspect_image(image_path)
    _, labels = parse_yolo_label(label_path_for_image(image_path), width, height)
    return bool(set(labels.tolist()) & MINORITY_CLASS_IDS)

def build_oversampled_entries(image_paths):
    entries = []
    minority_count = 0
    for image_path in tqdm(image_paths, desc="oversampling train"):
        repeat = MINORITY_REPEAT_FACTOR if image_contains_minority_class(image_path) else 1
        if repeat > 1:
            minority_count += 1
        entries.extend([image_path] * repeat)
    return entries, minority_count

oversampled_train_images, minority_image_count = build_oversampled_entries(train_images)
print("minority train images", minority_image_count)
print("oversampled train entries", len(oversampled_train_images))

def train_entries_for_experiment(config):
    if config.use_oversampling:
        return oversampled_train_images
    return train_images

for config in EXPERIMENTS:
    print(config.name, "train entries", len(train_entries_for_experiment(config)))
```

- [ ] **Bước 2: Chạy kiểm tra train-entry**

Kỳ vọng:

```text
minority train images 1002
oversampled train entries 8483
dfine_m_baseline_896 train entries 6479
dfine_m_oversample_minority_896_es train entries 8483
```

- [ ] **Bước 3: Thêm augmentation factory và class dataset**

```python
def bbox_params(min_visibility):
    return A.BboxParams(
        format="pascal_voc",
        label_fields=["class_labels"],
        min_visibility=min_visibility,
    )

def build_train_transform(config):
    if config.augmentation_policy == "baseline":
        transforms = [
            A.HorizontalFlip(p=0.5),
        ]
    elif config.augmentation_policy == "tuned":
        transforms = [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.12, contrast_limit=0.12, p=0.25),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=12, val_shift_limit=8, p=0.15),
        ]
    elif config.augmentation_policy == "scale_jitter":
        transforms = [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.12, contrast_limit=0.12, p=0.25),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=12, val_shift_limit=8, p=0.15),
            A.RandomScale(scale_limit=(-0.15, 0.20), p=0.30),
            A.RandomSizedBBoxSafeCrop(
                height=config.image_size,
                width=config.image_size,
                erosion_rate=0.08,
                p=0.25,
            ),
        ]
    else:
        raise ValueError(f"Unknown augmentation policy: {config.augmentation_policy}")

    return A.Compose(transforms, bbox_params=bbox_params(min_visibility=0.15))

def build_eval_transform():
    return A.Compose([], bbox_params=bbox_params(min_visibility=0.0))

def create_processor(config):
    return AutoImageProcessor.from_pretrained(
        MODEL_ID,
        do_resize=True,
        size={"height": config.image_size, "width": config.image_size},
        do_pad=False,
    )

class SH17DFineDataset(Dataset):
    def __init__(self, image_paths, processor, transform=None):
        self.image_paths = list(image_paths)
        self.processor = processor
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image_id = idx
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        boxes_xyxy, labels = parse_yolo_label(label_path_for_image(image_path), width, height)

        image_np = np.asarray(image)
        if self.transform is not None and len(boxes_xyxy) > 0:
            transformed = self.transform(
                image=image_np,
                bboxes=boxes_xyxy.tolist(),
                class_labels=labels.tolist(),
            )
            image_np = transformed["image"]
            boxes_xyxy = np.asarray(transformed["bboxes"], dtype=np.float32)
            labels = np.asarray(transformed["class_labels"], dtype=np.int64)

        coco_annotations = []
        for box, label in zip(boxes_xyxy, labels):
            x1, y1, x2, y2 = map(float, box)
            coco_annotations.append(
                {
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": max(0.0, x2 - x1) * max(0.0, y2 - y1),
                    "iscrowd": 0,
                }
            )

        encoded = self.processor(
            images=Image.fromarray(image_np),
            annotations={"image_id": image_id, "annotations": coco_annotations},
            return_tensors="pt",
        )

        target_boxes = torch.tensor(boxes_xyxy, dtype=torch.float32)
        target_labels = torch.tensor(labels, dtype=torch.long)
        return {
            "image_id": int(image_id),
            "image_path": str(image_path),
            "orig_size": torch.tensor([height, width], dtype=torch.long),
            "pixel_values": encoded["pixel_values"][0],
            "labels": encoded["labels"][0],
            "target_boxes_xyxy": target_boxes,
            "target_labels": target_labels,
        }

def collate_fn(batch):
    output = {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "labels": [item["labels"] for item in batch],
        "image_id": [item["image_id"] for item in batch],
        "image_path": [item["image_path"] for item in batch],
        "orig_size": torch.stack([item["orig_size"] for item in batch]),
        "target_boxes_xyxy": [item["target_boxes_xyxy"] for item in batch],
        "target_labels": [item["target_labels"] for item in batch],
    }
    return output

def build_train_dataset(config, processor):
    return SH17DFineDataset(
        train_entries_for_experiment(config),
        processor=processor,
        transform=build_train_transform(config),
    )

def build_val_dataset(processor):
    return SH17DFineDataset(
        val_images,
        processor=processor,
        transform=build_eval_transform(),
    )

debug_config = EXPERIMENTS[0]
debug_processor = create_processor(debug_config)
debug_train_dataset = build_train_dataset(debug_config, debug_processor)
debug_val_dataset = build_val_dataset(debug_processor)

sample = debug_train_dataset[0]
print(sample["pixel_values"].shape)
print(sample["labels"].keys())
print(sample["target_boxes_xyxy"].shape, sample["target_labels"].shape)
```

- [ ] **Bước 4: Validate một batch**

```python
train_loader_debug = DataLoader(
    debug_train_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=2,
    collate_fn=collate_fn,
)
debug_batch = next(iter(train_loader_debug))
print(debug_batch["pixel_values"].shape)
print(len(debug_batch["labels"]))
print(debug_batch["orig_size"])
```

Kỳ vọng:

```text
torch.Size([1, 3, 896, 896])
1
```

- [ ] **Bước 5: Commit dataset và collator**

```bash
git add SH17_dfine_m_896_benchmark.ipynb
git commit -m "feat: add D-FINE SH17 dataset adapter"
```

---

### Công Việc 4: Thêm Model Factory D-FINE-M Và TPU Smoke Test

**File:**
- Chỉnh sửa: `/Users/tranquangtrong/SH17/SH17_dfine_m_896_benchmark.ipynb`

- [ ] **Bước 1: Thêm device và model factory**

```python
if not XLA_AVAILABLE:
    raise RuntimeError(f"torch_xla is not available: {XLA_IMPORT_ERROR}")

device = xm.xla_device()
print("device", device)

def create_model(config, device):
    model = DFineForObjectDetection.from_pretrained(
        MODEL_ID,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    )
    model.to(device)
    return model

debug_model = create_model(EXPERIMENTS[0], device)

total_params = sum(param.numel() for param in debug_model.parameters())
trainable_params = sum(param.numel() for param in debug_model.parameters() if param.requires_grad)
print("total params", total_params)
print("trainable params", trainable_params)
```

- [ ] **Bước 2: Thêm helper đệ quy để chuyển tensor sang TPU**

```python
def move_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    return value

def model_inputs_from_batch(batch, device):
    inputs = {
        "pixel_values": batch["pixel_values"],
        "labels": batch["labels"],
    }
    if "pixel_mask" in batch:
        inputs["pixel_mask"] = batch["pixel_mask"]
    return move_to_device(inputs, device)
```

- [ ] **Bước 3: Thêm TPU smoke test một bước**

```python
smoke_loader = DataLoader(
    debug_train_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=2,
    collate_fn=collate_fn,
)

debug_model.train()
smoke_batch = next(iter(smoke_loader))
smoke_inputs = model_inputs_from_batch(smoke_batch, device)
smoke_outputs = debug_model(**smoke_inputs)
smoke_loss = smoke_outputs.loss
print("smoke loss", float(smoke_loss.detach().cpu()))
assert torch.isfinite(smoke_loss.detach().cpu()), "Smoke loss is not finite"

smoke_loss.backward()
xm.mark_step()
debug_model.zero_grad(set_to_none=True)
print("TPU forward/backward smoke test passed")
```

- [ ] **Bước 4: Chạy TPU smoke test**

Kỳ vọng:

```text
TPU forward/backward smoke test passed
```

Nếu batch size 1 ở 896 bị lỗi out-of-memory, giữ nguyên 896 và bật chế độ tiết kiệm memory trước khi thử lại:

```python
if hasattr(debug_model, "gradient_checkpointing_enable"):
    debug_model.gradient_checkpointing_enable()
torch.cuda.empty_cache() if torch.cuda.is_available() else None
```

- [ ] **Bước 5: Commit model smoke test**

```bash
git add SH17_dfine_m_896_benchmark.ipynb
git commit -m "feat: add D-FINE-M TPU smoke test"
```

---

### Công Việc 5: Thêm Runner Train Theo Biến Thể

**File:**
- Chỉnh sửa: `/Users/tranquangtrong/SH17/SH17_dfine_m_896_benchmark.ipynb`

- [ ] **Bước 1: Thêm utility EMA và early stopping**

```python
class EMAState:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

    def update(self, model):
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad or name not in self.shadow:
                continue
            self.shadow[name].mul_(self.decay).add_(parameter.detach(), alpha=1.0 - self.decay)

    def apply_shadow(self, model):
        backup = {}
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad or name not in self.shadow:
                continue
            backup[name] = parameter.detach().clone()
            parameter.data.copy_(self.shadow[name].data)
        return backup

    def restore(self, model, backup):
        for name, parameter in model.named_parameters():
            if name in backup:
                parameter.data.copy_(backup[name].data)

    def state_dict(self):
        return {
            "decay": self.decay,
            "shadow": {name: tensor.detach().cpu() for name, tensor in self.shadow.items()},
        }

    def load_state_dict(self, state):
        self.decay = state["decay"]
        self.shadow = {name: tensor.to(device) for name, tensor in state["shadow"].items()}

class EarlyStopping:
    def __init__(self, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.best = -float("inf")
        self.bad_rounds = 0

    def should_stop(self, metric_value):
        if metric_value > self.best + self.min_delta:
            self.best = metric_value
            self.bad_rounds = 0
            return False
        self.bad_rounds += 1
        return self.bad_rounds >= self.patience
```

- [ ] **Bước 2: Thêm helper dataloader, optimizer, scheduler và checkpoint**

```python
def create_dataloaders(config, processor):
    train_dataset = build_train_dataset(config, processor)
    val_dataset = build_val_dataset(processor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.physical_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
    )
    return train_dataset, val_dataset, train_loader, val_loader

def create_optimizer_and_scheduler(config, model, train_loader):
    backbone_keywords = ("backbone", "encoder")
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(keyword in name.lower() for keyword in backbone_keywords):
            backbone_params.append(param)
        else:
            head_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": config.backbone_lr},
            {"params": head_params, "lr": config.head_lr},
        ],
        weight_decay=config.weight_decay,
    )
    steps_per_epoch = math.ceil(len(train_loader) / config.grad_accum_steps)
    total_train_steps = config.epochs * steps_per_epoch
    warmup_steps = int(total_train_steps * config.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_train_steps,
    )
    return optimizer, scheduler, steps_per_epoch, total_train_steps, warmup_steps

def move_optimizer_state_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)

def torch_load_training_state(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")

def replace_checkpoint_dir(tmp_path, final_path):
    backup_path = final_path.with_name(f"{final_path.name}.bak")
    if backup_path.exists():
        shutil.rmtree(backup_path)
    if final_path.exists():
        final_path.rename(backup_path)
    tmp_path.rename(final_path)
    if backup_path.exists():
        shutil.rmtree(backup_path)

def resolve_resume_checkpoint(variant_output_dir):
    checkpoint_dir = variant_output_dir / "checkpoints"
    for checkpoint_name in ("last", "last.bak", "best"):
        checkpoint_path = checkpoint_dir / checkpoint_name
        if (checkpoint_path / "training_state.pt").exists():
            return checkpoint_path
    return None

def save_checkpoint(
    config,
    variant_output_dir,
    model,
    processor,
    optimizer,
    scheduler,
    epoch,
    metric_value,
    checkpoint_name,
    history=None,
    best_metrics=None,
    best_map50_95=-1.0,
    best_epoch=0,
    stopped_early=False,
    ema_state=None,
):
    checkpoint_dir = variant_output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / checkpoint_name
    tmp_path = checkpoint_dir / f"{checkpoint_name}.tmp"
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    cpu_model = model.cpu()
    cpu_model.save_pretrained(tmp_path)
    processor.save_pretrained(tmp_path)
    torch.save(
        {
            "experiment": config.__dict__,
            "epoch": epoch,
            "metric_value": metric_value,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "history": history or [],
            "best_metrics": best_metrics or {},
            "best_map50_95": best_map50_95,
            "best_epoch": best_epoch,
            "stopped_early": stopped_early,
            "ema": ema_state.state_dict() if ema_state is not None else None,
        },
        tmp_path / "training_state.pt",
    )
    cpu_model.to(device)
    replace_checkpoint_dir(tmp_path, checkpoint_path)
    print("saved", checkpoint_path)

def load_checkpoint_if_available(config, variant_output_dir, model, optimizer, scheduler, ema_state=None):
    if not config.resume_from_checkpoint:
        return {
            "start_epoch": 1,
            "history": [],
            "best_metrics": {},
            "best_map50_95": -1.0,
            "best_epoch": 0,
            "stopped_early": False,
            "checkpoint_path": None,
        }

    checkpoint_path = resolve_resume_checkpoint(variant_output_dir)
    if checkpoint_path is None:
        return {
            "start_epoch": 1,
            "history": [],
            "best_metrics": {},
            "best_map50_95": -1.0,
            "best_epoch": 0,
            "stopped_early": False,
            "checkpoint_path": None,
        }

    resumed_model = DFineForObjectDetection.from_pretrained(checkpoint_path).to(device)
    model.load_state_dict(resumed_model.state_dict())
    training_state = torch_load_training_state(checkpoint_path / "training_state.pt")
    optimizer.load_state_dict(training_state["optimizer"])
    scheduler.load_state_dict(training_state["scheduler"])
    move_optimizer_state_to_device(optimizer, device)
    if ema_state is not None and training_state.get("ema") is not None:
        ema_state.load_state_dict(training_state["ema"])

    start_epoch = int(training_state["epoch"]) + 1
    print(f"resume {config.name} from {checkpoint_path} at epoch {start_epoch}/{config.epochs}")
    return {
        "start_epoch": start_epoch,
        "history": training_state.get("history", []),
        "best_metrics": training_state.get("best_metrics", {}),
        "best_map50_95": float(training_state.get("best_map50_95", training_state.get("metric_value", -1.0))),
        "best_epoch": int(training_state.get("best_epoch", training_state["epoch"])),
        "stopped_early": bool(training_state.get("stopped_early", False)),
        "checkpoint_path": str(checkpoint_path),
    }
```

- [ ] **Bước 3: Thêm runner train một epoch**

```python
def run_train_epoch(model, train_loader, optimizer, scheduler, config, ema_state=None, max_batches=None):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    optimizer_steps = 0

    progress = tqdm(train_loader, desc="train dry run" if max_batches else "train")
    for batch_idx, batch in enumerate(progress, start=1):
        inputs = model_inputs_from_batch(batch, device)
        outputs = model(**inputs)
        loss = outputs.loss / config.grad_accum_steps
        loss.backward()
        running_loss += float(loss.detach().cpu()) * config.grad_accum_steps

        if batch_idx % config.grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            xm.optimizer_step(optimizer)
            scheduler.step()
            if ema_state is not None:
                ema_state.update(model)
            optimizer.zero_grad(set_to_none=True)
            xm.mark_step()
            optimizer_steps += 1

        progress.set_postfix(loss=running_loss / batch_idx, opt_steps=optimizer_steps)
        if max_batches is not None and batch_idx >= max_batches:
            break

    return running_loss / max(1, min(len(train_loader), max_batches or len(train_loader)))
```

- [ ] **Bước 4: Chạy dry run**

```python
dry_config = EXPERIMENTS[0]
dry_processor = create_processor(dry_config)
dry_model = create_model(dry_config, device)
_, _, dry_train_loader, _ = create_dataloaders(dry_config, dry_processor)
dry_optimizer, dry_scheduler, dry_steps_per_epoch, dry_total_steps, dry_warmup_steps = create_optimizer_and_scheduler(
    dry_config,
    dry_model,
    dry_train_loader,
)
print("dry steps_per_epoch", dry_steps_per_epoch)
print("dry total_train_steps", dry_total_steps)
print("dry warmup_steps", dry_warmup_steps)
dry_loss = run_train_epoch(
    dry_model,
    dry_train_loader,
    dry_optimizer,
    dry_scheduler,
    dry_config,
    ema_state=None,
    max_batches=5,
)
print("dry_loss", dry_loss)
```

Kỳ vọng:

```text
dry_loss <finite number>
```

- [ ] **Bước 5: Thêm full experiment runner**

Cell này phụ thuộc vào `evaluate_model` ở Công Việc 6. Thêm cell ngay ở bước này, nhưng chỉ execute sau khi Công Việc 6 đã tồn tại.

```python
def train_one_experiment(config):
    variant_output_dir = experiment_output_dir(config)
    processor = create_processor(config)
    model = create_model(config, device)
    train_dataset, val_dataset, train_loader, val_loader = create_dataloaders(config, processor)
    optimizer, scheduler, steps_per_epoch, total_train_steps, warmup_steps = create_optimizer_and_scheduler(
        config,
        model,
        train_loader,
    )
    ema_state = EMAState(model, config.ema_decay) if config.use_ema else None
    early_stopper = EarlyStopping(config.early_stopping_patience, config.early_stopping_min_delta) if config.early_stopping else None
    total_params = sum(param.numel() for param in model.parameters())
    resume_state = load_checkpoint_if_available(config, variant_output_dir, model, optimizer, scheduler, ema_state)

    print("experiment", config.name)
    print("train entries", len(train_dataset))
    print("val images", len(val_dataset))
    print("steps_per_epoch", steps_per_epoch)
    print("total_train_steps", total_train_steps)
    print("warmup_steps", warmup_steps)
    print("resume checkpoint", resume_state["checkpoint_path"])

    start_epoch = resume_state["start_epoch"]
    history = resume_state["history"]
    best_map50_95 = resume_state["best_map50_95"]
    best_metrics = resume_state["best_metrics"]
    best_epoch = resume_state["best_epoch"]
    stopped_early = resume_state["stopped_early"]

    if start_epoch > config.epochs:
        print(f"{config.name} already reached epoch {config.epochs}. Skip training.")

    epoch_progress = tqdm(
        range(start_epoch, config.epochs + 1),
        desc=f"{config.name} epochs",
        total=max(0, config.epochs - start_epoch + 1),
    )

    for epoch in epoch_progress:
        started = time.time()
        print(f"[{config.name}] epoch {epoch}/{config.epochs} started")
        train_loss = run_train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            config,
            ema_state=ema_state,
            max_batches=None,
        )
        row = {
            "experiment": config.name,
            "epoch": epoch,
            "train_loss": train_loss,
            "elapsed_min": (time.time() - started) / 60.0,
        }

        should_eval = epoch % config.eval_every_epochs == 0 or epoch == config.epochs
        if should_eval:
            backup = ema_state.apply_shadow(model) if ema_state is not None else None
            metrics, per_class_df, predictions = evaluate_model(
                model,
                val_loader,
                device,
                processor,
                output_dir=variant_output_dir,
            )
            if ema_state is not None:
                ema_state.restore(model, backup)

            row.update(metrics)
            if metrics["mAP50-95"] > best_map50_95:
                best_map50_95 = metrics["mAP50-95"]
                best_metrics = dict(metrics)
                best_epoch = epoch
                backup = ema_state.apply_shadow(model) if ema_state is not None else None
                save_checkpoint(
                    config,
                    variant_output_dir,
                    model,
                    processor,
                    optimizer,
                    scheduler,
                    epoch,
                    best_map50_95,
                    "best",
                    history=history + [row],
                    best_metrics=best_metrics,
                    best_map50_95=best_map50_95,
                    best_epoch=best_epoch,
                    stopped_early=stopped_early,
                    ema_state=ema_state,
                )
                if ema_state is not None:
                    ema_state.restore(model, backup)
                pd.DataFrame([{**metrics, "experiment": config.name, "epoch": epoch}]).to_csv(
                    variant_output_dir / "best_overall_metrics.csv",
                    index=False,
                )
                per_class_df.to_csv(variant_output_dir / "best_per_class_metrics.csv", index=False)
                pd.DataFrame(predictions).to_json(variant_output_dir / "best_predictions.json", orient="records")

            if early_stopper is not None and early_stopper.should_stop(metrics["mAP50-95"]):
                stopped_early = True
                row["early_stopped"] = True
                history.append(row)
                pd.DataFrame(history).to_csv(variant_output_dir / "training_history.csv", index=False)
                save_checkpoint(
                    config,
                    variant_output_dir,
                    model,
                    processor,
                    optimizer,
                    scheduler,
                    epoch,
                    row.get("mAP50-95", best_map50_95),
                    "last",
                    history=history,
                    best_metrics=best_metrics,
                    best_map50_95=best_map50_95,
                    best_epoch=best_epoch,
                    stopped_early=stopped_early,
                    ema_state=ema_state,
                )
                print(row)
                epoch_progress.set_postfix(
                    loss=train_loss,
                    best=best_map50_95,
                    stopped=True,
                )
                break

        if epoch % config.save_every_epochs == 0:
            save_checkpoint(
                config,
                variant_output_dir,
                model,
                processor,
                optimizer,
                scheduler,
                epoch,
                row.get("mAP50-95", -1.0),
                f"epoch_{epoch:03d}",
                history=history + [row],
                best_metrics=best_metrics,
                best_map50_95=best_map50_95,
                best_epoch=best_epoch,
                stopped_early=stopped_early,
                ema_state=ema_state,
            )

        history.append(row)
        pd.DataFrame(history).to_csv(variant_output_dir / "training_history.csv", index=False)
        if config.save_last_every_epoch:
            save_checkpoint(
                config,
                variant_output_dir,
                model,
                processor,
                optimizer,
                scheduler,
                epoch,
                row.get("mAP50-95", best_map50_95),
                "last",
                history=history,
                best_metrics=best_metrics,
                best_map50_95=best_map50_95,
                best_epoch=best_epoch,
                stopped_early=stopped_early,
                ema_state=ema_state,
            )
        epoch_progress.set_postfix(
            loss=train_loss,
            best=best_map50_95,
            epoch=f"{epoch}/{config.epochs}",
        )
        print(row)

    summary = {
        "experiment": config.name,
        "display_name": config.display_name,
        "mirrors": config.mirrors,
        "params_m": round(total_params / 1_000_000, 1),
        "best_epoch": best_epoch,
        "P": best_metrics.get("P", np.nan),
        "R": best_metrics.get("R", np.nan),
        "mAP50": best_metrics.get("mAP50", np.nan),
        "mAP50-95": best_metrics.get("mAP50-95", best_map50_95),
        "best_conf_threshold": best_metrics.get("best_conf_threshold", np.nan),
        "stopped_early": stopped_early,
        "train_entries": len(train_dataset),
        "output_dir": str(variant_output_dir),
    }
    (variant_output_dir / "run_manifest.json").write_text(json.dumps({**config.__dict__, **summary, "best_metrics": best_metrics}, indent=2))
    return summary
```

- [ ] **Bước 6: Thêm danh sách run được kiểm soát**

```python
RUN_EXPERIMENT_NAMES = [
    "dfine_m_baseline_896",                 # Baseline / Variant 0
    "dfine_m_tuned_896_es",                 # Variant 1
    "dfine_m_scale_jitter_896_es",          # Variant 2
    "dfine_m_oversample_minority_896_es",   # Variant 3
]

experiment_lookup = {config.name: config for config in EXPERIMENTS}
variant_summaries = []

for experiment_name in RUN_EXPERIMENT_NAMES:
    config = experiment_lookup[experiment_name]
    display_variant_markdown(config)
    variant_summaries.append(train_one_experiment(config))

pd.DataFrame(variant_summaries).to_csv(OUTPUT_ROOT / "variant_training_summaries.csv", index=False)
display(pd.DataFrame(variant_summaries))
```

- [ ] **Bước 7: Kiểm tra progress log và checkpoint resume**

Sau khi chạy ít nhất một epoch của một variant, notebook phải có các output sau trong log:

```text
[dfine_m_baseline_896] epoch 1/100 started
train: ... loss=...
saved /content/drive/MyDrive/SH17_outputs/dfine_m_896_variants/dfine_m_baseline_896/checkpoints/last
```

File `training_history.csv` của variant đó phải có ít nhất các cột này sau epoch đầu:

```text
experiment
epoch
train_loss
elapsed_min
```

Sau epoch có evaluation, thường là epoch 5 vì `eval_every_epochs=5`, `training_history.csv` phải có thêm các cột metric:

```text
P
R
mAP50
mAP50-95
```

Nếu runtime bị dừng sau epoch 1 rồi chạy lại cell train, notebook phải in ra dạng:

```text
resume dfine_m_baseline_896 from .../checkpoints/last at epoch 2/100
```

- [ ] **Bước 8: Commit variant training runner**

```bash
git add SH17_dfine_m_896_benchmark.ipynb
git commit -m "feat: add D-FINE-M variant training runner"
```

---

### Công Việc 6: Thêm Metric Validation

**File:**
- Chỉnh sửa: `/Users/tranquangtrong/SH17/SH17_dfine_m_896_benchmark.ipynb`

- [ ] **Bước 1: Thêm builder COCO ground-truth**

```python
def build_coco_ground_truth(dataset):
    images = []
    annotations = []
    ann_id = 1

    for idx, image_path in enumerate(tqdm(dataset.image_paths, desc="building COCO gt")):
        width, height = inspect_image(image_path)
        boxes, labels = parse_yolo_label(label_path_for_image(image_path), width, height)
        images.append({"id": idx, "width": width, "height": height, "file_name": Path(image_path).name})
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = map(float, box)
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": idx,
                    "category_id": int(label),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": max(0.0, x2 - x1) * max(0.0, y2 - y1),
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    coco = COCO()
    coco.dataset = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": idx, "name": name} for idx, name in ID2LABEL.items()],
    }
    coco.createIndex()
    return coco

coco_gt = build_coco_ground_truth(debug_val_dataset)
```

- [ ] **Bước 2: Thêm bước thu thập prediction**

```python
def collect_predictions(model, val_loader, device, processor, score_threshold=0.001):
    model.eval()
    predictions = []
    targets_for_pr = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="validation"):
            pixel_values = batch["pixel_values"].to(device)
            outputs = model(pixel_values=pixel_values)
            target_sizes = batch["orig_size"].to(device)
            results = processor.post_process_object_detection(
                outputs,
                threshold=score_threshold,
                target_sizes=target_sizes,
            )

            for item_idx, result in enumerate(results):
                image_id = int(batch["image_id"][item_idx])
                boxes = result["boxes"].detach().cpu().numpy()
                scores = result["scores"].detach().cpu().numpy()
                labels = result["labels"].detach().cpu().numpy()

                for box, score, label in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = map(float, box)
                    predictions.append(
                        {
                            "image_id": image_id,
                            "category_id": int(label),
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "score": float(score),
                            "box_xyxy": [x1, y1, x2, y2],
                        }
                    )

                targets_for_pr.append(
                    {
                        "image_id": image_id,
                        "boxes": batch["target_boxes_xyxy"][item_idx].numpy(),
                        "labels": batch["target_labels"][item_idx].numpy(),
                    }
                )

            xm.mark_step()

    return predictions, targets_for_pr
```

- [ ] **Bước 3: Thêm phần tính COCO AP**

```python
def run_coco_eval(coco_gt, predictions):
    if len(predictions) == 0:
        raise ValueError("No predictions were produced for validation.")

    coco_predictions = [
        {
            "image_id": item["image_id"],
            "category_id": item["category_id"],
            "bbox": item["bbox"],
            "score": item["score"],
        }
        for item in predictions
    ]
    coco_dt = coco_gt.loadRes(coco_predictions)
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.catIds = list(ID2LABEL.keys())
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    precision = evaluator.eval["precision"]
    per_class_rows = []
    for class_idx, class_name in ID2LABEL.items():
        class_precision = precision[:, :, class_idx, 0, -1]
        class_precision = class_precision[class_precision > -1]
        class_ap = float(np.mean(class_precision)) * 100.0 if class_precision.size else 0.0

        class_precision_50 = precision[0, :, class_idx, 0, -1]
        class_precision_50 = class_precision_50[class_precision_50 > -1]
        class_ap50 = float(np.mean(class_precision_50)) * 100.0 if class_precision_50.size else 0.0

        per_class_rows.append(
            {
                "class_id": class_idx,
                "class": class_name,
                "mAP50": class_ap50,
                "mAP50-95": class_ap,
            }
        )

    overall = {
        "mAP50-95": float(evaluator.stats[0]) * 100.0,
        "mAP50": float(evaluator.stats[1]) * 100.0,
    }
    return overall, pd.DataFrame(per_class_rows), evaluator
```

- [ ] **Bước 4: Thêm phần tính IoU và P/R**

```python
def box_iou_matrix(boxes_a, boxes_b):
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    boxes_a = np.asarray(boxes_a, dtype=np.float32)
    boxes_b = np.asarray(boxes_b, dtype=np.float32)

    area_a = np.maximum(0.0, boxes_a[:, 2] - boxes_a[:, 0]) * np.maximum(0.0, boxes_a[:, 3] - boxes_a[:, 1])
    area_b = np.maximum(0.0, boxes_b[:, 2] - boxes_b[:, 0]) * np.maximum(0.0, boxes_b[:, 3] - boxes_b[:, 1])

    lt = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    rb = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    wh = np.maximum(0.0, rb - lt)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-9)

def match_predictions_at_threshold(predictions, targets, score_threshold, iou_threshold=0.5):
    targets_by_image = {item["image_id"]: item for item in targets}
    predictions_by_image = defaultdict(list)
    for pred in predictions:
        if pred["score"] >= score_threshold:
            predictions_by_image[pred["image_id"]].append(pred)

    totals = {class_idx: {"tp": 0, "fp": 0, "fn": 0} for class_idx in ID2LABEL}

    for image_id, target in targets_by_image.items():
        target_boxes = np.asarray(target["boxes"], dtype=np.float32)
        target_labels = np.asarray(target["labels"], dtype=np.int64)
        image_predictions = sorted(predictions_by_image.get(image_id, []), key=lambda item: item["score"], reverse=True)

        for class_idx in ID2LABEL:
            gt_mask = target_labels == class_idx
            gt_boxes = target_boxes[gt_mask]
            pred_boxes = np.asarray(
                [pred["box_xyxy"] for pred in image_predictions if pred["category_id"] == class_idx],
                dtype=np.float32,
            )
            matched_gt = set()
            if len(pred_boxes) > 0 and len(gt_boxes) > 0:
                ious = box_iou_matrix(pred_boxes, gt_boxes)
            else:
                ious = np.zeros((len(pred_boxes), len(gt_boxes)), dtype=np.float32)

            for pred_idx in range(len(pred_boxes)):
                if len(gt_boxes) == 0:
                    totals[class_idx]["fp"] += 1
                    continue
                best_gt = int(np.argmax(ious[pred_idx]))
                best_iou = float(ious[pred_idx, best_gt])
                if best_iou >= iou_threshold and best_gt not in matched_gt:
                    totals[class_idx]["tp"] += 1
                    matched_gt.add(best_gt)
                else:
                    totals[class_idx]["fp"] += 1

            totals[class_idx]["fn"] += max(0, len(gt_boxes) - len(matched_gt))

    total_tp = sum(item["tp"] for item in totals.values())
    total_fp = sum(item["fp"] for item in totals.values())
    total_fn = sum(item["fn"] for item in totals.values())
    precision = total_tp / max(1, total_tp + total_fp)
    recall = total_tp / max(1, total_tp + total_fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)

    per_class_pr = []
    for class_idx, counts in totals.items():
        cls_precision = counts["tp"] / max(1, counts["tp"] + counts["fp"])
        cls_recall = counts["tp"] / max(1, counts["tp"] + counts["fn"])
        per_class_pr.append(
            {
                "class_id": class_idx,
                "class": ID2LABEL[class_idx],
                "P": cls_precision * 100.0,
                "R": cls_recall * 100.0,
            }
        )

    return {
        "threshold": score_threshold,
        "P": precision * 100.0,
        "R": recall * 100.0,
        "F1": f1 * 100.0,
        "per_class_pr": pd.DataFrame(per_class_pr),
    }

def find_best_pr_threshold(predictions, targets):
    thresholds = np.concatenate(
        [
            np.linspace(0.001, 0.05, 10),
            np.linspace(0.06, 0.50, 23),
            np.linspace(0.55, 0.95, 9),
        ]
    )
    rows = [match_predictions_at_threshold(predictions, targets, float(thr)) for thr in thresholds]
    best = max(rows, key=lambda row: row["F1"])
    pr_curve = pd.DataFrame([{key: value for key, value in row.items() if key != "per_class_pr"} for row in rows])
    return best, pr_curve
```

- [ ] **Bước 5: Thêm function evaluation đầy đủ**

```python
def evaluate_model(model, val_loader, device, processor, output_dir=None):
    predictions, targets_for_pr = collect_predictions(model, val_loader, device, processor, score_threshold=0.001)
    coco_metrics, per_class_ap, evaluator = run_coco_eval(coco_gt, predictions)
    best_pr, pr_curve = find_best_pr_threshold(predictions, targets_for_pr)

    metrics = {
        "P": best_pr["P"],
        "R": best_pr["R"],
        "mAP50": coco_metrics["mAP50"],
        "mAP50-95": coco_metrics["mAP50-95"],
        "best_conf_threshold": best_pr["threshold"],
        "F1": best_pr["F1"],
    }

    per_class_df = best_pr["per_class_pr"].merge(per_class_ap, on=["class_id", "class"], how="left")
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        pr_curve.to_csv(Path(output_dir) / "last_pr_curve.csv", index=False)
    return metrics, per_class_df, predictions
```

- [ ] **Bước 6: Chạy validation trên smoke model**

```python
smoke_eval_dir = OUTPUT_ROOT / "_smoke_eval"
debug_val_loader = DataLoader(
    debug_val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=2,
    collate_fn=collate_fn,
)
smoke_metrics, smoke_per_class, smoke_predictions = evaluate_model(
    debug_model,
    debug_val_loader,
    device,
    debug_processor,
    output_dir=smoke_eval_dir,
)
print(smoke_metrics)
print(smoke_per_class.head())
```

Kỳ vọng:

```text
Cell hoàn tất mà không có lỗi metric-shape.
Các chỉ số có thể thấp trước khi training.
```

- [ ] **Bước 7: Commit metric validation**

```bash
git add SH17_dfine_m_896_benchmark.ipynb
git commit -m "feat: add D-FINE validation metrics"
```

---

### Công Việc 7: Thêm Bảng So Sánh Khớp Format Trong Slide

**File:**
- Chỉnh sửa: `/Users/tranquangtrong/SH17/SH17_dfine_m_896_benchmark.ipynb`

- [ ] **Bước 1: Thêm các dòng baseline từ slide/docs hiện có**

```python
REFERENCE_RESULTS = [
    {
        "model": "YOLOv9s baseline 640",
        "Params(M)": 7.2,
        "P": 73.6,
        "R": 60.2,
        "mAP50": 65.3,
        "mAP50-95": 42.9,
        "source": "current slide/docs",
    },
    {
        "model": "YOLOv9s tuned 640",
        "Params(M)": 7.2,
        "P": 75.2,
        "R": 61.8,
        "mAP50": 66.8,
        "mAP50-95": 44.2,
        "source": "current slide/docs",
    },
    {
        "model": "YOLOv9s multiscale 960",
        "Params(M)": 7.2,
        "P": 77.0,
        "R": 64.0,
        "mAP50": 69.0,
        "mAP50-95": 46.6,
        "source": "current slide/docs",
    },
    {
        "model": "YOLOv9s oversample minority 960",
        "Params(M)": 7.2,
        "P": 76.8,
        "R": 65.6,
        "mAP50": 70.2,
        "mAP50-95": 47.5,
        "source": "current slide/docs",
    },
    {
        "model": "YOLOv9-e paper",
        "Params(M)": 58.1,
        "P": 81.0,
        "R": 65.0,
        "mAP50": 70.9,
        "mAP50-95": 48.7,
        "source": "SH17 paper",
    },
]

def load_dfine_variant_rows():
    rows = []
    for config in EXPERIMENTS:
        metrics_path = experiment_output_dir(config) / "best_overall_metrics.csv"
        if not metrics_path.exists():
            rows.append(
                {
                    "model": config.display_name,
                    "Params(M)": np.nan,
                    "P": np.nan,
                    "R": np.nan,
                    "mAP50": np.nan,
                    "mAP50-95": np.nan,
                    "best_conf_threshold": np.nan,
                    "source": "this notebook",
                    "mirrors": config.mirrors,
                    "status": "not_run",
                }
            )
            continue
        metrics = pd.read_csv(metrics_path).iloc[0].to_dict()
        manifest_path = experiment_output_dir(config) / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        rows.append(
            {
                "model": config.display_name,
                "Params(M)": manifest.get("params_m", np.nan),
                "P": metrics["P"],
                "R": metrics["R"],
                "mAP50": metrics["mAP50"],
                "mAP50-95": metrics["mAP50-95"],
                "best_conf_threshold": metrics.get("best_conf_threshold", np.nan),
                "source": "this notebook",
                "mirrors": config.mirrors,
                "status": "complete",
            }
        )
    return rows

def build_overall_comparison():
    comparison = pd.DataFrame(REFERENCE_RESULTS + load_dfine_variant_rows())
    comparison["delta_vs_yolov9s_v3_mAP50"] = comparison["mAP50"] - 70.2
    comparison["delta_vs_yolov9s_v3_mAP50-95"] = comparison["mAP50-95"] - 47.5
    comparison["delta_vs_yolov9e_mAP50"] = comparison["mAP50"] - 70.9
    comparison["delta_vs_yolov9e_mAP50-95"] = comparison["mAP50-95"] - 48.7
    return comparison
```

- [ ] **Bước 2: Thêm bảng reference per-class cho biến thể YOLOv9s mạnh nhất**

```python
YOLOV9S_V3_PER_CLASS = pd.DataFrame(
    [
        {"class": "hands", "yolov9s_v3_mAP50": 88.2, "yolov9s_v3_mAP50-95": 62.6},
        {"class": "person", "yolov9s_v3_mAP50": 90.2, "yolov9s_v3_mAP50-95": 75.3},
        {"class": "head", "yolov9s_v3_mAP50": 91.7, "yolov9s_v3_mAP50-95": 71.9},
        {"class": "face", "yolov9s_v3_mAP50": 92.1, "yolov9s_v3_mAP50-95": 71.4},
        {"class": "ear", "yolov9s_v3_mAP50": 82.9, "yolov9s_v3_mAP50-95": 53.1},
        {"class": "shoes", "yolov9s_v3_mAP50": 69.2, "yolov9s_v3_mAP50-95": 41.3},
        {"class": "tool", "yolov9s_v3_mAP50": 41.1, "yolov9s_v3_mAP50-95": 25.3},
        {"class": "gloves", "yolov9s_v3_mAP50": 64.8, "yolov9s_v3_mAP50-95": 41.5},
        {"class": "glasses", "yolov9s_v3_mAP50": 75.0, "yolov9s_v3_mAP50-95": 45.1},
        {"class": "helmet", "yolov9s_v3_mAP50": 77.2, "yolov9s_v3_mAP50-95": 57.1},
        {"class": "face-mask", "yolov9s_v3_mAP50": 73.9, "yolov9s_v3_mAP50-95": 47.4},
        {"class": "foot", "yolov9s_v3_mAP50": 31.6, "yolov9s_v3_mAP50-95": 15.5},
        {"class": "safety-vest", "yolov9s_v3_mAP50": 58.8, "yolov9s_v3_mAP50-95": 38.5},
        {"class": "ear-mufs", "yolov9s_v3_mAP50": 58.1, "yolov9s_v3_mAP50-95": 40.9},
        {"class": "safety-suit", "yolov9s_v3_mAP50": 56.7, "yolov9s_v3_mAP50-95": 36.5},
        {"class": "medical-suit", "yolov9s_v3_mAP50": 69.4, "yolov9s_v3_mAP50-95": 41.0},
        {"class": "face-guard", "yolov9s_v3_mAP50": 72.5, "yolov9s_v3_mAP50-95": 43.1},
    ]
)

def build_per_class_comparison(per_class_df):
    comparison = per_class_df.merge(YOLOV9S_V3_PER_CLASS, on="class", how="left")
    comparison["delta_mAP50_vs_yolov9s_v3"] = comparison["mAP50"] - comparison["yolov9s_v3_mAP50"]
    comparison["delta_mAP50-95_vs_yolov9s_v3"] = comparison["mAP50-95"] - comparison["yolov9s_v3_mAP50-95"]
    return comparison.sort_values("mAP50-95", ascending=False)
```

- [ ] **Bước 3: Thêm cell tạo bảng cuối**

```python
def best_completed_dfine_experiment(overall_comparison):
    dfine_rows = overall_comparison[
        (overall_comparison["source"] == "this notebook")
        & (overall_comparison["status"] == "complete")
    ].copy()
    if dfine_rows.empty:
        raise RuntimeError("No completed D-FINE variant found. Run at least one experiment before building final tables.")
    return dfine_rows.sort_values("mAP50-95", ascending=False).iloc[0]["model"]

def load_per_class_for_display_name(display_name):
    display_to_config = {config.display_name: config for config in EXPERIMENTS}
    config = display_to_config[display_name]
    path = experiment_output_dir(config) / "best_per_class_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)

overall_comparison = build_overall_comparison()
best_dfine_display_name = best_completed_dfine_experiment(overall_comparison)
best_dfine_per_class = load_per_class_for_display_name(best_dfine_display_name)
per_class_comparison = build_per_class_comparison(best_dfine_per_class)

overall_comparison.to_csv(OUTPUT_ROOT / "overall_comparison.csv", index=False)
per_class_comparison.to_csv(OUTPUT_ROOT / "per_class_comparison.csv", index=False)

display(overall_comparison.round(2))
print("Best D-FINE variant:", best_dfine_display_name)
display(per_class_comparison.round(2))
```

- [ ] **Bước 4: Kiểm tra chấp nhận**

Notebook phải hiển thị:

```text
D-FINE-M baseline 896
D-FINE-M tuned 896 + early stopping
D-FINE-M scale-jitter 896 + early stopping
D-FINE-M oversample minority 896 + early stopping
P
R
mAP50
mAP50-95
delta_vs_yolov9s_v3_mAP50
delta_vs_yolov9e_mAP50
```

- [ ] **Bước 5: Commit bảng so sánh**

```bash
git add SH17_dfine_m_896_benchmark.ipynb
git commit -m "feat: add D-FINE comparison tables"
```

---

### Công Việc 8: Thêm Cell Phân Tích Lỗi

**File:**
- Chỉnh sửa: `/Users/tranquangtrong/SH17/SH17_dfine_m_896_benchmark.ipynb`

- [ ] **Bước 1: Thêm summary điểm yếu theo class**

```python
def summarize_weak_classes(per_class_comparison, top_n=8):
    cols = [
        "class",
        "P",
        "R",
        "mAP50",
        "mAP50-95",
        "delta_mAP50_vs_yolov9s_v3",
        "delta_mAP50-95_vs_yolov9s_v3",
    ]
    weak_by_map = per_class_comparison.sort_values("mAP50-95", ascending=True)[cols].head(top_n)
    weak_by_delta = per_class_comparison.sort_values("delta_mAP50-95_vs_yolov9s_v3", ascending=True)[cols].head(top_n)
    return weak_by_map, weak_by_delta

weak_by_map, weak_by_delta = summarize_weak_classes(per_class_comparison)
display(weak_by_map.round(2))
display(weak_by_delta.round(2))
```

- [ ] **Bước 2: Thêm cell visualization định tính**

```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_predictions(image_path, predictions, max_predictions=30, score_threshold=0.25):
    image = Image.open(image_path).convert("RGB")
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image)
    ax.axis("off")

    shown = 0
    image_name = Path(image_path).name
    image_idx = val_images.index(Path(image_path)) if Path(image_path) in val_images else None
    for pred in sorted(predictions, key=lambda item: item["score"], reverse=True):
        if image_idx is None or pred["image_id"] != image_idx or pred["score"] < score_threshold:
            continue
        x, y, w, h = pred["bbox"]
        label = ID2LABEL[pred["category_id"]]
        rect = patches.Rectangle((x, y), w, h, linewidth=1.5, edgecolor="lime", facecolor="none")
        ax.add_patch(rect)
        ax.text(x, max(0, y - 4), f"{label} {pred['score']:.2f}", color="black", fontsize=8, backgroundcolor="lime")
        shown += 1
        if shown >= max_predictions:
            break
    plt.show()

def load_predictions_for_display_name(display_name):
    display_to_config = {config.display_name: config for config in EXPERIMENTS}
    config = display_to_config[display_name]
    path = experiment_output_dir(config) / "best_predictions.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())

best_variant_predictions = load_predictions_for_display_name(best_dfine_display_name)
best_variant_threshold = float(
    overall_comparison.loc[
        overall_comparison["model"] == best_dfine_display_name,
        "best_conf_threshold",
    ].iloc[0]
)

for image_path in random.sample(val_images, k=min(6, len(val_images))):
    draw_predictions(image_path, best_variant_predictions, score_threshold=best_variant_threshold)
```

- [ ] **Bước 3: Commit phần phân tích lỗi**

```bash
git add SH17_dfine_m_896_benchmark.ipynb
git commit -m "feat: add D-FINE error analysis cells"
```

---

### Công Việc 9: Thêm Export Cuối Và Hướng Dẫn Run

**File:**
- Chỉnh sửa: `/Users/tranquangtrong/SH17/SH17_dfine_m_896_benchmark.ipynb`

- [ ] **Bước 1: Thêm export final manifest**

```python
final_manifest = {
    "model_id": MODEL_ID,
    "default_image_size": DEFAULT_IMAGE_SIZE,
    "num_classes": NUM_CLASSES,
    "class_names": CLASS_NAMES,
    "minority_class_ids": sorted(MINORITY_CLASS_IDS),
    "minority_repeat_factor": MINORITY_REPEAT_FACTOR,
    "train_images": len(train_images),
    "val_images": len(val_images),
    "oversampled_train_entries": len(oversampled_train_images),
    "experiments": [config.__dict__ for config in EXPERIMENTS],
    "run_experiment_names": RUN_EXPERIMENT_NAMES,
    "best_dfine_variant": best_dfine_display_name,
    "output_root": str(OUTPUT_ROOT),
}

(OUTPUT_ROOT / "final_run_manifest.json").write_text(json.dumps(final_manifest, indent=2))
print(json.dumps(final_manifest, indent=2))
```

- [ ] **Bước 2: Thêm cell kết luận cuối dễ đọc**

```python
best_row = overall_comparison[overall_comparison["model"] == best_dfine_display_name].iloc[0]
print("Best D-FINE-M 896 variant")
print(f"Variant: {best_dfine_display_name}")
print(f"P: {best_row['P']:.2f}")
print(f"R: {best_row['R']:.2f}")
print(f"mAP50: {best_row['mAP50']:.2f}")
print(f"mAP50-95: {best_row['mAP50-95']:.2f}")
print(f"Delta vs YOLOv9s V3 mAP50: {best_row['delta_vs_yolov9s_v3_mAP50']:.2f}")
print(f"Delta vs YOLOv9s V3 mAP50-95: {best_row['delta_vs_yolov9s_v3_mAP50-95']:.2f}")
print(f"Delta vs YOLOv9e mAP50: {best_row['delta_vs_yolov9e_mAP50']:.2f}")
print(f"Delta vs YOLOv9e mAP50-95: {best_row['delta_vs_yolov9e_mAP50-95']:.2f}")
```

- [ ] **Bước 3: Chạy notebook từ TPU runtime mới**

Output thành công kỳ vọng:

```text
train images 6479
val images 1620
total instances 75994
minority train images 1002
oversampled train entries 8483
TPU forward/backward smoke test passed
checkpoints/last/training_state.pt
checkpoints/best/training_state.pt
training_history.csv
variant_training_summaries.csv
overall_comparison.csv
per_class_comparison.csv
final_run_manifest.json
```

- [ ] **Bước 4: Commit notebook cuối**

```bash
git add SH17_dfine_m_896_benchmark.ipynb
git commit -m "feat: finalize D-FINE-M 896 SH17 notebook"
```

---

## Ghi Chú Khi Triển Khai

- Giữ image size ở 896 cho toàn bộ biến thể D-FINE-M. Nếu thiếu memory, giảm áp lực bằng physical batch size 1, gradient checkpointing, giảm số dataloader worker và giảm tần suất evaluation trước khi cân nhắc đổi image size.
- Giữ oversampling giống experiment YOLOv9s V3, nhưng chỉ áp dụng cho `dfine_m_oversample_minority_896_es`; baseline và hai biến thể đầu phải dùng train split gốc 6,479 ảnh.
- Giữ Variant 1 là tuned recipe đầu tiên: AdamW parameter groups, cosine schedule, warmup dài hơn, gradient clipping, EMA, augmentation màu nhẹ và early stopping theo validation `mAP50-95`.
- Giữ Variant 2 là recipe cho small-object/high-resolution: output cố định 896 với scale jitter và bbox-safe crop, tránh shape ảnh động trên TPU vì có thể làm tăng overhead do recompile.
- Mỗi variant phải lưu `checkpoints/last` sau từng epoch. Nếu runtime sập, lần chạy sau phải tự resume từ `checkpoints/last`; nếu `last` lỗi trong lúc ghi, fallback sang `checkpoints/last.bak`, rồi mới tới `checkpoints/best`.
- `training_history.csv` phải được cập nhật sau từng epoch để biết model đang train tới epoch nào, loss bao nhiêu và metric validation gần nhất là gì.
- Báo cáo P/R bằng confidence threshold chọn theo F1 tại IoU 0.50. Báo cáo mAP bằng COCOeval.
- Không khẳng định bất kỳ biến thể D-FINE nào vượt YOLOv9s hoặc YOLOv9-e cho đến khi `overall_comparison.csv` được tạo từ checkpoint đã train.
- Lưu toàn bộ metric artifacts trong `OUTPUT_ROOT` để có thể dựng lại các bảng slide từ output notebook.

## Tự Review

- Độ phủ yêu cầu: plan dùng D-FINE-M, input 896px, manifest split SH17, baseline cộng ba biến thể, Variant 1 có early stopping/tuning methods, minority oversampling chỉ cho Variant 3, TPU smoke test, training, validation và bảng kiểu slide.
- Ràng buộc một notebook: phần triển khai chỉ chạm vào `/Users/tranquangtrong/SH17/SH17_dfine_m_896_benchmark.ipynb`; runtime artifacts là output, không phải implementation module.
- Tính nhất quán biến thể: bốn dòng D-FINE phản chiếu bốn dòng YOLOv9s trong slide: baseline, tuned, scale/multiscale và minority oversampling.
- Tính rõ ràng trong notebook: mỗi biến thể có markdown section riêng và loop train cũng hiển thị markdown của biến thể trước khi chạy.
- Tính bền khi train lâu: runner lưu `last` checkpoint sau từng epoch, lưu backup `last.bak`, lưu optimizer/scheduler/EMA/history và có resume tự động.
- Tính nhất quán metric: bảng overall có P, R, mAP50, mAP50-95, số parameter, status và delta so với YOLOv9s V3/YOLOv9-e; bảng per-class dùng biến thể D-FINE tốt nhất đã hoàn tất và có P, R, mAP50, mAP50-95, delta so với YOLOv9s V3.
- Rủi ro: fine-tune D-FINE bằng Hugging Face trên TPU phải được validate bằng smoke test trước khi train đầy đủ. Nếu XLA reject một operation, ghi rõ failure trong notebook và chỉ rerun trên Colab GPU sau khi xác nhận với user.
