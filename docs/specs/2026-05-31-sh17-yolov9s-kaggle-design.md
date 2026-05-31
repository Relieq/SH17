# SH17 YOLOv9s Kaggle Notebook Design

## Goal

Tạo một notebook Kaggle self-contained cho `yolov9s` trên SH17. Toàn bộ helper function, config variants, logic oversampling, checkpoint resume, và leaderboard đều nằm trong một file `.ipynb`, không phụ thuộc bất kỳ file local nào trong repo.

## Scope

Trong phạm vi lần này:

- Tạo một notebook mới dành riêng cho Kaggle
- Chỉ hỗ trợ `yolov9s`
- Nhúng toàn bộ helper/config vào notebook
- Giữ đủ 4 biến thể benchmark đang dùng:
  - `baseline_640`
  - `tuned_640`
  - `multiscale_960`
  - `oversample_minority_960`

Ngoài phạm vi:

- Không thêm `yolov9c`
- Không tách helper/config ra file phụ
- Không sửa notebook local đang chạy

## Architecture

Notebook Kaggle sẽ có các phần:

1. Setup và config Kaggle
2. Helper functions self-contained
3. Dataset preparation từ `/kaggle/input/...`
4. Variant preparation
5. Training/evaluation loop
6. Leaderboard/result review

## Kaggle-specific decisions

- Dữ liệu đọc từ `/kaggle/input/<dataset-name>`
- Output lưu vào `/kaggle/working/sh17_yolov9s_*`
- Không import từ `scripts/` hoặc `configs/`
- Có assert train/val disjoint để giảm nguy cơ leakage từ split lỗi

## Checkpoint behavior

- Mỗi variant là run độc lập
- Resume chỉ dùng checkpoint của chính variant đó
- Nếu run đã hoàn tất đủ epoch mục tiêu thì evaluate lại từ `best.pt`, không cố resume tiếp

## Variant logic

- `baseline_640 = base`
- `tuned_640 = base + training tuning`
- `multiscale_960 = tuned + scale changes`
- `oversample_minority_960 = multiscale + oversampling`

## Verification

Notebook cần được kiểm tra ở mức cấu trúc:

- hợp lệ về định dạng `.ipynb`
- không chứa import/references tới `scripts/` hay `configs/`
- có chứa cả 4 experiment `yolov9s`
- có path Kaggle trong notebook
