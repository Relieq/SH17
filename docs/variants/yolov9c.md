# YOLOv9c Variants

Nguồn cấu hình hiện tại: [sh17_yolov9c_experiments.yaml](D:\DS-AI\SH17\configs\sh17_yolov9c_experiments.yaml)

## Tổng quan

Notebook `yolov9c` hiện có 4 biến thể:

1. `yolov9c_baseline_640`
2. `yolov9c_tuned_640`
3. `yolov9c_multiscale_960`
4. `yolov9c_oversample_minority_960`

Mục tiêu của chuỗi biến thể này là đi từ baseline gần paper nhất sang các cải tiến nhắm vào:

- training strategy rõ ràng hơn
- small-object handling tốt hơn
- class imbalance của các lớp hiếm

## Tóm tắt nhanh

| Variant | Vai trò | Khác baseline | Khác biến thể trước |
| --- | --- | --- | --- |
| `yolov9c_baseline_640` | Mốc so sánh gốc | Không áp dụng | Không áp dụng |
| `yolov9c_tuned_640` | Tối ưu training strategy ở cùng độ phân giải | Đổi optimizer/scheduler/warmup/patience | Thêm tuning training, giữ `imgsz=640` |
| `yolov9c_multiscale_960` | Ưu tiên small objects | Tăng `imgsz`, bật `multi_scale`, giữ tuning chính | So với `tuned_640`: tăng độ phân giải, bật multi-scale, bỏ override `patience=30` |
| `yolov9c_oversample_minority_960` | Ưu tiên minority classes | Giữ cấu hình high-resolution và thêm oversampling | So với `multiscale_960`: chỉ thêm oversampling manifest |

## 1. `yolov9c_baseline_640`

### Mục đích

Đây là biến thể baseline để làm mốc so sánh cho toàn bộ các biến thể còn lại.

### Cấu hình chính

- `weights: yolov9c.pt`
- `imgsz: 640`
- `epochs: 200`
- `batch: -1`
- `workers: 8`
- `patience: 40`
- `save_period: 10`

### Khác gì so với baseline

- Không có khác biệt vì đây chính là baseline.

### Khác gì so với biến thể trước

- Không áp dụng vì đây là biến thể đầu tiên.

## 2. `yolov9c_tuned_640`

### Mục đích

Giữ nguyên độ phân giải `640` nhưng làm rõ chiến lược train để thử xem chỉ tuning optimizer/lịch học có đủ kéo điểm lên hay không.

### Thay đổi so với baseline

- `optimizer: AdamW`
- `lr0: 0.001`
- `lrf: 0.01`
- `warmup_epochs: 5`
- `cos_lr: true`
- `close_mosaic: 10`
- `patience: 30` thay cho `40`

### Khác gì so với baseline

- Không đổi model weight
- Không đổi `imgsz`
- Không bật `multi_scale`
- Chỉ thay phần training strategy

### Khác gì so với biến thể trước

So với `yolov9c_baseline_640`, đây là biến thể đầu tiên thêm tuning training nhưng vẫn giữ nguyên độ phân giải `640`.

## 3. `yolov9c_multiscale_960`

### Mục đích

Tăng khả năng học object nhỏ bằng cách tăng độ phân giải train và bật multi-scale.

### Thay đổi so với baseline

- `imgsz: 960` thay cho `640`
- `multi_scale: true`
- `optimizer: AdamW`
- `lr0: 0.001`
- `lrf: 0.01`
- `warmup_epochs: 5`
- `cos_lr: true`
- `close_mosaic: 10`

### Khác gì so với baseline

- Tăng độ phân giải train
- Bật `multi_scale`
- Vẫn có training tuning như biến thể tuned
- Không bật oversampling

### Khác gì so với biến thể trước

So với `yolov9c_tuned_640`:

- `imgsz` tăng từ `640` lên `960`
- thêm `multi_scale: true`
- không giữ override `patience: 30`, nên quay về mặc định `patience: 40`

## 4. `yolov9c_oversample_minority_960`

### Mục đích

Giữ cấu hình high-resolution và bổ sung oversampling để tăng tần suất xuất hiện của các lớp hiếm trong train manifest.

### Thay đổi so với baseline

- `imgsz: 960`
- `multi_scale: true`
- `optimizer: AdamW`
- `lr0: 0.001`
- `lrf: 0.01`
- `warmup_epochs: 5`
- `cos_lr: true`
- `close_mosaic: 10`
- `use_oversampled_train_manifest: true`

### Khác gì so với baseline

- Có toàn bộ thay đổi của biến thể `multiscale_960`
- Thêm oversampling cho minority classes

### Khác gì so với biến thể trước

So với `yolov9c_multiscale_960`, chỉ có một thay đổi:

- thêm `use_oversampled_train_manifest: true`

### Minority classes đang được oversample

Manifest oversample hiện tập trung vào các class id:

- `2`: `ear-mufs`
- `4`: `face-guard`
- `6`: `foot`
- `10`: `helmet`
- `13`: `medical-suit`
- `16`: `safety-vest`

## Ghi chú đọc kết quả

Khi so sánh kết quả:

- So `tuned_640` với `baseline_640` để biết training strategy có đủ hiệu quả không
- So `multiscale_960` với `tuned_640` để biết small-object handling có phải nút thắt chính không
- So `oversample_minority_960` với `multiscale_960` để biết imbalance có đang kìm các class hiếm hay không
