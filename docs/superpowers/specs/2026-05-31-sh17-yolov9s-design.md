# SH17 YOLOv9s Benchmark Design

## Goal

Tạo một notebook benchmark riêng cho `yolov9s` trên SH17, dùng chung helper pipeline hiện tại với `yolov9c`, nhưng tách riêng hoàn toàn config, notebook, run path, artifact path, và weight file. Notebook mới phải hỗ trợ baseline và các cải tiến bám theo `SH17_paper_summary.md`: tuning training strategy, multi-scale/high-resolution cho small objects, và oversampling cho minority classes.

## Scope

Trong phạm vi lần này:

- Tạo notebook mới `SH17_yolov9s_benchmark.ipynb`
- Tạo config mới `configs/sh17_yolov9s_experiments.yaml`
- Refactor nhẹ helper hiện tại để hỗ trợ cả `yolov9c` và `yolov9s`
- Giữ nguyên notebook `yolov9c` đang chạy, không đổi path artifact/run của nó
- Bổ sung test để đảm bảo helper chung không bị vỡ khi thêm `yolov9s`

Ngoài phạm vi lần này:

- Không thay đổi cấu trúc dataset SH17
- Không tạo pipeline riêng hoàn toàn cho `yolov9s`
- Không thêm detector hai giai đoạn, tiling phức tạp, hay crop-stage phụ

## Architecture

### 1. Shared helper, separate entrypoints

Giữ `scripts/sh17_yolov9c_pipeline.py` làm helper dùng chung trong ngắn hạn, nhưng refactor các hàm còn hard-code `yolov9c` để nhận tên weight/model động từ config. Notebook `yolov9s` chỉ là một entrypoint mới dùng helper này.

### 2. Isolated output paths

`yolov9s` phải tách riêng hoàn toàn:

- weight: `E:\models\yolov9\yolov9s.pt`
- runs: `E:\models\sh17_yolov9s_runs`
- artifacts: `D:\DS-AI\SH17\artifacts\sh17_yolov9s`

Việc tách path đảm bảo có thể chạy `yolov9c` và `yolov9s` song song mà không đè checkpoint, manifest override, leaderboard, hay ảnh plot.

### 3. Experiment matrix

Notebook `yolov9s` sẽ có 4 run chính:

- `yolov9s_baseline_640`
- `yolov9s_tuned_640`
- `yolov9s_multiscale_960`
- `yolov9s_oversample_minority_960`

Mục tiêu là giữ cùng triết lý benchmark như `yolov9c` để dễ so sánh, nhưng đổi base model sang `yolov9s` vốn nhẹ hơn và phù hợp hơn với giới hạn thời gian huấn luyện hiện tại.

## Data Flow

1. Notebook load config `yolov9s`
2. Helper tạo manifest train/val từ `E:\data\SH17`
3. Helper đảm bảo `yolov9s.pt` tồn tại trong `E:\models\yolov9`
4. Helper expand config thành danh sách experiments
5. Nếu có run oversample, notebook tạo `train_oversampled.txt`
6. Mỗi experiment:
   - resume từ checkpoint giữa chừng nếu run chưa hoàn tất
   - reuse `best.pt` nếu run đã hoàn tất
   - train nếu cần
   - val và ghi leaderboard

## Checkpoint Handling

Luật checkpoint sẽ giống notebook hiện tại:

- Resume ưu tiên `epoch*.pt` mới nhất
- Nếu run đã hoàn tất đủ epoch thì không `resume=True`, mà load `best.pt` để evaluate lại
- Luôn giữ `best.pt`, `last.pt`, và checkpoint định kỳ qua `save_period`

Điểm này rất quan trọng vì smoke test trước đó cho thấy `last.pt` sau khi strip optimizer không phải lúc nào cũng resumable đúng nghĩa.

## Testing

Phần thay đổi phải có test trước:

- test cho config dynamic path/weight name
- test generic weight resolution cho `yolov9s.pt`
- test `expand_experiments()` giữ đúng path riêng của `yolov9s`

Notebook mới sẽ được smoke-check bằng:

- validate file `.ipynb`
- load config
- build dataset manifest
- đảm bảo đường dẫn weight `yolov9s.pt` được resolve đúng

## Risks

- Helper hiện đang mang tên `sh17_yolov9c_pipeline.py`, nên nội dung sẽ phải trung tính hơn tên file
- `yolov9s` cần tách path hoàn toàn để không đụng run đang chạy của `yolov9c`
- Nếu venv hiện tại thiếu weight `yolov9s.pt`, notebook phải tự tải đúng file mà không ảnh hưởng `yolov9c.pt`

## Decision

Thực hiện theo hướng `1`: dùng chung helper pipeline hiện tại, tách riêng notebook + config cho `yolov9s`, và chỉ refactor ở mức tối thiểu cần thiết để helper hỗ trợ cả hai model.
