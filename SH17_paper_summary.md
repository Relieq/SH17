# SH17 Paper Summary

Nguồn chính: `2407.04590v1.pdf`, 19 trang. Các số liệu dưới đây chỉ lấy từ paper. Chỗ nào paper không nêu rõ được ghi là "Not specified in paper".

## 1. Thông tin tổng quan paper

| Mục | Nội dung |
| --- | --- |
| Tên paper | **SH17: A Dataset for Human Safety and Personal Protective Equipment Detection in Manufacturing Industry** |
| Tác giả | Hafiz Mughees Ahmad, Afshin Rahimi |
| Mục tiêu chính | Đề xuất dataset SH17 cho phát hiện con người, bộ phận cơ thể và PPE trong môi trường sản xuất/công nghiệp; benchmark các object detection model hiện đại; kiểm tra khả năng tổng quát hóa sang dataset khác. |
| Dataset SH17 dùng để làm gì? | Train và đánh giá object detection model để phát hiện PPE/body parts phục vụ kiểm tra tuân thủ an toàn lao động trong manufacturing industry. |
| Bài toán chính | **Object detection**, không phải classification/segmentation. Output là bounding boxes và class labels cho 17 lớp. |
| Dataset/public code | Dataset được công bố tại GitHub: `https://github.com/ahmadmughees/sh17dataset`. |
| Vị trí trong paper | Abstract, Section I, Section III, Section IV, Conclusion. |

## 2. Tổng hợp dataset SH17

### 2.1. Tóm tắt số liệu dataset

| Thuộc tính | Số liệu / mô tả | Vị trí |
| --- | --- | --- |
| Tổng số ảnh | 8,099 images | Abstract; Section I; Section III-A; Conclusion |
| Tổng số object instances | 75,994 annotated instances | Abstract; Section I; Table I |
| Số class | 17 classes | Abstract; Table I; Section III-C |
| Số instance trung bình mỗi ảnh | 9.38 instances/image | Section III-D, page 6 |
| Nguồn dữ liệu | Pexels, với các query như `manufacturing worker`, `industrial worker`, `human worker`, `labor`, v.v. | Section III-A, page 5 |
| Quy trình lọc dữ liệu | Sau khi bỏ ảnh trùng lặp, có khoảng 11,000 samples; khoảng 26% ảnh rỗng không chứa object thuộc target classes bị loại; còn 8,099 images. | Section III-A, page 5 |
| Annotation | 4 human annotators; 3 annotators gán nhãn ban đầu; team lead kiểm tra/sửa; graduate student kiểm tra cuối. | Section III-B, page 5 |
| Annotation tools | DarkLabel và LabelImg | Section III-B, pages 5-6 |
| Kích thước ảnh gốc | Native resolution; max 8,192 x 5,462; min 1,920 x 1,002 | Section III-D, page 6 |
| Hướng ảnh | Có cả landscape và portrait | Section III-D, page 6 |
| Object nhỏ | 39,764 annotations có area < 1% ảnh; 59,025 annotations có area < 5% ảnh | Section III-D, page 8 |
| Imbalance | Có imbalance rõ rệt: `hands` nhiều nhất 15,850 instances (20.9%); `face-guard` ít nhất 134 instances (0.2%); `helmet` 927 instances (~1.2%). | Section III-D; Figure 2; Table II |
| Train/test split | 80% train, 20% test; test set trong Table III có 1,620 images và 15,358 instances. Paper không nêu validation split. | Section IV, page 9; Appendix C, page 19; Table III; Table VIII |
| Validation split | Not specified in paper |
| Data format | Object detection annotations/bounding boxes; metadata fields được mô tả ở Table VI. File annotation format cụ thể không nêu rõ trong paper. | Section III-B; Appendix A; Table VI |
| Preprocessing khi train | Resize/fixed image size = 640 cho tất cả models vì giới hạn bộ nhớ. Normalization/cropping/other preprocessing: Not specified in paper. | Section IV, page 9 |
| Augmentation | Mosaic 4 images + horizontal flipping khi train. Dataset gốc có augmentation hay không: Not specified in paper. | Section IV, page 9 |

### 2.2. So sánh SH17 với các dataset PPE/helmet khác

Nguồn: Table I, page 3.

| Dataset | Classes | Images | Instances | Available | Paper |
| --- | ---: | ---: | ---: | :---: | --- |
| Pictor-PPE | 3 | 784 | - | Yes | [4] |
| SHW | 1 | 7,581 | 120,558 | Yes | [12] |
| CHV | 6 | 1,330 | - | Yes | [14] |
| TCRSF | 7 | 12,373 | 50,558 | No | [5] |
| GDUT-HWD | 5 | 3,174 | 18,893 | Yes | [11] |
| SHD | 3 | 5,000 | - | Yes | - |
| SHEL5K | 5 | 5,000 | 75,570 | Yes | [15] |
| SH17 (proposed) | 17 | 8,099 | 75,994 | Yes | - |

### 2.3. Danh sách 17 class và số instance

Nguồn: Table II, page 7.

| ID | Class | Additional tags | Instances | Mô tả |
| ---: | --- | --- | ---: | --- |
| 1 | Person | male, female, children | 13,802 | Uses visible features for classification. |
| 2 | Head | - | 11,985 | Any view of the head: front, back, top, etc. |
| 3 | Face | - | 8,950 | Visible only when the nose is visible. |
| 4 | Glasses | on, off, safety, vision | 1,945 | Detection of safety glasses. |
| 5 | Face-mask-medical | on, off | 669 | Detection of medical face masks. |
| 6 | Face-guard | on, off | 134 | Detection of faceguards. |
| 7 | Ear | - | 7,730 | Focused on ears for safety equipment detection. |
| 8 | Earmuffs / ear-mufs | on, off | 318 | Detection of earmuffs; over-ear headphones labeled as earmuffs. |
| 9 | Hands | - | 15,850 | Focus on hands for safety equipment detection. |
| 10 | Gloves | on, off | 2,790 | Detection of gloves. |
| 11 | Foot | - | 796 | Visible when there are no shoes; each foot annotated regardless of socks. |
| 12 | Shoes | on, off, safety, other | 4,560 | Safety shoes/thick joggers vs slippers/sneakers/other footwear. |
| 13 | Safety-vest | on, off | 530 | Detection of safety vests. |
| 14 | Tools | on, off | 4,647 | Tools being held; `on` means in hand; pencils/laptops not considered tools. |
| 15 | Helmet | on, off, white, red, black, yellow, blue | 927 | Helmet detection; tags also contain helmet color. |
| 16 | Medical-suit | on, off | 155 | Detection of medical suits. |
| 17 | Safety-suit | on, off | 530 | Detection of safety suits. |

### 2.4. Train/test instance split theo class

Nguồn: Appendix C, Table VIII, page 19.

| Class | Train instances | Test instances |
| --- | ---: | ---: |
| face-guard | 110 | 24 |
| medical-suit | 114 | 43 |
| safety-suit | 195 | 45 |
| ear-mufs | 269 | 49 |
| safety-vest | 433 | 97 |
| face-mask-medical | 519 | 151 |
| foot | 610 | 149 |
| helmet | 773 | 154 |
| glasses | 1,547 | 398 |
| gloves | 2,261 | 529 |
| shoes | 3,604 | 956 |
| tools | 3,724 | 923 |
| ear | 6,118 | 1,612 |
| face | 7,095 | 1,855 |
| head | 9,558 | 2,427 |
| person | 11,068 | 2,734 |
| hands | 12,638 | 3,212 |

### 2.5. Metadata fields

Nguồn: Appendix A, Table VI, page 18.

| Field | Description |
| --- | --- |
| Unique Identifier | Unique code assigned to each image. |
| Width and Height | Image dimensions in pixels. |
| URL | Web address where the image can be accessed. |
| Photographer Name | Name of the photographer. |
| Photographer URL | Photographer profile/portfolio URL. |
| Photographer ID | Unique photographer identifier. |
| Average Color | Average color as hexadecimal code. |
| Source | Origin/platform from which the image was obtained. |
| Liked | Boolean-like field indicating whether image was liked/favored on the platform. |
| Description | Summary/narrative of image content/context/theme. |

### 2.6. Person demographic tags

Nguồn: Appendix B, Table VII, page 18. Paper cảnh báo các tags này dựa trên đặc điểm nhìn thấy được nên có thể không hoàn toàn chính xác.

| Group | Male | Female | Children | Total |
| --- | ---: | ---: | ---: | ---: |
| White | 2,432 | 2,032 | 37 | 4,501 |
| Black | 1,098 | 776 | 7 | 1,881 |
| Brown | 577 | 218 | 12 | 807 |
| Asian | 963 | 1,272 | 52 | 2,287 |
| Total | 5,070 | 4,298 | 108 | 9,476 |

## 3. Tổng hợp các model trong paper

Nguồn chính: Section IV, page 9 và Table III, page 10.

Các điểm train chung paper nêu rõ:

- Model family: YOLOv8, YOLOv9, YOLOv10 với nhiều size variants.
- Input size: fixed image size 640 cho tất cả models.
- Pretraining: transfer learning từ model pretrained trên MS-COCO.
- Epochs: 200.
- Hardware: 2x NVIDIA RTX GPUs.
- Batch size: paper nói batch size từ 128 cho nano models đến 32 cho bigger-scale models; mapping chính xác cho từng variant không được nêu.
- Augmentation: mosaic 4 images + horizontal flipping.
- Hyperparameters: default hyperparameters từ open-source codebase của authors; paper ghi follow [44].
- NMS: dùng cho post-processing output của YOLOv8 và YOLOv9 variants; YOLOv10 được mô tả là loại bỏ NMS trong Section II-B.
- Optimizer, learning rate, scheduler, exact loss function trong experiment: Not specified in paper.

| Model name | Backbone / architecture | Input size | Pretrained hay scratch | Loss function | Optimizer | Learning rate | Batch size | Epochs | Augmentation | Metric đánh giá | Kết quả chính |
| --- | --- | --- | --- | --- | --- | --- | --- | ---: | --- | --- | --- |
| Yolo-8-n | YOLOv8 nano one-stage object detector; exact backbone not specified in paper | 640 | MS-COCO pretrained, transfer learning | Not specified in paper | Not specified in paper | Not specified in paper | Paper states 128 for nano models; exact per-model mapping not fully specified | 200 | Mosaic 4 images + horizontal flip | P, R, mAP@50, mAP50-95 | mAP@50 58.0; mAP50-95 36.6 |
| Yolo-8-s | YOLOv8 small; exact backbone not specified in paper | 640 | MS-COCO pretrained, transfer learning | Not specified in paper | Not specified in paper | Not specified in paper | Not specified per model; paper states 128 to 32 depending on scale | 200 | Mosaic 4 images + horizontal flip | P, R, mAP@50, mAP50-95 | mAP@50 63.7; mAP50-95 41.7 |
| Yolo-8-m | YOLOv8 medium; exact backbone not specified in paper | 640 | MS-COCO pretrained, transfer learning | Not specified in paper | Not specified in paper | Not specified in paper | Not specified per model; paper states 128 to 32 depending on scale | 200 | Mosaic 4 images + horizontal flip | P, R, mAP@50, mAP50-95 | mAP@50 66.6; mAP50-95 45.7 |
| Yolo-8-l | YOLOv8 large; exact backbone not specified in paper | 640 | MS-COCO pretrained, transfer learning | Not specified in paper | Not specified in paper | Not specified in paper | Not specified per model; paper states 128 to 32 depending on scale | 200 | Mosaic 4 images + horizontal flip | P, R, mAP@50, mAP50-95 | mAP@50 68.0; mAP50-95 47.0 |
| Yolo-8-x | YOLOv8 extra-large; exact backbone not specified in paper | 640 | MS-COCO pretrained, transfer learning | Not specified in paper | Not specified in paper | Not specified in paper | Not specified per model; paper states 128 to 32 depending on scale | 200 | Mosaic 4 images + horizontal flip | P, R, mAP@50, mAP50-95 | mAP@50 69.3; mAP50-95 47.2 |
| Yolo-9-t | YOLOv9 tiny; paper discusses PGI + GELAN for YOLOv9 | 640 | MS-COCO pretrained, transfer learning | Not specified in paper | Not specified in paper | Not specified in paper | Not specified per model; paper states 128 to 32 depending on scale | 200 | Mosaic 4 images + horizontal flip | P, R, mAP@50, mAP50-95 | mAP@50 58.5; mAP50-95 37.5 |
| Yolo-9-s | YOLOv9 small; PGI + GELAN | 640 | MS-COCO pretrained, transfer learning | Not specified in paper | Not specified in paper | Not specified in paper | Not specified per model; paper states 128 to 32 depending on scale | 200 | Mosaic 4 images + horizontal flip | P, R, mAP@50, mAP50-95 | mAP@50 65.3; mAP50-95 42.9 |
| Yolo-9-m | YOLOv9 medium; PGI + GELAN | 640 | MS-COCO pretrained, transfer learning | Not specified in paper | Not specified in paper | Not specified in paper | Not specified per model; paper states 128 to 32 depending on scale | 200 | Mosaic 4 images + horizontal flip | P, R, mAP@50, mAP50-95 | mAP@50 68.6; mAP50-95 46.5 |
| Yolo-9-c | YOLOv9 compact; PGI + GELAN | 640 | MS-COCO pretrained, transfer learning | Not specified in paper | Not specified in paper | Not specified in paper | Not specified per model; paper states 128 to 32 depending on scale | 200 | Mosaic 4 images + horizontal flip | P, R, mAP@50, mAP50-95 | mAP@50 67.7; mAP50-95 46.5 |
| Yolo-9-e | YOLOv9 extended/large variant; PGI + GELAN | 640 | MS-COCO pretrained, transfer learning | Not specified in paper | Not specified in paper | Not specified in paper | Not specified per model; paper states 128 to 32 depending on scale | 200 | Mosaic 4 images + horizontal flip | P, R, mAP@50, mAP50-95 | **Best: mAP@50 70.9; mAP50-95 48.7** |
| Yolo-10-n | YOLOv10 nano; NMS-free/end-to-end design via consistent dual assignment as described in Section II-B | 640 | MS-COCO pretrained, transfer learning | Not specified in paper | Not specified in paper | Not specified in paper | Paper states 128 for nano models; exact per-model mapping not fully specified | 200 | Mosaic 4 images + horizontal flip | P, R, mAP@50, mAP50-95 | mAP@50 57.2; mAP50-95 35.9 |
| Yolo-10-s | YOLOv10 small; NMS-free/end-to-end design | 640 | MS-COCO pretrained, transfer learning | Not specified in paper | Not specified in paper | Not specified in paper | Not specified per model; paper states 128 to 32 depending on scale | 200 | Mosaic 4 images + horizontal flip | P, R, mAP@50, mAP50-95 | mAP@50 62.7; mAP50-95 40.9 |
| Yolo-10-m | YOLOv10 medium; NMS-free/end-to-end design | 640 | MS-COCO pretrained, transfer learning | Not specified in paper | Not specified in paper | Not specified in paper | Not specified per model; paper states 128 to 32 depending on scale | 200 | Mosaic 4 images + horizontal flip | P, R, mAP@50, mAP50-95 | mAP@50 65.7; mAP50-95 43.8 |
| Yolo-10-b | YOLOv10 balanced/base variant; NMS-free/end-to-end design | 640 | MS-COCO pretrained, transfer learning | Not specified in paper | Not specified in paper | Not specified in paper | Not specified per model; paper states 128 to 32 depending on scale | 200 | Mosaic 4 images + horizontal flip | P, R, mAP@50, mAP50-95 | mAP@50 65.8; mAP50-95 45.1 |
| Yolo-10-l | YOLOv10 large; NMS-free/end-to-end design | 640 | MS-COCO pretrained, transfer learning | Not specified in paper | Not specified in paper | Not specified in paper | Not specified per model; paper states 128 to 32 depending on scale | 200 | Mosaic 4 images + horizontal flip | P, R, mAP@50, mAP50-95 | mAP@50 67.4; mAP50-95 46.0 |
| Yolo-10-x | YOLOv10 extra-large; NMS-free/end-to-end design | 640 | MS-COCO pretrained, transfer learning | Not specified in paper | Not specified in paper | Not specified in paper | Not specified per model; paper states 128 to 32 depending on scale | 200 | Mosaic 4 images + horizontal flip | P, R, mAP@50, mAP50-95 | mAP@50 67.8; mAP50-95 46.7 |

## 4. Số liệu performance của từng model

### 4.1. Benchmark trên SH17 test set

Nguồn: Table III, page 10. Test set: 1,620 images, 15,358 instances.

| Model | Params (M) | Images | Instances | Precision | Recall | mAP@50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Yolo-8-n | 3.2 | 1,620 | 15,358 | 67.5 | 53.6 | 58.0 | 36.6 |
| Yolo-8-s | 11.2 | 1,620 | 15,358 | 81.5 | 55.7 | 63.7 | 41.7 |
| Yolo-8-m | 25.9 | 1,620 | 15,358 | 77.1 | 60.5 | 66.6 | 45.7 |
| Yolo-8-l | 43.7 | 1,620 | 15,358 | 76.7 | 62.9 | 68.0 | 47.0 |
| Yolo-8-x | 68.2 | 1,620 | 15,358 | 77.1 | 63.1 | 69.3 | 47.2 |
| Yolo-9-t | 2.0 | 1,620 | 15,358 | 75.0 | 52.6 | 58.5 | 37.5 |
| Yolo-9-s | 7.2 | 1,620 | 15,358 | 73.6 | 60.2 | 65.3 | 42.9 |
| Yolo-9-m | 20.1 | 1,620 | 15,358 | 77.4 | 62.0 | 68.6 | 46.5 |
| Yolo-9-c | 25.5 | 1,620 | 15,358 | 79.6 | 60.8 | 67.7 | 46.5 |
| **Yolo-9-e** | **58.1** | **1,620** | **15,358** | **81.0** | **65.0** | **70.9** | **48.7** |
| Yolo-10-n | 2.3 | 1,620 | 15,358 | 66.8 | 53.2 | 57.2 | 35.9 |
| Yolo-10-s | 7.2 | 1,620 | 15,358 | 75.8 | 57.0 | 62.7 | 40.9 |
| Yolo-10-m | 15.4 | 1,620 | 15,358 | 71.4 | 61.4 | 65.7 | 43.8 |
| Yolo-10-b | 19.1 | 1,620 | 15,358 | 77.7 | 59.1 | 65.8 | 45.1 |
| Yolo-10-l | 24.4 | 1,620 | 15,358 | 76.0 | 61.8 | 67.4 | 46.0 |
| Yolo-10-x | 29.5 | 1,620 | 15,358 | 76.8 | 62.8 | 67.8 | 46.7 |

Các metric không có trong paper:

| Metric / thông tin | Trạng thái |
| --- | --- |
| Accuracy classification | Not specified in paper; paper dùng detection metrics. |
| F1-score | Not specified in paper |
| IoU score theo từng model | Not specified in paper; paper chỉ định nghĩa IoU trong Section III-E. |
| AUC | Not specified in paper |
| FLOPs | Not specified in paper |
| Inference time / FPS | Not specified in paper |

### 4.2. Model tốt nhất

Nguồn: Section IV, page 9; Table III, page 10.

| Câu hỏi | Kết luận |
| --- | --- |
| Model tốt nhất | **Yolo-9-e / YOLOv9-e** |
| Metric chính chứng minh tốt nhất | mAP@50 và mAP50-95 trên SH17 test set |
| Giá trị cụ thể | mAP@50 = **70.9**; mAP50-95 = **48.7** |
| Model đứng thứ hai theo mAP@50 | Yolo-8-x, mAP@50 = 69.3 |
| Khoảng hơn model thứ hai | +1.6 điểm mAP@50; +1.5 điểm mAP50-95 so với Yolo-8-x |
| Tham số | YOLOv9-e có 58.1M params; YOLOv8-x có 68.2M params. Paper kết luận YOLOv9-e ít params hơn khoảng 15%. |

Lưu ý: Table III in đậm YOLOv9-e là best overall. Tuy nhiên riêng cột Precision, Yolo-8-s có P = 81.5 cao hơn Yolo-9-e P = 81.0. Vì paper xác định best theo mAP, kết luận "model tốt nhất" nên dựa trên mAP.

### 4.3. Per-class performance của YOLOv9-e trên SH17

Nguồn: Table IV, page 10.

| Class | Images | Instances | Precision | Recall | mAP@50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 1,620 | 15,358 | 81.0 | 65.0 | 70.9 | 48.7 |
| hands | 1,284 | 3,212 | 91.4 | 83.9 | 89.8 | 64.8 |
| person | 1,515 | 2,734 | 90.9 | 89.2 | 92.1 | 77.9 |
| head | 1,314 | 2,427 | 94.8 | 89.1 | 93.5 | 74.3 |
| face | 1,155 | 1,855 | 96.0 | 88.1 | 93.8 | 73.8 |
| ear | 987 | 1,612 | 91.2 | 75.5 | 84.3 | 55.0 |
| shoes | 320 | 956 | 79.2 | 62.8 | 70.8 | 43.2 |
| tool | 455 | 923 | 67.4 | 39.1 | 43.2 | 27.6 |
| gloves | 254 | 529 | 81.6 | 58.9 | 66.5 | 43.5 |
| glasses | 323 | 398 | 87.4 | 72.6 | 76.4 | 46.9 |
| helmet | 93 | 154 | 81.3 | 67.8 | 77.0 | 57.6 |
| face-mask | 75 | 151 | 88.8 | 73.2 | 75.5 | 49.2 |
| foot | 64 | 149 | 51.7 | 22.1 | 29.3 | 14.0 |
| safety-vest | 45 | 97 | 66.4 | 55.0 | 57.7 | 38.1 |
| ear-mufs | 38 | 49 | 79.6 | 46.9 | 57.1 | 40.5 |
| safety-suit | 28 | 45 | 65.7 | 53.3 | 58.5 | 38.3 |
| medical-suit | 30 | 43 | 86.0 | 65.1 | 68.5 | 40.6 |
| face-guard | 23 | 24 | 76.8 | 62.5 | 71.7 | 42.8 |

### 4.4. Cross-domain validation trên Pictor-PPE

Nguồn: Section IV-A, pages 11-12; Table V, page 12. Paper map Pictor-PPE `worker`, `hat`, `vest` sang SH17 `Person`, `Helmet`, `Safety-vest`.

| Class | Images | Instances | Precision | Recall | mAP@50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 654 | 3,477 | 73.5 | 50.3 | 58.9 | 37.6 |
| person | 654 | 2,080 | 83.6 | 80.7 | 85.5 | 61.6 |
| helmet | 451 | 1,369 | 94.6 | 23.7 | 55.5 | 30.0 |
| safety-vest | 12 | 28 | 42.4 | 46.4 | 35.5 | 21.2 |

Text của paper nói YOLOv9-e đạt 58.8% mAP trên Pictor-PPE, còn Table V ghi `all` mAP@50 = 58.9. Đây có thể là khác biệt làm tròn hoặc typo; nên dùng Table V khi cần số trong bảng.

### 4.5. Confusion matrix

Nguồn: Figure 4, page 12. Figure 4 là confusion matrix cho YOLOv9-e trên Pictor-PPE, không phải SH17 test set.

Các giá trị dễ đọc trong Figure 4:

| True class | Predicted class | Value |
| --- | --- | ---: |
| person | person | 0.82 |
| person | background | 0.18 |
| helmet | helmet | 0.22 |
| helmet | head | 0.37 |
| helmet | background | 0.41 |
| safety-vest | safety-vest | 0.46 |
| safety-vest | background | 0.54 |
| background | person | 0.08 |
| background | ear | 0.07 |
| background | face | 0.10 |
| background | foot | 0.01 |
| background | tool | 0.02 |
| background | glasses | 0.01 |
| background | gloves | 0.04 |
| background | helmet | 0.01 |
| background | hands | 0.18 |
| background | head | 0.23 |
| background | shoes | 0.24 |

Paper nhận xét: person đôi khi bị xem là background; helmet thường bị nhầm với head. Vị trí: Section IV-A, page 11; Figure 4, page 12.

## 5. So sánh cách train giữa các model

Nguồn: Section IV, page 9.

| Thành phần train | YOLOv8 variants | YOLOv9 variants | YOLOv10 variants | Nhận xét |
| --- | --- | --- | --- | --- |
| Train/test split | 80% train, 20% test | 80% train, 20% test | 80% train, 20% test | Cùng split theo paper. Validation split không được nêu. |
| Test set | 1,620 images, 15,358 instances | 1,620 images, 15,358 instances | 1,620 images, 15,358 instances | Cùng test set trong Table III. |
| Input size | 640 | 640 | 640 | Cùng fixed image size. |
| Pretrained | MS-COCO pretrained, transfer learning | MS-COCO pretrained, transfer learning | MS-COCO pretrained, transfer learning | Cùng chiến lược transfer learning. |
| Epochs | 200 | 200 | 200 | Cùng số epochs. |
| Augmentation | Mosaic 4 images + horizontal flip | Mosaic 4 images + horizontal flip | Mosaic 4 images + horizontal flip | Cùng augmentation theo paper. |
| Optimizer | Not specified in paper | Not specified in paper | Not specified in paper | Không đủ thông tin để xác nhận giống hệt. |
| Learning rate | Not specified in paper | Not specified in paper | Not specified in paper | Không đủ thông tin để xác nhận giống hệt. |
| Batch size | 128 cho nano tới 32 cho bigger-scale models; per-model mapping not specified | 128 tới 32 tùy scale; per-model mapping not specified | 128 tới 32 tùy scale; per-model mapping not specified | Batch size khác theo scale, không hoàn toàn đồng nhất giữa mọi model. |
| Loss function | Not specified in paper | Not specified in paper | Not specified in paper | Paper không cho bảng loss theo model. |
| Post-processing | NMS | NMS | YOLOv10 được mô tả là loại bỏ NMS | Khác biệt kiến trúc/post-processing, đặc biệt YOLOv10. |
| Hyperparameters | Default hyperparameters từ open-source codebase; follow [44] | Default hyperparameters từ open-source codebase; follow [44] | Default hyperparameters từ open-source codebase; follow [44] | Cùng mô tả chung, nhưng exact values không được nêu. |
| Hardware | 2x NVIDIA RTX GPUs | 2x NVIDIA RTX GPUs | 2x NVIDIA RTX GPUs | Cùng hardware theo paper. |

Kết luận:

- Các model được train theo **pipeline chung giống nhau ở mức cao**: cùng split, input size 640, transfer learning từ MS-COCO, 200 epochs, augmentation mosaic + horizontal flip, cùng test set.
- Tuy nhiên **không thể kết luận mọi hyperparameter giống hệt nhau** vì paper không nêu optimizer, learning rate, scheduler, exact loss, exact batch size cho từng variant.
- Batch size thay đổi theo model scale; YOLOv10 khác YOLOv8/9 ở thiết kế NMS-free.
- So sánh tương đối công bằng vì cùng dataset/test set/epochs/input/augmentation, nhưng có khoảng trống reproducibility và fairness do thiếu exact hyperparameters và vì batch size thay đổi theo scale.

## 6. Phân tích điểm mạnh và điểm yếu của model tốt nhất

Model tốt nhất: **YOLOv9-e**.

### 6.1. Vì sao YOLOv9-e có thể đạt kết quả tốt nhất?

- Paper giải thích YOLOv9 dùng **Programmable Gradient Information (PGI)** để giữ thông tin quan trọng trong quá trình detection và **GELAN** để cân bằng parameter count, complexity, accuracy và inference speed. Nguồn: Section II-B, page 4.
- YOLOv9-e là biến thể lớn/mạnh trong nhóm YOLOv9, đạt mAP@50 = 70.9 và mAP50-95 = 48.7 trên SH17. Nguồn: Section IV; Table III.
- Paper cũng nhận xét YOLOv9-e có 58.1M parameters, ít hơn YOLOv8-x 68.2M parameters khoảng 15% nhưng mAP cao hơn. Nguồn: Section IV, page 9.

### 6.2. Điểm mạnh

- Best overall theo mAP: 70.9 mAP@50 và 48.7 mAP50-95.
- Recall cao nhất trong Table III: 65.0.
- Mạnh ở các class nhiều mẫu và/hoặc đặc trưng rõ: `face` mAP@50 93.8, `head` 93.5, `person` 92.1, `hands` 89.8, `ear` 84.3.
- Cross-domain vẫn có khả năng dùng trực tiếp: trên Pictor-PPE, `all` mAP@50 = 58.9; `person` mAP@50 = 85.5.

### 6.3. Điểm yếu

- Một số class có mAP thấp: `foot` mAP@50 29.3, `tool` 43.2, `ear-mufs` 57.1, `safety-vest` 57.7.
- Recall thấp ở các class khó: `foot` recall 22.1, `tool` recall 39.1, `ear-mufs` recall 46.9.
- Dataset có nhiều object nhỏ: 39,764 annotations < 1% area và 59,025 annotations < 5% area, gây khó cho detection.
- Dataset imbalance mạnh: các class PPE quan trọng như `face-guard`, `medical-suit`, `earmuffs`, `safety-vest`, `helmet` ít hơn nhiều so với `hands`, `person`, `head`, `face`.

### 6.4. Class dự đoán kém hơn và lỗi thường gặp

Nguồn: Section IV, pages 9-11; Figure 5; Section IV-A; Figure 4.

- Paper nêu `tools` và `foot` có mAP thấp vì sample rất đa dạng: shoes có nhiều kiểu như safety shoes, slippers, joggers, runners; tools có thể là bất cứ vật gì người lao động đang dùng.
- Figure 5: sample (2), YOLOv8-m và YOLOv8-x có false positive `tool` trên mặt đất; sample (3), YOLOv8-n có false positive `face`; row 5, tất cả models không phát hiện được tool trong tay worker.
- Cross-domain Pictor-PPE: helmet thường bị nhầm với head; person đôi khi bị xem là background.

## 7. Gợi ý cải tiến pipeline để vượt model tốt nhất trong paper

Các đề xuất dưới đây là hướng triển khai thực tế dựa trên vấn đề paper nêu: class imbalance, small objects, minority PPE classes, tool/foot diversity, và thiếu ablation/hyperparameter detail.

### 7.1. Cải tiến dữ liệu

| Hướng cải tiến | Lý do |
| --- | --- |
| Data cleaning và annotation audit cho minority classes | `face-guard`, `medical-suit`, `earmuffs`, `safety-vest`, `helmet`, `foot` có ít samples; kiểm tra nhãn sai/thiếu có thể tăng mAP rõ hơn so với chỉ đổi model. |
| Class balancing bằng oversampling hoặc class-aware sampling | Giảm bias về các class nhiều mẫu như `hands`, `person`, `head`, `face`. |
| Hard example mining | Tập trung vào ảnh có object nhỏ, occlusion, tool trong tay, helmet/head dễ nhầm, vest/background. |

### 7.2. Cải tiến preprocessing

| Hướng cải tiến | Lý do |
| --- | --- |
| Multi-scale training thay vì chỉ fixed 640 | SH17 có nhiều object rất nhỏ; multi-scale có thể giúp robust hơn với scale. |
| Tiling/slicing cho ảnh high-resolution | Ảnh gốc rất lớn nhưng train resize 640 có thể làm mất chi tiết PPE nhỏ như ear/earmuffs/glasses/tools. |
| Resize giữ aspect ratio + letterbox nhất quán | Tránh méo hình khi resize ảnh landscape/portrait. Paper không nêu chi tiết resize strategy. |
| Crop quanh person/hands/head như stage phụ | Có thể tăng độ phân giải local cho PPE nhỏ, đặc biệt glove/tool/helmet/faceguard. |
| Normalization theo pretrained backbone | Paper không nêu normalization; cần kiểm tra code để đảm bảo đúng chuẩn model pretrained. |
| Enhancement nhẹ cho low-light/industrial scenes | Có thể giúp PPE nhỏ hoặc màu sắc mờ, nhưng cần ablation để tránh làm lệch domain. |

### 7.3 Cải tiến training strategy

| Hướng cải tiến | Lý do |
| --- | --- |
| Learning rate scheduler + warmup rõ ràng | Paper không nêu exact scheduler; Figure 3 cho thấy plateau khoảng epoch 170, có thể tối ưu lịch LR/early stopping. |
| Early stopping hoặc best checkpoint theo mAP50-95 | Tránh train thêm sau khi mAP plateau. |
| Optimizer/weight decay tuning | Paper không nêu optimizer/LR/weight decay; đây là khoảng trống lớn để cải thiện. |
| Focal loss hoặc class-balanced loss | Phù hợp với imbalance mạnh giữa các class. Cần test vì paper không nêu loss. |
| Label smoothing | Có thể giảm overconfidence trong classes dễ nhầm như helmet/head và vest/background. |
| Copy-paste/cutmix có kiểm soát cho PPE nhỏ | Tăng exposure cho minority small objects; cần giữ realistic placement. |
| Mixed precision + gradient accumulation | Cho phép tăng effective batch size hoặc input resolution khi bị giới hạn GPU. |
