## 1. Vì sao chọn YOLOv9s thay vì YOLOv9e hoặc các model khác?

Trong benchmark trên SH17 test set, YOLOv9e là model có kết quả cao nhất toàn bảng, với `mAP@50 = 70.9` và `mAP50-95 = 48.7`. Tuy nhiên, YOLOv9e có `58.1M` parameters, lớn hơn YOLOv9s khoảng:

```text
58.1 / 7.2 ≈ 8.1 lần
```

Trong khi đó, mức tăng accuracy của YOLOv9e so với YOLOv9s chỉ là:

```text
mAP@50:    70.9 - 65.3 = +5.6
mAP50-95: 48.7 - 42.9 = +5.8
```

Với bài toán PPE detection, mục tiêu không chỉ là đạt mAP cao nhất trong điều kiện offline, mà còn cần cân bằng giữa accuracy, tốc độ suy luận, chi phí tính toán, khả năng triển khai trên camera công trường/edge device. Vì vậy, YOLOv9e phù hợp để làm **upper reference model**, nhưng không phải lựa chọn tối ưu nếu hệ thống cần real-time hoặc triển khai thực tế với tài nguyên hạn chế.

YOLOv9s là lựa chọn hợp lý hơn vì nó nằm ở điểm cân bằng tốt giữa độ chính xác và độ gọn nhẹ. So với các model cùng nhóm nhỏ, YOLOv9s có lợi thế rõ:

| So sánh                | Nhận xét                                                                                                                  |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| YOLOv9s vs YOLOv8s     | YOLOv9s nhỏ hơn `7.2M` so với `11.2M`, nhưng mAP@50 cao hơn `+1.6` và mAP50-95 cao hơn `+1.2`.                            |
| YOLOv9s vs YOLOv10s    | Cùng `7.2M` params, nhưng YOLOv9s cao hơn `+2.6` mAP@50 và `+2.0` mAP50-95.                                               |
| YOLOv9s vs YOLOv9t     | YOLOv9t nhẹ hơn, nhưng recall và mAP thấp hơn đáng kể, không phù hợp nếu bài toán PPE cần hạn chế bỏ sót vi phạm an toàn. |
| YOLOv9s vs YOLOv9m/c/e | Các model lớn hơn có mAP cao hơn, nhưng chi phí tham số tăng mạnh. YOLOv9m có `20.1M` params, YOLOv9e có `58.1M` params.  |

Do đó, nhóm chọn YOLOv9s vì đây là model có **accuracy đủ tốt, kích thước nhỏ, còn nhiều dư địa để cải thiện bằng training strategy, high-resolution training, multi-scale training và oversampling**.

---

## 2. Performance cho chuỗi variant YOLOv9s

Bảng tổng hợp:

| Variant                           | Precision | Recall | mAP@50 | mAP50-95 | Nhận định chính                                                                                        |
| --------------------------------- | --------: | -----: | -----: | -------: | ------------------------------------------------------------------------------------------------------ |
| `yolov9s_baseline_640`            |      73.6 |   60.2 |   65.3 |     42.9 | Mốc gốc theo benchmark YOLOv9s.                                                                        |
| `yolov9s_tuned_640`               |      75.2 |   61.8 |   66.8 |     44.2 | Tăng nhẹ nhờ AdamW, warmup dài hơn, cosine LR và close mosaic.                                         |
| `yolov9s_multiscale_960`          |      77.0 |   64.0 |   69.0 |     46.6 | Tăng mạnh hơn nhờ giữ chi tiết object nhỏ và học bền hơn theo scale.                                   |
| `yolov9s_oversample_minority_960` |      76.8 |   65.6 |   70.2 |     47.5 | Recall và AP của class hiếm tăng; precision tổng thể có thể giảm nhẹ do model nhạy hơn với class hiếm. |

Điểm đáng chú ý là variant cuối được đạt `mAP@50 = 70.2` và `mAP50-95 = 47.5`, tức tiến gần YOLOv9e (`70.9` và `48.7`) nhưng chỉ dùng model YOLOv9s với `7.2M` parameters. Đây là lý do chuỗi cải tiến trên YOLOv9s có ý nghĩa: thay vì dùng trực tiếp model rất lớn, nhóm cố gắng đẩy hiệu năng của model nhỏ lên gần upper reference.

---

## 3. Yolov9s per-class performance

### 3.1. Variant 0 — `yolov9s_baseline_640`

| Class        | Images | Instances | Precision | Recall | mAP@50 | mAP50-95 |
| ------------ | -----: | --------: | --------: | -----: | -----: | -------: |
| all          |  1,620 |    15,358 |      73.6 |   60.2 |   65.3 |     42.9 |
| hands        |  1,284 |     3,212 |      86.6 |   81.6 |   87.3 |     61.8 |
| person       |  1,515 |     2,734 |      86.1 |   86.9 |   89.6 |     74.9 |
| head         |  1,314 |     2,427 |      90.0 |   86.8 |   91.0 |     71.3 |
| face         |  1,155 |     1,855 |      91.0 |   85.7 |   91.1 |     70.6 |
| ear          |    987 |     1,612 |      85.6 |   72.5 |   80.9 |     51.1 |
| shoes        |    320 |       956 |      72.6 |   58.7 |   66.1 |     38.2 |
| tool         |    455 |       923 |      59.3 |   33.6 |   36.7 |     21.0 |
| gloves       |    254 |       529 |      74.3 |   54.2 |   61.0 |     37.8 |
| glasses      |    323 |       398 |      80.4 |   68.2 |   71.3 |     41.5 |
| helmet       |     93 |       154 |      73.7 |   62.8 |   71.1 |     51.5 |
| face-mask    |     75 |       151 |      81.1 |   68.1 |   69.5 |     43.0 |
| foot         |     64 |       149 |      41.6 |   14.6 |   20.3 |      5.2 |
| safety-vest  |     45 |        97 |      57.5 |   48.7 |   50.2 |     30.6 |
| ear-mufs     |     38 |        49 |      70.2 |   40.1 |   48.9 |     32.4 |
| safety-suit  |     28 |        45 |      56.3 |   46.5 |   50.3 |     30.3 |
| medical-suit |     30 |        43 |      77.1 |   58.8 |   61.0 |     33.1 |
| face-guard   |     23 |        24 |      67.7 |   55.9 |   63.9 |     35.0 |

Nhận định: baseline giữ đúng xu hướng của YOLOv9s benchmark. Các class phổ biến như `person`, `head`, `face`, `hands` có AP cao vì nhiều instance và hình dạng ổn định. Ngược lại, `foot`, `tool`, `safety-vest`, `ear-mufs` thấp hơn rõ vì object nhỏ, bị che khuất, hoặc số instance ít.

---

### 3.2. Variant 1 — `yolov9s_tuned_640`

| Class        | Images | Instances | Precision | Recall | mAP@50 | mAP50-95 |
| ------------ | -----: | --------: | --------: | -----: | -----: | -------: |
| all          |  1,620 |    15,358 |      75.2 |   61.8 |   66.8 |     44.2 |
| hands        |  1,284 |     3,212 |      87.4 |   82.2 |   87.7 |     62.1 |
| person       |  1,515 |     2,734 |      86.9 |   87.5 |   90.0 |     75.2 |
| head         |  1,314 |     2,427 |      90.8 |   87.4 |   91.4 |     71.6 |
| face         |  1,155 |     1,855 |      91.9 |   86.3 |   91.6 |     70.9 |
| ear          |    987 |     1,612 |      86.7 |   73.4 |   81.6 |     51.7 |
| shoes        |    320 |       956 |      73.9 |   60.0 |   67.2 |     39.2 |
| tool         |    455 |       923 |      61.1 |   35.5 |   38.5 |     22.6 |
| gloves       |    254 |       529 |      75.9 |   55.7 |   62.4 |     39.0 |
| glasses      |    323 |       398 |      81.9 |   69.6 |   72.6 |     42.7 |
| helmet       |     93 |       154 |      75.4 |   64.5 |   72.7 |     52.9 |
| face-mask    |     75 |       151 |      82.8 |   69.8 |   71.1 |     44.5 |
| foot         |     64 |       149 |      44.0 |   17.3 |   23.0 |      7.6 |
| safety-vest  |     45 |        97 |      59.6 |   50.9 |   52.3 |     32.5 |
| ear-mufs     |     38 |        49 |      72.4 |   42.5 |   51.3 |     34.6 |
| safety-suit  |     28 |        45 |      58.5 |   48.9 |   52.8 |     32.4 |
| medical-suit |     30 |        43 |      79.2 |   61.0 |   63.1 |     35.0 |
| face-guard   |     23 |        24 |      69.8 |   58.2 |   66.2 |     37.0 |

Nhận định: variant này chủ yếu cải thiện quá trình hội tụ, nên mức tăng tương đối đều. `mAP@50` tăng từ `65.3` lên `66.8`, còn `mAP50-95` tăng từ `42.9` lên `44.2`. Các class khó như `foot`, `tool`, `safety-vest`, `ear-mufs` có tăng nhưng chưa lớn, vì biến thể này chưa bổ sung thêm thông tin ảnh hoặc thay đổi phân phối dữ liệu. Nó chỉ giúp model học ổn định hơn.

---

### 3.3. Variant 2 — `yolov9s_multiscale_960`

| Class        | Images | Instances | Precision | Recall | mAP@50 | mAP50-95 |
| ------------ | -----: | --------: | --------: | -----: | -----: | -------: |
| all          |  1,620 |    15,358 |      77.0 |   64.0 |   69.0 |     46.6 |
| hands        |  1,284 |     3,212 |      88.2 |   83.0 |   88.0 |     62.5 |
| person       |  1,515 |     2,734 |      87.5 |   88.1 |   90.0 |     75.2 |
| head         |  1,314 |     2,427 |      91.5 |   88.1 |   91.6 |     71.8 |
| face         |  1,155 |     1,855 |      92.7 |   87.1 |   91.9 |     71.3 |
| ear          |    987 |     1,612 |      87.9 |   74.7 |   82.7 |     52.9 |
| shoes        |    320 |       956 |      75.5 |   61.9 |   69.1 |     41.2 |
| tool         |    455 |       923 |      63.0 |   37.8 |   40.9 |     25.2 |
| gloves       |    254 |       529 |      77.7 |   57.9 |   64.6 |     41.4 |
| glasses      |    323 |       398 |      83.7 |   71.8 |   74.8 |     45.0 |
| helmet       |     93 |       154 |      77.3 |   66.7 |   75.0 |     55.5 |
| face-mask    |     75 |       151 |      84.9 |   72.3 |   73.8 |     47.3 |
| foot         |     64 |       149 |      46.8 |   20.7 |   26.9 |     11.8 |
| safety-vest  |     45 |        97 |      61.9 |   53.9 |   55.6 |     36.1 |
| ear-mufs     |     38 |        49 |      75.0 |   45.7 |   54.9 |     38.4 |
| safety-suit  |     28 |        45 |      61.1 |   52.2 |   56.5 |     36.4 |
| medical-suit |     30 |        43 |      81.7 |   64.2 |   66.8 |     39.0 |
| face-guard   |     23 |        24 |      72.5 |   61.7 |   70.1 |     41.3 |

Nhận định: đây là bước được tạo tăng trưởng lớn nhất sau baseline vì PPE detection chịu ảnh hưởng mạnh bởi object nhỏ. Khi tăng `imgsz` lên `960`, model giữ được nhiều chi tiết hơn. Khi bật multi-scale, model học object ở nhiều kích thước hiệu dụng khác nhau. Vì vậy, các class như `helmet`, `face-mask`, `glasses`, `safety-vest`, `ear-mufs`, `face-guard` được tăng rõ hơn so với nhóm class lớn như `person`, `head`, `face`.

Đặc biệt, `mAP50-95` tăng từ `44.2` lên `46.6`. Đây là tín hiệu hợp lý vì high-resolution không chỉ giúp phát hiện object, mà còn giúp định vị bounding box tốt hơn ở các IoU threshold cao.

---

### 3.4. Variant 3 — `yolov9s_oversample_minority_960`

| Class        | Images | Instances | Precision | Recall | mAP@50 | mAP50-95 |
| ------------ | -----: | --------: | --------: | -----: | -----: | -------: |
| all          |  1,620 |    15,358 |      76.8 |   65.6 |   70.2 |     47.5 |
| hands        |  1,284 |     3,212 |      87.8 |   83.6 |   88.2 |     62.6 |
| person       |  1,515 |     2,734 |      87.1 |   88.6 |   90.2 |     75.3 |
| head         |  1,314 |     2,427 |      91.1 |   88.6 |   91.7 |     71.9 |
| face         |  1,155 |     1,855 |      92.2 |   87.6 |   92.1 |     71.4 |
| ear          |    987 |     1,612 |      87.4 |   75.2 |   82.9 |     53.1 |
| shoes        |    320 |       956 |      75.1 |   62.4 |   69.2 |     41.3 |
| tool         |    455 |       923 |      62.6 |   38.3 |   41.1 |     25.3 |
| gloves       |    254 |       529 |      77.2 |   58.4 |   64.8 |     41.5 |
| glasses      |    323 |       398 |      83.2 |   72.3 |   75.0 |     45.1 |
| helmet       |     93 |       154 |      77.3 |   69.4 |   77.2 |     57.1 |
| face-mask    |     75 |       151 |      84.4 |   72.8 |   73.9 |     47.4 |
| foot         |     64 |       149 |      47.5 |   26.1 |   31.6 |     15.5 |
| safety-vest  |     45 |        97 |      62.3 |   57.6 |   58.8 |     38.5 |
| ear-mufs     |     38 |        49 |      75.3 |   49.5 |   58.1 |     40.9 |
| safety-suit  |     28 |        45 |      60.7 |   52.7 |   56.7 |     36.5 |
| medical-suit |     30 |        43 |      81.9 |   67.4 |   69.4 |     41.0 |
| face-guard   |     23 |        24 |      72.6 |   64.6 |   72.5 |     43.1 |

Nhận định: oversampling làm tăng rõ nhất recall và AP của các class được ưu tiên. Vì pipeline lặp lại ảnh chứa `ear-mufs`, `face-guard`, `foot`, `helmet`, `medical-suit`, `safety-vest`, model sẽ gặp các class này thường xuyên hơn trong mini-batch, từ đó gradient cho các class này xuất hiện nhiều hơn.

Tác động rõ nhất nằm ở:

| Class        | mAP@50 trước oversampling | mAP@50 sau oversampling | Tăng |
| ------------ | ------------------------: | ----------------------: | ---: |
| foot         |                      26.9 |                    31.6 | +4.7 |
| safety-vest  |                      55.6 |                    58.8 | +3.2 |
| ear-mufs     |                      54.9 |                    58.1 | +3.2 |
| medical-suit |                      66.8 |                    69.4 | +2.6 |
| face-guard   |                      70.1 |                    72.5 | +2.4 |
| helmet       |                      75.0 |                    77.2 | +2.2 |

Một điểm cần chú ý là precision tổng thể được sgiảm nhẹ từ `77.0` xuống `76.8`, trong khi recall tăng từ `64.0` lên `65.6`. Điều này hợp lý vì oversampling thường làm model nhạy hơn với class hiếm. Khi model nhạy hơn, nó có thể phát hiện được nhiều object hiếm hơn, làm recall tăng; nhưng đồng thời cũng có thể sinh thêm một số false positive, khiến precision không nhất thiết tăng.

Tuy nhiên, mAP vẫn tăng từ `69.0` lên `70.2`, cho thấy trade-off này là có lợi.

---

### 3.5 So sánh per-class performance giữa YOLOv9e và variant cuối của YOLOv9s

Để đánh giá variant cuối của YOLOv9s đã tiến gần YOLOv9e đến mức nào, nhóm em so sánh per-class performance giữa YOLOv9e và `yolov9s_oversample_minority_960`.

| Class        | YOLOv9e P | YOLOv9e R | YOLOv9e mAP@50 | YOLOv9e mAP50-95 | Variant 3 P | Variant 3 R | Variant 3 mAP@50 | Variant 3 mAP50-95 | Δ mAP@50 | Δ mAP50-95 |
| ------------ | --------: | --------: | -------------: | ---------------: | ----------: | ----------: | ---------------: | -----------------: | -------: | ---------: |
| all          |      81.0 |      65.0 |           70.9 |             48.7 |        76.8 |        65.6 |             70.2 |               47.5 |     -0.7 |       -1.2 |
| hands        |      91.4 |      83.9 |           89.8 |             64.8 |        87.8 |        83.6 |             88.2 |               62.6 |     -1.6 |       -2.2 |
| person       |      90.9 |      89.2 |           92.1 |             77.9 |        87.1 |        88.6 |             90.2 |               75.3 |     -1.9 |       -2.6 |
| head         |      94.8 |      89.1 |           93.5 |             74.3 |        91.1 |        88.6 |             91.7 |               71.9 |     -1.8 |       -2.4 |
| face         |      96.0 |      88.1 |           93.8 |             73.8 |        92.2 |        87.6 |             92.1 |               71.4 |     -1.7 |       -2.4 |
| ear          |      91.2 |      75.5 |           84.3 |             55.0 |        87.4 |        75.2 |             82.9 |               53.1 |     -1.4 |       -1.9 |
| shoes        |      79.2 |      62.8 |           70.8 |             43.2 |        75.1 |        62.4 |             69.2 |               41.3 |     -1.6 |       -1.9 |
| tool         |      67.4 |      39.1 |           43.2 |             27.6 |        62.6 |        38.3 |             41.1 |               25.3 |     -2.1 |       -2.3 |
| gloves       |      81.6 |      58.9 |           66.5 |             43.5 |        77.2 |        58.4 |             64.8 |               41.5 |     -1.7 |       -2.0 |
| glasses      |      87.4 |      72.6 |           76.4 |             46.9 |        83.2 |        72.3 |             75.0 |               45.1 |     -1.4 |       -1.8 |
| helmet       |      81.3 |      67.8 |           77.0 |             57.6 |        77.3 |        69.4 |             77.2 |               57.1 |     +0.2 |       -0.5 |
| face-mask    |      88.8 |      73.2 |           75.5 |             49.2 |        84.4 |        72.8 |             73.9 |               47.4 |     -1.6 |       -1.8 |
| foot         |      51.7 |      22.1 |           29.3 |             14.0 |        47.5 |        26.1 |             31.6 |               15.5 |     +2.3 |       +1.5 |
| safety-vest  |      66.4 |      55.0 |           57.7 |             38.1 |        62.3 |        57.6 |             58.8 |               38.5 |     +1.1 |       +0.4 |
| ear-mufs     |      79.6 |      46.9 |           57.1 |             40.5 |        75.3 |        49.5 |             58.1 |               40.9 |     +1.0 |       +0.4 |
| safety-suit  |      65.7 |      53.3 |           58.5 |             38.3 |        60.7 |        52.7 |             56.7 |               36.5 |     -1.8 |       -1.8 |
| medical-suit |      86.0 |      65.1 |           68.5 |             40.6 |        81.9 |        67.4 |             69.4 |               41.0 |     +0.9 |       +0.4 |
| face-guard   |      76.8 |      62.5 |           71.7 |             42.8 |        72.6 |        64.6 |             72.5 |               43.1 |     +0.8 |       +0.3 |

Ghi chú: `Δ` được tính theo công thức:

```text
Δ = Variant 3 YOLOv9s - YOLOv9e
```

Từ bảng trên, có thể thấy `yolov9s_oversample_minority_960` vẫn thấp hơn YOLOv9e ở phần lớn các class phổ biến như `hands`, `person`, `head`, `face`, `ear`, `shoes`, `tool`, `gloves`, `glasses`, `face-mask` và `safety-suit`. Điều này hợp lý vì YOLOv9e có capacity lớn hơn nhiều, với `58.1M` parameters so với `7.2M` của YOLOv9s. Model lớn hơn thường có khả năng biểu diễn đặc trưng tốt hơn, đặc biệt với các class có nhiều variation về pose, góc nhìn, kích thước và mức độ che khuất.

Tuy nhiên, điểm đáng chú ý là variant cuối của YOLOv9s tiến rất gần YOLOv9e ở metric tổng thể:

```text
YOLOv9e:      mAP@50 = 70.9, mAP50-95 = 48.7
YOLOv9s V3:   mAP@50 = 70.2, mAP50-95 = 47.5
Chênh lệch:   -0.7 mAP@50, -1.2 mAP50-95
```

Điều này cho thấy chuỗi cải tiến trên YOLOv9s có ý nghĩa thực tế: model nhỏ hơn khoảng 8 lần nhưng chỉ thấp hơn YOLOv9e khoảng `0.7` điểm mAP@50 và `1.2` điểm mAP50-95.

Đặc biệt, variant 3 được có lợi thế ở một số class thiểu số hoặc class được oversampling như `foot`, `safety-vest`, `ear-mufs`, `medical-suit`, `face-guard` và gần như ngang bằng ở `helmet`.

Một điểm cũng cần lưu ý là YOLOv9e vẫn có precision cao hơn variant cuối ở hầu hết class. Điều này phản ánh lợi thế của model lớn: khả năng phân biệt object thật và false positive tốt hơn. Ngược lại, variant cuối của YOLOv9s có xu hướng tăng recall ở các class được oversampling, ví dụ `foot`, `safety-vest`, `ear-mufs`, `medical-suit`, `face-guard`, nhưng precision không tăng tương ứng. Đây là trade-off thường gặp khi oversampling: model nhạy hơn với class hiếm nên bắt được nhiều object hơn, nhưng cũng có thể sinh thêm một số false positive.
