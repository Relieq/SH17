# Example Images Notes

This folder contains 10 sample images from the SH17 `val/test` split.

The layout is now organized by subfolder:

- `image_01` to `image_10`: each folder contains the image file, its matching ground-truth label `.txt`, and its prediction summary `.md`
- each `image_0X` folder also contains a `renders/` subfolder with the overlay images for each model
- `all_predic.md`: combined prediction notes for `image_04` to `image_10`
- `image_notes.md`: this reference file

Inside each `image_0X` folder, you will typically find:

```text
image_0X_*.jpeg
image_0X_*.txt
image_0X_predicted.md
renders/
```

Inside `renders/`, the generated files are:

```text
yolov9s_baseline_overlay.jpg
yolov9s_variant3_overlay.jpg
yolov9e_overlay.jpg
```

## Label Format

Each line in a `.txt` label file has the following format:

```text
<class_id> <x_center> <y_center> <width> <height>
```

- This is the YOLO detection label format.
- All coordinates are normalized to the image size and fall in the `0..1` range.
- To read a label file quickly, map `class_id` using the table below.

## Class Id -> Class Name

| Id | Class name |
| -- | ---------- |
| 0 | person |
| 1 | ear |
| 2 | ear-mufs |
| 3 | face |
| 4 | face-guard |
| 5 | face-mask |
| 6 | foot |
| 7 | tool |
| 8 | glasses |
| 9 | gloves |
| 10 | helmet |
| 11 | hands |
| 12 | head |
| 13 | medical-suit |
| 14 | shoes |
| 15 | safety-suit |
| 16 | safety-vest |

## Minority Classes Prioritized by Oversampling

The oversampling pipeline in the notebook increases the sampling frequency of images that contain at least one of the following classes:

| Id | Class name |
| -- | ---------- |
| 2 | ear-mufs |
| 4 | face-guard |
| 6 | foot |
| 10 | helmet |
| 13 | medical-suit |
| 16 | safety-vest |

## Quick Notes For Each Image

| Folder | Image file | Report role | Key classes to check quickly |
| ------ | ---------- | ----------- | ---------------------------- |
| `image_01` | `image_01_normal_case.jpeg` | Normal case | `person`, `helmet`, `safety-vest`, `gloves`, `tool` |
| `image_02` | `image_02_crowded_scene.jpeg` | Crowded scene | `person`, `head`, `hands`, `shoes`, `safety-vest`, `face-mask`, `foot` |
| `image_03` | `image_03_small_ppe_far_camera.jpeg` | Small PPE / far camera | `person`, `helmet`, `safety-vest`, `shoes`, `safety-suit` |
| `image_04` | `image_04_foot_shoes_case.jpeg` | Foot / shoes case | `foot`, `shoes`, `tool`, `person`, `hands` |
| `image_05` | `image_05_safety_vest_case.jpeg` | Safety-vest case | `safety-vest`, `tool`, `shoes`, `person`, `hands` |
| `image_06` | `image_06_ear_mufs_case.jpeg` | Minority class case | `ear-mufs`, `helmet`, `safety-vest`, `glasses`, `person` |
| `image_07` | `image_07_face_guard_hard_view.jpeg` | Face-guard / hard-view case | `face-guard`, `tool`, `hands`, `person` |
| `image_08` | `image_08_medical_suit_case.jpeg` | Medical-suit case | `medical-suit`, `face-mask`, `glasses`, `shoes`, `person` |
| `image_09` | `image_09_occlusion_case.jpeg` | Occlusion case | `person`, `helmet`, `hands`, `face`, `foot` |
| `image_10` | `image_10_failure_case_tool_confusion.jpeg` | Failure-case candidate | `tool`, `person`, `face`, `hands`, `glasses` |

## Quick Usage During The Presentation

- Open the `image_0X` folder.
- Open `image_0X_*.jpeg`.
- Open the matching `image_0X_*.txt` label file.
- If needed, open `image_0X_predicted.md` to compare the projected outputs of YOLOv9s baseline, YOLOv9s variant 3, and YOLOv9e.
- To view the rendered bounding boxes directly, open the matching files inside `renders/`.
- Map each `class_id` to its class name using the `Class Id -> Class Name` table above.
- If you want to emphasize the effect of oversampling, prioritize `image_04`, `image_05`, `image_06`, `image_07`, and `image_08` because they contain minority classes or harder PPE categories.
- `image_10_failure_case_tool_confusion.jpeg` is a strong failure-case candidate, but to confirm it is a true model failure you should still compare it against the prediction overlay.
