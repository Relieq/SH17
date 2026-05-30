from pathlib import Path

from scripts.sh17_yolov9c_pipeline import (
    DatasetManifest,
    build_dataset_manifest,
    build_oversampled_train_manifest,
    ensure_model_weights,
    expand_experiments,
    load_experiment_config,
    pick_checkpoint_for_resume,
)


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


def test_pick_checkpoint_for_resume_prefers_last_then_best(tmp_path):
    run_dir = tmp_path / "train" / "yolov9c_baseline_640" / "weights"
    run_dir.mkdir(parents=True)
    best = run_dir / "best.pt"
    last = run_dir / "last.pt"
    best.write_bytes(b"best")
    last.write_bytes(b"last")

    selected = pick_checkpoint_for_resume(run_dir.parent)

    assert selected == last


def test_pick_checkpoint_for_resume_prefers_latest_epoch_checkpoint(tmp_path):
    run_dir = tmp_path / "train" / "yolov9c_baseline_640" / "weights"
    run_dir.mkdir(parents=True)
    (run_dir / "best.pt").write_bytes(b"best")
    (run_dir / "last.pt").write_bytes(b"last")
    epoch_3 = run_dir / "epoch3.pt"
    epoch_7 = run_dir / "epoch7.pt"
    epoch_3.write_bytes(b"epoch3")
    epoch_7.write_bytes(b"epoch7")

    selected = pick_checkpoint_for_resume(run_dir.parent)

    assert selected == epoch_7


def test_ensure_model_weights_uses_requested_weight_name(tmp_path, monkeypatch):
    requested = []

    def fake_download(url, output_path):
        requested.append((url, Path(output_path)))
        Path(output_path).write_bytes(b"weights")

    monkeypatch.setattr("scripts.sh17_yolov9c_pipeline.urllib.request.urlretrieve", fake_download)

    path = ensure_model_weights(tmp_path, "yolov9s.pt")

    assert path.name == "yolov9s.pt"
    assert path.exists()
    assert requested[0][0].endswith("/yolov9s.pt")


def test_expand_experiments_keeps_model_specific_paths(tmp_path):
    config = {
        "paths": {
            "dataset_root": Path("E:/data/SH17"),
            "weights_dir": Path("E:/models/yolov9"),
            "artifact_root": Path("D:/DS-AI/SH17/artifacts/sh17_yolov9s"),
            "runs_root": Path("E:/models/sh17_yolov9s_runs"),
        },
        "defaults": {"epochs": 200},
        "experiments": [{"name": "yolov9s_baseline_640", "weights": "yolov9s.pt"}],
    }
    manifest = DatasetManifest(
        train_manifest=tmp_path / "train.txt",
        val_manifest=tmp_path / "val.txt",
        data_yaml=tmp_path / "sh17.yaml",
        train_count=1,
        val_count=1,
    )

    expanded = expand_experiments(config, manifest)

    assert expanded[0]["weights"] == "yolov9s.pt"
    assert expanded[0]["artifact_root"].endswith("sh17_yolov9s")
    assert expanded[0]["runs_root"].endswith("sh17_yolov9s_runs")
