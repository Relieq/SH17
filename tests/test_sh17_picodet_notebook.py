import ast
import json
from pathlib import Path


def _notebook_payload():
    notebook_path = Path("SH17_picodet_l_paddledet_portable.ipynb")
    return json.loads(notebook_path.read_text(encoding="utf-8"))


def _notebook_sources():
    payload = _notebook_payload()
    return "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])


def test_picodet_notebook_contains_expected_experiments_and_class_order():
    sources = _notebook_sources()

    assert "PP-PicoDet-L" in sources
    assert "picodet_l_320_baseline_50e" in sources
    assert "picodet_l_640_scale_50e" in sources
    assert "picodet_l_640_oversample_minority_50e" in sources
    assert "picodet_l_640_zoom_crop_50e" in sources
    assert "picodet_l_416_scale" not in sources
    assert "PaddleDetection" in sources
    assert "SSDLite320" not in sources
    assert "create_model" not in sources
    assert "YOLO(" not in sources

    expected_class_order = [
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
    for class_name in expected_class_order:
        assert f'"{class_name}"' in sources


def test_picodet_notebook_normalizes_data_root_before_manifest_paths():
    sources = _notebook_sources()

    assert "DATA_ROOT_CANDIDATES = _data_root_candidates()" in sources
    assert "def _resolve_data_root()" in sources
    assert 'Path("E:/data/SH17")' in sources
    assert 'Path("D:/data/SH17")' in sources
    assert 'PROJECT_ROOT / "datasets" / "sh17-dataset-for-ppe-detection"' in sources
    assert 'PROJECT_ROOT / "datasets" / "mugheesahmad" / "sh17-dataset-for-ppe-detection"' in sources
    assert "DATA_ROOT = Path(DATA_ROOT).expanduser()" in sources
    assert 'TRAIN_MANIFEST = Path(os.environ.get("SH17_TRAIN_MANIFEST", DATA_ROOT / "train_files.txt"))' in sources
    assert "Missing dataset manifests." in sources
    assert "Set DATA_ROOT to the SH17 dataset folder" in sources


def test_picodet_notebook_imports_helper_from_project_scripts():
    sources = _notebook_sources()

    assert "SELF_CONTAINED_HELPER_SOURCE" not in sources
    assert "def _helper_roots()" in sources
    assert 'PROJECT_ROOT / "SH17"' in sources
    assert 'root / "scripts" / "sh17_picodet_dataset.py"' in sources
    assert "Run this notebook from the SH17 project folder" in sources
    assert "from scripts.sh17_picodet_dataset import" in sources


def test_picodet_notebook_uses_paddledetection_train_cli_without_output_dir():
    sources = _notebook_sources()

    assert '"--output_dir"' not in sources
    assert '"tools/train.py"' in sources
    assert "last_checkpoint_base_path" in sources
    assert 'command.extend(["-r", str(last_checkpoint)])' in sources


def test_picodet_notebook_has_local_friendly_dependency_setup():
    sources = _notebook_sources()

    assert 'print("PYTHON_EXECUTABLE:", sys.executable)' in sources
    assert "def command_env(extra_env=None)" in sources
    assert 'base_env.setdefault("PYTHONUTF8", "1")' in sources
    assert 'base_env.setdefault("PYTHONIOENCODING", "utf-8")' in sources
    assert 'encoding="utf-8"' in sources
    assert 'errors="replace"' in sources
    assert "INSTALL_PADDLEDET_REQUIREMENTS = _env_flag" in sources
    assert 'BOOTSTRAP_PIP_PACKAGES = ["setuptools<81", "wheel"]' in sources
    assert "PADDLEDET_REQUIREMENTS_ONLY_BINARY = _env_flag" in sources
    assert "def ensure_supported_requirements_python()" in sources
    assert "sys.version_info >= (3, 13)" in sources
    assert "PaddleDetection requirements pin numpy<2.0" in sources
    assert "def paddledet_requirements_install_command(requirements)" in sources
    assert 'PADDLEDET_BINARY_ONLY_PACKAGES = ["numpy", "scipy", "opencv-python", "pycocotools", "shapely"]' in sources
    assert '"--only-binary=" + ",".join(PADDLEDET_BINARY_ONLY_PACKAGES)' in sources
    assert "PADDLE_INSTALL_INDEX_URL" in sources
    assert '"-i", PADDLE_INSTALL_INDEX_URL' in sources
    assert "PADDLE_INSTALL_PACKAGE" in sources
    assert "AUTO_REINSTALL_PADDLE_PACKAGE = _env_flag" in sources
    assert "def ensure_paddle_package()" in sources
    assert "paddlepaddle-gpu" in sources
    assert "pip\", \"uninstall\", \"-y\"" in sources
    assert "pkg_resources" in sources
    assert "SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL" in sources
    assert 'run_command([sys.executable, "-m", "pip", "install", *BOOTSTRAP_PIP_PACKAGES], env=pip_env)' in sources
    assert "run_command(paddledet_requirements_install_command(requirements), env=pip_env)" in sources
    assert '"pip", "install", "-q", "-r"' not in sources


def test_picodet_notebook_can_fallback_to_cpu_when_gpu_is_unavailable():
    sources = _notebook_sources()

    assert "VERIFY_PADDLE_GPU_RUNTIME = _env_flag" in sources
    assert "def verify_paddle_gpu_runtime(paddle, device)" in sources
    assert "paddle.utils.run_check()" in sources
    assert "cudnn64_8.dll" in sources
    assert "ALLOW_CPU_FALLBACK = os.environ.get" in sources
    assert "if USE_GPU and \"gpu\" not in device.lower():" in sources
    assert "if ALLOW_CPU_FALLBACK:" in sources
    assert 'paddle.set_device("cpu")' in sources
    assert "Falling back to CPU" in sources
    assert "TRAIN_WITH_AMP" in sources
    assert 'command.append("--amp")' in sources


def test_picodet_notebook_code_cells_are_valid_python():
    payload = _notebook_payload()

    for index, cell in enumerate(payload["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        ast.parse(source, filename=f"SH17_picodet_l_paddledet_portable.ipynb cell {index}")


def test_picodet_helper_converts_yolo_manifests_to_coco_and_oversamples(tmp_path):
    from scripts.sh17_picodet_dataset import (
        CLASS_NAMES,
        MINORITY_CLASS_IDS_ZERO_BASED,
        build_oversampled_coco,
        convert_manifest_to_coco,
    )

    data_root = tmp_path / "sh17"
    image_dir = data_root / "images"
    label_dir = data_root / "labels"
    image_dir.mkdir(parents=True)
    label_dir.mkdir()

    train_lines = []
    for index in range(8):
        image_path = image_dir / f"train_{index}.jpg"
        image_path.write_bytes(b"fake")
        train_lines.append(f"images/{image_path.name}")
        class_id = 2 if index == 0 else 0
        (label_dir / f"train_{index}.txt").write_text(
            f"{class_id} 0.5 0.5 0.2 0.4\n",
            encoding="utf-8",
        )

    val_lines = []
    for index in range(4):
        image_path = image_dir / f"val_{index}.jpg"
        image_path.write_bytes(b"fake")
        val_lines.append(f"images/{image_path.name}")
        (label_dir / f"val_{index}.txt").write_text(
            "10 0.5 0.5 0.1 0.2\n",
            encoding="utf-8",
        )

    train_manifest = data_root / "train_files.txt"
    val_manifest = data_root / "val_files.txt"
    train_manifest.write_text("\n".join(train_lines), encoding="utf-8")
    val_manifest.write_text("\n".join(val_lines), encoding="utf-8")

    annotations_dir = tmp_path / "annotations"
    train_json = annotations_dir / "instances_train.json"
    val_json = annotations_dir / "instances_val.json"
    resolver = lambda _: (100, 200)

    train_stats = convert_manifest_to_coco(
        manifest_path=train_manifest,
        dataset_root=data_root,
        output_json=train_json,
        image_size_resolver=resolver,
    )
    val_stats = convert_manifest_to_coco(
        manifest_path=val_manifest,
        dataset_root=data_root,
        output_json=val_json,
        image_size_resolver=resolver,
    )

    train_payload = json.loads(train_json.read_text(encoding="utf-8"))
    val_payload = json.loads(val_json.read_text(encoding="utf-8"))

    assert train_stats["images"] == 8
    assert train_stats["instances"] == 8
    assert val_stats["images"] == 4
    assert len(train_payload["categories"]) == len(CLASS_NAMES) == 17
    assert train_payload["annotations"][0]["bbox"] == [40.0, 60.0, 20.0, 80.0]
    assert train_payload["annotations"][0]["category_id"] == 3
    assert val_payload["annotations"][0]["category_id"] == 11

    oversampled_json = annotations_dir / "instances_train_oversampled.json"
    oversampled_stats = build_oversampled_coco(
        train_json=train_json,
        output_json=oversampled_json,
        minority_class_ids_zero_based=MINORITY_CLASS_IDS_ZERO_BASED,
        repeat_factor=3,
    )
    oversampled_payload = json.loads(oversampled_json.read_text(encoding="utf-8"))

    assert oversampled_stats["images"] == 10
    assert oversampled_stats["instances"] == 10
    assert len({image["id"] for image in oversampled_payload["images"]}) == 10
    assert len({ann["id"] for ann in oversampled_payload["annotations"]}) == 10


def test_picodet_config_generation_uses_sh17_dataset_and_variant_settings(tmp_path):
    from scripts.sh17_picodet_dataset import build_picodet_config_text, picodet_experiments

    experiments = {experiment.name: experiment for experiment in picodet_experiments()}
    assert {
        name: experiment.config_file_name
        for name, experiment in experiments.items()
    } == {
        "picodet_l_320_baseline_50e": "picodet_l_320_sh17.yml",
        "picodet_l_640_scale_50e": "picodet_l_640_sh17.yml",
        "picodet_l_640_oversample_minority_50e": "picodet_l_640_oversample_sh17.yml",
        "picodet_l_640_zoom_crop_50e": "picodet_l_640_zoom_crop_sh17.yml",
    }
    config_text = build_picodet_config_text(
        experiment=experiments["picodet_l_640_oversample_minority_50e"],
        dataset_dir=tmp_path / "dataset",
        output_dir=tmp_path / "runs",
        paddledet_root=tmp_path / "PaddleDetection",
        epochs=50,
        snapshot_epoch=5,
        worker_num=6,
        use_shared_memory=True,
    )

    assert "picodet_l_640_coco_lcnet.yml" in config_text
    assert "num_classes: 17" in config_text
    assert "epoch: 50" in config_text
    assert "use_ema: true" in config_text
    assert "use_gpu: true" in config_text
    assert "snapshot_epoch: 5" in config_text
    assert "worker_num: 6" in config_text
    assert "use_shared_memory: true" in config_text
    assert f"save_dir: {(tmp_path / 'runs' / 'picodet_l_640_oversample_minority_50e').as_posix()}" in config_text
    assert "base_lr: 0.06" in config_text
    assert "batch_size: 12" in config_text
    assert "instances_train_oversampled.json" in config_text

    cpu_config_text = build_picodet_config_text(
        experiment=experiments["picodet_l_320_baseline_50e"],
        dataset_dir=tmp_path / "dataset",
        output_dir=tmp_path / "runs",
        paddledet_root=tmp_path / "PaddleDetection",
        use_gpu=False,
    )
    assert "use_gpu: false" in cpu_config_text

    zoom_config_text = build_picodet_config_text(
        experiment=experiments["picodet_l_640_zoom_crop_50e"],
        dataset_dir=tmp_path / "dataset",
        output_dir=tmp_path / "runs",
        paddledet_root=tmp_path / "PaddleDetection",
        epochs=50,
        snapshot_epoch=5,
    )
    assert "BatchRandomResize: {target_size: [512, 544, 576, 608, 640, 672, 704, 736, 768]" in zoom_config_text
    assert "picodet_l_640_zoom_crop_50e" in zoom_config_text


def test_picodet_analyzer_notebook_uses_pending_for_missing_real_metrics():
    analyzer_path = Path("analyze_picodet_l_variants.ipynb")
    payload = json.loads(analyzer_path.read_text(encoding="utf-8"))
    sources = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])

    assert "pending" in sources
    assert "picodet_l_640_zoom_crop_50e" in sources
    assert "summary_metrics.csv" in sources
    assert "per_class_metrics.csv" in sources
    assert "variant_map_comparison.png" in sources
    assert "random.uniform" not in sources
    assert "np.random" not in sources
