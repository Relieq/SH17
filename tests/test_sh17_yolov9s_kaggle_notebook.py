import json
from pathlib import Path


def test_kaggle_notebook_is_self_contained_and_uses_kaggle_paths():
    notebook_path = Path("SH17_yolov9s_kaggle.ipynb")
    payload = json.loads(notebook_path.read_text(encoding="utf-8"))
    sources = "\n".join(
        "".join(cell.get("source", []))
        for cell in payload["cells"]
    )

    assert "/kaggle/input/" in sources
    assert "/kaggle/working/" in sources
    assert "from scripts.sh17_yolov9c_pipeline import" not in sources
    assert "configs/sh17_yolov9s_experiments.yaml" not in sources
    assert "yolov9s_baseline_640" in sources
    assert "yolov9s_tuned_640" in sources
    assert "yolov9s_multiscale_960" in sources
    assert "yolov9s_oversample_minority_960" in sources


def test_kaggle_notebook_uses_fixed_safe_batch_sizes():
    notebook_path = Path("SH17_yolov9s_kaggle.ipynb")
    payload = json.loads(notebook_path.read_text(encoding="utf-8"))
    sources = "\n".join(
        "".join(cell.get("source", []))
        for cell in payload["cells"]
    )

    assert '"batch": -1' not in sources
    assert '"batch": 8' in sources
    assert '"batch": 4' in sources
