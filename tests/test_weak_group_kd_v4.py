from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import torch
import yaml
import numpy as np

from src.build_hard_example_manifest import build_manifest
from src.build_weak_group_matrix import long_trials, main as build_matrix_main, short_trials
from src.distillation import DistillationLoss, HardExampleStore, StudentDistillAddons, TeacherSignalStore
from src.distill_trainer import kd_objective_fingerprint


def base_config(tmp_path: Path) -> dict:
    return {
        "paths": {"project_root": str(tmp_path), "data_yaml": str(tmp_path / "dataset.yaml")},
        "dataset": {
            "image_size": 64,
            "nc": 25,
            "class_groups": {"ship": [0, 1, 2, 3], "aircraft": list(range(4, 24)), "vehicle": [24]},
        },
        "teacher": {"feature_channels": 32},
        "student": {"epochs": 8, "kd_warmup_epochs": 1},
        "runtime": {"experiment": "fk"},
        "distillation": {
            "temperature": 6.0,
            "feature_scale_weights": [0.6, 0.3, 0.1],
            "foreground_weight": 1.0,
            "boundary_weight": 0.3,
            "background_weight": 0.05,
            "teacher_confidence_floor": 0.0,
            "roi_match_iou_floor": 0.5,
        },
    }


def test_hard_manifest_builder_accepts_train_and_rejects_validation(tmp_path: Path) -> None:
    train_dir = tmp_path / "dataset" / "images" / "train"
    val_dir = tmp_path / "dataset" / "images" / "val"
    train_dir.mkdir(parents=True); val_dir.mkdir(parents=True)
    import cv2

    train_image = train_dir / "a.jpg"; cv2.imwrite(str(train_image), np.zeros((64, 64, 3), dtype=np.uint8))
    val_image = val_dir / "v.jpg"; cv2.imwrite(str(val_image), np.zeros((64, 64, 3), dtype=np.uint8))
    (tmp_path / "dataset.yaml").write_text(
        f"path: {tmp_path / 'dataset'}\ntrain: images/train\nval: images/val\n", encoding="utf-8"
    )
    fields = [
        "model", "image", "gt_index", "fine_class", "coarse_group", "size", "crowded", "edge",
        "error_type", "gt_x", "gt_y", "gt_w", "gt_h",
    ]
    csv_path = tmp_path / "instances.csv"

    def write(image: Path) -> None:
        with csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
            writer.writerow({
                "model": "C0", "image": str(image), "gt_index": 0, "fine_class": 24,
                "coarse_group": "vehicle", "size": "small", "crowded": "True", "edge": "False",
                "error_type": "low_confidence", "gt_x": .5, "gt_y": .5, "gt_w": .1, "gt_h": .1,
            })

    cfg = base_config(tmp_path)
    write(train_image)
    manifest = build_manifest(cfg, csv_path, "C0")
    assert manifest["split"] == "train"
    assert "a.jpg" in manifest["images"]
    assert manifest["error_counts"] == {"low_confidence": 1}
    assert np.allclose(manifest["images"]["a.jpg"]["objects"][0]["box_xywhn"], [.5, .5, .1, .1])

    write(val_image)
    try:
        build_manifest(cfg, csv_path, "C0")
    except RuntimeError as error:
        assert "data leakage" in str(error)
    else:
        raise AssertionError("Validation diagnostics must never become a training manifest.")


def test_group_weighting_redistributes_but_preserves_mean_budget(tmp_path: Path) -> None:
    cfg = base_config(tmp_path)
    cfg["distillation"]["group_distill_weights"] = {"ship": 1.28, "aircraft": 0.70, "vehicle": 1.75}
    wrapper = DistillationLoss(
        lambda _preds, _batch: None,
        StudentDistillAddons([16, 32, 64], 32, 25),
        TeacherSignalStore(tmp_path),
        cfg,
    )
    wrapper.features = [torch.randn(1, 16, 8, 8, requires_grad=True)]
    boxes = torch.tensor([[.25, .5, .1, .1], [.75, .5, .1, .1]])
    batch = {
        "img": torch.zeros(1, 3, 64, 64), "bboxes": boxes, "cls": torch.tensor([[0.0], [4.0]]),
        "batch_idx": torch.tensor([0, 0]), "im_file": [str(tmp_path / "images" / "train" / "a.jpg")],
    }
    signals = [{"boxes_xywhn": boxes, "classes": torch.tensor([0, 4])}]
    weights = wrapper._batch_object_weights(batch, signals, "feature")
    assert torch.allclose(weights.mean(), torch.tensor(1.0))
    assert torch.allclose(weights[0] / weights[1], torch.tensor(1.28 / .70))


def test_hard_store_is_relocation_safe_and_rejects_non_train_manifest(tmp_path: Path) -> None:
    path = tmp_path / "hard.json"
    path.write_text(json.dumps({
        "format": 1, "split": "train",
        "images": {"nested/a.jpg": {"objects": [{"gt_index": 0, "fine_class": 24, "error_type": "low_confidence"}]}},
    }), encoding="utf-8")
    store = HardExampleStore(path)
    found = store.object("/new/root/images/train/nested/a.jpg", 0, 24)
    assert found and found["error_type"] == "low_confidence"
    path.write_text(json.dumps({"format": 1, "split": "val", "images": {"a.jpg": {}}}), encoding="utf-8")
    try:
        HardExampleStore(path)
    except RuntimeError as error:
        assert "split='train'" in str(error)
    else:
        raise AssertionError("A validation manifest must be rejected.")


def test_hard_store_recovers_from_duplicate_label_index_shift_by_geometry(tmp_path: Path) -> None:
    path = tmp_path / "hard.json"
    path.write_text(json.dumps({
        "format": 1, "split": "train",
        "images": {"a.jpg": {"objects": [
            {"gt_index": 0, "fine_class": 3, "box_xywhn": [.25, .5, .1, .1], "error_type": "detected"},
            {"gt_index": 1, "fine_class": 3, "box_xywhn": [.75, .5, .1, .1], "error_type": "low_confidence"},
        ]}},
    }), encoding="utf-8")
    store = HardExampleStore(path)
    # A duplicate cached before the second object shifted its teacher index to
    # 2. Geometry must recover the second object instead of trusting gt_index.
    found = store.object(
        "/root/dataset/images/train/a.jpg", 2, 3, torch.tensor([.75, .5, .1, .1])
    )
    assert found and found["error_type"] == "low_confidence"


def test_v4_matrix_is_bounded_and_long_phase_keeps_reference_controls() -> None:
    short = short_trials("hard.json")
    assert [trial["name"] for trial in short] == ["v4_gb_fk", "v4_eh_fk", "v4_eh_f"]
    assert all(trial["updates"]["feature_gradient_ratio"] == .03 for trial in short)
    assert all(trial["updates"].get("cls_gradient_ratio") == .005 for trial in short)
    long = long_trials("hard.json")
    assert len(long) == 5
    assert {trial["experiment"] for trial in long} == {"c0", "f", "k", "fk"}
    assert long[-1]["name"] == "v4l_wt_fk"


def test_v4_generator_writes_resume_safe_scripts(tmp_path: Path, monkeypatch) -> None:
    cfg = base_config(tmp_path)
    cfg["paths"].update(baseline_weights="baseline.pt", dino_weights="dino.pt", dino_repo="external/dinov3")
    config_path = tmp_path / "direction1.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    hard = tmp_path / "hard.json"
    hard.write_text('{"format":1,"split":"train","images":{"a.jpg":{"objects":[]}}}', encoding="utf-8")
    output = tmp_path / "matrix"
    monkeypatch.setattr(sys, "argv", [
        "build_weak_group_matrix", "--base", str(config_path), "--hard-manifest", str(hard),
        "--phase", "short", "--out", str(output),
    ])
    build_matrix_main()
    script = (output / "run.sh").read_text(encoding="utf-8")
    matrix = json.loads((output / "matrix.json").read_text(encoding="utf-8"))
    assert len(matrix) == 3
    assert "DIRECTION1 RESUME" in script
    assert "v4_eh_fk" in script
    generated = yaml.safe_load((output / "v4_eh_fk.yaml").read_text(encoding="utf-8"))
    assert generated["distillation"]["hard_example_manifest"] == str(hard)
    assert generated["student"]["epochs"] == 8


def test_objective_fingerprint_changes_with_manifest_content(tmp_path: Path) -> None:
    cfg = base_config(tmp_path)
    manifest = tmp_path / "hard.json"
    manifest.write_text('{"format":1,"split":"train","images":{"a.jpg":{"objects":[]}}}', encoding="utf-8")
    cfg["distillation"]["hard_example_manifest"] = str(manifest)
    first = kd_objective_fingerprint(cfg)
    manifest.write_text('{"format":1,"split":"train","images":{"b.jpg":{"objects":[]}}}', encoding="utf-8")
    second = kd_objective_fingerprint(cfg)
    assert first != second
