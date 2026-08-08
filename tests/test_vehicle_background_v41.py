from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from src.build_vehicle_bg_matrix import vehicle_bg_trials
from src.build_vehicle_negative_manifest import build_vehicle_negative_manifest
from src.common import stable_image_key
from src.distill_trainer import (
    HardImageReplayDataset,
    export_plain_yolo_checkpoint,
    kd_objective_fingerprint,
    restore_legacy_kd_calibration,
    sync_kd_calibration_buffers,
)
from src.distillation import DistillationLoss, StudentDistillAddons, TeacherSignalStore, VehicleNegativeStore


def config(tmp_path: Path) -> dict:
    return {
        "paths": {"project_root": str(tmp_path), "data_yaml": str(tmp_path / "dataset.yaml")},
        "dataset": {"image_size": 64, "nc": 25, "class_groups": {"vehicle": [24]}},
        "teacher": {"feature_channels": 32},
        "student": {"epochs": 8, "kd_warmup_epochs": 1},
        "runtime": {"experiment": "fk"},
        "distillation": {
            "weighting_mode": "gradient_calibrated", "temperature": 6.0,
            "feature_scale_weights": [.6, .3, .1], "foreground_weight": 1.0,
            "boundary_weight": .3, "background_weight": .05,
            "teacher_confidence_floor": 0.0, "roi_match_iou_floor": .5,
            "vehicle_bg_enabled": True,
        },
    }


def write_fp_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "model", "image", "fine_class", "coarse_group", "score", "box_x1", "box_y1",
        "box_x2", "box_y2", "reason", "predicted_size",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def test_negative_manifest_keeps_only_train_vehicle_background_and_letterboxes(tmp_path: Path) -> None:
    train = tmp_path / "dataset" / "images" / "train"; train.mkdir(parents=True)
    image_path = train / "wide.jpg"
    cv2.imwrite(str(image_path), np.zeros((100, 200, 3), dtype=np.uint8))
    (tmp_path / "dataset.yaml").write_text(
        f"path: {tmp_path / 'dataset'}\ntrain: images/train\nval: images/val\n", encoding="utf-8"
    )
    csv_path = tmp_path / "fp.csv"
    base = {"model": "B0", "image": str(image_path), "fine_class": 24, "coarse_group": "vehicle", "score": .6,
            "box_x1": 50, "box_y1": 25, "box_x2": 100, "box_y2": 75, "predicted_size": "medium"}
    write_fp_csv(csv_path, [{**base, "reason": "background"}, {**base, "reason": "localization"}])
    manifest = build_vehicle_negative_manifest(config(tmp_path), csv_path, "B0", .35)
    assert manifest["negative_boxes"] == 1
    assert manifest["excluded"]["localization"] == 1
    box = manifest["images"]["wide.jpg"]["boxes"][0]["box_xywhn"]
    assert np.allclose(box, [.375, .5, .25, .25])


def test_vehicle_background_loss_reaches_p3_and_auxiliary_head(tmp_path: Path) -> None:
    manifest_path = tmp_path / "negative.json"
    manifest_path.write_text(json.dumps({
        "format": 1, "kind": "vehicle_background", "split": "train",
        "images": {"a.jpg": {"boxes": [{"box_xywhn": [.75, .5, .2, .2], "score": .8}]}},
    }), encoding="utf-8")
    cfg = config(tmp_path); cfg["distillation"]["vehicle_negative_manifest"] = str(manifest_path)
    addons = StudentDistillAddons([16, 32, 64], 32, 25, enable_vehicle_bg=True)
    wrapper = DistillationLoss(lambda _p, _b: None, addons, TeacherSignalStore(tmp_path), cfg)
    p3 = torch.randn(1, 16, 8, 8, requires_grad=True); wrapper.features = [p3]
    batch = {
        "img": torch.zeros(1, 3, 64, 64), "im_file": ["/data/images/train/a.jpg"],
        "bboxes": torch.tensor([[.25, .5, .2, .2]]), "batch_idx": torch.tensor([0]),
        "cls": torch.tensor([[24.0]]),
    }
    loss, positives, negatives = wrapper.vehicle_background_loss(batch)
    loss.backward()
    assert positives == negatives == 1
    assert float(loss.detach()) > 0
    assert p3.grad is not None and float(p3.grad.norm()) > 0
    assert next(addons.vehicle_bg_head.parameters()).grad is not None


def test_vehicle_background_branch_survives_full_detector_backward(tmp_path: Path) -> None:
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import DetectionModel

    image_path = tmp_path / "dataset" / "images" / "train" / "a.jpg"
    image_path.parent.mkdir(parents=True)
    cv2.imwrite(str(image_path), np.zeros((64, 64, 3), dtype=np.uint8))
    cache = tmp_path / "cache"; cache.mkdir()
    negative_manifest = tmp_path / "negative.json"
    negative_manifest.write_text(json.dumps({
        "format": 1, "kind": "vehicle_background", "split": "train",
        "images": {"a.jpg": {"boxes": [{"box_xywhn": [.75, .5, .2, .2], "score": .8}]}},
    }), encoding="utf-8")
    cfg = config(tmp_path)
    cfg["distillation"].update({
        "vehicle_negative_manifest": str(negative_manifest), "calibration_batches": 1,
        "gradient_probe_batches": 1, "feature_gradient_ratio": .03,
        "cls_gradient_ratio": .003, "vehicle_bg_gradient_ratio": .002,
        "feature_weight_bounds": [1e-6, 1e6], "cls_weight_bounds": [1e-6, 1e6],
        "vehicle_bg_weight_bounds": [1e-6, 1e6],
    })
    model = DetectionModel("yolo11n.yaml", ch=3, nc=25, verbose=False).train()
    model.args = get_cfg()
    with torch.no_grad():
        feature_shapes = model(torch.zeros(1, 3, 64, 64))["feats"]
    channels = [int(feature.shape[1]) for feature in feature_shapes]
    addons = StudentDistillAddons(channels, 32, 25, enable_vehicle_bg=True)
    model.add_module("distill_addons", addons)
    box = torch.tensor([[.25, .5, .2, .2]], dtype=torch.float32)
    torch.save({
        "path": str(image_path.resolve()), "boxes_xywhn": box, "classes": torch.tensor([24]),
        "roi_logits": torch.randn(1, 25),
        "p3": torch.randn(32, *feature_shapes[0].shape[-2:]),
        "p4": torch.randn(32, *feature_shapes[1].shape[-2:]),
        "p5": torch.randn(32, *feature_shapes[2].shape[-2:]),
    }, cache / f"{stable_image_key(image_path)}.pt")
    wrapper = DistillationLoss(model.init_criterion(), addons, TeacherSignalStore(cache), cfg)
    model.criterion = wrapper
    batch = {
        "img": torch.rand(1, 3, 64, 64), "batch_idx": torch.tensor([0]),
        "cls": torch.tensor([[24.0]]), "bboxes": box, "im_file": [str(image_path)],
    }
    loss, _items = model(batch)
    loss.backward()
    assert wrapper.last["vehicle_bg_raw"] > 0
    assert wrapper.last["kd"] > 0
    assert int(addons.vehicle_bg_kd_calibration_count) == 1
    assert next(addons.vehicle_bg_head.parameters()).grad is not None


def test_sparse_background_health_patience_restarts_each_epoch(tmp_path: Path) -> None:
    manifest_path = tmp_path / "negative.json"
    manifest_path.write_text(json.dumps({
        "format": 1, "kind": "vehicle_background", "split": "train",
        "images": {"a.jpg": {"boxes": [{"box_xywhn": [.5, .5, .2, .2], "score": .8}]}},
    }), encoding="utf-8")
    cfg = config(tmp_path)
    cfg["runtime"]["experiment"] = "c0"
    cfg["distillation"].update({
        "vehicle_negative_manifest": str(manifest_path),
        "vehicle_bg_health_patience_batches": 128,
        "vehicle_bg_schedule": {"start_epoch": 1, "warmup_epochs": 1, "hold_epochs": 1, "decay_epochs": 1},
    })
    addons = StudentDistillAddons([16, 32, 64], 32, 25, enable_vehicle_bg=True)
    wrapper = DistillationLoss(lambda _p, _b: None, addons, TeacherSignalStore(tmp_path), cfg)
    # Simulate a complete first epoch, then the beginning of the first active
    # background epoch. Cumulative calls are already large, current-epoch calls
    # are not, so health must keep waiting instead of raising a false failure.
    wrapper.batch_calls = 700
    wrapper.set_epoch(1)
    wrapper.epoch_totals["batches"] = 10
    wrapper.assert_health(10)


def test_negative_manifest_rejects_validation_image(tmp_path: Path) -> None:
    train = tmp_path / "dataset" / "images" / "train"; train.mkdir(parents=True)
    val = tmp_path / "dataset" / "images" / "val"; val.mkdir(parents=True)
    cv2.imwrite(str(train / "train.jpg"), np.zeros((64, 64, 3), dtype=np.uint8))
    val_image = val / "val.jpg"
    cv2.imwrite(str(val_image), np.zeros((64, 64, 3), dtype=np.uint8))
    (tmp_path / "dataset.yaml").write_text(
        f"path: {tmp_path / 'dataset'}\ntrain: images/train\nval: images/val\n", encoding="utf-8"
    )
    csv_path = tmp_path / "fp.csv"
    write_fp_csv(csv_path, [{
        "model": "B0", "image": str(val_image), "fine_class": 24, "coarse_group": "vehicle",
        "score": .6, "box_x1": 10, "box_y1": 10, "box_x2": 20, "box_y2": 20,
        "reason": "background", "predicted_size": "small",
    }])
    with pytest.raises(RuntimeError, match="data leakage"):
        build_vehicle_negative_manifest(config(tmp_path), csv_path, "B0", .35)


def test_replay_dataset_appends_only_manifest_images() -> None:
    class Base:
        im_files = ["/d/images/train/a.jpg", "/d/images/train/b.jpg"]
        def __len__(self): return 2
        def __getitem__(self, index): return index
    replay = HardImageReplayDataset(Base(), {"b.jpg"}, repeats=3)
    assert len(replay) == 5
    assert [replay[index] for index in range(5)] == [0, 1, 1, 1, 1]
    assert replay.replay_images == 1


def test_v41_matrix_reallocates_fixed_half_percent_k_family_budget() -> None:
    trials = vehicle_bg_trials("hard.json", "negative.json")
    assert [trial["name"] for trial in trials] == ["v41_replay_fk", "v41_k003_bg002", "v41_k001_bg004"]
    assert trials[0]["updates"]["cls_gradient_ratio"] == .005
    assert trials[0]["updates"]["vehicle_bg_enabled"] is False
    for trial in trials[1:]:
        updates = trial["updates"]
        assert updates["vehicle_bg_enabled"] is True
        assert abs(updates["cls_gradient_ratio"] + updates["vehicle_bg_gradient_ratio"] - .005) < 1e-12


def test_vehicle_negative_manifest_changes_resume_fingerprint(tmp_path: Path) -> None:
    manifest = tmp_path / "negative.json"
    manifest.write_text('{"one":1}', encoding="utf-8")
    cfg = config(tmp_path)
    cfg["paths"]["baseline_weights"] = "baseline.pt"
    cfg["distillation"]["vehicle_negative_manifest"] = str(manifest)
    first = kd_objective_fingerprint(cfg)
    manifest.write_text('{"two":2}', encoding="utf-8")
    assert kd_objective_fingerprint(cfg) != first


def test_deployment_export_removes_vehicle_background_head(tmp_path: Path) -> None:
    from ultralytics.nn.tasks import DetectionModel
    from src.distill_trainer import infer_student_channels

    model = DetectionModel("yolo11n.yaml", nc=25, ch=3, verbose=False)
    channels = infer_student_channels(model, 64)
    model.add_module("distill_addons", StudentDistillAddons(channels, 32, 25, enable_vehicle_bg=True))
    detector_before = {
        name: value.clone() for name, value in model.state_dict().items()
        if not name.startswith("distill_addons.")
    }
    source, destination = tmp_path / "best.pt", tmp_path / "best_deploy.pt"
    torch.save({"model": model, "optimizer": None, "epoch": -1}, source)
    assert export_plain_yolo_checkpoint(source, destination) > 0
    deployed = torch.load(destination, map_location="cpu", weights_only=False)["model"]
    assert not hasattr(deployed, "distill_addons")
    assert detector_before.keys() == deployed.state_dict().keys()
    assert all(torch.equal(detector_before[name], deployed.state_dict()[name]) for name in detector_before)


def test_calibration_buffers_bypass_ema_and_legacy_checkpoint_is_recoverable(tmp_path: Path) -> None:
    live = StudentDistillAddons([16, 32, 64], 32, 25, enable_vehicle_bg=True)
    ema = StudentDistillAddons([16, 32, 64], 32, 25, enable_vehicle_bg=True)
    with torch.no_grad():
        live.feature_kd_weight.fill_(1.9); live.feature_kd_log_sum.fill_(82.0); live.feature_kd_calibration_count.fill_(128)
        live.cls_kd_weight.fill_(.016); live.cls_kd_log_sum.fill_(-529.0); live.cls_kd_calibration_count.fill_(128)
        live.vehicle_bg_kd_weight.fill_(.2); live.vehicle_bg_kd_log_sum.fill_(-96.0); live.vehicle_bg_kd_calibration_count.fill_(60)
    assert sync_kd_calibration_buffers(live, ema) == 9
    for name, value in live.named_buffers():
        if "_kd_" in name:
            assert torch.equal(value, dict(ema.named_buffers())[name])

    model = torch.nn.Module(); model.add_module("distill_addons", ema)
    # Simulate the old EMA corruption: floating state moved but integer count
    # remained at its initialization value.
    ema.feature_kd_weight.fill_(1.2); ema.feature_kd_log_sum.fill_(5.0); ema.feature_kd_calibration_count.zero_()
    health = tmp_path / "kd_health.jsonl"
    health.write_text(json.dumps({
        "epoch": 1,
        "feature_kd_weight": 1.9, "feature_kd_calibration_count": 128,
        "cls_kd_weight": .016, "cls_kd_calibration_count": 128,
        "vehicle_bg_kd_weight": .2, "vehicle_bg_kd_calibration_count": 60,
    }) + "\n", encoding="utf-8")
    assert restore_legacy_kd_calibration(model, health, checkpoint_epoch=0) == 9
    assert int(ema.feature_kd_calibration_count) == 128
    assert float(ema.feature_kd_weight) == pytest.approx(1.9)
    assert float(ema.feature_kd_log_sum) == pytest.approx(128 * np.log(1.9), rel=1e-6)
    assert model.kd_calibration_buffers_exact is True
    assert restore_legacy_kd_calibration(model, health, checkpoint_epoch=0) == 0


def test_inconsistent_ema_calibration_state_fails_with_branch_name(tmp_path: Path) -> None:
    manifest_path = tmp_path / "negative.json"
    manifest_path.write_text(json.dumps({
        "format": 1, "kind": "vehicle_background", "split": "train",
        "images": {"a.jpg": {"boxes": [{"box_xywhn": [.5, .5, .2, .2], "score": .8}]}},
    }), encoding="utf-8")
    cfg = config(tmp_path); cfg["distillation"]["vehicle_negative_manifest"] = str(manifest_path)
    addons = StudentDistillAddons([16, 32, 64], 32, 25, enable_vehicle_bg=True)
    addons.cls_kd_calibration_count.zero_(); addons.cls_kd_log_sum.fill_(5.0)
    wrapper = DistillationLoss(lambda _p, _b: None, addons, TeacherSignalStore(tmp_path), cfg)
    gradients = (torch.ones(1),)
    with pytest.raises(RuntimeError, match="inconsistent cls calibration state"):
        wrapper._calibrated_weight("cls", gradients, gradients)


def test_legacy_recovery_runs_after_ultralytics_rebuilds_real_trainer_model(tmp_path: Path) -> None:
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import DetectionModel
    from src.distill_trainer import DistillationDetectionTrainer, infer_student_channels

    source = DetectionModel("yolo11n.yaml", ch=3, nc=25, verbose=False)
    source.args = get_cfg()
    channels = infer_student_channels(source, 64)
    source.add_module("distill_addons", StudentDistillAddons(channels, 32, 25, enable_vehicle_bg=True))
    # This is the exact old checkpoint inconsistency reported by the server.
    source.distill_addons.feature_kd_calibration_count.zero_()
    source.distill_addons.feature_kd_log_sum.fill_(95.0625)
    health = tmp_path / "kd_health.jsonl"
    health.write_text(json.dumps({
        "epoch": 1,
        "feature_kd_weight": 2.1, "feature_kd_calibration_count": 128,
        "cls_kd_weight": .02, "cls_kd_calibration_count": 128,
        "vehicle_bg_kd_weight": .2, "vehicle_bg_kd_calibration_count": 60,
    }) + "\n", encoding="utf-8")
    cfg = config(tmp_path)
    cfg["runtime"].update({
        "cache_manifest": {},
        "legacy_calibration_recovery": {"health_file": str(health), "checkpoint_epoch": 0},
    })
    DistillationDetectionTrainer.configure(cfg, tmp_path)
    trainer = object.__new__(DistillationDetectionTrainer)
    trainer.data = {"nc": 25, "channels": 3}
    rebuilt = trainer.get_model(cfg=source.yaml, weights=source, verbose=False)
    assert int(rebuilt.distill_addons.feature_kd_calibration_count) == 128
    assert float(rebuilt.distill_addons.feature_kd_weight) == pytest.approx(2.1)
    assert float(rebuilt.distill_addons.feature_kd_log_sum) == pytest.approx(128 * np.log(2.1), rel=1e-6)
    assert rebuilt.kd_calibration_buffers_exact is True
