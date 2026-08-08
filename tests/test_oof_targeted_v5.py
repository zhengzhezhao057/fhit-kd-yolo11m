from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from src.approve_oof_annotation_policy import approve_manifest
from src.build_hard_replay_manifest import build_replay_manifest
from src.build_oof_targeted_matrix import detector_state_fingerprint, trial_definitions
from src.distill_trainer import (
    HardImageReplayDataset,
    kd_objective_fingerprint,
    load_hard_replay_images,
    reset_kd_calibration_buffers,
)


def test_annotation_policy_approval_preserves_geometry_and_removes_review_gate() -> None:
    candidate = {
        "format": 1, "kind": "vehicle_background", "split": "train", "review_required": True,
        "negative_boxes": 2, "images": {"a.jpg": {"boxes": [{}, {}]}},
    }
    summary = {
        "safe_vehicle_background_candidates": {"boxes": 2},
        "checkpoint_policy": "fixed_epoch_last_pt", "fold_inventory_fingerprint": "folds",
    }
    approved = approve_manifest(candidate, summary, "abc")
    assert approved["review_required"] is False
    assert approved["human_reviewed"] is False
    assert approved["negative_boxes"] == 2
    assert approved["annotation_policy"] == "dataset_d0_exhaustive_positive_annotations"


def test_replay_manifest_unions_hard_ship_vehicle_and_approved_background() -> None:
    hard = {"format": 1, "split": "train", "images": {
        "ship.jpg": {"objects": [{"coarse_group": "ship", "error_type": "low_confidence"}]},
        "plane.jpg": {"objects": [{"coarse_group": "aircraft", "error_type": "low_confidence"}]},
        "easy.jpg": {"objects": [{"coarse_group": "vehicle", "error_type": "detected"}]},
    }}
    negative = {
        "format": 1, "kind": "vehicle_background", "split": "train", "review_required": False,
        "annotation_policy": "test", "images": {"negative.jpg": {"boxes": [{"score": .8}]}},
    }
    replay = build_replay_manifest(hard, negative, "h", "n")
    assert set(replay["images"]) == {"ship.jpg", "negative.jpg"}
    assert replay["images"]["ship.jpg"]["hard_positive_objects"] == 1
    assert replay["images"]["negative.jpg"]["vehicle_background_boxes"] == 1


def test_replay_loader_and_dataset_are_deterministic(tmp_path: Path) -> None:
    manifest = tmp_path / "replay.json"
    manifest.write_text(json.dumps({
        "format": 1, "kind": "hard_image_replay", "split": "train", "images": {"b.jpg": {}},
    }), encoding="utf-8")
    cfg = {"paths": {"project_root": str(tmp_path)}, "distillation": {"hard_image_replay_manifest": str(manifest)}}
    assert load_hard_replay_images(cfg) == {"b.jpg"}
    class Base:
        im_files = ["/d/images/train/a.jpg", "/d/images/train/b.jpg"]
        def __len__(self): return 2
        def __getitem__(self, index): return index
    replay = HardImageReplayDataset(Base(), {"b.jpg"}, 1)
    assert [replay[index] for index in range(len(replay))] == [0, 1, 1]


def test_v5_trials_share_replay_and_activate_background_at_epoch_zero() -> None:
    trials = trial_definitions("hard.json", "negative.json", "replay.json")
    assert [item["name"] for item in trials] == ["v5_oof_replay_c0", "v5_oof_replay_fk", "v5_oof_replay_fk_bg"]
    assert all(item["updates"]["hard_image_replay_manifest"] == "replay.json" for item in trials)
    assert trials[2]["updates"]["vehicle_bg_schedule"]["start_epoch"] == 0
    assert trials[2]["updates"]["vehicle_bg_gradient_ratio"] == pytest.approx(.001)
    assert trials[1]["updates"]["reset_kd_calibration_on_start"] is True


def test_new_objective_resets_calibration_but_not_model_parameters() -> None:
    from src.distillation import StudentDistillAddons

    addons = StudentDistillAddons([16, 32, 64], 32, 25, enable_vehicle_bg=True)
    parameter = next(addons.parameters()).detach().clone()
    with torch.no_grad():
        addons.feature_kd_weight.fill_(2.0); addons.feature_kd_log_sum.fill_(8.0); addons.feature_kd_calibration_count.fill_(4)
        addons.cls_kd_weight.fill_(.2); addons.cls_kd_log_sum.fill_(-4.0); addons.cls_kd_calibration_count.fill_(3)
    assert reset_kd_calibration_buffers(addons) == 9
    assert int(addons.feature_kd_calibration_count) == 0 and float(addons.feature_kd_weight) == 1.0
    assert int(addons.cls_kd_calibration_count) == 0 and float(addons.cls_kd_log_sum) == 0.0
    assert torch.equal(parameter, next(addons.parameters()).detach())


def test_detector_fingerprint_ignores_training_only_addons(tmp_path: Path) -> None:
    full = torch.nn.Module(); full.add_module("detector", torch.nn.Linear(2, 2)); full.add_module("distill_addons", torch.nn.Linear(2, 1))
    deploy = torch.nn.Module(); deploy.add_module("detector", torch.nn.Linear(2, 2))
    deploy.detector.load_state_dict(full.detector.state_dict())
    full_path, deploy_path = tmp_path / "full.pt", tmp_path / "deploy.pt"
    torch.save({"model": full}, full_path); torch.save({"model": deploy}, deploy_path)
    assert detector_state_fingerprint(full_path) == detector_state_fingerprint(deploy_path)


def test_replay_manifest_content_changes_kd_resume_fingerprint(tmp_path: Path) -> None:
    replay = tmp_path / "replay.json"; replay.write_text('{"one":1}', encoding="utf-8")
    cfg = {
        "runtime": {"experiment": "fk"}, "dataset": {"nc": 25, "image_size": 640, "class_groups": {}},
        "student": {"epochs": 6}, "paths": {"project_root": str(tmp_path)},
        "distillation": {"hard_image_replay_manifest": str(replay)},
    }
    first = kd_objective_fingerprint(cfg)
    replay.write_text('{"two":2}', encoding="utf-8")
    assert kd_objective_fingerprint(cfg) != first
