from pathlib import Path

import torch
import yaml

from src.build_direction1_matrix import resumable_screen_prelude, trial_matrix
from src.teacher_diagnostics import analyse_entries, size_bucket


def test_teacher_diagnostics_reports_class_coarse_and_size() -> None:
    logits = torch.full((3, 25), -4.0)
    logits[0, 0] = 4.0
    logits[1, 5] = 4.0
    logits[2, 0] = 4.0  # vehicle truth intentionally misclassified as a ship
    entries = [{
        "classes": torch.tensor([0, 5, 24]),
        "roi_logits": logits,
        "boxes_xywhn": torch.tensor([[0.5, 0.5, 0.02, 0.02], [0.5, 0.5, 0.1, 0.1], [0.5, 0.5, 0.3, 0.3]]),
    }]

    report = analyse_entries(entries, 25, 640)

    assert report["instances"] == 3
    assert report["correct"] == 2
    assert report["per_coarse_group"]["ship"]["accuracy"] == 1.0
    assert report["per_coarse_group"]["aircraft"]["accuracy"] == 1.0
    assert report["per_coarse_group"]["vehicle"]["accuracy"] == 0.0
    assert report["per_size"]["small"]["instances"] == 1
    assert report["per_size"]["medium"]["instances"] == 1
    assert report["per_size"]["large"]["instances"] == 1
    assert set(report["temperature_profiles"]) == {"1", "2", "3", "4", "6"}
    assert report["temperature_profiles"]["6"]["top1_probability_quantiles"]["p50"] < report["temperature_profiles"]["1"]["top1_probability_quantiles"]["p50"]


def test_direction1_screen_is_bounded_and_contains_all_ablation_groups() -> None:
    trials = trial_matrix({})
    assert len(trials) == 8
    assert {trial["experiment"] for trial in trials} == {"c0", "f", "k", "fk"}
    assert len({trial["name"] for trial in trials}) == len(trials)
    k_trials = [trial for trial in trials if trial["experiment"] == "k"]
    assert {(trial["updates"]["temperature"], trial["updates"]["cls_gradient_ratio"]) for trial in k_trials} == {
        (4.0, 0.005), (6.0, 0.005), (6.0, 0.01)
    }
    assert len({trial["updates"]["teacher_confidence_floor"] for trial in k_trials}) == 1


def test_direction1_continuation_does_not_use_destructive_bias_warmup() -> None:
    config_path = Path(__file__).parents[1] / "configs" / "direction1.example.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    student = config["student"]
    assert student["warmup_epochs"] == 0.0
    assert student["warmup_bias_lr"] == student["lr0"]


def test_direction1_screen_runner_skips_complete_and_resumes_checkpointed_runs() -> None:
    script = "\n".join(resumable_screen_prelude(8))
    assert "TARGET_EPOCHS=8" in script
    assert "DIRECTION1 SKIP" in script
    assert "DIRECTION1 RESUME" in script
    assert "args+=(--resume)" in script
    assert "has no resumable last.pt" in script
    assert 'local run_dir="runs/$name" last=' not in script
    assert 'local run_dir="runs/$name"\n  local last="$run_dir/weights/last.pt"' in script


def test_size_bucket_uses_coco_pixel_area_thresholds() -> None:
    assert size_bucket(torch.tensor([0.5, 0.5, 0.01, 0.01]), 640) == "small"
    assert size_bucket(torch.tensor([0.5, 0.5, 0.1, 0.1]), 640) == "medium"
    assert size_bucket(torch.tensor([0.5, 0.5, 0.2, 0.2]), 640) == "large"
