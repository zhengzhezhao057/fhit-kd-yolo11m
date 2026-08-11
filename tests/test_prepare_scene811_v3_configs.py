from __future__ import annotations

import json
from pathlib import Path

import yaml

from src.prepare_scene811_v3_configs import generate_configs


FINGERPRINT = "1" * 64
SPLIT_FINGERPRINT = "2" * 64


def test_baseline_matrix_is_portable_and_has_two_matched_recipes(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "audit_d0.json").write_text(
        json.dumps(
            {
                "training_ready": True,
                "dataset_id": "scene811_v3_grouped_clean_r10",
                "dataset_fingerprint": FINGERPRINT,
            }
        ),
        encoding="utf-8",
    )
    (dataset / "dataset_fingerprint.json").write_text(
        json.dumps(
            {
                "dataset_id": "scene811_v3_grouped_clean_r10",
                "dataset_fingerprint": FINGERPRINT,
                "split_fingerprint": SPLIT_FINGERPRINT,
            }
        ),
        encoding="utf-8",
    )
    for name in ("dataset.yaml", "dataset_official.yaml"):
        (dataset / name).write_text("train: train.txt\nval: images/val\n", encoding="utf-8")
    (dataset / "split_manifest.csv").write_text("split,image\n", encoding="utf-8")
    baseline_template = tmp_path / "baseline.yaml"
    baseline_template.write_text("optimizer: AdamW\npatience: 0\n", encoding="utf-8")
    experiment_template = tmp_path / "experiment.yaml"
    experiment_template.write_text("dataset: {}\n", encoding="utf-8")
    output = tmp_path / "generated"

    result = generate_configs(
        dataset_root=dataset,
        output_dir=output,
        baseline_template=baseline_template,
        experiment_template=experiment_template,
        seeds=(42, 3407, 20260809),
    )

    assert len(result["baseline_trials"]) == 6
    assert {item["recipe"] for item in result["baseline_trials"]} == {"official", "mix"}
    assert {item["seed"] for item in result["baseline_trials"]} == {42, 3407, 20260809}
    generated = yaml.safe_load((output / "baseline_mix_seed42.yaml").read_text("utf-8"))
    assert generated["dataset_id"] == "scene811_v3_grouped_clean_r10"
    assert generated["patience"] == 0
    assert generated["dataset_fingerprint"] == FINGERPRINT
    assert generated["data"] == str((dataset / "dataset.yaml").resolve())


def test_config_generation_refuses_failed_d0(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "audit_d0.json").write_text(
        json.dumps(
            {
                "training_ready": False,
                "dataset_id": "scene811_v3_grouped_clean",
                "dataset_fingerprint": FINGERPRINT,
            }
        ),
        encoding="utf-8",
    )
    (dataset / "dataset_fingerprint.json").write_text(
        json.dumps(
            {
                "dataset_id": "scene811_v3_grouped_clean",
                "dataset_fingerprint": FINGERPRINT,
                "split_fingerprint": SPLIT_FINGERPRINT,
            }
        ),
        encoding="utf-8",
    )
    baseline = tmp_path / "baseline.yaml"
    baseline.write_text("{}\n", encoding="utf-8")
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text("{}\n", encoding="utf-8")
    try:
        generate_configs(
            dataset_root=dataset,
            output_dir=tmp_path / "out",
            baseline_template=baseline,
            experiment_template=experiment,
        )
    except RuntimeError as error:
        assert "D0 did not pass" in str(error)
    else:
        raise AssertionError("failed D0 must block config generation")
