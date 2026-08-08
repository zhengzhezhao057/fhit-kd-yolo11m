from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.oof_training import (
    classify_run,
    fold_summary,
    load_fold_manifest,
    objective_payload,
    validate_initial_checkpoint_name,
    verify_saved_provenance,
)


def make_fold_manifest(tmp_path: Path) -> tuple[Path, dict]:
    folds_dir = tmp_path / "oof3"
    folds_dir.mkdir()
    summaries = []
    for fold in range(3):
        train = folds_dir / f"fold{fold}_train.txt"
        val = folds_dir / f"fold{fold}_val.txt"
        yaml_path = folds_dir / f"fold{fold}.yaml"
        train.write_text(f"/data/product{fold + 1}.jpg\n", encoding="utf-8")
        val.write_text(f"/data/product{fold}.jpg\n", encoding="utf-8")
        yaml_path.write_text(f"path: /data\ntrain: {train}\nval: {val}\n", encoding="utf-8")
        summaries.append({
            "fold": fold, "train_images": 1, "val_images": 1,
            "train_list": str(train), "val_list": str(val), "data_yaml": str(yaml_path),
        })
    manifest = {
        "kind": "dataset_d0_product_grouped_oof", "read_only_source": True,
        "folds": 3, "seed": 7, "train_inventory_fingerprint": "abc", "fold_summaries": summaries,
    }
    (folds_dir / "folds.json").write_text(json.dumps(manifest), encoding="utf-8")
    return folds_dir, manifest


def test_run_state_is_start_resume_skip_or_error(tmp_path: Path) -> None:
    run = tmp_path / "run"
    assert classify_run(run, 3).action == "start"
    (run / "weights").mkdir(parents=True)
    assert classify_run(run, 3).action == "error"
    (run / "weights" / "last.pt").write_bytes(b"checkpoint")
    assert classify_run(run, 3).action == "resume"
    (run / "results.csv").write_text("epoch,metric\n0,0.1\n1,0.2\n2,0.3\n", encoding="utf-8")
    (run / "weights" / "best.pt").write_bytes(b"best")
    state = classify_run(run, 3)
    assert state.action == "skip" and state.completed_epochs == 3


def test_only_official_yolo11m_name_is_accepted() -> None:
    validate_initial_checkpoint_name("yolo11m.pt")
    validate_initial_checkpoint_name("/models/yolo11m.pt")
    with pytest.raises(ValueError, match="forbidden"):
        validate_initial_checkpoint_name("runs/v4_eh_fk/weights/best_deploy.pt")


def test_fold_manifest_and_resume_provenance_are_strict(tmp_path: Path) -> None:
    folds_dir, manifest = make_fold_manifest(tmp_path)
    loaded = load_fold_manifest(folds_dir)
    row = fold_summary(loaded, 0)
    objective = objective_payload(loaded, row, 0, 30, 16, 4, "0", "yolo11m.pt")
    provenance = tmp_path / "provenance.json"
    provenance.write_text(json.dumps(objective), encoding="utf-8")
    assert verify_saved_provenance(provenance, objective)["fold"] == 0
    changed = dict(objective, epochs=31)
    with pytest.raises(RuntimeError, match="differs"):
        verify_saved_provenance(provenance, changed)


def test_fold_list_leakage_is_rejected(tmp_path: Path) -> None:
    folds_dir, _ = make_fold_manifest(tmp_path)
    manifest = load_fold_manifest(folds_dir)
    row = manifest["fold_summaries"][0]
    Path(row["val_list"]).write_text(Path(row["train_list"]).read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(RuntimeError, match="leakage"):
        fold_summary(manifest, 0)
