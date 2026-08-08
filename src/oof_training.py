from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


OFFICIAL_INITIAL_CHECKPOINT = "yolo11m.pt"
OOF_RUN_PREFIX = "oof_d0_product_fold"


@dataclass(frozen=True)
class RunState:
    action: str
    completed_epochs: int
    checkpoint: Path | None = None
    reason: str = ""


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def completed_epochs(results_csv: Path) -> int:
    if not results_csv.exists():
        return 0
    with results_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        return sum(1 for row in csv.DictReader(stream) if any(str(value).strip() for value in row.values()))


def classify_run(run_dir: Path, target_epochs: int) -> RunState:
    rows = completed_epochs(run_dir / "results.csv")
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    if rows >= target_epochs and best.exists():
        return RunState("skip", rows, best, "target epochs and best.pt are present")
    if last.exists():
        return RunState("resume", rows, last, "an incomplete run has a resumable candidate")
    if not run_dir.exists():
        return RunState("start", 0, None, "run directory does not exist")
    return RunState(
        "error", rows, None,
        "run directory exists without a usable last.pt; preserve it and inspect the failure manually",
    )


def load_fold_manifest(folds_dir: Path, expected_folds: int = 3) -> dict[str, Any]:
    path = folds_dir / "folds.json"
    if not path.exists():
        raise FileNotFoundError(f"OOF manifest missing: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "kind": "dataset_d0_product_grouped_oof",
        "read_only_source": True,
        "folds": expected_folds,
    }
    mismatches = {key: (manifest.get(key), value) for key, value in expected.items() if manifest.get(key) != value}
    if mismatches:
        raise RuntimeError(f"Incompatible OOF fold manifest: {mismatches}")
    summaries = manifest.get("fold_summaries", [])
    if len(summaries) != expected_folds or {int(row["fold"]) for row in summaries} != set(range(expected_folds)):
        raise RuntimeError("OOF manifest does not contain exactly one summary for every fold.")
    return manifest


def fold_summary(manifest: dict[str, Any], fold: int) -> dict[str, Any]:
    matches = [row for row in manifest["fold_summaries"] if int(row["fold"]) == fold]
    if len(matches) != 1:
        raise RuntimeError(f"OOF fold {fold} is missing or duplicated in folds.json.")
    row = matches[0]
    for key in ("train_list", "val_list", "data_yaml"):
        if not Path(row[key]).exists():
            raise FileNotFoundError(f"OOF fold {fold} {key} is missing: {row[key]}")
    train_lines = {line for line in Path(row["train_list"]).read_text(encoding="utf-8").splitlines() if line}
    val_lines = {line for line in Path(row["val_list"]).read_text(encoding="utf-8").splitlines() if line}
    if not train_lines or not val_lines or train_lines.intersection(val_lines):
        raise RuntimeError(f"OOF fold {fold} has an empty list or direct train/val image leakage.")
    if len(train_lines) != int(row["train_images"]) or len(val_lines) != int(row["val_images"]):
        raise RuntimeError(f"OOF fold {fold} list sizes differ from folds.json.")
    return row


def validate_initial_checkpoint_name(value: str | Path) -> None:
    if Path(value).name.lower() != OFFICIAL_INITIAL_CHECKPOINT:
        raise ValueError(
            "OOF detectors must start from the clean official yolo11m.pt checkpoint. "
            "B0/V3/V4 checkpoints saw all Dataset-D0 and are forbidden here."
        )


def objective_payload(
    manifest: dict[str, Any], row: dict[str, Any], fold: int, epochs: int, batch: int,
    workers: int, device: str, initial_checkpoint: str | Path,
) -> dict[str, Any]:
    return {
        "format": 1,
        "kind": "dataset_d0_product_grouped_oof_detector",
        "fold": fold,
        "fold_seed": int(manifest["seed"]),
        "train_inventory_fingerprint": manifest["train_inventory_fingerprint"],
        "train_images": int(row["train_images"]),
        "val_images": int(row["val_images"]),
        "data_yaml": str(Path(row["data_yaml"]).resolve()),
        "data_yaml_sha256": sha256_file(Path(row["data_yaml"])),
        "initial_checkpoint_name": Path(initial_checkpoint).name,
        "epochs": int(epochs),
        "imgsz": 640,
        "batch": int(batch),
        "workers": int(workers),
        "device": str(device),
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.01,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "cos_lr": True,
        "seed": int(manifest["seed"]) + fold,
        "augmentation": {
            "mosaic": 1.0, "close_mosaic": 10, "mixup": 0.0, "cutmix": 0.0,
            "copy_paste": 0.0, "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
            "degrees": 0.0, "translate": 0.1, "scale": 0.5, "shear": 0.0,
            "perspective": 0.0, "fliplr": 0.5, "flipud": 0.5,
        },
    }


def verify_saved_provenance(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"OOF provenance missing: {path}; exact resume is forbidden.")
    saved = json.loads(path.read_text(encoding="utf-8"))
    mismatches = {key: (saved.get(key), value) for key, value in expected.items() if saved.get(key) != value}
    if mismatches:
        raise RuntimeError(f"OOF resume objective differs from the checkpoint run: {mismatches}")
    return saved
