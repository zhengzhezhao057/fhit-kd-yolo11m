from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from .artifact_paths import manifests_dir, run_dir
from .common import json_dump, load_config
from .dataset_d0 import file_sha256
from .oof_training import classify_run, validate_initial_checkpoint_name

MANIFEST_FORMAT = 1


@dataclass(frozen=True)
class BaselinePlan:
    action: str
    completed_epochs: int
    checkpoint: Path | None
    manifest: dict


def load_fingerprint_summary() -> dict:
    path = manifests_dir() / "dataset_fingerprint.json"
    if not path.is_file():
        raise FileNotFoundError(f"Scene811 fingerprint summary missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def plan_baseline(config: dict, output_dir: Path, *, seed: int | None = None, initial_checkpoint: str = "yolo11m.pt") -> BaselinePlan:
    validate_initial_checkpoint_name(initial_checkpoint)
    seed = int(seed if seed is not None else config.get("seed", 0))
    epochs = int(config.get("epochs", 50))
    state = classify_run(output_dir, epochs)
    fingerprint = load_fingerprint_summary()
    detector_path = Path(initial_checkpoint)
    manifest = {
        "format": MANIFEST_FORMAT,
        "kind": "scene811_baseline_run",
        "dataset_id": "scene811_v1",
        "dataset_fingerprint": fingerprint["dataset_fingerprint"],
        "class_mapping_fingerprint": fingerprint["class_mapping_fingerprint"],
        "split_fingerprint": fingerprint["split_fingerprint"],
        "replay_manifest_sha256": None,
        "initial_detector_sha256": file_sha256(detector_path) if detector_path.is_file() else "unavailable",
        "seed": seed,
        "epochs": epochs,
        "experiment": output_dir.name,
        "config_sha256": file_sha256(Path(config["__config_path__"])) if config.get("__config_path__") else None,
    }
    return BaselinePlan(state.action, state.completed_epochs, state.checkpoint, manifest)


def write_run_manifest(manifest: dict, output_dir: Path) -> Path:
    target = output_dir / "run_manifest.json"
    json_dump(manifest, target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Scene811 baseline from the official yolo11m.pt with full provenance.")
    parser.add_argument("--config", default="configs/scene811_baselines/baseline_seed0.yaml")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpoint", default="yolo11m.pt")
    parser.add_argument("--execute", action="store_true", help="Run Ultralytics training; otherwise only plan.")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    config["__config_path__"] = str(config_path)
    seed = int(args.seed if args.seed is not None else config.get("seed", 0))
    experiment = args.run_name or f"baseline_yolo11m_seed{seed}"
    output_dir = run_dir(experiment)
    plan = plan_baseline(config, output_dir, seed=seed, initial_checkpoint=args.checkpoint)
    write_run_manifest(plan.manifest, output_dir)
    print(f"action={plan.action} completed_epochs={plan.completed_epochs} output={output_dir}")
    if plan.action == "skip":
        return
    if plan.action == "error":
        raise RuntimeError(f"Run directory exists without a usable checkpoint: {output_dir}")
    if not args.execute:
        print("dry-run; pass --execute to start training")
        return
    from ultralytics import YOLO
    model = YOLO(args.checkpoint)
    kwargs = dict(
        data=config["data"], epochs=config.get("epochs", 50), imgsz=config.get("imgsz", 640),
        batch=config.get("batch", 16), workers=config.get("workers", 4),
        optimizer=config.get("optimizer", "AdamW"), seed=seed,
        save_period=config.get("save_period", 1), device=config.get("device", 0),
        project=str(output_dir.parent), name=experiment, exist_ok=True,
    )
    if plan.action == "resume" and plan.checkpoint is not None:
        kwargs["resume"] = str(plan.checkpoint)
    model.train(**kwargs)


if __name__ == "__main__":
    main()
