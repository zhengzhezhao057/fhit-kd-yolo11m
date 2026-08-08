from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or exactly resume the source-aware YOLO11m baseline.")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--checkpoint", default="yolo11m.pt")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if Path(args.run_name).name != args.run_name:
        parser.error("--run-name must be one safe directory name")
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parents[1]
    dataset_id = str(config.get("dataset_id", "scene811_v2"))
    run_dir = root / "runs" / dataset_id / args.run_name
    last = run_dir / "weights" / "last.pt"
    target_epochs = int(args.epochs or config.get("epochs", 120))
    from ultralytics import YOLO
    if args.resume:
        if not last.is_file():
            raise FileNotFoundError(f"Exact resume requires {last}")
        model = YOLO(last)
        if not model.ckpt or model.ckpt.get("optimizer") is None:
            raise RuntimeError(f"Checkpoint is stripped and cannot exactly resume: {last}")
        print(f"BASELINE RESUME: {last}")
        model.train(resume=str(last), epochs=target_epochs)
        return
    if run_dir.exists():
        raise FileExistsError(f"{run_dir} already exists; use --resume or choose a new run name.")
    checkpoint = Path(args.checkpoint)
    model = YOLO(str(checkpoint))
    seed = int(args.seed if args.seed is not None else config.get("seed", 0))
    kwargs = {
        "data": config["data"],
        "epochs": target_epochs,
        "imgsz": int(config.get("imgsz", 640)),
        "batch": int(config.get("batch", 16)),
        "workers": int(config.get("workers", 4)),
        "device": config.get("device", 0),
        "optimizer": config.get("optimizer", "AdamW"),
        "lr0": float(config.get("lr0", 0.001)),
        "lrf": float(config.get("lrf", 0.01)),
        "weight_decay": float(config.get("weight_decay", 0.0005)),
        "warmup_epochs": float(config.get("warmup_epochs", 3.0)),
        "patience": int(config.get("patience", 30)),
        "save_period": int(config.get("save_period", -1)),
        "seed": seed,
        "project": str(root / "runs" / dataset_id),
        "name": args.run_name,
        "exist_ok": True,
        "plots": True,
    }
    manifest = {
        "format": 1,
        "kind": "source_aware_baseline",
        "dataset_id": dataset_id,
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint) if checkpoint.is_file() else "ultralytics-managed",
        "seed": seed,
        "target_epochs": target_epochs,
    }
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    model.train(**kwargs)


if __name__ == "__main__":
    main()
