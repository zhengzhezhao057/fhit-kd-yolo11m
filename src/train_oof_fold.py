from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import load_config
from .oof_training import (
    OOF_RUN_PREFIX,
    classify_run,
    fold_summary,
    load_fold_manifest,
    objective_payload,
    sha256_file,
    validate_initial_checkpoint_name,
    verify_saved_provenance,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one leakage-free product-grouped OOF YOLO11m detector.")
    parser.add_argument("--config", default="configs/direction1.yaml")
    parser.add_argument("--folds-dir", default="reports/dataset_d0/oof3")
    parser.add_argument("--fold", type=int, required=True, choices=(0, 1, 2))
    parser.add_argument("--initial-weights", default="yolo11m.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    validate_initial_checkpoint_name(args.initial_weights)
    cfg = load_config(args.config)
    root = Path(cfg["paths"]["project_root"]).resolve()
    folds_dir = Path(args.folds_dir)
    folds_dir = folds_dir if folds_dir.is_absolute() else root / folds_dir
    manifest = load_fold_manifest(folds_dir)
    row = fold_summary(manifest, args.fold)
    run_name = f"{OOF_RUN_PREFIX}{args.fold}"
    run_dir = root / "runs" / run_name
    provenance_path = run_dir / "oof_provenance.json"
    objective = objective_payload(
        manifest, row, args.fold, args.epochs, args.batch, args.workers, args.device, args.initial_weights
    )
    state = classify_run(run_dir, args.epochs)

    from ultralytics import YOLO

    if args.resume:
        if state.action != "resume" or state.checkpoint is None:
            raise RuntimeError(f"Cannot resume {run_name}: action={state.action}, reason={state.reason}")
        verify_saved_provenance(provenance_path, objective)
        model = YOLO(state.checkpoint)
        if not model.ckpt or model.ckpt.get("optimizer") is None or int(model.ckpt.get("epoch", -1)) < 0:
            raise RuntimeError(f"{state.checkpoint} is not a full optimizer/EMA/scaler resumable checkpoint.")
        print(f"OOF RESUME: fold={args.fold} completed_epochs={state.completed_epochs} checkpoint={state.checkpoint}")
        model.train(resume=str(state.checkpoint), compile=False)
        return

    if state.action != "start":
        raise RuntimeError(
            f"Refuse to start {run_name}: action={state.action}, completed={state.completed_epochs}, reason={state.reason}. "
            "Use the serial runner for automatic skip/resume."
        )
    model = YOLO(args.initial_weights)
    checkpoint_path = Path(model.ckpt_path).resolve()
    validate_initial_checkpoint_name(checkpoint_path)
    if len(model.names) != 80:
        raise RuntimeError(
            f"Expected the official 80-class COCO yolo11m.pt, found {len(model.names)} classes. "
            "A Dataset-D0-trained checkpoint must not initialize OOF models."
        )
    objective["initial_checkpoint_path"] = str(checkpoint_path)
    objective["initial_checkpoint_sha256"] = sha256_file(checkpoint_path)
    run_dir.mkdir(parents=True, exist_ok=False)
    provenance_path.write_text(json.dumps(objective, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"OOF START: fold={args.fold} train={row['train_images']} val={row['val_images']} "
        f"initial_sha256={objective['initial_checkpoint_sha256']}"
    )
    aug = objective["augmentation"]
    model.train(
        data=str(Path(row["data_yaml"]).resolve()), epochs=args.epochs, imgsz=640, batch=args.batch,
        device=args.device, workers=args.workers, optimizer="AdamW", lr0=0.001, lrf=0.01,
        weight_decay=0.0005, warmup_epochs=3.0, cos_lr=True, seed=int(manifest["seed"]) + args.fold,
        project=str(root / "runs"), name=run_name, exist_ok=True, resume=False, amp=True,
        plots=True, save_period=-1, compile=False, patience=args.epochs, deterministic=True, cache=False,
        mosaic=aug["mosaic"], close_mosaic=aug["close_mosaic"], mixup=aug["mixup"],
        cutmix=aug["cutmix"], copy_paste=aug["copy_paste"], hsv_h=aug["hsv_h"],
        hsv_s=aug["hsv_s"], hsv_v=aug["hsv_v"], degrees=aug["degrees"],
        translate=aug["translate"], scale=aug["scale"], shear=aug["shear"],
        perspective=aug["perspective"], fliplr=aug["fliplr"], flipud=aug["flipud"],
    )


if __name__ == "__main__":
    main()
