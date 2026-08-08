from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import load_config
from .distill_trainer import (
    DistillationDetectionTrainer,
    ReplayDetectionTrainer,
    kd_objective_fingerprint,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run C0/F/K/FK fair continuation experiments from the fixed YOLO11m baseline.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--exp", required=True, choices=("c0", "f", "k", "fk", "e0", "e1", "e2", "e3"), help="e0/e1/e2/e3 are backward-compatible aliases for c0/f/k/fk.")
    parser.add_argument("--cache", default=None, help="Teacher train cache. Required for f/k/fk.")
    parser.add_argument("--resume", nargs="?", const="auto", default=None, help="Resume the full Ultralytics last.pt checkpoint. Omit a value to use runs/<exp>/weights/last.pt.")
    parser.add_argument("--run-name", default=None, help="Exact directory under runs/. Defaults to c0/f/k/fk; never auto-increments names.")
    parser.add_argument("--epochs", type=int, default=None, help="Override student epochs, useful for a 1-epoch KD health check.")
    parser.add_argument("--health-batches", type=int, default=None, help="KD smoke test: stop cleanly after this many batches (must be >= health patience).")
    args = parser.parse_args()
    aliases = {"e0": "c0", "e1": "f", "e2": "k", "e3": "fk"}
    experiment = aliases.get(args.exp, args.exp)
    cfg = load_config(args.config); cfg["runtime"] = {"experiment": experiment, "health_batches": args.health_batches}
    from ultralytics import YOLO
    root = Path(cfg["paths"]["project_root"]); student_cfg = cfg["student"]
    output = root / "runs"; run_name = args.run_name or experiment
    if Path(run_name).name != run_name:
        raise ValueError("--run-name must be a single directory name, not a path.")
    run_dir = output / run_name
    resume_path = run_dir / "weights" / "last.pt" if args.resume == "auto" else (Path(args.resume) if args.resume else None)
    cfg["runtime"]["fresh_start"] = resume_path is None
    cfg["runtime"]["reset_kd_calibration"] = bool(
        resume_path is None and cfg["distillation"].get("reset_kd_calibration_on_start", False)
    )
    if resume_path is not None:
        if not resume_path.exists():
            raise FileNotFoundError(f"Student resume checkpoint not found: {resume_path}")
        model = YOLO(resume_path)
        if not model.ckpt or model.ckpt.get("optimizer") is None or model.ckpt.get("epoch", -1) < 0:
            raise RuntimeError(f"{resume_path} is not a full resumable Ultralytics checkpoint (optimizer/epoch missing).")
        print(f"Resuming {experiment} exactly from {resume_path}; optimizer, EMA, scaler, scheduler and epoch state come from last.pt.")
    else:
        if run_dir.exists():
            raise FileExistsError(f"{run_dir} already exists. Choose a new explicit --run-name or use --resume; automatic '-2' directories are forbidden.")
        model = YOLO(cfg["paths"]["baseline_weights"])
    if experiment != "c0":
        cache_dir = Path(args.cache) if args.cache else root / "cache" / "teacher_signals" / "train"
        if not cache_dir.exists():
            raise FileNotFoundError(f"Teacher cache missing: {cache_dir}. Run cache_teacher_signals.py first.")
        manifest = cache_dir / "manifest.json"
        if not manifest.exists():
            raise FileNotFoundError(f"Teacher cache manifest missing: {manifest}. Refuse to train from an unaudited cache.")
        try:
            manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Invalid teacher cache manifest {manifest}: {error}") from error
        expected_manifest = {
            "format": 2,
            "split": "train",
            "image_size": int(cfg["dataset"]["image_size"]),
            "feature_channels": int(cfg["teacher"]["feature_channels"]),
            "num_classes": int(cfg["dataset"]["nc"]),
        }
        mismatches = {key: (manifest_data.get(key), value) for key, value in expected_manifest.items() if manifest_data.get(key) != value}
        if mismatches:
            raise RuntimeError(f"Teacher cache manifest is incompatible with this experiment: {mismatches}")
        cfg["runtime"]["cache_manifest"] = manifest_data
        if resume_path is not None:
            if not hasattr(model.model, "distill_addons"):
                raise RuntimeError(
                    f"{resume_path} has no distill_addons and came from the old silently-disabled KD path. "
                    "It cannot be resumed; start a new F/K/FK run with a new --run-name."
                )
            saved_experiment = getattr(model.model, "kd_experiment", None)
            if saved_experiment != experiment:
                raise RuntimeError(f"Checkpoint experiment is {saved_experiment!r}, not requested {experiment!r}; cross-group resume is forbidden.")
            if getattr(model.model, "kd_cache_manifest", None) != manifest_data:
                raise RuntimeError("Checkpoint teacher-cache manifest differs from the selected cache; exact KD resume is forbidden.")
            expected_fingerprint = kd_objective_fingerprint(cfg)
            saved_fingerprint = getattr(model.model, "kd_objective_fingerprint", None)
            targeted_v4 = bool(
                cfg["distillation"].get("group_distill_weights")
                or cfg["distillation"].get("hard_example_manifest")
                or cfg["distillation"].get("vehicle_negative_manifest")
            )
            if saved_fingerprint is None and targeted_v4:
                raise RuntimeError(
                    "Checkpoint predates the v4 KD objective fingerprint. Exact targeted-KD resume cannot be proven; "
                    "preserve this run and start a new explicit --run-name."
                )
            if saved_fingerprint is not None and saved_fingerprint != expected_fingerprint:
                raise RuntimeError("Checkpoint KD objective fingerprint differs from the selected config; exact resume is forbidden.")
            if not bool(getattr(model.model, "kd_calibration_buffers_exact", False)):
                # Passing resume=<path> makes Ultralytics reload the checkpoint
                # from disk and discard mutations made to this outer YOLO
                # instance. Defer legacy-state reconstruction until the custom
                # trainer has rebuilt its real model in get_model().
                cfg["runtime"]["legacy_calibration_recovery"] = {
                    "health_file": str(run_dir / "kd_health.jsonl"),
                    "checkpoint_epoch": int(model.ckpt.get("epoch", -1)),
                }
                print(
                    "Scheduled pre-fix KD calibration recovery inside the real trainer model "
                    f"from {run_dir / 'kd_health.jsonl'}."
                )
        DistillationDetectionTrainer.configure(cfg, cache_dir)
        patience = int(cfg["distillation"].get("health_patience_batches", 10))
        if args.health_batches is not None and args.health_batches < patience:
            raise ValueError(f"--health-batches must be >= health_patience_batches ({patience}).")
        if bool(cfg["distillation"].get("vehicle_bg_enabled", False)) and args.health_batches is not None:
            vehicle_patience = int(cfg["distillation"].get("vehicle_bg_health_patience_batches", 128))
            if args.health_batches < vehicle_patience:
                raise ValueError(
                    f"vehicle-background health check requires --health-batches >= {vehicle_patience} "
                    "because replay negatives are intentionally sparse."
                )
    elif args.health_batches is not None:
        raise ValueError("--health-batches is only meaningful for F/K/FK KD runs.")
    replay_control = experiment == "c0" and bool(cfg["distillation"].get("hard_image_replay_manifest"))
    if replay_control:
        # Fail before Ultralytics starts if the manifest is missing or unsafe.
        from .distill_trainer import load_hard_replay_images
        if not load_hard_replay_images(cfg):
            raise RuntimeError("C0 hard replay was requested but selected zero images.")
        ReplayDetectionTrainer.configure(cfg)
    # The cache was made from deterministic 640 letterbox images. Every image-changing augmentation must be off
    # in *all* four continuation groups, including C0, or the KD comparisons are not fair/aligned.
    deterministic_kd_args = dict(
        mosaic=0.0, mixup=0.0, cutmix=0.0, copy_paste=0.0, close_mosaic=0,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, bgr=0.0,
        fliplr=0.0, flipud=0.0, degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
        auto_augment=None, erasing=0.0, multi_scale=0.0,
    )
    train_kwargs = dict(
        data=cfg["paths"]["data_yaml"], epochs=args.epochs or student_cfg["epochs"], imgsz=cfg["dataset"]["image_size"], batch=student_cfg["batch"],
        device=0, workers=student_cfg["workers"], optimizer=student_cfg["optimizer"], lr0=student_cfg["lr0"], lrf=student_cfg["lrf"],
        weight_decay=student_cfg["weight_decay"], warmup_epochs=student_cfg["warmup_epochs"], cos_lr=True, seed=student_cfg["seed"],
        project=str(output), name=run_name, exist_ok=True, resume=False, amp=True, plots=True,
        # KD gradient calibration intentionally performs retained autograd.grad
        # calls before the optimizer backward. Whole-model torch.compile is not
        # compatible with that lifecycle and provides no benefit in 10-batch
        # health checks, so keep every ablation group eager and comparable.
        compile=False,
        save_period=int(student_cfg.get("save_period", -1)), **deterministic_kd_args,
    )
    if "warmup_bias_lr" in student_cfg:
        train_kwargs["warmup_bias_lr"] = float(student_cfg["warmup_bias_lr"])
    if resume_path is not None:
        # Let Ultralytics consume the checkpoint's saved args and restore optimizer/EMA/scaler/scheduler exactly.
        if experiment == "c0":
            if replay_control:
                model.train(trainer=ReplayDetectionTrainer, resume=str(resume_path), compile=False)
            else:
                model.train(resume=str(resume_path), compile=False)
        else:
            model.train(trainer=DistillationDetectionTrainer, resume=str(resume_path), compile=False)
    else:
        if experiment == "c0":
            if replay_control:
                model.train(trainer=ReplayDetectionTrainer, **train_kwargs)
            else:
                model.train(**train_kwargs)
        else:
            model.train(trainer=DistillationDetectionTrainer, **train_kwargs)


if __name__ == "__main__":
    main()
