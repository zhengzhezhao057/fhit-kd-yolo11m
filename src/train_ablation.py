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
from .provenance import (
    artifact_namespace,
    portable_dataset_identity,
    prepare_run_lineage,
    resolve_dataset_identity,
    student_runs_root,
    teacher_cache_dir,
    validate_cache_manifest,
    verify_cache_sample,
    file_sha256,
)
from .prototype_bank import PrototypeBank


FHIT_V2_EXPERIMENTS = {"g", "p", "gp"}


def require_fhit_cache_manifest(
    manifest: dict, experiment: str, *, prototype_dim: int = 512
) -> None:
    """Reject legacy caches before a G/P/GP run can silently degrade."""

    if experiment not in FHIT_V2_EXPERIMENTS:
        return
    if manifest.get("format") != 3:
        raise RuntimeError(
            f"{experiment.upper()} requires cache format=3 provenance; rebuild the V3 teacher cache."
        )
    if experiment in {"p", "gp"} and int(manifest.get("roi_embedding_dim", 0)) != int(prototype_dim):
        raise RuntimeError(
            f"{experiment.upper()} requires {prototype_dim}-D cached penultimate roi_embeddings; "
            "the selected cache is old or incompatible."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run legacy C0/F/K/FK or FHiT-KD v2 G/P/GP continuation experiments.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--exp", required=True, choices=("c0", "f", "k", "fk", "g", "p", "gp", "e0", "e1", "e2", "e3"), help="e0/e1/e2/e3 are backward-compatible aliases for c0/f/k/fk.")
    parser.add_argument("--cache", default=None, help="Teacher train cache. Required for every KD experiment.")
    parser.add_argument("--prototype-bank", default=None, help="Validated V3 leave-one-scene-out bank. Required for p/gp; overrides distillation.prototype_bank.")
    parser.add_argument("--resume", nargs="?", const="auto", default=None, help="Resume the full Ultralytics last.pt checkpoint. V3 resolves it inside the dataset-fingerprint namespace.")
    parser.add_argument("--run-name", default=None, help="Exact directory under the dataset-namespaced runs root. Defaults to the experiment name; never auto-increments names.")
    parser.add_argument("--epochs", type=int, default=None, help="Override student epochs, useful for a 1-epoch KD health check.")
    parser.add_argument("--health-batches", type=int, default=None, help="KD smoke test: stop cleanly after this many batches (must be >= health patience).")
    args = parser.parse_args()
    aliases = {"e0": "c0", "e1": "f", "e2": "k", "e3": "fk"}
    experiment = aliases.get(args.exp, args.exp)
    cfg = load_config(args.config)
    identity = resolve_dataset_identity(cfg)
    cfg["runtime"] = {
        "experiment": experiment,
        "health_batches": args.health_batches,
        "dataset_identity": identity,
        "artifact_namespace": artifact_namespace(identity),
    }
    if experiment in FHIT_V2_EXPERIMENTS and not identity.get("strict"):
        raise RuntimeError(
            "G/P/GP require a fingerprint-enforced V3 dataset with split_manifest.scene_id."
        )
    if args.prototype_bank:
        cfg["distillation"]["prototype_bank"] = args.prototype_bank
    from ultralytics import YOLO
    root = Path(cfg["paths"]["project_root"]); student_cfg = cfg["student"]
    if experiment in {"p", "gp"} and not cfg["distillation"].get("prototype_bank"):
        cfg["distillation"]["prototype_bank"] = str(
            root
            / "cache"
            / "prototype_banks"
            / artifact_namespace(identity)
            / "leave_one_scene_out.pt"
        )
    output = student_runs_root(cfg, identity); run_name = args.run_name or experiment
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
        cache_dir = Path(args.cache) if args.cache else teacher_cache_dir(cfg, "train", identity)
        if not cache_dir.exists():
            raise FileNotFoundError(f"Teacher cache missing: {cache_dir}. Run cache_teacher_signals.py first.")
        manifest = cache_dir / "manifest.json"
        if not manifest.exists():
            raise FileNotFoundError(f"Teacher cache manifest missing: {manifest}. Refuse to train from an unaudited cache.")
        try:
            manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Invalid teacher cache manifest {manifest}: {error}") from error
        validate_cache_manifest(cfg, manifest_data, "train", identity)
        require_fhit_cache_manifest(
            manifest_data,
            experiment,
            prototype_dim=int(cfg["distillation"].get("prototype_embedding_dim", 512)),
        )
        provenance_checked = verify_cache_sample(
            cfg, cache_dir, manifest_data, samples=64
        )
        if identity.get("strict"):
            print(
                f"V3 cache provenance verified: namespace={artifact_namespace(identity)} "
                f"samples={provenance_checked}"
            )
        cfg["runtime"]["cache_manifest"] = manifest_data
        if experiment in {"p", "gp"}:
            bank_value = cfg["distillation"].get("prototype_bank")
            if not bank_value:
                raise RuntimeError(
                    "P/GP requires --prototype-bank, distillation.prototype_bank, or the "
                    "fingerprint-namespaced default bank. "
                    "Build it with: python -m src.prototype_bank build ..."
                )
            bank_path = Path(bank_value)
            if not bank_path.is_absolute():
                bank_path = root / bank_path
            bank_path = bank_path.resolve()
            if not bank_path.is_file():
                raise FileNotFoundError(f"Prototype bank missing: {bank_path}")
            bank = PrototypeBank.load(bank_path, cfg, manifest_data)
            cfg["distillation"]["prototype_bank"] = str(bank_path)
            cfg["runtime"]["prototype_bank"] = {
                "bank_fingerprint": bank.value["bank_fingerprint"],
                "sha256": file_sha256(bank_path),
                "embedding_dim": bank.embedding_dim,
                "min_count": bank.min_count,
            }
        if resume_path is not None:
            if not hasattr(model.model, "distill_addons"):
                raise RuntimeError(
                    f"{resume_path} has no distill_addons and came from the old silently-disabled KD path. "
                    "It cannot be resumed; start a new KD run with a new --run-name."
                )
            saved_experiment = getattr(model.model, "kd_experiment", None)
            if saved_experiment != experiment:
                raise RuntimeError(f"Checkpoint experiment is {saved_experiment!r}, not requested {experiment!r}; cross-group resume is forbidden.")
            if getattr(model.model, "kd_cache_manifest", None) != manifest_data:
                raise RuntimeError("Checkpoint teacher-cache manifest differs from the selected cache; exact KD resume is forbidden.")
            if getattr(model.model, "kd_prototype_bank", None) != cfg["runtime"].get("prototype_bank"):
                raise RuntimeError(
                    "Checkpoint prototype-bank identity differs from the selected bank; exact KD resume is forbidden."
                )
            if identity.get("strict"):
                saved_identity = getattr(model.model, "kd_dataset_identity", None)
                if saved_identity != portable_dataset_identity(identity):
                    raise RuntimeError(
                        "Checkpoint dataset fingerprint differs from V3 config; cross-dataset resume is forbidden."
                    )
                if getattr(model.model, "kd_artifact_namespace", None) != artifact_namespace(identity):
                    raise RuntimeError(
                        "Checkpoint artifact namespace differs from V3 config; exact resume is forbidden."
                    )
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
        raise ValueError("--health-batches is only meaningful for KD runs (F/K/FK/G/P/GP).")
    replay_control = experiment == "c0" and bool(cfg["distillation"].get("hard_image_replay_manifest"))
    if replay_control:
        # Fail before Ultralytics starts if the manifest is missing or unsafe.
        from .distill_trainer import load_hard_replay_images
        if not load_hard_replay_images(cfg):
            raise RuntimeError("C0 hard replay was requested but selected zero images.")
        ReplayDetectionTrainer.configure(cfg)
    prepare_run_lineage(
        cfg,
        run_dir,
        experiment=experiment,
        initial_checkpoint=Path(cfg["paths"]["baseline_weights"]),
        resume_checkpoint=resume_path,
        cache_manifest_data=cfg["runtime"].get("cache_manifest"),
        identity=identity,
    )
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
