from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import yaml

from .build_direction1_matrix import resumable_screen_prelude
from .build_weak_group_matrix import targeted_updates
from .common import load_config


def detector_state_fingerprint(checkpoint: Path) -> str:
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = payload.get("model")
    if model is None:
        raise RuntimeError(f"Checkpoint has no model: {checkpoint}")
    digest = hashlib.sha256()
    detector_items = [(name, value) for name, value in model.state_dict().items() if not name.startswith("distill_addons.")]
    if not detector_items:
        raise RuntimeError(f"Checkpoint has no detector tensors: {checkpoint}")
    for name, value in sorted(detector_items):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8")); digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii")); digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def trial_definitions(hard: str, negative: str, replay: str) -> list[dict]:
    schedules = {
        "feature_schedule": {"start_epoch": 0, "warmup_epochs": 1, "hold_epochs": 1, "decay_epochs": 2},
        "cls_schedule": {"start_epoch": 0, "warmup_epochs": 1, "hold_epochs": 1, "decay_epochs": 2},
        "vehicle_bg_schedule": {"start_epoch": 0, "warmup_epochs": 1, "hold_epochs": 1, "decay_epochs": 2},
    }
    replay_common = {"hard_image_replay_manifest": replay, "hard_image_replay_repeats": 1}
    kd_common = {
        **targeted_updates(hard), **replay_common, **schedules,
        # V4's stored gradient calibration belongs to a different object
        # distribution. Fresh V5 trials recalibrate; exact last.pt resumes do not.
        "reset_kd_calibration_on_start": True,
    }
    bg = {
        "vehicle_negative_manifest": negative,
        "vehicle_bg_enabled": True,
        "vehicle_bg_gradient_ratio": 0.001,
        "vehicle_bg_health_patience_batches": 128,
        "vehicle_bg_max_positive_rois": 64,
        "vehicle_bg_max_negatives_per_image": 4,
        "vehicle_bg_score_power": 1.0,
        "vehicle_bg_weight_bounds": [0.000001, 10.0],
    }
    return [
        {"name": "v5_oof_replay_c0", "experiment": "c0", "source": "deploy", "updates": replay_common},
        {"name": "v5_oof_replay_fk", "experiment": "fk", "source": "distill", "updates": kd_common},
        {"name": "v5_oof_replay_fk_bg", "experiment": "fk", "source": "distill", "updates": {**kd_common, **bg}},
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the bounded OOF hard-positive/background v5 short screen.")
    parser.add_argument("--base", default="configs/direction1.yaml")
    parser.add_argument("--hard-manifest", default="reports/dataset_d0/oof_mining_v1/hard_examples_oof.json")
    parser.add_argument("--negative-manifest", default="reports/dataset_d0/oof_mining_v1/vehicle_background_annotation_policy.json")
    parser.add_argument("--replay-manifest", default="reports/dataset_d0/oof_training_v5/hard_replay_images.json")
    parser.add_argument("--distill-checkpoint", default="runs/v4_eh_fk/weights/best.pt")
    parser.add_argument("--deploy-checkpoint", default="runs/v4_eh_fk/weights/best_deploy.pt")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--out", default="configs/oof_targeted_v5")
    args = parser.parse_args()
    base_path = Path(args.base).resolve(); base = load_config(base_path)
    root = Path(base["paths"]["project_root"]).resolve()
    def resolve(value: str) -> Path:
        path = Path(value); return path if path.is_absolute() else root / path
    hard, negative, replay = map(resolve, (args.hard_manifest, args.negative_manifest, args.replay_manifest))
    distill_checkpoint, deploy_checkpoint = map(resolve, (args.distill_checkpoint, args.deploy_checkpoint))
    for path in (hard, negative, replay, distill_checkpoint, deploy_checkpoint):
        if not path.exists(): raise FileNotFoundError(path)
    distill_hash = detector_state_fingerprint(distill_checkpoint)
    deploy_hash = detector_state_fingerprint(deploy_checkpoint)
    if distill_hash != deploy_hash:
        raise RuntimeError(
            "V5 control and KD initialization detector tensors differ; fair comparison is forbidden. "
            f"distill={distill_hash}, deploy={deploy_hash}"
        )
    output = resolve(args.out); output.mkdir(parents=True, exist_ok=True)
    trials = trial_definitions(str(hard), str(negative), str(replay))
    commands = resumable_screen_prelude(args.epochs)
    evaluation = ["set -euo pipefail", 'PYTHON_BIN="${PYTHON_BIN:-python}"', "mkdir -p reports/oof_targeted_v5", ""]
    models: list[str] = []
    rows = []
    base_relative = base_path.relative_to(root).as_posix()
    for trial in trials:
        cfg = copy.deepcopy(base)
        cfg["student"]["epochs"] = args.epochs
        cfg["student"]["save_period"] = 1
        cfg["paths"]["baseline_weights"] = str(deploy_checkpoint if trial["source"] == "deploy" else distill_checkpoint)
        cfg["distillation"].update(trial["updates"])
        path = output / f"{trial['name']}.yaml"
        path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
        relative = path.relative_to(root).as_posix()
        commands.append(f"run_trial {trial['name']} {trial['experiment']} {relative}")
        best_name = "best.pt" if trial["experiment"] == "c0" else "best_deploy.pt"
        last_name = "last.pt" if trial["experiment"] == "c0" else "last_deploy.pt"
        for selection, weight_name in (("best", best_name), ("last", last_name)):
            label = f"{trial['name']}_{selection}"
            weight = f"runs/{trial['name']}/weights/{weight_name}"
            models.append(f"--model {label}={weight}")
            evaluation.append(
                f'"$PYTHON_BIN" -m src.competition_eval --config {base_relative} --model {weight} '
                f'--split val --class-aware --out reports/oof_targeted_v5/{label}_competition.json'
            )
        rows.append({**trial, "config": relative})
    evaluation.extend(["", '"$PYTHON_BIN" -m src.validate_models ' + " ".join(models) + f" --config {base_relative} --out reports/oof_targeted_v5/native.json"])
    health = [
        "set -euo pipefail", 'PYTHON_BIN="${PYTHON_BIN:-python}"', 'CACHE_DIR="${CACHE_DIR:-cache/teacher_signals/train}"',
        f'"$PYTHON_BIN" -m src.train_ablation --config {rows[2]["config"]} --exp fk --cache "$CACHE_DIR" --run-name health_v5_oof_fk_bg --epochs 1 --health-batches 128',
    ]
    (output / "matrix.json").write_text(json.dumps({
        "detector_initialization_sha256": distill_hash, "epochs": args.epochs, "trials": rows,
        "selection_policy": "evaluate both best and fixed final epoch; auxiliary branches are active from epoch 0",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "health.sh").write_text("\n".join(health) + "\n", encoding="utf-8", newline="\n")
    (output / "run.sh").write_text("\n".join(commands) + "\n", encoding="utf-8", newline="\n")
    (output / "evaluate.sh").write_text("\n".join(evaluation) + "\n", encoding="utf-8", newline="\n")
    print(f"OOF V5 MATRIX: trials={len(trials)} epochs={args.epochs} detector_sha256={distill_hash} output={output}")
    print("Run health.sh first. Every branch is active from epoch 0; evaluate both best and fixed last checkpoints.")


if __name__ == "__main__":
    main()
