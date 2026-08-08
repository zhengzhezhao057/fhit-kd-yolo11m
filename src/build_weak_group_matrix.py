from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

from .build_direction1_matrix import resumable_screen_prelude
from .common import load_config


GROUP_WEIGHTS = {"ship": 1.28, "aircraft": 0.70, "vehicle": 1.75}
FEATURE_ERROR_WEIGHTS = {
    "detected": 0.80,
    "low_confidence": 1.40,
    "nms_suppressed": 1.10,
    "localization": 1.60,
    "no_candidate": 1.70,
    "wrong_group": 1.00,
}
CLS_ERROR_WEIGHTS = {
    "detected": 0.80,
    "low_confidence": 1.60,
    "nms_suppressed": 1.00,
    "localization": 0.90,
    "no_candidate": 0.80,
    "wrong_group": 1.30,
}


def targeted_updates(hard_manifest: str | None) -> dict:
    updates = {
        "temperature": 6.0,
        "feature_gradient_ratio": 0.03,
        "cls_gradient_ratio": 0.005,
        "group_distill_weights": GROUP_WEIGHTS,
    }
    if hard_manifest:
        updates.update({
            "hard_example_manifest": hard_manifest,
            "feature_error_weights": FEATURE_ERROR_WEIGHTS,
            "cls_error_weights": CLS_ERROR_WEIGHTS,
            "object_weight_bounds": [0.25, 4.0],
            "feature_crowded_multiplier": 1.15,
            "cls_crowded_multiplier": 1.10,
            "feature_edge_multiplier": 1.10,
            "cls_edge_multiplier": 1.05,
            "feature_small_multiplier": 1.20,
            "cls_small_multiplier": 1.10,
        })
    return updates


def short_trials(hard_manifest: str) -> list[dict]:
    return [
        {"name": "v4_gb_fk", "experiment": "fk", "updates": targeted_updates(None)},
        {"name": "v4_eh_fk", "experiment": "fk", "updates": targeted_updates(hard_manifest)},
        {"name": "v4_eh_f", "experiment": "f", "updates": targeted_updates(hard_manifest)},
    ]


def long_trials(hard_manifest: str) -> list[dict]:
    return [
        {"name": "v4l_c0", "experiment": "c0", "updates": {}},
        {"name": "v4l_f_r001", "experiment": "f", "updates": {"feature_gradient_ratio": 0.01}},
        {"name": "v4l_k_t6_r005", "experiment": "k", "updates": {"temperature": 6.0, "cls_gradient_ratio": 0.005}},
        {"name": "v4l_fk", "experiment": "fk", "updates": {"temperature": 6.0, "feature_gradient_ratio": 0.03, "cls_gradient_ratio": 0.005}},
        {"name": "v4l_wt_fk", "experiment": "fk", "updates": targeted_updates(hard_manifest)},
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate leakage-safe weak-group KD v4 experiments.")
    parser.add_argument("--base", default="configs/direction1.yaml")
    parser.add_argument("--hard-manifest", default="reports/weak_group_train/hard_examples.json")
    parser.add_argument("--phase", choices=("short", "long"), default="short")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    base_path = Path(args.base).resolve()
    base = load_config(base_path)
    root = Path(base["paths"]["project_root"])
    output = Path(args.out or f"configs/weak_group_v4_{args.phase}")
    if not output.is_absolute():
        output = root / output
    output.mkdir(parents=True, exist_ok=True)
    hard_manifest = Path(args.hard_manifest)
    if not hard_manifest.is_absolute():
        hard_manifest = root / hard_manifest
    if not hard_manifest.exists():
        raise FileNotFoundError(
            f"TRAIN hard-example manifest missing: {hard_manifest}. Run src.build_hard_example_manifest first."
        )

    trials = short_trials(str(hard_manifest)) if args.phase == "short" else long_trials(str(hard_manifest))
    target_epochs = 8 if args.phase == "short" else 20
    commands = resumable_screen_prelude(target_epochs)
    evaluation = ["set -euo pipefail", 'PYTHON_BIN="${PYTHON_BIN:-python}"', "mkdir -p reports/weak_group_v4", ""]
    manifest_rows = []
    native_arguments = []
    base_relative = base_path.relative_to(root).as_posix()
    for trial in trials:
        config = copy.deepcopy(base)
        config["student"]["epochs"] = target_epochs
        if args.phase == "long":
            config["distillation"]["feature_schedule"] = {"start_epoch": 0, "warmup_epochs": 1, "hold_epochs": 5, "decay_epochs": 10}
            config["distillation"]["cls_schedule"] = {"start_epoch": 1, "warmup_epochs": 1, "hold_epochs": 4, "decay_epochs": 10}
        config["distillation"].update(trial["updates"])
        config_path = output / f"{trial['name']}.yaml"
        config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        relative = config_path.relative_to(root).as_posix()
        commands.append(f"run_trial {trial['name']} {trial['experiment']} {relative}")
        weight_name = "best.pt" if trial["experiment"] == "c0" else "best_deploy.pt"
        weight = f"runs/{trial['name']}/weights/{weight_name}"
        native_arguments.append(f"--model {trial['name']}={weight}")
        evaluation.append(
            f'"$PYTHON_BIN" -m src.competition_eval --config {base_relative} --model {weight} '
            f'--split val --class-aware --out reports/weak_group_v4/{trial["name"]}_competition.json'
        )
        manifest_rows.append({**trial, "config": relative})
    evaluation.extend([
        "",
        '"$PYTHON_BIN" -m src.validate_models ' + " ".join(native_arguments) + f" --config {base_relative} --out reports/weak_group_v4/{args.phase}_native.json",
    ])
    (output / "matrix.json").write_text(json.dumps(manifest_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "run.sh").write_text("\n".join(commands) + "\n", encoding="utf-8", newline="\n")
    (output / "evaluate.sh").write_text("\n".join(evaluation) + "\n", encoding="utf-8", newline="\n")
    print(f"Generated {len(trials)} {args.phase} trials in {output}")
    print(f"Review {output / 'matrix.json'} before running {output / 'run.sh'}")


if __name__ == "__main__":
    main()
