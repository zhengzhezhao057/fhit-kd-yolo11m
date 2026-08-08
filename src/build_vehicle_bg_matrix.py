from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

from .build_direction1_matrix import resumable_screen_prelude
from .build_weak_group_matrix import targeted_updates
from .common import load_config


def vehicle_bg_trials(hard_manifest: str, negative_manifest: str) -> list[dict]:
    common = targeted_updates(hard_manifest)
    common.update({
        "vehicle_negative_manifest": negative_manifest,
        "vehicle_negative_replay_repeats": 3,
        "vehicle_bg_health_patience_batches": 128,
        "vehicle_bg_max_positive_rois": 64,
        "vehicle_bg_max_negatives_per_image": 4,
        "vehicle_bg_score_power": 1.0,
        "vehicle_bg_weight_bounds": [0.000001, 10.0],
        "vehicle_bg_schedule": {"start_epoch": 1, "warmup_epochs": 1, "hold_epochs": 1, "decay_epochs": 3},
    })
    return [
        {
            "name": "v41_replay_fk",
            "experiment": "fk",
            "updates": {**common, "vehicle_bg_enabled": False, "cls_gradient_ratio": 0.005},
        },
        {
            "name": "v41_k003_bg002",
            "experiment": "fk",
            "updates": {
                **common, "vehicle_bg_enabled": True,
                "cls_gradient_ratio": 0.003, "vehicle_bg_gradient_ratio": 0.002,
            },
        },
        {
            "name": "v41_k001_bg004",
            "experiment": "fk",
            "updates": {
                **common, "vehicle_bg_enabled": True,
                "cls_gradient_ratio": 0.001, "vehicle_bg_gradient_ratio": 0.004,
            },
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the bounded v4.1 vehicle hard-negative matrix.")
    parser.add_argument("--base", default="configs/direction1.yaml")
    parser.add_argument("--hard-manifest", default="reports/weak_group_train/hard_examples.json")
    parser.add_argument("--negative-manifest", default="reports/weak_group_train_c035/vehicle_background_negatives.json")
    parser.add_argument("--out", default="configs/vehicle_bg_v41")
    args = parser.parse_args()
    base_path = Path(args.base).resolve()
    base = load_config(base_path)
    root = Path(base["paths"]["project_root"])

    def resolved(path_value: str) -> Path:
        path = Path(path_value)
        return path if path.is_absolute() else root / path

    hard_manifest = resolved(args.hard_manifest)
    negative_manifest = resolved(args.negative_manifest)
    for path in (hard_manifest, negative_manifest):
        if not path.exists():
            raise FileNotFoundError(f"Required TRAIN manifest missing: {path}")
    output = resolved(args.out)
    output.mkdir(parents=True, exist_ok=True)
    trials = vehicle_bg_trials(str(hard_manifest), str(negative_manifest))
    commands = resumable_screen_prelude(8)
    evaluation = ["set -euo pipefail", 'PYTHON_BIN="${PYTHON_BIN:-python}"', "mkdir -p reports/vehicle_bg_v41", ""]
    manifest_rows = []
    native_arguments = []
    base_relative = base_path.relative_to(root).as_posix()
    for trial in trials:
        config = copy.deepcopy(base)
        config["student"]["epochs"] = 8
        config["distillation"].update(trial["updates"])
        config_path = output / f"{trial['name']}.yaml"
        config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        relative = config_path.relative_to(root).as_posix()
        commands.append(f"run_trial {trial['name']} {trial['experiment']} {relative}")
        weight = f"runs/{trial['name']}/weights/best_deploy.pt"
        native_arguments.append(f"--model {trial['name']}={weight}")
        evaluation.append(
            f'"$PYTHON_BIN" -m src.competition_eval --config {base_relative} --model {weight} '
            f'--split val --class-aware --out reports/vehicle_bg_v41/{trial["name"]}_competition.json'
        )
        manifest_rows.append({**trial, "config": relative})
    evaluation.extend([
        "",
        '"$PYTHON_BIN" -m src.validate_models ' + " ".join(native_arguments)
        + f" --config {base_relative} --out reports/vehicle_bg_v41/native.json",
    ])
    (output / "matrix.json").write_text(json.dumps(manifest_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "run.sh").write_text("\n".join(commands) + "\n", encoding="utf-8", newline="\n")
    (output / "evaluate.sh").write_text("\n".join(evaluation) + "\n", encoding="utf-8", newline="\n")
    print(f"Generated {len(trials)} v4.1 trials in {output}")
    print("K-family budget stays fixed at 0.5%: replay=0.5+0.0, bg02=0.3+0.2, bg04=0.1+0.4.")


if __name__ == "__main__":
    main()
