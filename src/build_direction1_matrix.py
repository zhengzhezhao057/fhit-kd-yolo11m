from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

from .common import load_config


def trial_matrix(base: dict) -> list[dict]:
    return [
        {"name": "d1_c0", "experiment": "c0", "updates": {}},
        {"name": "d1_f_r001", "experiment": "f", "updates": {"feature_gradient_ratio": 0.01}},
        {"name": "d1_f_r003", "experiment": "f", "updates": {"feature_gradient_ratio": 0.03}},
        {"name": "d1_f_r005", "experiment": "f", "updates": {"feature_gradient_ratio": 0.05}},
        # Native teacher confidence is saturated (train p10=0.9999), so 0.7
        # versus 0.9 is not an informative experiment.  T4/T6 at one shared
        # gradient ratio isolates temperature; the two T6 runs then isolate
        # classification-KD strength.
        {"name": "d1_k_t4_r005", "experiment": "k", "updates": {"temperature": 4.0, "teacher_confidence_floor": 0.7, "cls_gradient_ratio": 0.005}},
        {"name": "d1_k_t6_r005", "experiment": "k", "updates": {"temperature": 6.0, "teacher_confidence_floor": 0.7, "cls_gradient_ratio": 0.005}},
        {"name": "d1_k_t6_r01", "experiment": "k", "updates": {"temperature": 6.0, "teacher_confidence_floor": 0.7, "cls_gradient_ratio": 0.01}},
        {"name": "d1_fk_t6_r03_r005", "experiment": "fk", "updates": {"temperature": 6.0, "feature_gradient_ratio": 0.03, "cls_gradient_ratio": 0.005}},
    ]


def resumable_screen_prelude(target_epochs: int) -> list[str]:
    return [
        "# Run from the repository root. Set PYTHON_BIN if python is not the intended environment.",
        "set -euo pipefail",
        'PYTHON_BIN="${PYTHON_BIN:-python}"',
        'CACHE_DIR="${CACHE_DIR:-cache/teacher_signals/train}"',
        f"TARGET_EPOCHS={target_epochs}",
        "",
        "run_trial() {",
        '  local name="$1" exp="$2" config="$3"',
        # Keep dependent assignments on separate commands. With `set -u`, Bash
        # expands every word in a single `local` command before `run_dir` has
        # been assigned, so referencing `$run_dir` in that same command aborts.
        '  local run_dir="runs/$name"',
        '  local last="$run_dir/weights/last.pt"',
        '  local marker="$run_dir/weights/best.pt"',
        '  if [[ "$exp" != "c0" ]]; then marker="$run_dir/weights/best_deploy.pt"; fi',
        "  local rows=0",
        '  if [[ -f "$run_dir/results.csv" ]]; then rows=$(($(wc -l < "$run_dir/results.csv") - 1)); fi',
        '  if (( rows >= TARGET_EPOCHS )) && [[ -f "$marker" ]]; then',
        '    echo "DIRECTION1 SKIP: $name already completed ($rows/$TARGET_EPOCHS epochs)."',
        "    return 0",
        "  fi",
        '  local args=(-m src.train_ablation --config "$config" --exp "$exp" --run-name "$name")',
        '  if [[ "$exp" != "c0" ]]; then args+=(--cache "$CACHE_DIR"); fi',
        '  if [[ -f "$last" ]]; then',
        '    echo "DIRECTION1 RESUME: $name from $last ($rows/$TARGET_EPOCHS completed epochs)."',
        '    args+=(--resume)',
        '  elif [[ -e "$run_dir" ]]; then',
        '    echo "DIRECTION1 ERROR: $run_dir exists but has no resumable last.pt. Preserve it and choose a recovery action manually." >&2',
        "    return 1",
        "  else",
        '    echo "DIRECTION1 START: $name"',
        "  fi",
        '  "$PYTHON_BIN" "${args[@]}"',
        "}",
        "",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the bounded Direction-1 eight-run screening matrix.")
    parser.add_argument("--base", default="configs/direction1.yaml")
    parser.add_argument("--out", default="configs/direction1_trials")
    args = parser.parse_args()
    base_path = Path(args.base).resolve(); base = load_config(base_path)
    project_root = Path(base["paths"]["project_root"])
    output = Path(args.out)
    if not output.is_absolute():
        output = project_root / output
    output.mkdir(parents=True, exist_ok=True)
    manifest = []
    commands = resumable_screen_prelude(int(base["student"]["epochs"]))
    evaluation_commands = [
        "# Evaluate only after all screening runs complete.",
        "set -euo pipefail",
        'PYTHON_BIN="${PYTHON_BIN:-python}"',
        "mkdir -p reports/direction1",
        "",
    ]
    native_model_arguments = []
    base_config_relative = base_path.relative_to(project_root).as_posix()
    for trial in trial_matrix(base):
        config = copy.deepcopy(base)
        config["distillation"].update(trial["updates"])
        config_path = output / f"{trial['name']}.yaml"
        config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        relative_config = config_path.relative_to(project_root).as_posix()
        command = f'run_trial {trial["name"]} {trial["experiment"]} {relative_config}'
        commands.append(command)
        weight_name = "best.pt" if trial["experiment"] == "c0" else "best_deploy.pt"
        weight_path = f"runs/{trial['name']}/weights/{weight_name}"
        native_model_arguments.append(f"--model {trial['name']}={weight_path}")
        evaluation_commands.append(
            f'"$PYTHON_BIN" -m src.competition_eval --config {base_config_relative} '
            f'--model {weight_path} --split val --class-aware --out reports/direction1/{trial["name"]}_competition.json'
        )
        manifest.append({**trial, "config": relative_config, "command": command})
    native_command = (
        '"$PYTHON_BIN" -m src.validate_models '
        + " ".join(native_model_arguments)
        + f" --config {base_config_relative} --out reports/direction1/native_val.json"
    )
    evaluation_commands.extend([
        "",
        native_command,
        '"$PYTHON_BIN" -m src.summarize_direction1 --runs runs --pattern "d1_*" --output reports/direction1/screen_summary.json',
    ])
    (output / "matrix.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "run_screen.sh").write_text("\n".join(commands) + "\n", encoding="utf-8", newline="\n")
    (output / "evaluate_screen.sh").write_text("\n".join(evaluation_commands) + "\n", encoding="utf-8", newline="\n")
    print(f"Generated {len(manifest)} trials in {output}")
    print(f"Review {output / 'matrix.json'} before running {output / 'run_screen.sh'}")


if __name__ == "__main__":
    main()
