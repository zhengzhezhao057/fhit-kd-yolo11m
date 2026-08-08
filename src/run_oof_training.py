from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .common import load_config
from .oof_training import OOF_RUN_PREFIX, classify_run, load_fold_manifest, validate_initial_checkpoint_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all three Dataset-D0 product-grouped OOF detectors serially.")
    parser.add_argument("--config", default="configs/direction1.yaml")
    parser.add_argument("--folds-dir", default="reports/dataset_d0/oof3")
    parser.add_argument("--initial-weights", default="yolo11m.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    validate_initial_checkpoint_name(args.initial_weights)
    cfg = load_config(args.config)
    root = Path(cfg["paths"]["project_root"]).resolve()
    folds_dir = Path(args.folds_dir)
    folds_dir = folds_dir if folds_dir.is_absolute() else root / folds_dir
    load_fold_manifest(folds_dir)

    for fold in range(3):
        run_dir = root / "runs" / f"{OOF_RUN_PREFIX}{fold}"
        state = classify_run(run_dir, args.epochs)
        print(
            f"OOF PLAN: fold={fold} action={state.action} completed_epochs={state.completed_epochs} "
            f"reason={state.reason}", flush=True,
        )
        if state.action == "skip":
            continue
        if state.action == "error":
            raise RuntimeError(f"OOF fold {fold}: {state.reason}")
        command = [
            sys.executable, "-m", "src.train_oof_fold", "--config", args.config,
            "--folds-dir", str(folds_dir), "--fold", str(fold),
            "--initial-weights", args.initial_weights, "--epochs", str(args.epochs),
            "--batch", str(args.batch), "--workers", str(args.workers), "--device", args.device,
        ]
        if state.action == "resume":
            command.append("--resume")
        subprocess.run(command, cwd=root, check=True)
    print("OOF TRAINING COMPLETE: all three folds reached the fixed target.")


if __name__ == "__main__":
    main()
