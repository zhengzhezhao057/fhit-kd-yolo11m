from __future__ import annotations

import argparse
from pathlib import Path

from .common import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Check files, package versions and CUDA before any experiment.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    args = parser.parse_args(); cfg = load_config(args.config)
    import torch
    import torchvision
    import ultralytics
    failures = []
    print(f"Python environment check")
    print(f"torch={torch.__version__}, torchvision={torchvision.__version__}, ultralytics={ultralytics.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu={torch.cuda.get_device_name(0)}, vram_gb={torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}")
    else:
        failures.append("CUDA is unavailable. Do not run teacher or student training on CPU.")
    for label, value in (("data_yaml", cfg["paths"]["data_yaml"]), ("baseline_weights", cfg["paths"]["baseline_weights"]), ("dino_weights", cfg["paths"]["dino_weights"]), ("dino_repo", cfg["paths"]["dino_repo"])):
        ok = Path(value).exists(); print(f"{label}: {'OK' if ok else 'MISSING'} -> {value}")
        if not ok: failures.append(f"Missing {label}: {value}")
    if str(ultralytics.__version__) != "8.4.90":
        failures.append("This project is validated with ultralytics 8.4.90. Activate the yolo11 environment or install requirements.txt before training.")
    if failures:
        print("FAILED")
        for item in failures: print(f"- {item}")
        raise SystemExit(1)
    print("PASS: environment is ready")


if __name__ == "__main__":
    main()
