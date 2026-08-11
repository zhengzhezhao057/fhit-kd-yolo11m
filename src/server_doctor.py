"""Portable preflight and optional dependency bootstrap for a new training server."""
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_PACKAGES = {
    "torch": "2.4.0",
    "torchvision": "0.19.0",
    "ultralytics": "8.4.90",
    "opencv-python": "4.10.0",
    "Pillow": "10.0.0",
    "PyYAML": "6.0.2",
    "tqdm": "4.66.0",
    "pytest": "8.0.0",
}
DINO_REQUIRED_MODULES = {
    "ftfy": "ftfy",
    "omegaconf": "omegaconf",
    "regex": "regex",
    "scikit-learn": "sklearn",
    "submitit": "submitit",
    "termcolor": "termcolor",
    "torchmetrics": "torchmetrics",
}
MIN_VRAM_GB = 24
MIN_FREE_GB = 120


def vram_capacity(total_bytes: int) -> tuple[float, float, bool]:
    """Return decimal GB, binary GiB and whether a marketed 24 GB card is sufficient.

    NVIDIA/PyTorch commonly reports a 24 GB RTX 4090 as about 23.5 GiB. Comparing
    that binary value with the decimal product label 24 incorrectly rejects the card.
    """
    decimal_gb = total_bytes / 1_000_000_000
    binary_gib = total_bytes / 1024**3
    return decimal_gb, binary_gib, decimal_gb >= MIN_VRAM_GB


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    required: bool = True


def version_at_least(actual: str, minimum: str) -> bool:
    """Dependency-free comparison for the stable numeric package versions used here."""
    def numeric(value: str) -> tuple[int, ...]:
        numbers = re.findall(r"\d+", value.split("+")[0])
        return tuple(int(number) for number in numbers)
    current, target = numeric(actual), numeric(minimum)
    width = max(len(current), len(target))
    return current + (0,) * (width - len(current)) >= target + (0,) * (width - len(target))


def package_checks() -> list[Check]:
    checks: list[Check] = []
    for package, minimum in REQUIRED_PACKAGES.items():
        module = {"opencv-python": "cv2", "Pillow": "PIL", "PyYAML": "yaml"}.get(package, package)
        if importlib.util.find_spec(module) is None:
            checks.append(Check(package, False, f"missing (requires >= {minimum})"))
            continue
        try:
            actual = importlib.metadata.version(package)
            exact = package == "ultralytics"
            ok = actual == minimum if exact else version_at_least(actual, minimum)
            relation = "==" if exact else ">="
            checks.append(Check(package, ok, f"{actual} (requires {relation} {minimum})"))
        except importlib.metadata.PackageNotFoundError:
            checks.append(Check(package, False, "installed module has no package metadata"))
    for package, module in DINO_REQUIRED_MODULES.items():
        if importlib.util.find_spec(module) is None:
            checks.append(Check(package, False, "missing (required by DINOv3)"))
            continue
        try:
            checks.append(Check(package, True, f"{importlib.metadata.version(package)} (required by DINOv3)"))
        except importlib.metadata.PackageNotFoundError:
            checks.append(Check(package, False, "installed module has no package metadata"))
    return checks


def run(command: list[str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, check=True)


def install_dependencies(cuda_index: str) -> None:
    """Install the supported Python packages. Driver installation remains a manual admin task."""
    run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    run([sys.executable, "-m", "pip", "install", "--upgrade", "torch", "torchvision", "--index-url", f"https://download.pytorch.org/whl/{cuda_index}"])
    run([sys.executable, "-m", "pip", "install", "--upgrade", "-r", str(PROJECT_ROOT / "requirements.txt")])
    dino_requirements = PROJECT_ROOT / "external" / "dinov3" / "requirements.txt"
    if not dino_requirements.exists():
        raise FileNotFoundError(f"DINOv3 requirements missing at {dino_requirements}; initialize the git submodule first.")
    run([sys.executable, "-m", "pip", "install", "--upgrade", "-r", str(dino_requirements)])


def full_checks(config_path: Path, require_config: bool, min_free_gb: float = MIN_FREE_GB) -> list[Check]:
    checks: list[Check] = []
    if not (sys.version_info >= (3, 10) and sys.version_info < (3, 14)):
        checks.append(Check("Python", False, f"{sys.version.split()[0]}; use Python 3.10-3.13"))
    else:
        checks.append(Check("Python", True, f"{sys.version.split()[0]}"))

    try:
        import torch
        if not torch.cuda.is_available():
            checks.append(Check("CUDA", False, "PyTorch cannot see an NVIDIA CUDA device"))
        else:
            props = torch.cuda.get_device_properties(0)
            vram_gb, vram_gib, enough_vram = vram_capacity(props.total_memory)
            checks.append(Check("CUDA", True, f"{torch.cuda.get_device_name(0)}, CUDA {torch.version.cuda}"))
            checks.append(Check("GPU VRAM", enough_vram, f"{vram_gb:.1f} GB ({vram_gib:.1f} GiB; requires >= {MIN_VRAM_GB} GB)"))
    except Exception as exc:
        checks.append(Check("CUDA", False, f"torch import/check failed: {exc}"))

    free_gb = shutil.disk_usage(PROJECT_ROOT).free / 1024**3
    checks.append(Check("Free disk", free_gb >= min_free_gb, f"{free_gb:.1f} GiB at {PROJECT_ROOT} (requires >= {min_free_gb:g} GiB)"))
    dino_source = PROJECT_ROOT / "external" / "dinov3" / "hubconf.py"
    checks.append(Check("DINOv3 submodule", dino_source.exists(), str(dino_source)))

    if config_path.exists():
        from .common import load_config
        config = load_config(config_path)
        for key, value in config.get("paths", {}).items():
            if key == "project_root":
                continue
            checks.append(Check(f"config:{key}", Path(value).exists(), str(value)))
    else:
        checks.append(Check("experiment config", not require_config, f"missing {config_path}; create it with python -m src.prepare_config", required=require_config))
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description="Check or bootstrap a cloned DINOv3-YOLO11m experiment server.")
    parser.add_argument("--full", action="store_true", help="Also check GPU, VRAM, disk, submodule and configured data/weight paths.")
    parser.add_argument("--require-config", action="store_true", help="Treat a missing configs/experiment.yaml as an error (implies --full).")
    parser.add_argument("--config", default="configs/experiment.yaml", help="Machine-specific experiment config path.")
    parser.add_argument("--install", action="store_true", help="Install/upgrade Python dependencies before checking. Does not install GPU drivers.")
    parser.add_argument("--cuda-index", default="cu126", choices=("cu118", "cu121", "cu124", "cu126"), help="PyTorch wheel CUDA index used with --install.")
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=MIN_FREE_GB,
        help="Required free space on the project filesystem. Keep the 120 GiB default unless storage has been budgeted explicitly; this dataset supports 45 GiB compact mode.",
    )
    args = parser.parse_args()
    if args.min_free_gb <= 0:
        parser.error("--min-free-gb must be positive")
    if args.install:
        install_dependencies(args.cuda_index)

    checks = package_checks()
    if args.full or args.require_config:
        config = Path(args.config)
        if not config.is_absolute():
            config = PROJECT_ROOT / config
        checks.extend(full_checks(config, args.require_config, args.min_free_gb))

    print("\nServer doctor report")
    for check in checks:
        state = "PASS" if check.ok else ("FAIL" if check.required else "WARN")
        print(f"[{state}] {check.name}: {check.detail}")
    failures = [check for check in checks if check.required and not check.ok]
    if failures:
        print(f"\nFAILED: {len(failures)} required check(s) need attention.")
        if not args.install:
            print("For missing or old Python packages, rerun with --install. GPU drivers must be installed by the server administrator.")
        raise SystemExit(1)
    print("\nPASS: server is ready for the selected check level.")


if __name__ == "__main__":
    main()
