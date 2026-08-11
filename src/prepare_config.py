from __future__ import annotations

import argparse
from pathlib import Path

from .common import load_config
from .provenance import inject_discovered_fingerprints


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a machine-specific experiment.yaml without editing source-controlled files."
    )
    parser.add_argument("--data-yaml", required=True, help="Absolute path to the YOLO dataset.yaml")
    parser.add_argument("--baseline-weights", required=True, help="Absolute path to the fixed pure-YOLO best.pt")
    parser.add_argument("--dino-weights", required=True, help="Absolute path to the DINOv3-SAT ViT-L/16 .pth")
    parser.add_argument("--dino-repo", default=None, help="DINOv3 checkout; defaults to external/dinov3 in this repository")
    parser.add_argument(
        "--template",
        default="configs/experiment.example.yaml",
        help="Repository-relative or absolute YAML template. Use configs/direction1.example.yaml for the new short KD study.",
    )
    parser.add_argument("--out", default="configs/experiment.yaml", help="Output config path")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    example = Path(args.template)
    if not example.is_absolute():
        example = project_root / example
    if not example.exists():
        parser.error(f"Configuration template not found: {example}")
    config = load_config(example)
    config["paths"] = {
        "data_yaml": str(Path(args.data_yaml).resolve()),
        "baseline_weights": str(Path(args.baseline_weights).resolve()),
        "dino_weights": str(Path(args.dino_weights).resolve()),
        "dino_repo": str(Path(args.dino_repo).resolve()) if args.dino_repo else str((project_root / "external" / "dinov3").resolve()),
        "project_root": str(project_root),
    }
    missing = [name for name, value in config["paths"].items() if name != "project_root" and not Path(value).exists()]
    if missing:
        parser.error(f"Missing required path(s): {', '.join(missing)}")
    try:
        inject_discovered_fingerprints(config)
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(f"Dataset provenance validation failed: {error}")

    import yaml
    output = Path(args.out)
    if not output.is_absolute():
        output = project_root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
