from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

from .common import load_config
from .provenance import inject_discovered_fingerprints, resolve_dataset_identity


DEFAULT_SEEDS = (42, 3407, 20260809)


def _require_file(path: Path, description: str) -> Path:
    value = path.expanduser().resolve()
    if not value.is_file():
        raise FileNotFoundError(f"{description} is missing: {value}")
    return value


def _load_training_ready_dataset(root: Path) -> tuple[Path, dict, dict]:
    root = root.expanduser().resolve()
    audit_path = _require_file(root / "audit_d0.json", "Scene811 V3 D0 audit")
    fingerprint_path = _require_file(
        root / "dataset_fingerprint.json", "Scene811 V3 dataset fingerprint"
    )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    fingerprint = json.loads(fingerprint_path.read_text(encoding="utf-8"))
    if audit.get("training_ready") is not True:
        raise RuntimeError(f"Dataset D0 did not pass: {audit_path}")
    audit_dataset_id = str(audit.get("dataset_id", ""))
    fingerprint_dataset_id = str(fingerprint.get("dataset_id", ""))
    if not audit_dataset_id or audit_dataset_id != fingerprint_dataset_id:
        raise RuntimeError(
            "D0 and dataset_fingerprint.json identify different dataset ids: "
            f"{audit_dataset_id!r} != {fingerprint_dataset_id!r}."
        )
    if audit.get("dataset_fingerprint") != fingerprint.get("dataset_fingerprint"):
        raise RuntimeError("D0 and dataset_fingerprint.json identify different datasets.")
    for name in ("dataset.yaml", "dataset_official.yaml", "split_manifest.csv"):
        _require_file(root / name, f"Scene811 V3 {name}")
    return root, audit, fingerprint


def _parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("--seeds must contain one or more distinct comma-separated integers")
    return seeds


def generate_configs(
    *,
    dataset_root: Path,
    output_dir: Path,
    baseline_template: Path,
    experiment_template: Path,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    epochs: int = 120,
    image_size: int = 640,
    batch: int = 16,
    workers: int = 4,
    device: str = "0",
    baseline_weights: Path | None = None,
    dino_weights: Path | None = None,
    dino_repo: Path | None = None,
) -> dict:
    project_root = Path(__file__).resolve().parents[1]
    dataset_root, audit, fingerprint = _load_training_ready_dataset(dataset_root)
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_base = load_config(baseline_template)
    dataset_id = str(fingerprint["dataset_id"])
    recipes = {
        "official": dataset_root / "dataset_official.yaml",
        "mix": dataset_root / "dataset.yaml",
    }
    matrix: list[dict] = []
    for recipe, data_yaml in recipes.items():
        for seed in seeds:
            config = copy.deepcopy(baseline_base)
            config.update(
                {
                    "dataset_id": dataset_id,
                    "dataset_fingerprint": fingerprint["dataset_fingerprint"],
                    "split_fingerprint": fingerprint["split_fingerprint"],
                    "recipe": recipe,
                    "data": str(data_yaml.resolve()),
                    "epochs": int(epochs),
                    "imgsz": int(image_size),
                    "batch": int(batch),
                    "workers": int(workers),
                    "device": str(device),
                    "seed": int(seed),
                }
            )
            filename = f"baseline_{recipe}_seed{seed}.yaml"
            destination = output_dir / filename
            destination.write_text(
                yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
            )
            matrix.append(
                {
                    "recipe": recipe,
                    "seed": seed,
                    "run_name": f"b_{recipe}_s{seed}",
                    "config": str(destination),
                    "data_yaml": str(data_yaml.resolve()),
                }
            )

    # Baseline validation must be possible before a KD starting checkpoint has
    # been selected.  Keep this deliberately small: it contains only the
    # machine-resolved fields consumed by validate_models, competition_eval
    # and weak_group_diagnostics, and no teacher/checkpoint placeholder.
    evaluation = {
        "paths": {
            "data_yaml": str(recipes["mix"].resolve()),
            "project_root": str(project_root.resolve()),
        },
        "dataset": {
            "id": dataset_id,
            "image_size": int(image_size),
        },
        "student": {"batch": int(batch)},
        "evaluation": {
            "batch": min(int(batch), 8),
            "max_det": 10_000,
            "nms_iou": 0.50,
        },
    }
    evaluation_path = output_dir / "evaluation_v3.yaml"
    evaluation_path.write_text(
        yaml.safe_dump(evaluation, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    experiment_path: Path | None = None
    if any(value is not None for value in (baseline_weights, dino_weights, dino_repo)):
        if baseline_weights is None or dino_weights is None:
            raise ValueError(
                "--baseline-weights and --dino-weights must be supplied together when generating KD config"
            )
        baseline_weights = _require_file(baseline_weights, "fixed B-mix baseline checkpoint")
        dino_weights = _require_file(dino_weights, "DINOv3-SAT checkpoint")
        resolved_dino_repo = (
            dino_repo.expanduser().resolve()
            if dino_repo is not None
            else (project_root / "external" / "dinov3").resolve()
        )
        _require_file(resolved_dino_repo / "hubconf.py", "DINOv3 hubconf.py")
        experiment = load_config(experiment_template)
        experiment["paths"] = {
            "data_yaml": str(recipes["mix"].resolve()),
            "baseline_weights": str(baseline_weights),
            "dino_weights": str(dino_weights),
            "dino_repo": str(resolved_dino_repo),
            "project_root": str(project_root),
        }
        experiment.setdefault("dataset", {})["id"] = dataset_id
        experiment["dataset"]["image_size"] = int(image_size)
        experiment.setdefault("student", {})["seed"] = int(seeds[0])
        inject_discovered_fingerprints(experiment)
        identity = resolve_dataset_identity(experiment)
        namespace = f"{identity['dataset_id']}__{identity['dataset_fingerprint'][:12]}"
        experiment.setdefault("distillation", {})["prototype_bank"] = str(
            project_root / "cache" / "prototype_banks" / namespace / "leave_one_scene_out.pt"
        )
        experiment_path = output_dir / "experiment_v3.yaml"
        experiment_path.write_text(
            yaml.safe_dump(experiment, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )

    result = {
        "format": 1,
        "dataset_id": dataset_id,
        "dataset_root": str(dataset_root),
        "dataset_fingerprint": fingerprint["dataset_fingerprint"],
        "split_fingerprint": fingerprint["split_fingerprint"],
        "training_ready": bool(audit["training_ready"]),
        "baseline_trials": matrix,
        "evaluation_config": str(evaluation_path),
        "experiment_config": str(experiment_path) if experiment_path else None,
    }
    (output_dir / "matrix.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate machine-resolved Scene811 V3 baseline and optional KD configs."
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--out", default="configs/generated/scene811_v3")
    parser.add_argument("--baseline-template", default="configs/scene811_v3.baseline.example.yaml")
    parser.add_argument("--experiment-template", default="configs/scene811_v3.experiment.example.yaml")
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--baseline-weights", default=None)
    parser.add_argument("--dino-weights", default=None)
    parser.add_argument("--dino-repo", default=None)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]

    def resolve_project_path(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else root / path

    result = generate_configs(
        dataset_root=Path(args.dataset_root),
        output_dir=resolve_project_path(args.out),
        baseline_template=resolve_project_path(args.baseline_template),
        experiment_template=resolve_project_path(args.experiment_template),
        seeds=_parse_seeds(args.seeds),
        epochs=args.epochs,
        image_size=args.image_size,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        baseline_weights=Path(args.baseline_weights) if args.baseline_weights else None,
        dino_weights=Path(args.dino_weights) if args.dino_weights else None,
        dino_repo=Path(args.dino_repo) if args.dino_repo else None,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
