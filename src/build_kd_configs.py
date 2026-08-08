from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml


def build(base: dict, mode: str, hard_manifest: str | None = None) -> dict:
    config = copy.deepcopy(base)
    distill = config["distillation"]
    if mode == "global":
        distill["hard_example_manifest"] = None
        distill["group_distill_weights"] = {}
        distill["feature_error_weights"] = {}
        distill["cls_error_weights"] = {}
        distill["object_weight_bounds"] = [0.25, 4.0]
        distill.setdefault("hierarchical_kl", {})["enabled"] = False
    elif mode == "fah":
        if not hard_manifest:
            raise ValueError("FAH-KD requires --hard-manifest from leakage-safe OOF mining.")
        path = Path(hard_manifest)
        if not path.is_absolute():
            path = Path(config["paths"]["project_root"]) / path
        if not path.is_file():
            raise FileNotFoundError(path)
        distill["hard_example_manifest"] = str(path.resolve())
        distill.setdefault("hierarchical_kl", {})["enabled"] = True
    else:
        raise ValueError(mode)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate immutable Global-KD or failure-aware hierarchical KD configs.")
    parser.add_argument("--base", default="configs/experiment.yaml")
    parser.add_argument("--mode", required=True, choices=("global", "fah"))
    parser.add_argument("--hard-manifest", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    base = yaml.safe_load(Path(args.base).read_text(encoding="utf-8"))
    config = build(base, args.mode, args.hard_manifest)
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"wrote {destination} mode={args.mode}")


if __name__ == "__main__":
    main()
