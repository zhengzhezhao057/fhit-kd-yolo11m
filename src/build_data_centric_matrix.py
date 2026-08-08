from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .common import json_dump, load_config

GENERATOR_VERSION = 1
SCREEN_ROOT = Path("configs/scene811_screens")

ROUND1 = {
    "scene811_r0": {"hard_positive": 0.00, "hard_negative": 0.00, "purpose": "no-replay continuation control"},
    "scene811_hp10": {"hard_positive": 0.10, "hard_negative": 0.00, "purpose": "hard positive contribution"},
    "scene811_hn10": {"hard_positive": 0.00, "hard_negative": 0.10, "purpose": "hard background contribution"},
    "scene811_mix10": {"hard_positive": 0.05, "hard_negative": 0.05, "purpose": "low-budget joint"},
    "scene811_mix20": {"hard_positive": 0.10, "hard_negative": 0.10, "purpose": "high-budget joint"},
}

ROUND2 = {
    "replay_ship40": {"composition": {"ship_ratio": 0.40}, "purpose": "ship:vehicle replay ratio 40/60"},
    "replay_ship60": {"composition": {"ship_ratio": 0.60}, "purpose": "ship:vehicle replay ratio 60/40"},
    "replay_bg_crop": {"composition": {"background_style": "empty_label_crop"}, "purpose": "empty-label crops instead of full-image background"},
    "replay_small_vehicle_w15": {"composition": {"small_vehicle_weight": 1.5}, "purpose": "small vehicle replay weight"},
    "replay_crowded_ship_w15": {"composition": {"crowded_ship_weight": 1.5}, "purpose": "crowded ship replay weight"},
    "replay_20": {"hard_positive": 0.10, "hard_negative": 0.10, "purpose": "replay budget 20% with winning composition"},
}

ROUND3 = {
    "aug_none": {"augmentation": {"mode": "none"}, "purpose": "all image-augmentation transforms off"},
    "aug_mild": {"augmentation": {"mode": "mild", "translate": 0.1, "scale": 0.5, "hsv": 0.015}, "purpose": "mild translate/scale/HSV"},
    "aug_context": {"augmentation": {"mode": "context", "translate": 0.1, "scale": 0.5, "hsv": 0.015}, "purpose": "mild augmentation plus background context crops"},
}

ROUND4 = {
    "res_640": {"imgsz": 640, "purpose": "baseline resolution"},
    "res_832": {"imgsz": 832, "purpose": "higher resolution"},
    "res_1024": {"imgsz": 1024, "purpose": "max resolution"},
}

ROUNDS = {1: ROUND1, 2: ROUND2, 3: ROUND3, 4: ROUND4}


def trial_config(base: dict, name: str, overrides: dict, round_number: int, kind: str) -> dict:
    config = {
        "screen": {
            "round": round_number,
            "kind": kind,
            "epochs": 8,
            "augment": "none",
            "initial_weights": base.get("model", "yolo11m.pt"),
            "imgsz": int(overrides.get("imgsz", base.get("imgsz", 640))),
        },
        "model": base.get("model", "yolo11m.pt"),
        "data": base.get("data", "configs/datasets/scene811.server.yaml"),
        "imgsz": int(overrides.get("imgsz", base.get("imgsz", 640))),
        "batch": base.get("batch", 16),
        "optimizer": base.get("optimizer", "AdamW"),
        "replay": {
            "hard_positive_fraction": float(overrides.get("hard_positive", 0.0)),
            "hard_negative_fraction": float(overrides.get("hard_negative", 0.0)),
            "max_repeat_per_image": 3,
            "max_replay_fraction": 0.20,
            "composition": overrides.get("composition", {}),
            "pools": "artifacts/scene811_v1/replay/replay_inventory.json",
        },
        "augmentation": overrides.get("augmentation", {"mode": "none"}),
        "purpose": overrides.get("purpose", ""),
    }
    if round_number == 1:
        config["screen"]["kind"] = "sample_type"
    elif round_number == 2:
        config["screen"]["kind"] = "composition"
    elif round_number == 3:
        config["screen"]["kind"] = "augmentation"
        config["screen"]["augment"] = config["augmentation"]["mode"]
    elif round_number == 4:
        config["screen"]["kind"] = "resolution"
    return config


def build_matrix(base_config: Path, round_number: int, out_dir: Path) -> dict:
    base = load_config(base_config)
    experiments = ROUNDS[round_number]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, overrides in experiments.items():
        config = trial_config(base, name, overrides, round_number, "")
        target = out / f"{name}.yaml"
        target.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        written.append(name)
    matrix = {
        "format": 1,
        "kind": "scene811_data_centric_matrix",
        "generator_version": GENERATOR_VERSION,
        "round": round_number,
        "base_config": str(Path(base_config).resolve()),
        "experiments": written,
        "spec": experiments,
        "rules": {
            "round1": "8 epochs, augmentation off, same yolo11m.pt init",
            "round2": "winning composition from round 1 only",
            "round3": "winning replay from round 2 only",
            "round4": "winning replay and augmentation only; report small vehicle, speed, VRAM and mAP50-95",
        },
    }
    json_dump(matrix, out / "matrix.json")
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Scene811 data-centric screening matrices (design section 12).")
    parser.add_argument("--base", default="configs/scene811_baselines/baseline_seed0.yaml")
    parser.add_argument("--round", type=int, choices=sorted(ROUNDS), default=1)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    out = Path(args.out) if args.out else Path(__file__).resolve().parents[1] / SCREEN_ROOT / f"round{args.round}"
    matrix = build_matrix(Path(args.base), args.round, out)
    print(f"saved {out}/matrix.json")
    for name in matrix["experiments"]:
        print(name)


if __name__ == "__main__":
    main()
