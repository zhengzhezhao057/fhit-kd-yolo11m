from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch

from .common import load_config, resolve_data_yaml, stable_image_key
from .train_teacher import YoloRoiDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify that cached teacher labels match the deterministic letterbox dataset.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--cache", default=None)
    parser.add_argument("--samples", type=int, default=64)
    args = parser.parse_args()
    cfg = load_config(args.config); data = resolve_data_yaml(cfg)
    root = Path(cfg["paths"]["project_root"])
    cache_dir = Path(args.cache) if args.cache else root / "cache" / "teacher_signals" / args.split
    dataset = YoloRoiDataset(data, args.split, cfg["dataset"]["image_size"])
    if not cache_dir.exists():
        raise FileNotFoundError(f"Cache missing: {cache_dir}")
    indices = list(range(len(dataset)))
    random.Random(0).shuffle(indices)
    checked = 0
    for index in indices[: min(args.samples, len(indices))]:
        item = dataset[index]
        cache_path = cache_dir / f"{stable_image_key(item['path'])}.pt"
        if not cache_path.exists():
            raise FileNotFoundError(f"Cache miss: {cache_path}")
        signal = torch.load(cache_path, map_location="cpu", weights_only=False)
        if str(Path(signal["path"]).resolve()) != str(Path(item["path"]).resolve()):
            raise RuntimeError(f"Path mismatch for {item['path']}")
        if not torch.equal(signal["classes"].long(), item["classes"].long()):
            raise RuntimeError(f"Class ordering mismatch for {item['path']}")
        if not torch.allclose(signal["boxes_xywhn"].float(), item["boxes"].float(), atol=1e-5, rtol=0):
            raise RuntimeError(f"Letterbox box mismatch for {item['path']}")
        checked += 1
    print(f"PASS: {checked} cached samples match deterministic teacher letterbox labels.")
    print("C0/F/K/FK additionally disable every photometric and geometric augmentation in train_ablation.py.")


if __name__ == "__main__":
    main()
