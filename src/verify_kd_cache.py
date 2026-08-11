from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from .common import load_config, resolve_data_yaml, stable_image_key
from .provenance import (
    resolve_dataset_identity,
    teacher_cache_dir,
    validate_cache_manifest,
    verify_cache_sample,
)
from .train_teacher import YoloRoiDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify that cached teacher labels match the deterministic letterbox dataset.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--cache", default=None)
    parser.add_argument("--samples", type=int, default=64)
    args = parser.parse_args()
    cfg = load_config(args.config); data = resolve_data_yaml(cfg)
    identity = resolve_dataset_identity(cfg)
    cache_dir = Path(args.cache) if args.cache else teacher_cache_dir(cfg, args.split, identity)
    dataset = YoloRoiDataset(data, args.split, cfg["dataset"]["image_size"])
    if not cache_dir.exists():
        raise FileNotFoundError(f"Cache missing: {cache_dir}")
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Cache manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_cache_manifest(cfg, manifest, args.split, identity)
    provenance_checked = verify_cache_sample(
        cfg, cache_dir, manifest, samples=args.samples
    )
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
    print(
        f"PASS: {checked} cached samples match deterministic teacher letterbox labels; "
        f"{provenance_checked} samples also match image/label/manifest hashes."
    )
    print("C0/F/K/FK additionally disable every photometric and geometric augmentation in train_ablation.py.")


if __name__ == "__main__":
    main()
