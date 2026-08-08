from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .common import json_dump, load_config


DEFAULT_GROUPS = {"ship", "vehicle"}
DEFAULT_ERRORS = {"low_confidence", "nms_suppressed", "localization", "no_candidate", "wrong_group"}


def build_replay_manifest(hard: dict, negative: dict, hard_sha256: str, negative_sha256: str) -> dict:
    if hard.get("format") != 1 or hard.get("split") != "train":
        raise RuntimeError("Hard-example source must be a format=1 TRAIN manifest.")
    if negative.get("format") != 1 or negative.get("kind") != "vehicle_background" or negative.get("split") != "train":
        raise RuntimeError("Negative source must be a format=1 TRAIN vehicle_background manifest.")
    if negative.get("review_required") is not False:
        raise RuntimeError("Negative source has not been explicitly approved by review or annotation policy.")
    images: dict[str, dict] = {}
    for relative, entry in hard.get("images", {}).items():
        selected = [
            item for item in entry.get("objects", [])
            if item.get("coarse_group") in DEFAULT_GROUPS and item.get("error_type") in DEFAULT_ERRORS
        ]
        if selected:
            images[relative] = {
                "hard_positive_objects": len(selected),
                "groups": sorted({item["coarse_group"] for item in selected}),
                "errors": sorted({item["error_type"] for item in selected}),
                "vehicle_background_boxes": 0,
            }
    for relative, entry in negative.get("images", {}).items():
        target = images.setdefault(relative, {
            "hard_positive_objects": 0, "groups": [], "errors": [], "vehicle_background_boxes": 0,
        })
        target["vehicle_background_boxes"] = len(entry.get("boxes", []))
    if not images:
        raise RuntimeError("No hard TRAIN images survived replay selection.")
    for relative in images:
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise RuntimeError(f"Unsafe TRAIN-relative replay path: {relative!r}")
    return {
        "format": 1, "kind": "hard_image_replay", "split": "train",
        "groups": sorted(DEFAULT_GROUPS), "errors": sorted(DEFAULT_ERRORS),
        "images": dict(sorted(images.items())), "image_count": len(images),
        "source_hard_manifest_sha256": hard_sha256,
        "source_negative_manifest_sha256": negative_sha256,
        "negative_annotation_policy": negative.get("annotation_policy", "human_reviewed"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build one shared OOF hard-image replay population for fair controls.")
    parser.add_argument("--config", default="configs/direction1.yaml")
    parser.add_argument("--hard-manifest", default="reports/dataset_d0/oof_mining_v1/hard_examples_oof.json")
    parser.add_argument("--negative-manifest", default="reports/dataset_d0/oof_mining_v1/vehicle_background_annotation_policy.json")
    parser.add_argument("--out", default="reports/dataset_d0/oof_training_v5/hard_replay_images.json")
    args = parser.parse_args()
    config = load_config(args.config); root = Path(config["paths"]["project_root"]).resolve()
    def resolve(value: str) -> Path:
        path = Path(value); return path if path.is_absolute() else root / path
    hard_path, negative_path = resolve(args.hard_manifest), resolve(args.negative_manifest)
    for path in (hard_path, negative_path):
        if not path.exists(): raise FileNotFoundError(path)
    manifest = build_replay_manifest(
        json.loads(hard_path.read_text(encoding="utf-8")),
        json.loads(negative_path.read_text(encoding="utf-8")),
        hashlib.sha256(hard_path.read_bytes()).hexdigest(),
        hashlib.sha256(negative_path.read_bytes()).hexdigest(),
    )
    destination = resolve(args.out); json_dump(manifest, destination)
    positive_images = sum(int(item["hard_positive_objects"] > 0) for item in manifest["images"].values())
    negative_images = sum(int(item["vehicle_background_boxes"] > 0) for item in manifest["images"].values())
    print(
        f"HARD REPLAY MANIFEST: images={manifest['image_count']} hard_positive_images={positive_images} "
        f"vehicle_background_images={negative_images} output={destination}"
    )


if __name__ == "__main__":
    main()
