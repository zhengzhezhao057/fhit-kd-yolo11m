from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from .common import load_config, resolve_data_yaml, split_image_dir, stable_image_key, xyxy_to_xywhn


REQUIRED_COLUMNS = {
    "model", "image", "fine_class", "coarse_group", "score",
    "box_x1", "box_y1", "box_x2", "box_y2", "reason", "predicted_size",
}


def build_vehicle_negative_manifest(
    config: dict,
    false_positive_csv: Path,
    model_name: str,
    minimum_score: float,
) -> dict:
    """Convert TRAIN vehicle-background false positives to teacher-letterbox boxes."""
    if not 0.0 <= minimum_score <= 1.0:
        raise ValueError("minimum_score must be between 0 and 1")
    data = resolve_data_yaml(config)
    train_dir = split_image_dir(data, "train").resolve()
    valid_images = {
        stable_image_key(path): path.resolve()
        for path in train_dir.rglob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    }
    with false_positive_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"{false_positive_csv} lacks required columns {sorted(missing)}")
        source_rows = list(reader)

    images: dict[str, dict] = {}
    excluded = Counter()
    kept_scores: list[float] = []
    image_size = int(config["dataset"]["image_size"])
    shapes: dict[str, tuple[int, int]] = {}
    for row in source_rows:
        if row["model"] != model_name:
            excluded["other_model"] += 1
            continue
        key = stable_image_key(row["image"])
        if key not in valid_images:
            raise RuntimeError(f"Refusing data leakage: {row['image']} is not a member of the configured TRAIN split.")
        if row["coarse_group"] != "vehicle":
            excluded["other_group"] += 1
            continue
        if row["reason"] != "background":
            # localization can be a shifted real vehicle and duplicate is a
            # post-processing event; neither is a safe negative label.
            excluded[row["reason"]] += 1
            continue
        score = float(row["score"])
        if score < minimum_score:
            excluded["below_minimum_score"] += 1
            continue
        if key not in shapes:
            image = cv2.imread(str(valid_images[key]))
            if image is None:
                raise RuntimeError(f"Cannot read training image: {valid_images[key]}")
            shapes[key] = image.shape[:2]
        height, width = shapes[key]
        ratio = min(image_size / height, image_size / width)
        resized_width, resized_height = int(round(width * ratio)), int(round(height * ratio))
        left, top = (image_size - resized_width) // 2, (image_size - resized_height) // 2
        box = np.asarray([[float(row[name]) for name in ("box_x1", "box_y1", "box_x2", "box_y2")]], dtype=np.float32)
        box[:, [0, 2]] = box[:, [0, 2]] * ratio + left
        box[:, [1, 3]] = box[:, [1, 3]] * ratio + top
        box_xywhn = xyxy_to_xywhn(box, image_size, image_size)[0]
        relative = valid_images[key].relative_to(train_dir).as_posix()
        entry = images.setdefault(relative, {"boxes": []})
        entry["boxes"].append({
            "box_xywhn": [float(value) for value in box_xywhn],
            "score": score,
            "predicted_size": row["predicted_size"],
            "fine_class": int(row["fine_class"]),
        })
        kept_scores.append(score)
    for entry in images.values():
        entry["boxes"].sort(key=lambda item: (-item["score"], item["box_xywhn"]))
    if not kept_scores:
        raise RuntimeError("No safe vehicle-background negatives survived filtering.")
    return {
        "format": 1,
        "kind": "vehicle_background",
        "split": "train",
        "source_model": model_name,
        "source_csv": str(false_positive_csv.resolve()),
        "source_csv_sha256": hashlib.sha256(false_positive_csv.read_bytes()).hexdigest(),
        "minimum_score": minimum_score,
        "images": images,
        "negative_boxes": len(kept_scores),
        "excluded": dict(sorted(excluded.items())),
        "score_range": {"minimum": min(kept_scores), "maximum": max(kept_scores)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a leakage-safe TRAIN vehicle-background negative manifest.")
    parser.add_argument("--config", default="configs/direction1.yaml")
    parser.add_argument("--false-positives", required=True)
    parser.add_argument("--model", default="B0")
    parser.add_argument("--split", choices=("train",), default="train")
    parser.add_argument("--minimum-score", type=float, default=0.35)
    parser.add_argument("--out", default="reports/weak_group_train_c035/vehicle_background_negatives.json")
    args = parser.parse_args()
    manifest = build_vehicle_negative_manifest(
        load_config(args.config), Path(args.false_positives), args.model, args.minimum_score
    )
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"saved {destination}: {manifest['negative_boxes']} safe vehicle-background boxes "
        f"in {len(manifest['images'])} TRAIN images"
    )
    print(f"excluded={manifest['excluded']}")


if __name__ == "__main__":
    main()
