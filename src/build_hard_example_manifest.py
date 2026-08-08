from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .common import load_config, resolve_data_yaml, split_image_dir, stable_image_key, transform_boxes_to_letterbox


REQUIRED_COLUMNS = {
    "model", "image", "gt_index", "fine_class", "coarse_group", "size",
    "crowded", "edge", "error_type", "gt_x", "gt_y", "gt_w", "gt_h",
}


def build_manifest(config: dict, instances_csv: Path, model_name: str) -> dict:
    """Build a hard-example manifest from TRAIN diagnostics only.

    The membership check is intentionally strict: even a single validation or
    test image aborts the build so model-selection evidence cannot leak into
    the distillation objective.
    """
    data = resolve_data_yaml(config)
    train_dir = split_image_dir(data, "train").resolve()
    valid_keys = {
        stable_image_key(path): path.resolve()
        for path in train_dir.rglob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    }
    if not valid_keys:
        raise RuntimeError(f"No training images found below {train_dir}")

    with instances_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames or [])
        if missing:
            raise RuntimeError(
                f"{instances_csv} lacks {sorted(missing)}. Regenerate TRAIN diagnostics "
                "with the current weak_group_diagnostics.py."
            )
        rows = [row for row in reader if row["model"] == model_name]
    if not rows:
        raise RuntimeError(f"No rows for model {model_name!r} in {instances_csv}")

    images: dict[str, dict] = {}
    image_shapes: dict[str, tuple[int, int]] = {}
    seen: set[tuple[str, int]] = set()
    errors: Counter[str] = Counter()
    groups: Counter[str] = Counter()
    for row in rows:
        key = stable_image_key(row["image"])
        if key not in valid_keys:
            raise RuntimeError(
                f"Refusing data leakage: {row['image']} is not a member of the configured TRAIN split."
            )
        gt_index = int(row["gt_index"])
        identity = (key, gt_index)
        if identity in seen:
            raise RuntimeError(f"Duplicate diagnostic row for image key {key}, gt_index={gt_index}")
        seen.add(identity)
        error_type = row["error_type"]
        group = row["coarse_group"]
        errors[error_type] += 1
        groups[group] += 1
        relative_image = valid_keys[key].relative_to(train_dir).as_posix()
        entry = images.setdefault(relative_image, {"objects": []})
        if key not in image_shapes:
            import cv2

            image = cv2.imread(str(valid_keys[key]))
            if image is None:
                raise RuntimeError(f"Cannot read training image while building hard manifest: {valid_keys[key]}")
            image_shapes[key] = image.shape[:2]
        height, width = image_shapes[key]
        image_size = int(config["dataset"]["image_size"])
        ratio = min(image_size / height, image_size / width)
        resized_width, resized_height = int(round(width * ratio)), int(round(height * ratio))
        left, top = (image_size - resized_width) // 2, (image_size - resized_height) // 2
        original_box = np.asarray([[float(row[name]) for name in ("gt_x", "gt_y", "gt_w", "gt_h")]], dtype=np.float32)
        teacher_box = transform_boxes_to_letterbox(original_box, width, height, ratio, left, top, image_size)[0]
        entry["objects"].append({
            "gt_index": gt_index,
            "fine_class": int(row["fine_class"]),
            # Geometry is stored in the deterministic teacher-cache letterbox
            # coordinate system. This survives duplicate-label removal that
            # can shift gt_index between diagnostics and the cached teacher.
            "box_xywhn": [float(value) for value in teacher_box],
            "coarse_group": group,
            "size": row["size"],
            "crowded": row["crowded"].lower() == "true",
            "edge": row["edge"].lower() == "true",
            "error_type": error_type,
        })
    for entry in images.values():
        entry["objects"].sort(key=lambda item: item["gt_index"])

    digest = hashlib.sha256(instances_csv.read_bytes()).hexdigest()
    return {
        "format": 1,
        "split": "train",
        "source_model": model_name,
        "source_csv": str(instances_csv.resolve()),
        "source_csv_sha256": digest,
        "instances": len(rows),
        # Keys are paths relative to images/train, not absolute-path hashes, so
        # the artifact remains valid after cloning/moving the dataset.
        "images": images,
        "error_counts": dict(sorted(errors.items())),
        "group_counts": dict(sorted(groups.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a leakage-safe TRAIN hard-example manifest for targeted KD.")
    parser.add_argument("--config", default="configs/direction1.yaml")
    parser.add_argument("--instances", required=True, help="TRAIN <model>_instances.csv from weak_group_diagnostics.")
    parser.add_argument("--model", default="C0")
    parser.add_argument("--split", choices=("train",), default="train", help="Only TRAIN is accepted by design.")
    parser.add_argument("--out", default="reports/weak_group_train/hard_examples.json")
    args = parser.parse_args()
    manifest = build_manifest(load_config(args.config), Path(args.instances), args.model)
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {destination}: {manifest['instances']} TRAIN objects in {len(manifest['images'])} images")
    print(f"errors={manifest['error_counts']}")


if __name__ == "__main__":
    main()
