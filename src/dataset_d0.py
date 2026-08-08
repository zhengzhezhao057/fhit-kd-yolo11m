from __future__ import annotations

import hashlib
import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from .common import COARSE_NAMES, FINE_TO_COARSE, image_to_label_path, read_yolo_labels, xywhn_to_xyxy
from .weak_group_diagnostics import crowded_flags, edge_flags, size_bucket


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def image_files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def source_identities(image_path: str | Path) -> tuple[str, str]:
    """Return crop-parent scene and broader satellite-product identities."""
    stem = Path(image_path).stem
    scene = re.sub(r"_crop\d+$", "", stem, flags=re.IGNORECASE)
    product = re.sub(r"-CCD\d+_\d+$", "", scene, flags=re.IGNORECASE)
    product = re.sub(r"-PAN\d+$", "", product, flags=re.IGNORECASE)
    return scene, product


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory_fingerprint(rows: list[dict]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: item["relative_image"]):
        for key in ("relative_image", "image_sha256", "label_sha256"):
            digest.update(str(row[key]).encode("utf-8")); digest.update(b"\0")
    return digest.hexdigest()


def scan_split(
    dataset_root: Path,
    image_dir: Path,
    label_dir: Path,
    image_size: int,
    *,
    hash_images: bool,
) -> tuple[dict, list[dict]]:
    fine = Counter(); coarse = Counter(); sizes = Counter()
    scenes = Counter(); products = Counter()
    inventory: list[dict] = []
    backgrounds = corrupt = duplicates = invalid = edge_count = crowded_count = 0
    instance_count = 0
    for image_path in image_files(image_dir):
        relative = image_path.relative_to(dataset_root).as_posix()
        label_path = image_to_label_path(image_path, image_dir, label_dir)
        image = cv2.imread(str(image_path))
        if image is None:
            corrupt += 1
            continue
        height, width = image.shape[:2]
        raw_classes, raw_boxes = read_yolo_labels(label_path, deduplicate=False)
        classes, boxes = read_yolo_labels(label_path, deduplicate=True)
        duplicates += len(raw_classes) - len(classes)
        valid_mask = np.asarray([
            0 <= int(cls) < 25
            and np.isfinite(box).all()
            and 0.0 <= float(box[0]) <= 1.0
            and 0.0 <= float(box[1]) <= 1.0
            and 0.0 < float(box[2]) <= 1.0
            and 0.0 < float(box[3]) <= 1.0
            for cls, box in zip(classes, boxes)
        ], dtype=bool)
        invalid += int((~valid_mask).sum())
        classes, boxes = classes[valid_mask], boxes[valid_mask]
        if not len(classes):
            backgrounds += 1
        xyxy = xywhn_to_xyxy(boxes, width, height)
        crowded = crowded_flags(xyxy, classes)
        edges = edge_flags(xyxy, width, height)
        crowded_count += int(crowded.sum()); edge_count += int(edges.sum())
        for cls, box in zip(classes, xyxy):
            class_id = int(cls)
            fine[class_id] += 1
            coarse[COARSE_NAMES[FINE_TO_COARSE[class_id]]] += 1
            sizes[size_bucket(box, width, height, image_size)] += 1
        instance_count += len(classes)
        scene, product = source_identities(image_path)
        scenes[scene] += 1; products[product] += 1
        inventory.append({
            "relative_image": relative,
            "image": str(image_path.resolve()),
            "label": str(label_path.resolve()),
            "image_sha256": (
                file_sha256(image_path)
                if hash_images else f"unhashed:{relative}:{image_path.stat().st_size}"
            ),
            "label_sha256": file_sha256(label_path) if label_path.exists() else hashlib.sha256(b"").hexdigest(),
            "scene": scene,
            "product": product,
            "classes": [int(value) for value in classes],
        })
    summary = {
        "images": len(inventory), "instances": instance_count, "background_images": backgrounds,
        "corrupt_images": corrupt, "duplicate_labels": duplicates, "invalid_labels": invalid,
        "edge_instances": edge_count, "crowded_instances": crowded_count,
        "per_class": {str(index): fine[index] for index in range(25)},
        "per_group": {name: coarse[name] for name in COARSE_NAMES},
        "per_size": {name: sizes[name] for name in ("small", "medium", "large")},
        "scene_groups": len(scenes), "product_groups": len(products),
        "maximum_crops_per_scene": max(scenes.values(), default=0),
        "maximum_crops_per_product": max(products.values(), default=0),
    }
    return summary, inventory


def split_overlap(left: list[dict], right: list[dict]) -> dict:
    result = {}
    for key in ("image_sha256", "scene", "product"):
        left_groups = Counter(row[key] for row in left)
        right_groups = Counter(row[key] for row in right)
        overlap = sorted(set(left_groups).intersection(right_groups))
        result[key] = {
            "shared_groups": len(overlap),
            "left_images": sum(left_groups[value] for value in overlap),
            "right_images": sum(right_groups[value] for value in overlap),
            "examples": overlap[:20],
        }
    return result
