from __future__ import annotations

import argparse
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

from .common import COARSE_NAMES, FINE_TO_COARSE, json_dump, read_yolo_labels, xywhn_to_xyxy
from .artifact_paths import dataset_root, manifests_dir
from .dataset_d0 import file_sha256, image_files
from .dataset_registry import fingerprint_scene811, load_manifest, manifest_split_fingerprint, class_mapping_fingerprint
from .weak_group_diagnostics import crowded_flags, edge_flags, size_bucket


def label_duplicates(lines: list[str]) -> list[dict]:
    seen: dict[str, list[int]] = defaultdict(list)
    for index, line in enumerate(lines, start=1):
        text = line.strip()
        if text:
            seen[text].append(index)
    return [{"row": text, "line_numbers": positions} for text, positions in seen.items() if len(positions) > 1]


def valid_label_line(fields: list[str]) -> bool:
    if len(fields) < 5:
        return False
    cls, x, y, w, h = float(fields[0]), float(fields[1]), float(fields[2]), float(fields[3]), float(fields[4])
    return (
        0 <= int(cls) < 25
        and np.isfinite([x, y, w, h]).all()
        and 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0
        and 0.0 < w <= 1.0 and 0.0 < h <= 1.0
    )


def audit_scene811(
    dataset_root: Path,
    manifest_path: Path,
    *,
    hash_images: bool = True,
    image_size: int = 640,
    names: dict | None = None,
) -> dict:
    root = Path(dataset_root)
    rows = load_manifest(manifest_path)
    by_image: dict[str, dict] = {}
    scene_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_image[(row["split"], row["image"])] = row
        scene_splits[row["scene_id"]].add(row["split"])

    summaries: dict[str, dict] = {}
    duplicate_files: dict[str, dict] = {}
    empty_files: list[str] = []
    unmanifested: list[str] = []
    orphan_labels: list[str] = []
    all_relative: set[str] = set()
    for split in ("train", "val", "test"):
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        fine = Counter(); coarse = Counter(); sizes = Counter()
        scene_count = 0
        instances = backgrounds = dup_rows = invalid_rows = empty = missing_images = missing_labels = 0
        crowded = edge = 0
        seen_scenes = set()
        manifest_images = {(row["split"], row["image"]) for row in rows if row["split"] == split}
        for image_path in image_files(image_dir):
            relative = image_path.relative_to(root).as_posix()
            all_relative.add(relative)
            label_path = root / "labels" / split / image_path.relative_to(image_dir).with_suffix(".txt")
            key = (split, image_path.relative_to(image_dir).as_posix())
            entry = by_image.get(key)
            if entry is None:
                unmanifested.append(relative)
            scene_id = entry["scene_id"] if entry else "UNMANIFESTED"
            seen_scenes.add(scene_id)
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            height, width = image.shape[:2]
            lines = label_path.read_text(encoding="utf-8").splitlines() if label_path.exists() else []
            raw_classes, raw_boxes = read_yolo_labels(label_path, deduplicate=False)
            classes, boxes = read_yolo_labels(label_path, deduplicate=True)
            dup_rows += len(raw_classes) - len(classes)
            if dup_lines := label_duplicates(lines):
                duplicate_files[relative] = {
                    "duplicate_count": len(dup_lines),
                    "removed_rows": sum(len(item["line_numbers"]) - 1 for item in dup_lines),
                    "examples": dup_lines[:5],
                }
            valid = np.asarray([valid_label_line(line.split()) for line in lines], dtype=bool) if lines else np.zeros(0, dtype=bool)
            invalid_rows += int((~valid).sum())
            if len(valid):
                classes, boxes = classes[valid], boxes[valid]
            if not len(classes):
                backgrounds += 1
                if not lines:
                    empty += 1
                    empty_files.append(relative)
            xyxy = xywhn_to_xyxy(boxes, width, height)
            crowded += int(crowded_flags(xyxy, classes).sum())
            edge += int(edge_flags(xyxy, width, height).sum())
            for cls, box in zip(classes, xyxy):
                class_id = int(cls)
                fine[class_id] += 1
                coarse[COARSE_NAMES[FINE_TO_COARSE[class_id]]] += 1
                sizes[size_bucket(box, width, height, image_size)] += 1
            instances += len(classes)
        for row in rows:
            if row["split"] != split:
                continue
            if not (root / "images" / split / row["image"]).is_file():
                missing_images += 1
            if not (root / "labels" / split / row["label"]).is_file():
                missing_labels += 1
        for label_path in (label_dir.rglob("*") if label_dir.is_dir() else []):
            if label_path.is_file() and label_path.suffix == ".txt":
                rel_label = label_path.relative_to(label_dir).as_posix()
                if rel_label not in {row["label"] for row in rows if row["split"] == split}:
                    orphan_labels.append(label_path.relative_to(root).as_posix())
        summaries[split] = {
            "images": len(manifest_images), "instances": instances, "background_images": backgrounds,
            "empty_label_files": empty, "duplicate_label_rows": dup_rows,
            "invalid_label_rows": invalid_rows, "missing_images": missing_images, "missing_labels": missing_labels,
            "unmanifested_images": len([u for u in unmanifested if u.startswith(f"images/{split}/")]),
            "orphan_labels": len([o for o in orphan_labels if o.startswith(f"labels/{split}/")]),
            "scene_groups": len(seen_scenes), "per_class": {str(i): fine[i] for i in range(25)},
            "per_group": {name: coarse[name] for name in COARSE_NAMES},
            "per_size": {name: sizes[name] for name in ("small", "medium", "large")},
            "crowded_instances": crowded, "edge_instances": edge,
        }

    cross_split = {scene: sorted(parts) for scene, parts in scene_splits.items() if len(parts) > 1}
    if names is None:
        with (root / "dataset.yaml").open("r", encoding="utf-8") as stream:
            names = yaml.safe_load(stream)["names"]
    fingerprint = fingerprint_scene811(
        root, manifest_path, names,
        hash_images=hash_images,
        label_fix_manifest_path=manifests_dir() / "label_fix_manifest.json",
        background_confirmation="pending_human_review",
        non_l_scene_audit_path=manifests_dir() / "non_l_scene_audit.json",
    )
    report = {
        "format": 1,
        "kind": "scene811_v1_audit",
        "read_only": True,
        "dataset_root": str(root.resolve()),
        "manifest": str(Path(manifest_path).resolve()),
        "manifest_sha256": file_sha256(Path(manifest_path)),
        "splits": summaries,
        "cross_split_scenes": {"count": len(cross_split), "examples": sorted(cross_split.items())[:20]},
        "unmanifested_images": unmanifested,
        "orphan_labels": orphan_labels,
        "duplicate_row_files": duplicate_files,
        "empty_label_files": empty_files,
        "fingerprint": {
            "dataset_fingerprint": fingerprint["dataset_fingerprint"],
            "class_mapping_fingerprint": fingerprint["class_mapping_fingerprint"],
            "split_fingerprint": fingerprint["split_fingerprint"],
            "image_count": fingerprint["image_count"],
        },
        "interpretation": {
            "scene": "split_manifest.csv scene_id is authoritative; filename heuristics are not used for grouping",
            "duplicates": "exact duplicate label rows are removed by Ultralytics; a fix manifest is recorded before fingerprinting",
            "empty_labels": "empty label files are legal background images only after human confirmation",
            "non_l_scenes": "non-L-number scene ids are currently single-image families; near-duplicate audit is separate (audit_scene_groups)",
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only structural audit of Scene811 v1 (manifest-driven).")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--no-image-hash", action="store_true")
    parser.add_argument("--image-size", type=int, default=640)
    args = parser.parse_args()
    root = Path(args.dataset_root) if args.dataset_root else dataset_root()
    manifest = Path(args.manifest) if args.manifest else root / "split_manifest.csv"
    report = audit_scene811(root, manifest, hash_images=not args.no_image_hash, image_size=args.image_size)
    out = Path(args.out) if args.out else manifests_dir() / "audit_scene811.json"
    json_dump(report, out)
    print(f"saved {out}; read-only audit of {report['fingerprint']['image_count']} images")
    for split, summary in report["splits"].items():
        print(split, {key: summary[key] for key in (
            "images", "instances", "duplicate_label_rows", "invalid_label_rows", "empty_label_files",
            "missing_images", "missing_labels", "unmanifested_images", "orphan_labels",
        )})
    print("cross_split_scenes:", report["cross_split_scenes"]["count"])
    print("dataset_fingerprint:", report["fingerprint"]["dataset_fingerprint"])


if __name__ == "__main__":
    main()
