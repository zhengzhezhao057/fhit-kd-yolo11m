from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .artifact_paths import replay_dir
from .common import json_dump, read_yolo_labels, xywhn_to_xyxy
from .dataset_registry import load_manifest

CONTEXTS = (3, 7)
EDGE_MARGIN = 4


def expand_box(box_xyxy: np.ndarray, width: int, height: int, context: int) -> np.ndarray:
    cx = (box_xyxy[0] + box_xyxy[2]) / 2.0
    cy = (box_xyxy[1] + box_xyxy[3]) / 2.0
    bw = box_xyxy[2] - box_xyxy[0]
    bh = box_xyxy[3] - box_xyxy[1]
    nw = bw * context
    nh = bh * context
    x0 = np.clip(cx - nw / 2.0, 0.0, float(width))
    y0 = np.clip(cy - nh / 2.0, 0.0, float(height))
    x1 = np.clip(cx + nw / 2.0, 0.0, float(width))
    y1 = np.clip(cy + nh / 2.0, 0.0, float(height))
    return np.asarray([x0, y0, x1, y1], dtype=np.float32)


def intersects_gt(crop: np.ndarray, gt_boxes: np.ndarray) -> bool:
    if len(gt_boxes) == 0:
        return False
    cx0, cy0, cx1, cy1 = crop
    gx0, gy0, gx1, gy1 = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2], gt_boxes[:, 3]
    inter_w = np.maximum(0.0, np.minimum(cx1, gx1) - np.maximum(cx0, gx0))
    inter_h = np.maximum(0.0, np.minimum(cy1, gy1) - np.maximum(cy0, gy0))
    return bool(np.any((inter_w > 0) & (inter_h > 0)))


def plan_hard_negative_crops(
    dataset_root: Path,
    manifest_path: Path,
    background_manifest: dict,
    *,
    contexts: tuple[int, ...] = CONTEXTS,
    edge_margin: int = EDGE_MARGIN,
) -> dict:
    if background_manifest.get("format") != 1 or background_manifest.get("split") != "train":
        raise RuntimeError("Background source must be a format=1 TRAIN manifest.")
    rows = {row["image"]: row for row in load_manifest(manifest_path) if row["split"] == "train"}
    images: dict[str, dict] = {}
    for relative, entry in background_manifest.get("images", {}).items():
        row = rows.get(relative)
        if row is None:
            raise RuntimeError(f"Background image missing from TRAIN manifest: {relative}")
        image_path = dataset_root / "images" / "train" / relative
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Unreadable background image: {image_path}")
        height, width = image.shape[:2]
        label_path = dataset_root / "labels" / "train" / Path(relative).with_suffix(".txt")
        _, gt = read_yolo_labels(label_path, deduplicate=True)
        gt_xyxy = xywhn_to_xyxy(gt, width, height) if len(gt) else np.zeros((0, 4), dtype=np.float32)
        planned = []
        for box_xywhn in entry.get("boxes", []):
            box_xyxy = xywhn_to_xyxy(np.asarray([box_xywhn], dtype=np.float32), width, height)[0]
            for context in contexts:
                crop = expand_box(box_xyxy, width, height, context)
                if intersects_gt(crop, gt_xyxy):
                    continue
                if crop[0] < edge_margin or crop[1] < edge_margin or crop[2] > width - edge_margin or crop[3] > height - edge_margin:
                    continue
                planned.append({
                    "context": int(context),
                    "crop_xyxy": [float(value) for value in crop],
                    "crop_xywhn": [float(value) for value in xyxy_to_xywhn(crop, width, height)],
                    "empty_label": True,
                    "intersects_gt": False,
                    "edge_safe": True,
                })
        if planned:
            images[relative] = {"image_size": [int(width), int(height)], "crops": planned}
    return {
        "format": 1,
        "kind": "scene811_hard_negative_crops",
        "dataset_id": "scene811_v1",
        "split": "train",
        "images": dict(sorted(images.items())),
        "image_count": len(images),
    }


def xyxy_to_xywhn(box: np.ndarray, width: int, height: int) -> np.ndarray:
    out = np.zeros(4, dtype=np.float32)
    out[0] = (box[0] + box[2]) / 2.0 / width
    out[1] = (box[1] + box[3]) / 2.0 / height
    out[2] = (box[2] - box[0]) / width
    out[3] = (box[3] - box[1]) / height
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan 3x/7x empty-label context crops from confirmed vehicle backgrounds.")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--background-manifest", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    from .artifact_paths import dataset_root as resolve_root
    root = Path(args.dataset_root) if args.dataset_root else resolve_root()
    manifest = Path(args.manifest) if args.manifest else root / "split_manifest.csv"
    background = json.loads(Path(args.background_manifest).read_text(encoding="utf-8"))
    report = plan_hard_negative_crops(root, manifest, background)
    out = Path(args.out) if args.out else replay_dir() / "hard_negative_crops.json"
    json_dump(report, out)
    total = sum(len(entry["crops"]) for entry in report["images"].values())
    print(f"saved {out}; images={report['image_count']} crops={total}")


if __name__ == "__main__":
    main()
