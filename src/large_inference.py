from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .common import FINE_TO_COARSE, json_dump, load_config
from .competition_eval import box_iou_one_to_many


@dataclass
class TileDetection:
    box: np.ndarray
    score: float
    fine_class: int


def tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if tile_size <= 0 or stride <= 0:
        raise ValueError("tile_size and stride must be positive")
    if length <= tile_size:
        return [0]
    last = length - tile_size
    starts = list(range(0, last + 1, stride))
    if starts[-1] != last:
        starts.append(last)
    return starts


def global_nms(detections: list[TileDetection], iou_threshold: float, merge_mode: str) -> list[TileDetection]:
    """Remove cross-window duplicates after all boxes are mapped into full-image coordinates."""
    if merge_mode not in {"fine", "coarse"}:
        raise ValueError("merge_mode must be fine or coarse")
    kept: list[TileDetection] = []
    groups: dict[int, list[TileDetection]] = {}
    for detection in detections:
        key = detection.fine_class if merge_mode == "fine" else FINE_TO_COARSE[detection.fine_class]
        groups.setdefault(key, []).append(detection)
    for group in groups.values():
        pending = sorted(group, key=lambda item: item.score, reverse=True)
        while pending:
            leader = pending.pop(0)
            kept.append(leader)
            if not pending:
                continue
            boxes = np.stack([item.box for item in pending])
            ious = box_iou_one_to_many(leader.box, boxes)
            pending = [item for item, iou in zip(pending, ious) if float(iou) < iou_threshold]
    return sorted(kept, key=lambda item: item.score, reverse=True)


def list_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted(path for path in source.rglob("*") if path.suffix.lower() in suffixes)


def inference_for_image(model, image_path: Path, args) -> dict:
    import cv2
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot read {image_path}")
    height, width = image.shape[:2]
    xs, ys = tile_starts(width, args.tile_size, args.tile_stride), tile_starts(height, args.tile_size, args.tile_stride)
    tile_records: list[tuple[np.ndarray, int, int]] = []
    for y in ys:
        for x in xs:
            tile = image[y:min(y + args.tile_size, height), x:min(x + args.tile_size, width)]
            if tile.shape[:2] != (args.tile_size, args.tile_size):
                tile = cv2.copyMakeBorder(tile, 0, args.tile_size - tile.shape[0], 0, args.tile_size - tile.shape[1], cv2.BORDER_CONSTANT, value=(114, 114, 114))
            tile_records.append((tile, x, y))
    started = time.perf_counter()
    detections: list[TileDetection] = []
    for offset in range(0, len(tile_records), args.batch):
        chunk = tile_records[offset:offset + args.batch]
        results = model.predict([record[0] for record in chunk], imgsz=args.imgsz, conf=args.conf, iou=args.tile_nms_iou, device=args.device, verbose=False)
        for result, (_, x, y) in zip(results, chunk):
            if result.boxes is None or not len(result.boxes):
                continue
            boxes = result.boxes.xyxy.detach().cpu().numpy()
            scores = result.boxes.conf.detach().cpu().numpy()
            classes = result.boxes.cls.detach().cpu().numpy().astype(int)
            for box, score, fine_class in zip(boxes, scores, classes):
                full = box.astype(np.float32) + np.array([x, y, x, y], dtype=np.float32)
                full[[0, 2]] = np.clip(full[[0, 2]], 0, width)
                full[[1, 3]] = np.clip(full[[1, 3]], 0, height)
                if full[2] > full[0] and full[3] > full[1]:
                    detections.append(TileDetection(full, float(score), int(fine_class)))
    merged = global_nms(detections, args.global_nms_iou, args.merge_mode)
    elapsed = time.perf_counter() - started
    return {
        "image": str(image_path), "width": width, "height": height, "tiles": len(tile_records), "seconds_excluding_read": elapsed,
        "raw_detections": len(detections), "merged_detections": len(merged),
        "detections": [{"xyxy": item.box.round(3).tolist(), "score": item.score, "fine_class": item.fine_class} for item in merged],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch tiled inference for 10k remote-sensing images; outputs full-image coordinates.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True, help="One large image or a directory of images.")
    parser.add_argument("--out", default="reports/large_inference.json")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--tile-size", type=int, default=None)
    parser.add_argument("--tile-stride", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--conf", type=float, default=0.01)
    parser.add_argument("--tile-nms-iou", type=float, default=None)
    parser.add_argument("--global-nms-iou", type=float, default=None)
    parser.add_argument("--merge-mode", choices=("fine", "coarse"), default=None)
    parser.add_argument("--device", default=0)
    args = parser.parse_args()
    cfg = load_config(args.config)
    large = {"tile_size": 800, "tile_stride": 480, "infer_batch": 16, "tile_nms_iou": 0.5, "global_nms_iou": 0.65, "merge_mode": "coarse"} | cfg.get("large_image", {})
    args.tile_size = args.tile_size or large["tile_size"]; args.tile_stride = args.tile_stride or large["tile_stride"]
    args.batch = args.batch or large["infer_batch"]; args.imgsz = args.imgsz or cfg["dataset"]["image_size"]
    args.tile_nms_iou = args.tile_nms_iou if args.tile_nms_iou is not None else large["tile_nms_iou"]
    args.global_nms_iou = args.global_nms_iou if args.global_nms_iou is not None else large["global_nms_iou"]
    args.merge_mode = args.merge_mode or large["merge_mode"]
    from ultralytics import YOLO
    model = YOLO(args.model)
    images = list_images(Path(args.source))
    if not images:
        raise FileNotFoundError(f"No images found under {args.source}")
    records = [inference_for_image(model, image, args) for image in images]
    result = {"model": args.model, "tile_size": args.tile_size, "tile_stride": args.tile_stride, "imgsz": args.imgsz, "batch": args.batch, "tile_nms_iou": args.tile_nms_iou, "global_nms_iou": args.global_nms_iou, "merge_mode": args.merge_mode, "images": records}
    json_dump(result, args.out)
    print(f"saved {args.out}; images={len(records)}; mean_seconds={np.mean([record['seconds_excluding_read'] for record in records]):.3f}")


if __name__ == "__main__":
    main()
