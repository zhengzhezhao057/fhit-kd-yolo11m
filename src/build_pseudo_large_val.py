from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

from .common import image_to_label_path, load_config, read_yolo_labels, resolve_data_yaml, split_image_dir, xywhn_to_xyxy


def main() -> None:
    parser = argparse.ArgumentParser(description="Build labeled pseudo-large validation canvases to tune tiled inference before real large labels exist.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--out", default="pseudo_large/val")
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--canvas", type=int, default=3200)
    parser.add_argument("--source-size", type=int, default=800, help="Use native square source images of this size to preserve object scale.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    cfg = load_config(args.config); data = resolve_data_yaml(cfg)
    image_dir = split_image_dir(data, "val"); label_dir = Path(data["path"]) / "labels" / "val"
    sources = []
    for path in image_dir.rglob("*"):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is not None and image.shape[:2] == (args.source_size, args.source_size):
            sources.append((path, image))
    if not sources:
        raise RuntimeError(f"No {args.source_size}x{args.source_size} validation images found")
    if args.canvas % args.source_size:
        raise ValueError("canvas must be divisible by source-size")
    root = Path(args.out); image_out = root / "images"; label_out = root / "labels"; image_out.mkdir(parents=True, exist_ok=True); label_out.mkdir(parents=True, exist_ok=True)
    cells = args.canvas // args.source_size; rng = random.Random(args.seed)
    for canvas_index in range(args.count):
        canvas = np.full((args.canvas, args.canvas, 3), 114, dtype=np.uint8); labels: list[tuple[int, np.ndarray]] = []
        for row in range(cells):
            for col in range(cells):
                path, image = rng.choice(sources); x, y = col * args.source_size, row * args.source_size
                canvas[y:y + args.source_size, x:x + args.source_size] = image
                classes, boxes_n = read_yolo_labels(image_to_label_path(path, image_dir, label_dir))
                boxes = xywhn_to_xyxy(boxes_n, args.source_size, args.source_size)
                for fine_class, box in zip(classes, boxes):
                    box[[0, 2]] += x; box[[1, 3]] += y
                    labels.append((int(fine_class), box))
        stem = f"pseudo_{canvas_index:04d}"; cv2.imwrite(str(image_out / f"{stem}.jpg"), canvas)
        rows = []
        for fine_class, (x1, y1, x2, y2) in labels:
            rows.append(f"{fine_class} {(x1 + x2) / 2 / args.canvas:.8f} {(y1 + y2) / 2 / args.canvas:.8f} {(x2 - x1) / args.canvas:.8f} {(y2 - y1) / args.canvas:.8f}")
        (label_out / f"{stem}.txt").write_text("\n".join(rows), encoding="utf-8")
    print(f"wrote {args.count} pseudo-large {args.canvas}x{args.canvas} canvases to {root}")


if __name__ == "__main__":
    main()
