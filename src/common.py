from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


FINE_TO_COARSE = {**{i: 0 for i in range(0, 4)}, **{i: 1 for i in range(4, 24)}, 24: 2}
COARSE_NAMES = ("ship", "aircraft", "vehicle")


def load_config(path: str | Path) -> dict[str, Any]:
    import yaml
    with Path(path).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def resolve_data_yaml(config: dict[str, Any]) -> dict[str, Any]:
    import yaml
    p = Path(config["paths"]["data_yaml"])
    with p.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    root = Path(data["path"])
    data["path"] = str(root)
    return data


def split_image_dir(data: dict[str, Any], split: str) -> Path:
    return Path(data["path"]) / data[split]


def image_to_label_path(image_path: Path, image_dir: Path, label_dir: Path) -> Path:
    return label_dir / image_path.relative_to(image_dir).with_suffix(".txt")


def stable_image_key(image_path: str | Path) -> str:
    return hashlib.sha1(str(Path(image_path).resolve()).lower().encode("utf-8")).hexdigest()


def read_yolo_labels(label_path: Path, *, deduplicate: bool = False) -> tuple[np.ndarray, np.ndarray]:
    if not label_path.exists():
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 4), dtype=np.float32)
    rows: list[list[float]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = line.strip().split()
        if len(values) >= 5:
            rows.append([float(v) for v in values[:5]])
    if not rows:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 4), dtype=np.float32)
    array = np.asarray(rows, dtype=np.float32)
    if deduplicate:
        # Ultralytics removes exact duplicate labels while scanning a dataset.
        # Keep this opt-in so existing teacher caches retain their original
        # schema, while standalone competition evaluation can use the same GT
        # population as native validation.
        _, first_indices = np.unique(array, axis=0, return_index=True)
        if len(first_indices) != len(array):
            array = array[np.sort(first_indices)]
    return array[:, 0].astype(np.int64), array[:, 1:5]


def xywhn_to_xyxy(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    out = boxes.astype(np.float32).copy()
    out[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2.0) * width
    out[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2.0) * height
    out[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2.0) * width
    out[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2.0) * height
    return out


def xyxy_to_xywhn(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    out = boxes.astype(np.float32).copy()
    out[:, 0] = ((boxes[:, 0] + boxes[:, 2]) / 2.0) / width
    out[:, 1] = ((boxes[:, 1] + boxes[:, 3]) / 2.0) / height
    out[:, 2] = (boxes[:, 2] - boxes[:, 0]) / width
    out[:, 3] = (boxes[:, 3] - boxes[:, 1]) / height
    return out


def letterbox(image: np.ndarray, size: int = 640, color: int = 114) -> tuple[np.ndarray, float, int, int]:
    """Deterministic square letterbox compatible with the KD cache phase."""
    import cv2
    h, w = image.shape[:2]
    ratio = min(size / h, size / w)
    nw, nh = int(round(w * ratio)), int(round(h * ratio))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_w, pad_h = size - nw, size - nh
    left, top = pad_w // 2, pad_h // 2
    out = cv2.copyMakeBorder(resized, top, pad_h - top, left, pad_w - left, cv2.BORDER_CONSTANT, value=(color, color, color))
    return out, ratio, left, top


def transform_boxes_to_letterbox(boxes: np.ndarray, original_w: int, original_h: int, ratio: float, left: int, top: int, size: int) -> np.ndarray:
    xyxy = xywhn_to_xyxy(boxes, original_w, original_h)
    if len(xyxy) == 0:
        return xyxy_to_xywhn(xyxy, size, size)
    xyxy[:, [0, 2]] = xyxy[:, [0, 2]] * ratio + left
    xyxy[:, [1, 3]] = xyxy[:, [1, 3]] * ratio + top
    return xyxy_to_xywhn(xyxy, size, size)


def json_dump(value: Any, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
