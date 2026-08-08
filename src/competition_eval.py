from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .common import COARSE_NAMES, FINE_TO_COARSE, image_to_label_path, json_dump, load_config, read_yolo_labels, resolve_data_yaml, split_image_dir, xywhn_to_xyxy


@dataclass
class Detection:
    box: np.ndarray
    score: float
    fine_class: int

    @property
    def coarse_class(self) -> int:
        return FINE_TO_COARSE[self.fine_class]


def box_iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.float32)
    x1 = np.maximum(box[0], boxes[:, 0]); y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2]); y2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = np.maximum(0, box[2] - box[0]) * np.maximum(0, box[3] - box[1])
    area_b = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    return intersection / np.maximum(area_a + area_b - intersection, 1e-9)


def score_image(predictions: list[Detection], gt_boxes: np.ndarray, gt_fine_classes: np.ndarray, class_aware: bool = False) -> tuple[int, int, int]:
    """Competition-order matching: sort by confidence and match each GT at most once."""
    matched = np.zeros(len(gt_boxes), dtype=bool)
    tp = fp = 0
    for pred in sorted(predictions, key=lambda item: item.score, reverse=True):
        eligible = ~matched
        if class_aware:
            # dtype must remain bool even when this image/group has zero GTs;
            # np.array([]) defaults to float and cannot be combined with &=.
            class_matches = np.fromiter(
                (FINE_TO_COARSE[int(c)] == pred.coarse_class for c in gt_fine_classes),
                dtype=bool,
                count=len(gt_fine_classes),
            )
            eligible &= class_matches
        indices = np.flatnonzero(eligible)
        if not len(indices):
            fp += 1; continue
        ious = box_iou_one_to_many(pred.box, gt_boxes[indices])
        best_pos = int(np.argmax(ious))
        gt_index = int(indices[best_pos])
        threshold = 0.35 if FINE_TO_COARSE[int(gt_fine_classes[gt_index])] == 2 else 0.50
        if float(ious[best_pos]) >= threshold:
            matched[gt_index] = True; tp += 1
        else:
            fp += 1
    return tp, fp, int((~matched).sum())


def metric_dict(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    recall = tp / max(tp + fn, 1)
    false_alarm = fp / max(fp + tp, 1)
    precision = 1.0 - false_alarm
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "false_alarm_rate": false_alarm,
        "f1": f1,
    }


def confidence_range(start: float, stop: float, step: float) -> list[float]:
    """Return an inclusive, stable confidence grid for dense operating-point analysis."""
    if not (0.0 <= start <= stop <= 1.0) or step <= 0.0:
        raise ValueError("confidence range requires 0 <= START <= STOP <= 1 and STEP > 0")
    count = int(np.floor((stop - start) / step + 1e-9))
    values = [round(start + index * step, 10) for index in range(count + 1)]
    if values[-1] < stop - 1e-9:
        values.append(round(stop, 10))
    return values


def select_operating_points(candidates: list[dict], min_recall: float = 0.85, max_false_alarm: float = 0.20) -> dict[str, dict]:
    """Select distinct safety-gate and balanced competition operating points."""
    if not candidates:
        raise ValueError("at least one evaluation candidate is required")
    recall_valid = [item for item in candidates if item["overall"]["recall"] >= min_recall]
    gate_pool = recall_valid or candidates
    gate_min_fdr = min(
        gate_pool,
        key=lambda item: (
            item["overall"]["false_alarm_rate"],
            -item["overall"]["recall"],
            item["confidence"],
        ),
    )
    fully_valid = [
        item
        for item in candidates
        if item["overall"]["recall"] >= min_recall
        and item["overall"]["false_alarm_rate"] <= max_false_alarm
    ]
    balanced_pool = fully_valid or recall_valid or candidates
    best_f1 = max(
        balanced_pool,
        key=lambda item: (
            item["overall"]["f1"],
            item["overall"]["recall"],
            -item["overall"]["false_alarm_rate"],
        ),
    )
    false_alarm_valid = [
        item for item in candidates if item["overall"]["false_alarm_rate"] <= max_false_alarm
    ]
    recall_pool = false_alarm_valid or candidates
    max_recall_under_fdr = max(
        recall_pool,
        key=lambda item: (
            item["overall"]["recall"],
            -item["overall"]["false_alarm_rate"],
        ),
    )
    return {
        "gate_min_fdr": copy.deepcopy(gate_min_fdr),
        "best_f1": copy.deepcopy(best_f1),
        "max_recall_under_fdr": copy.deepcopy(max_recall_under_fdr),
    }


def collect_model_predictions(model, image_paths: list[Path], confidence: float, image_size: int, nms_iou: float) -> dict[Path, list[Detection]]:
    predictions: dict[Path, list[Detection]] = {}
    for image_path in image_paths:
        result = model.predict(str(image_path), imgsz=image_size, conf=confidence, iou=nms_iou, verbose=False)[0]
        detections: list[Detection] = []
        if result.boxes is not None and len(result.boxes):
            boxes = result.boxes.xyxy.detach().cpu().numpy()
            scores = result.boxes.conf.detach().cpu().numpy()
            labels = result.boxes.cls.detach().cpu().numpy().astype(int)
            detections = [Detection(box=box.astype(np.float32), score=float(score), fine_class=int(label)) for box, score, label in zip(boxes, scores, labels)]
        predictions[image_path] = detections
    return predictions


def evaluate_cached_predictions(predicted: dict[Path, list[Detection]], data: dict, split: str, confidence: float, class_aware: bool) -> dict:
    image_dir = split_image_dir(data, split)
    label_dir = Path(data["path"]) / "labels" / split
    image_paths = sorted([p for p in image_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}])
    overall = np.zeros(3, dtype=np.int64)
    per_group = {name: np.zeros(3, dtype=np.int64) for name in COARSE_NAMES}
    for image_path in image_paths:
        image = __import__("cv2").imread(str(image_path))
        height, width = image.shape[:2]
        classes, boxes_n = read_yolo_labels(
            image_to_label_path(image_path, image_dir, label_dir),
            deduplicate=True,
        )
        boxes = xywhn_to_xyxy(boxes_n, width, height)
        selected = [p for p in predicted[image_path] if p.score >= confidence]
        tp, fp, fn = score_image(selected, boxes, classes, class_aware)
        overall += np.asarray([tp, fp, fn])
        for group_idx, group_name in enumerate(COARSE_NAMES):
            gt_mask = np.fromiter(
                (FINE_TO_COARSE[int(c)] == group_idx for c in classes),
                dtype=bool,
                count=len(classes),
            )
            group_predictions = [p for p in selected if p.coarse_class == group_idx]
            a, b, c = score_image(group_predictions, boxes[gt_mask], classes[gt_mask], class_aware=True)
            per_group[group_name] += np.asarray([a, b, c])
    report = {"split": split, "confidence": confidence, "class_aware_matching": class_aware, "overall": metric_dict(*overall.tolist()), "per_group": {name: metric_dict(*values.tolist()) for name, values in per_group.items()}}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a YOLO checkpoint with the competition TP/FP/FN rules.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", default="val", choices=("val", "test"))
    parser.add_argument("--out", default="reports/baseline_val.json")
    parser.add_argument("--confidence", type=float, action="append", default=None, help="Evaluate this operating point. Repeat for multiple values; default uses evaluation.confidence_grid.")
    parser.add_argument(
        "--confidence-range",
        type=float,
        nargs=3,
        metavar=("START", "STOP", "STEP"),
        help="Evaluate an inclusive dense confidence range, for example: --confidence-range 0.30 0.70 0.01.",
    )
    parser.add_argument("--class-aware", action="store_true", help="Require the predicted ship/aircraft/vehicle group to match the GT group.")
    parser.add_argument("--class-agnostic", action="store_true", help="Diagnostic only: allow cross-group matches.")
    args = parser.parse_args()
    if args.class_aware and args.class_agnostic:
        parser.error("--class-aware and --class-agnostic cannot be used together")
    if args.confidence and args.confidence_range:
        parser.error("--confidence and --confidence-range cannot be used together")
    config = load_config(args.config)
    data = resolve_data_yaml(config)
    image_dir = split_image_dir(data, args.split)
    image_paths = sorted([p for p in image_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}])
    from ultralytics import YOLO
    model = YOLO(args.model)
    try:
        confidences = confidence_range(*args.confidence_range) if args.confidence_range else (args.confidence or config["evaluation"]["confidence_grid"])
    except ValueError as exc:
        parser.error(str(exc))
    if any(not 0.0 <= value <= 1.0 for value in confidences):
        parser.error("--confidence must be between 0 and 1")
    confidences = sorted(set(float(value) for value in confidences))
    configured_matching = bool(config["evaluation"].get("class_aware_matching", True))
    class_aware = False if args.class_agnostic else (True if args.class_aware else configured_matching)
    # Infer once at the lowest threshold, then reuse detections for every requested operating point.
    predictions = collect_model_predictions(model, image_paths, min(confidences), config["dataset"]["image_size"], config["evaluation"]["nms_iou"])
    candidates = [evaluate_cached_predictions(predictions, data, args.split, confidence, class_aware) for confidence in confidences]
    operating_points = select_operating_points(candidates)
    for point in operating_points.values():
        point["model"] = str(args.model)
    result = {
        # Keep selected as a backwards-compatible alias.  It is a conservative
        # hard-gate point, not the balanced competition optimum.
        "selected": operating_points["gate_min_fdr"],
        "operating_points": operating_points,
        "all_thresholds": candidates,
        "rules": {
            "class_aware_matching": class_aware,
            "nms_iou": config["evaluation"]["nms_iou"],
            "confidence_grid": confidences,
            "minimum_recall": 0.85,
            "maximum_false_alarm_rate": 0.20,
        },
    }
    json_dump(result, args.out)
    print(f"saved {args.out}")
    print({name: point for name, point in operating_points.items()})


if __name__ == "__main__":
    main()
