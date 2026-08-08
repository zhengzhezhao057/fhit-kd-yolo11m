from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .common import (
    COARSE_NAMES,
    FINE_TO_COARSE,
    image_to_label_path,
    json_dump,
    load_config,
    read_yolo_labels,
    resolve_data_yaml,
    split_image_dir,
    stable_image_key,
    xywhn_to_xyxy,
)
from .competition_eval import Detection, box_iou_one_to_many, metric_dict, score_image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
ERROR_TYPES = (
    "detected",
    "low_confidence",
    "nms_suppressed",
    "wrong_group",
    "localization",
    "no_candidate",
)
ERROR_COLORS = {
    "detected": (40, 190, 40),
    "low_confidence": (0, 210, 255),
    "nms_suppressed": (0, 120, 255),
    "wrong_group": (255, 120, 0),
    "localization": (220, 0, 220),
    "no_candidate": (0, 0, 255),
}
FP_REASONS = ("duplicate", "wrong_group", "localization", "background")


@dataclass(frozen=True)
class ImageGroundTruth:
    image_path: Path
    classes: np.ndarray
    boxes: np.ndarray
    width: int
    height: int


def parse_model_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("--model must use NAME=/path/to/weights.pt")
    name, path = value.split("=", 1)
    if not name.strip() or not path.strip():
        raise ValueError("--model must use a non-empty NAME and checkpoint path")
    return name.strip(), Path(path.strip())


def competition_iou_threshold(fine_class: int) -> float:
    return 0.35 if FINE_TO_COARSE[int(fine_class)] == 2 else 0.50


def match_ground_truths(
    predictions: list[Detection],
    gt_boxes: np.ndarray,
    gt_classes: np.ndarray,
    *,
    class_aware: bool,
    fixed_iou: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one-to-one prediction indices and IoUs using competition ordering."""
    matched_predictions = np.full(len(gt_boxes), -1, dtype=np.int64)
    matched_ious = np.zeros(len(gt_boxes), dtype=np.float32)
    for pred_index in sorted(range(len(predictions)), key=lambda index: predictions[index].score, reverse=True):
        prediction = predictions[pred_index]
        eligible = matched_predictions < 0
        if class_aware:
            eligible &= np.fromiter(
                (FINE_TO_COARSE[int(fine)] == prediction.coarse_class for fine in gt_classes),
                dtype=bool,
                count=len(gt_classes),
            )
        indices = np.flatnonzero(eligible)
        if not len(indices):
            continue
        ious = box_iou_one_to_many(prediction.box, gt_boxes[indices])
        best_position = int(np.argmax(ious))
        gt_index = int(indices[best_position])
        threshold = fixed_iou if fixed_iou is not None else competition_iou_threshold(int(gt_classes[gt_index]))
        if float(ious[best_position]) >= threshold:
            matched_predictions[gt_index] = pred_index
            matched_ious[gt_index] = float(ious[best_position])
    return matched_predictions, matched_ious


def size_bucket(box: np.ndarray, width: int, height: int, image_size: int) -> str:
    """COCO-style size after the same square letterbox scale used by YOLO."""
    scale = min(image_size / max(width, 1), image_size / max(height, 1))
    area = max(float(box[2] - box[0]), 0.0) * scale * max(float(box[3] - box[1]), 0.0) * scale
    if area < 32.0**2:
        return "small"
    if area < 96.0**2:
        return "medium"
    return "large"


def crowded_flags(boxes: np.ndarray, classes: np.ndarray, distance_factor: float = 2.0) -> np.ndarray:
    """Flag same-group instances whose centres are close relative to object scale."""
    flags = np.zeros(len(boxes), dtype=bool)
    if len(boxes) < 2:
        return flags
    centres = (boxes[:, :2] + boxes[:, 2:]) / 2.0
    diagonals = np.linalg.norm(np.maximum(boxes[:, 2:] - boxes[:, :2], 1e-6), axis=1)
    groups = np.asarray([FINE_TO_COARSE[int(value)] for value in classes], dtype=np.int64)
    for left in range(len(boxes)):
        for right in range(left + 1, len(boxes)):
            if groups[left] != groups[right]:
                continue
            distance = float(np.linalg.norm(centres[left] - centres[right]))
            scale = max(float(diagonals[left]), float(diagonals[right]), 1e-6)
            if distance <= distance_factor * scale:
                flags[left] = flags[right] = True
    return flags


def edge_flags(boxes: np.ndarray, width: int, height: int, edge_fraction: float = 0.02) -> np.ndarray:
    margin_x = width * edge_fraction
    margin_y = height * edge_fraction
    if not len(boxes):
        return np.zeros((0,), dtype=bool)
    return (
        (boxes[:, 0] <= margin_x)
        | (boxes[:, 1] <= margin_y)
        | (boxes[:, 2] >= width - margin_x)
        | (boxes[:, 3] >= height - margin_y)
    )


def best_candidate(
    predictions: list[Detection],
    gt_box: np.ndarray,
    gt_fine_class: int,
    *,
    class_aware: bool,
) -> tuple[int, float]:
    eligible = [
        index
        for index, prediction in enumerate(predictions)
        if not class_aware or prediction.coarse_class == FINE_TO_COARSE[int(gt_fine_class)]
    ]
    if not eligible:
        return -1, 0.0
    ious = box_iou_one_to_many(gt_box, np.asarray([predictions[index].box for index in eligible]))
    position = int(np.argmax(ious))
    return eligible[position], float(ious[position])


def classify_instances(
    current: list[Detection],
    low_confidence: list[Detection],
    loose_nms: list[Detection],
    gt_boxes: np.ndarray,
    gt_classes: np.ndarray,
    *,
    localization_iou_floor: float = 0.10,
) -> list[dict]:
    current_matches, current_ious = match_ground_truths(current, gt_boxes, gt_classes, class_aware=True)
    low_matches, low_ious = match_ground_truths(low_confidence, gt_boxes, gt_classes, class_aware=True)
    loose_matches, loose_ious = match_ground_truths(loose_nms, gt_boxes, gt_classes, class_aware=True)
    any_matches, _ = match_ground_truths(loose_nms, gt_boxes, gt_classes, class_aware=False)
    iou50_matches, _ = match_ground_truths(current, gt_boxes, gt_classes, class_aware=True, fixed_iou=0.50)
    iou75_matches, _ = match_ground_truths(current, gt_boxes, gt_classes, class_aware=True, fixed_iou=0.75)
    rows: list[dict] = []
    for gt_index, fine_class in enumerate(gt_classes):
        source: list[Detection]
        pred_index: int
        match_iou: float
        if current_matches[gt_index] >= 0:
            error_type = "detected"
            source = current
            pred_index = int(current_matches[gt_index])
            match_iou = float(current_ious[gt_index])
        elif low_matches[gt_index] >= 0:
            error_type = "low_confidence"
            source = low_confidence
            pred_index = int(low_matches[gt_index])
            match_iou = float(low_ious[gt_index])
        elif loose_matches[gt_index] >= 0:
            error_type = "nms_suppressed"
            source = loose_nms
            pred_index = int(loose_matches[gt_index])
            match_iou = float(loose_ious[gt_index])
        elif (
            any_matches[gt_index] >= 0
            and loose_nms[int(any_matches[gt_index])].coarse_class != FINE_TO_COARSE[int(fine_class)]
        ):
            error_type = "wrong_group"
            source = loose_nms
            pred_index = int(any_matches[gt_index])
            match_iou = float(box_iou_one_to_many(gt_boxes[gt_index], np.asarray([source[pred_index].box]))[0])
        else:
            pred_index, match_iou = best_candidate(
                loose_nms,
                gt_boxes[gt_index],
                int(fine_class),
                class_aware=True,
            )
            source = loose_nms
            error_type = "localization" if pred_index >= 0 and match_iou >= localization_iou_floor else "no_candidate"
        prediction = source[pred_index] if pred_index >= 0 else None
        rows.append({
            "error_type": error_type,
            "matched_iou": match_iou,
            "prediction_score": prediction.score if prediction is not None else 0.0,
            "predicted_fine_class": prediction.fine_class if prediction is not None else -1,
            "predicted_coarse_class": prediction.coarse_class if prediction is not None else -1,
            "fine_class_correct": bool(prediction is not None and prediction.fine_class == int(fine_class)),
            "matched_iou50": bool(iou50_matches[gt_index] >= 0),
            "matched_iou75": bool(iou75_matches[gt_index] >= 0),
        })
    return rows


def classify_false_positives(
    predictions: list[Detection],
    gt_boxes: np.ndarray,
    gt_classes: np.ndarray,
    *,
    localization_iou_floor: float = 0.10,
) -> list[dict]:
    """Describe every prediction left unmatched by competition scoring."""
    gt_matches, _ = match_ground_truths(predictions, gt_boxes, gt_classes, class_aware=True)
    matched_prediction_indices = {int(value) for value in gt_matches if value >= 0}
    rows: list[dict] = []
    for prediction_index, prediction in enumerate(predictions):
        if prediction_index in matched_prediction_indices:
            continue
        same_index, same_iou = -1, 0.0
        same_gt_indices = [
            index
            for index, fine_class in enumerate(gt_classes)
            if FINE_TO_COARSE[int(fine_class)] == prediction.coarse_class
        ]
        if same_gt_indices:
            same_ious = box_iou_one_to_many(prediction.box, gt_boxes[same_gt_indices])
            same_position = int(np.argmax(same_ious))
            same_index = int(same_gt_indices[same_position])
            same_iou = float(same_ious[same_position])
        any_index = -1
        any_iou = 0.0
        if len(gt_boxes):
            any_ious = box_iou_one_to_many(prediction.box, gt_boxes)
            any_index = int(np.argmax(any_ious))
            any_iou = float(any_ious[any_index])

        duplicate = same_index >= 0 and same_iou >= competition_iou_threshold(int(gt_classes[same_index]))
        wrong_group = (
            any_index >= 0
            and FINE_TO_COARSE[int(gt_classes[any_index])] != prediction.coarse_class
            and any_iou >= competition_iou_threshold(int(gt_classes[any_index]))
        )
        if duplicate:
            reason = "duplicate"
            nearest_index, nearest_iou = same_index, same_iou
        elif wrong_group:
            reason = "wrong_group"
            nearest_index, nearest_iou = any_index, any_iou
        elif same_index >= 0 and same_iou >= localization_iou_floor:
            reason = "localization"
            nearest_index, nearest_iou = same_index, same_iou
        else:
            reason = "background"
            nearest_index, nearest_iou = any_index, any_iou
        rows.append({
            "prediction_index": prediction_index,
            "fine_class": prediction.fine_class,
            "coarse_group": COARSE_NAMES[prediction.coarse_class],
            "score": prediction.score,
            "box_x1": float(prediction.box[0]),
            "box_y1": float(prediction.box[1]),
            "box_x2": float(prediction.box[2]),
            "box_y2": float(prediction.box[3]),
            "reason": reason,
            "nearest_gt_index": nearest_index,
            "nearest_gt_iou": nearest_iou,
            "nearest_gt_fine_class": int(gt_classes[nearest_index]) if nearest_index >= 0 else -1,
        })
    return rows


def load_ground_truth(config: dict, split: str) -> tuple[dict, list[ImageGroundTruth]]:
    import cv2

    data = resolve_data_yaml(config)
    image_dir = split_image_dir(data, split)
    label_dir = Path(data["path"]) / "labels" / split
    image_paths = sorted(path for path in image_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    items: list[ImageGroundTruth] = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"cannot read image: {image_path}")
        height, width = image.shape[:2]
        classes, normalized_boxes = read_yolo_labels(
            image_to_label_path(image_path, image_dir, label_dir),
            deduplicate=True,
        )
        items.append(ImageGroundTruth(image_path, classes, xywhn_to_xyxy(normalized_boxes, width, height), width, height))
    return data, items


def collect_predictions(
    model,
    items: list[ImageGroundTruth],
    *,
    image_size: int,
    confidence: float,
    nms_iou: float,
    batch: int,
    max_det: int,
) -> dict[Path, list[Detection]]:
    paths = [str(item.image_path) for item in items]
    if batch < 1:
        raise ValueError("batch must be >= 1")
    predictions: dict[Path, list[Detection]] = {}
    # Ultralytics treats an explicit list of image paths as one inference
    # batch in some versions, regardless of the `batch` argument. Slice the
    # list ourselves so a 671-image validation set cannot become one giant
    # allocation on GPU.
    for start in range(0, len(paths), batch):
        input_batch = paths[start : start + batch]
        results = list(model.predict(
            source=input_batch,
            imgsz=image_size,
            conf=confidence,
            iou=nms_iou,
            max_det=max_det,
            stream=True,
            verbose=False,
        ))
        if len(results) != len(input_batch):
            raise RuntimeError(f"model returned {len(results)} result(s) for a batch of {len(input_batch)} image(s)")
        # Bind by input order. Ultralytics' result.path normalization differs
        # between Windows and Linux, while result ordering follows the source
        # list on both platforms.
        for input_path, result in zip(input_batch, results):
            detections: list[Detection] = []
            if result.boxes is not None and len(result.boxes):
                boxes = result.boxes.xyxy.detach().cpu().numpy()
                scores = result.boxes.conf.detach().cpu().numpy()
                labels = result.boxes.cls.detach().cpu().numpy().astype(int)
                detections = [
                    Detection(box.astype(np.float32), float(score), int(label))
                    for box, score, label in zip(boxes, scores, labels)
                ]
            predictions[Path(input_path).resolve()] = detections
    expected = {item.image_path.resolve() for item in items}
    missing = expected - set(predictions)
    if missing:
        raise RuntimeError(f"model prediction omitted {len(missing)} image(s), first={next(iter(missing))}")
    return predictions


def evaluate_prediction_map(
    predictions: dict[Path, list[Detection]],
    items: list[ImageGroundTruth],
    confidence: float,
) -> dict:
    overall = np.zeros(3, dtype=np.int64)
    per_group = {name: np.zeros(3, dtype=np.int64) for name in COARSE_NAMES}
    for item in items:
        selected = [prediction for prediction in predictions[item.image_path.resolve()] if prediction.score >= confidence]
        overall += np.asarray(score_image(selected, item.boxes, item.classes, class_aware=True))
        for group_index, group_name in enumerate(COARSE_NAMES):
            mask = np.asarray([FINE_TO_COARSE[int(value)] == group_index for value in item.classes], dtype=bool)
            group_predictions = [prediction for prediction in selected if prediction.coarse_class == group_index]
            per_group[group_name] += np.asarray(score_image(group_predictions, item.boxes[mask], item.classes[mask], class_aware=True))
    return {
        "overall": metric_dict(*overall.tolist()),
        "per_group": {name: metric_dict(*values.tolist()) for name, values in per_group.items()},
    }


def summarize_instance_rows(rows: list[dict]) -> dict:
    def summarize(selected: Iterable[dict]) -> dict:
        values = list(selected)
        errors = Counter(value["error_type"] for value in values)
        detected = [value for value in values if value["error_type"] == "detected"]
        return {
            "instances": len(values),
            "detected": len(detected),
            "detected_rate": len(detected) / max(len(values), 1),
            "fine_class_correct_rate_among_detected": sum(value["fine_class_correct"] for value in detected) / max(len(detected), 1),
            "iou50_match_rate": sum(value["matched_iou50"] for value in values) / max(len(values), 1),
            "iou75_match_rate": sum(value["matched_iou75"] for value in values) / max(len(values), 1),
            "errors": {name: errors.get(name, 0) for name in ERROR_TYPES},
        }

    summary = {"overall": summarize(rows), "per_group": {}, "per_size": {}, "attributes": {}}
    for group in COARSE_NAMES:
        group_rows = [row for row in rows if row["coarse_group"] == group]
        group_summary = summarize(group_rows)
        group_summary["per_size"] = {
            size: summarize(row for row in group_rows if row["size"] == size)
            for size in ("small", "medium", "large")
        }
        group_summary["attributes"] = {
            attribute: summarize(row for row in group_rows if row[attribute])
            for attribute in ("crowded", "edge")
        }
        summary["per_group"][group] = group_summary
    for size in ("small", "medium", "large"):
        summary["per_size"][size] = summarize(row for row in rows if row["size"] == size)
    for attribute in ("crowded", "edge"):
        summary["attributes"][attribute] = summarize(row for row in rows if row[attribute])
    return summary


def summarize_false_positive_rows(rows: list[dict]) -> dict:
    def summarize(selected: Iterable[dict]) -> dict:
        values = list(selected)
        reasons = Counter(value["reason"] for value in values)
        return {
            "false_positives": len(values),
            "reasons": {name: reasons.get(name, 0) for name in FP_REASONS},
        }

    return {
        "overall": summarize(rows),
        "per_group": {
            group: summarize(row for row in rows if row["coarse_group"] == group)
            for group in COARSE_NAMES
        },
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_visuals(output: Path, model_name: str, rows: list[dict], items: list[ImageGroundTruth], max_visuals: int) -> None:
    import cv2

    weak_errors = [row for row in rows if row["coarse_group"] in {"ship", "vehicle"} and row["error_type"] != "detected"]
    error_counts = Counter(row["image"] for row in weak_errors)
    selected_images = [name for name, _ in error_counts.most_common(max_visuals)]
    rows_by_image: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        rows_by_image[row["image"]].append(row)
    items_by_name = {str(item.image_path): item for item in items}
    destination = output / "visuals" / model_name
    destination.mkdir(parents=True, exist_ok=True)
    for image_name in selected_images:
        item = items_by_name[image_name]
        image = cv2.imread(str(item.image_path))
        for row in rows_by_image[image_name]:
            if row["coarse_group"] not in {"ship", "vehicle"}:
                continue
            box = [int(round(value)) for value in item.boxes[int(row["gt_index"])]]
            color = ERROR_COLORS[row["error_type"]]
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)
            label = f'{row["coarse_group"]}:{row["error_type"]}'
            cv2.putText(image, label, (box[0], max(box[1] - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
        relative_name = f"{stable_image_key(item.image_path)[:10]}_{item.image_path.name}"
        cv2.imwrite(str(destination / relative_name), image)


def save_false_positive_visuals(
    output: Path,
    model_name: str,
    rows: list[dict],
    items: list[ImageGroundTruth],
    max_visuals: int,
) -> None:
    import cv2

    weak_rows = [row for row in rows if row["coarse_group"] in {"ship", "vehicle"}]
    counts = Counter(row["image"] for row in weak_rows)
    selected_images = [name for name, _ in counts.most_common(max_visuals)]
    rows_by_image: dict[str, list[dict]] = defaultdict(list)
    for row in weak_rows:
        rows_by_image[row["image"]].append(row)
    items_by_name = {str(item.image_path): item for item in items}
    destination = output / "false_positive_visuals" / model_name
    destination.mkdir(parents=True, exist_ok=True)
    for image_name in selected_images:
        item = items_by_name[image_name]
        image = cv2.imread(str(item.image_path))
        for row in rows_by_image[image_name]:
            box = [int(round(row[key])) for key in ("box_x1", "box_y1", "box_x2", "box_y2")]
            color = (0, 0, 255) if row["coarse_group"] == "vehicle" else (0, 128, 255)
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)
            label = f'{row["coarse_group"]}:{row["reason"]}:{row["score"]:.2f}'
            cv2.putText(image, label, (box[0], max(box[1] - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
        filename = f"{stable_image_key(item.image_path)[:10]}_{item.image_path.name}"
        cv2.imwrite(str(destination / filename), image)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose ship/vehicle misses by confidence, NMS, localization and class errors.")
    parser.add_argument("--config", default="configs/direction1.yaml")
    parser.add_argument("--model", action="append", required=True, help="Repeat NAME=/path/to/weights.pt.")
    # TRAIN is required for leakage-safe hard-example mining in v4. Model
    # selection still uses val; the manifest builder independently rejects
    # every row that is not a configured training image.
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--out", default="reports/weak_group_diagnostics")
    parser.add_argument("--confidence", type=float, action="append", default=None, help="Threshold matrix value; repeat as needed.")
    parser.add_argument("--operating-confidence", type=float, default=0.50)
    parser.add_argument("--confidence-floor", type=float, default=0.01)
    parser.add_argument("--nms-iou", type=float, action="append", default=None, help="NMS matrix value; repeat as needed.")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--max-det", type=int, default=3000)
    parser.add_argument("--localization-iou-floor", type=float, default=0.10)
    parser.add_argument("--crowd-distance-factor", type=float, default=2.0)
    parser.add_argument("--edge-fraction", type=float, default=0.02)
    parser.add_argument("--max-visuals", type=int, default=40)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not 0.0 <= args.confidence_floor <= args.operating_confidence <= 1.0:
        parser.error("require 0 <= confidence-floor <= operating-confidence <= 1")
    config = load_config(args.config)
    image_size = int(config["dataset"]["image_size"])
    confidences = sorted(set(args.confidence or [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]))
    nms_values = sorted(set(args.nms_iou or [float(config["evaluation"]["nms_iou"]), 0.65, 0.80]))
    if any(not 0.0 <= value <= 1.0 for value in confidences + nms_values):
        parser.error("confidence and NMS IoU values must be between 0 and 1")
    base_nms = min(nms_values, key=lambda value: abs(value - float(config["evaluation"]["nms_iou"])))
    loose_nms = max(nms_values)
    model_specs = [parse_model_spec(value) for value in args.model]
    if len({name for name, _ in model_specs}) != len(model_specs):
        parser.error("model names must be unique")
    for _, checkpoint in model_specs:
        if not checkpoint.exists():
            parser.error(f"model checkpoint does not exist: {checkpoint}")

    _, items = load_ground_truth(config, args.split)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    matrix_rows: list[dict] = []
    summaries: dict[str, dict] = {}
    from ultralytics import YOLO

    for model_name, checkpoint in model_specs:
        print(f"DIAGNOSTIC MODEL: {model_name} ({checkpoint})")
        model = YOLO(checkpoint)
        by_nms: dict[float, dict[Path, list[Detection]]] = {}
        for nms_iou in nms_values:
            print(f"  inference nms_iou={nms_iou:.2f} conf={args.confidence_floor:.3f}")
            by_nms[nms_iou] = collect_predictions(
                model,
                items,
                image_size=image_size,
                confidence=args.confidence_floor,
                nms_iou=nms_iou,
                batch=args.batch,
                max_det=args.max_det,
            )
            for confidence in confidences:
                report = evaluate_prediction_map(by_nms[nms_iou], items, confidence)
                for group, metrics in [("overall", report["overall"]), *report["per_group"].items()]:
                    matrix_rows.append({
                        "model": model_name,
                        "nms_iou": nms_iou,
                        "confidence": confidence,
                        "group": group,
                        **metrics,
                    })

        instance_rows: list[dict] = []
        false_positive_rows: list[dict] = []
        base_predictions = by_nms[base_nms]
        loose_predictions = by_nms[loose_nms]
        for item in items:
            key = item.image_path.resolve()
            current = [prediction for prediction in base_predictions[key] if prediction.score >= args.operating_confidence]
            low = base_predictions[key]
            loose = loose_predictions[key]
            classifications = classify_instances(
                current,
                low,
                loose,
                item.boxes,
                item.classes,
                localization_iou_floor=args.localization_iou_floor,
            )
            image_false_positives = classify_false_positives(
                current,
                item.boxes,
                item.classes,
                localization_iou_floor=args.localization_iou_floor,
            )
            crowded = crowded_flags(item.boxes, item.classes, args.crowd_distance_factor)
            edges = edge_flags(item.boxes, item.width, item.height, args.edge_fraction)
            for gt_index, result in enumerate(classifications):
                fine_class = int(item.classes[gt_index])
                gt_box = item.boxes[gt_index]
                gt_xywhn = [
                    float((gt_box[0] + gt_box[2]) / (2.0 * item.width)),
                    float((gt_box[1] + gt_box[3]) / (2.0 * item.height)),
                    float((gt_box[2] - gt_box[0]) / item.width),
                    float((gt_box[3] - gt_box[1]) / item.height),
                ]
                instance_rows.append({
                    "model": model_name,
                    "image": str(item.image_path),
                    "gt_index": gt_index,
                    "fine_class": fine_class,
                    "coarse_group": COARSE_NAMES[FINE_TO_COARSE[fine_class]],
                    "size": size_bucket(item.boxes[gt_index], item.width, item.height, image_size),
                    "crowded": bool(crowded[gt_index]),
                    "edge": bool(edges[gt_index]),
                    # Keep the original normalized GT geometry so a TRAIN-only
                    # diagnostic can be converted into a relocation-safe hard
                    # example manifest. Validation diagnostics remain analysis
                    # artifacts and are explicitly rejected by that builder.
                    "gt_x": gt_xywhn[0],
                    "gt_y": gt_xywhn[1],
                    "gt_w": gt_xywhn[2],
                    "gt_h": gt_xywhn[3],
                    **result,
                })
            for false_positive in image_false_positives:
                prediction_box = np.asarray([
                    false_positive["box_x1"],
                    false_positive["box_y1"],
                    false_positive["box_x2"],
                    false_positive["box_y2"],
                ], dtype=np.float32)
                false_positive_rows.append({
                    "model": model_name,
                    "image": str(item.image_path),
                    "predicted_size": size_bucket(prediction_box, item.width, item.height, image_size),
                    **false_positive,
                })
        write_csv(output / f"{model_name}_instances.csv", instance_rows)
        write_csv(output / f"{model_name}_false_positives.csv", false_positive_rows)
        if args.max_visuals > 0:
            save_visuals(output, model_name, instance_rows, items, args.max_visuals)
            save_false_positive_visuals(output, model_name, false_positive_rows, items, args.max_visuals)
        base_report = evaluate_prediction_map(base_predictions, items, args.operating_confidence)
        expected_false_positives = int(base_report["overall"]["FP"])
        if len(false_positive_rows) != expected_false_positives:
            raise RuntimeError(
                f"false-positive export mismatch for {model_name}: "
                f"exported={len(false_positive_rows)} evaluated={expected_false_positives}"
            )
        for group_name in COARSE_NAMES:
            exported_group_fp = sum(row["coarse_group"] == group_name for row in false_positive_rows)
            expected_group_fp = int(base_report["per_group"][group_name]["FP"])
            if exported_group_fp != expected_group_fp:
                raise RuntimeError(
                    f"false-positive export mismatch for {model_name}/{group_name}: "
                    f"exported={exported_group_fp} evaluated={expected_group_fp}"
                )
        model_summary = summarize_instance_rows(instance_rows)
        model_summary["false_positives"] = summarize_false_positive_rows(false_positive_rows)
        summaries[model_name] = model_summary
        print(f"  wrote {len(instance_rows)} GT rows and {len(false_positive_rows)} false-positive rows")

    write_csv(output / "threshold_nms_matrix.csv", matrix_rows)
    report = {
        "split": args.split,
        "models": {name: str(path) for name, path in model_specs},
        "rules": {
            "operating_confidence": args.operating_confidence,
            "confidence_floor": args.confidence_floor,
            "confidence_grid": confidences,
            "base_nms_iou": base_nms,
            "loose_nms_iou": loose_nms,
            "nms_iou_grid": nms_values,
            "localization_iou_floor": args.localization_iou_floor,
            "crowd_distance_factor": args.crowd_distance_factor,
            "edge_fraction": args.edge_fraction,
            "competition_iou": {"ship": 0.50, "aircraft": 0.50, "vehicle": 0.35},
        },
        "error_taxonomy": {
            "detected": "matched at operating confidence and base NMS",
            "low_confidence": "matched at confidence floor with base NMS",
            "nms_suppressed": "matched only after raising the NMS IoU",
            "wrong_group": "box matched only when coarse-class matching was disabled",
            "localization": "same-group candidate exists but IoU is below the competition threshold",
            "no_candidate": "no same-group candidate reaches localization_iou_floor",
        },
        "false_positive_taxonomy": {
            "duplicate": "an extra prediction overlaps an already matched same-group GT",
            "wrong_group": "prediction overlaps a GT from another coarse group",
            "localization": "same-group GT is nearby but IoU is below the competition threshold",
            "background": "no same-group GT reaches localization_iou_floor",
        },
        "summaries": summaries,
    }
    json_dump(report, output / "summary.json")
    print(f"saved diagnostics to {output.resolve()}")


if __name__ == "__main__":
    main()
