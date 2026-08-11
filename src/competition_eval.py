from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

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


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
OFFICIAL_IOU_THRESHOLDS = {"ship": 0.50, "aircraft": 0.50, "vehicle": 0.35}
MINIMUM_RECALL = 0.85
MAXIMUM_FALSE_ALARM_RATE = 0.20
MAXIMUM_SECONDS_PER_IMAGE = 20.0
DEFAULT_BATCH = 8
# Ultralytics defaults to 300 detections.  That can silently truncate dense
# remote-sensing scenes at a low confidence floor, changing both recall and
# false-alarm counts.  P0 therefore uses a deliberately high default and
# fails closed whenever the cap is reached.
DEFAULT_MAX_DET = 10_000


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
        gt_group = COARSE_NAMES[FINE_TO_COARSE[int(gt_fine_classes[gt_index])]]
        threshold = OFFICIAL_IOU_THRESHOLDS[gt_group]
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


def bounded_range(start: float, stop: float, step: float, *, name: str) -> list[float]:
    """Return an inclusive, numerically stable grid in ``[0, 1]``."""
    if not (0.0 <= start <= stop <= 1.0) or step <= 0.0 or not all(
        math.isfinite(value) for value in (start, stop, step)
    ):
        raise ValueError(f"{name} range requires 0 <= START <= STOP <= 1 and STEP > 0")
    count = int(np.floor((stop - start) / step + 1e-9))
    values = [round(start + index * step, 10) for index in range(count + 1)]
    if values[-1] < stop - 1e-9:
        values.append(round(stop, 10))
    return values


def confidence_range(start: float, stop: float, step: float) -> list[float]:
    """Return an inclusive confidence grid for operating-point analysis."""
    return bounded_range(start, stop, step, name="confidence")


def nms_iou_range(start: float, stop: float, step: float) -> list[float]:
    """Return an inclusive NMS-IoU grid for operating-point analysis."""
    return bounded_range(start, stop, step, name="NMS IoU")


def ensure_no_max_det_truncation(hit_images: Iterable[str], max_det: int) -> None:
    """Fail closed instead of reporting metrics from silently capped output."""
    hits = list(hit_images)
    if hits:
        preview = ", ".join(hits[:3])
        suffix = "" if len(hits) <= 3 else f" (+{len(hits) - 3} more)"
        raise RuntimeError(
            f"max_det={max_det} was reached by {len(hits)} image(s): {preview}{suffix}. "
            "Predictions may be truncated; rerun with a larger --max-det."
        )


def build_timing_report(
    *,
    image_count: int,
    post_read_batch_seconds: Iterable[float],
    batch_sizes: Iterable[int],
    image_read_seconds: float,
    stage_ms: dict[str, float],
    batch: int,
    image_size: int,
    max_det: int,
    nms_iou: float,
    warmup_iterations: int,
    device: str | None,
    model_load_seconds: float | None = None,
    source_shapes: Iterable[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Build the auditable inference-time schema required by the score sheet.

    The primary interval begins *after* each batch has been read into memory
    and ends after prediction boxes have been materialized as CPU result
    objects.  Image reading is reported separately and is never subtracted
    from an already combined wall-clock measurement.
    """
    durations = [float(value) for value in post_read_batch_seconds]
    sizes = [int(value) for value in batch_sizes]
    if image_count <= 0 or not durations or len(durations) != len(sizes):
        raise ValueError("timing requires a positive image count and paired batch durations/sizes")
    if sum(sizes) != image_count or any(size <= 0 for size in sizes):
        raise ValueError("timing batch sizes must be positive and sum to image_count")
    if any(value < 0 or not math.isfinite(value) for value in durations):
        raise ValueError("timing durations must be finite and non-negative")
    if image_read_seconds < 0 or not math.isfinite(image_read_seconds):
        raise ValueError("image_read_seconds must be finite and non-negative")
    total_seconds = float(sum(durations))
    # Assign the amortized batch latency to every image in that batch.  This
    # keeps quantiles meaningful for a batched benchmark without pretending
    # that asynchronous kernels were timed independently per image.
    per_image_ms = [duration * 1000.0 / size for duration, size in zip(durations, sizes) for _ in range(size)]
    normalized_stage_ms = {
        name: float(stage_ms.get(name, 0.0))
        for name in ("preprocess", "model", "postprocess")
    }
    stage_total_ms = sum(normalized_stage_ms.values())
    shapes = [(int(height), int(width)) for height, width in (source_shapes or [])]
    if shapes and len(shapes) != image_count:
        raise ValueError("source_shapes must contain one (height, width) pair per image")
    large_gate_applicable = bool(shapes) and all(height >= 10_000 and width >= 10_000 for height, width in shapes)
    large_gate_pass = total_seconds / image_count <= MAXIMUM_SECONDS_PER_IMAGE if large_gate_applicable else None
    report = {
        "schema_version": "fhit.competition_timing.v1",
        "protocol": "competition_no_image_io_v1",
        "excludes_image_read": True,
        "interval_start": "after_each_batch_image_reads_complete",
        "interval_end": "after_prediction_objects_materialized",
        "included_stages": ["preprocess", "model", "postprocess", "prediction_materialization"],
        "image_count": int(image_count),
        # These aliases intentionally match src.experiment_ledger.normalize_timing.
        "total_seconds": total_seconds,
        "mean_ms_per_image": total_seconds * 1000.0 / image_count,
        "p50_ms_per_image": float(np.percentile(per_image_ms, 50)),
        "p95_ms_per_image": float(np.percentile(per_image_ms, 95)),
        "seconds_per_image": total_seconds / image_count,
        "throughput_images_per_second": image_count / max(total_seconds, 1e-12),
        # The score sheet ranks total post-read inference time.  The separate
        # 20-second gate applies to a 10000x10000 competition image, so it must
        # not be claimed from an ordinary 640/800-pixel crop validation set.
        "ranking_total_seconds_excluding_image_read": total_seconds,
        "competition_large_image_gate": {
            "required_minimum_source_height": 10_000,
            "required_minimum_source_width": 10_000,
            "maximum_seconds_per_image": MAXIMUM_SECONDS_PER_IMAGE,
            "applicable": large_gate_applicable,
            "pass": large_gate_pass,
        },
        "competition_timing_pass": large_gate_pass,
        "image_read": {
            "filesystem_cache_state": "uncontrolled",
            "total_seconds": float(image_read_seconds),
            "mean_ms_per_image": float(image_read_seconds) * 1000.0 / image_count,
        },
        "end_to_end": {
            "scope": "image_read_plus_post_read_inference; model_load_and_warmup_excluded",
            "total_seconds": total_seconds + float(image_read_seconds),
            "mean_ms_per_image": (total_seconds + float(image_read_seconds)) * 1000.0 / image_count,
        },
        "ultralytics_stage_totals_ms": normalized_stage_ms,
        "ultralytics_stage_mean_ms_per_image": {
            name: value / image_count for name, value in normalized_stage_ms.items()
        },
        "python_orchestration_and_materialization_ms": max(0.0, total_seconds * 1000.0 - stage_total_ms),
        "batch_durations_seconds": durations,
        "batch_sizes": sizes,
        "batch": int(batch),
        "image_size": int(image_size),
        "max_det": int(max_det),
        "nms_iou": float(nms_iou),
        "warmup_iterations": int(warmup_iterations),
        "repetitions": 1,
        "device": device,
    }
    if shapes:
        heights = [height for height, _ in shapes]
        widths = [width for _, width in shapes]
        report["source_image_dimensions"] = {
            "min_height": min(heights),
            "max_height": max(heights),
            "min_width": min(widths),
            "max_width": max(widths),
            "images_at_least_10000x10000": sum(
                height >= 10_000 and width >= 10_000 for height, width in shapes
            ),
        }
    if model_load_seconds is not None:
        report["model_load_seconds_excluded"] = float(model_load_seconds)
    return report


def competition_ranking_items(report: dict[str, Any], timing: dict[str, Any]) -> dict[str, Any]:
    """Return the six coarse-group metrics plus the official timing item."""
    groups = report.get("per_group", {})
    missing = [name for name in COARSE_NAMES if name not in groups]
    if missing:
        raise ValueError(f"competition report is missing coarse groups: {missing}")
    items: dict[str, Any] = {}
    for name in COARSE_NAMES:
        metrics = groups[name]
        for field in ("recall", "false_alarm_rate"):
            value = metrics.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"competition report is missing numeric {name}.{field}")
        items[f"{name}_recall"] = float(metrics["recall"])
        items[f"{name}_false_alarm_rate"] = float(metrics["false_alarm_rate"])
    total_seconds = timing.get("total_seconds")
    if not isinstance(total_seconds, (int, float)) or not math.isfinite(float(total_seconds)):
        raise ValueError("timing is missing numeric total_seconds")
    items["total_inference_seconds_excluding_image_read"] = float(total_seconds)
    if len(items) != 7:
        raise AssertionError("official ranking item schema must contain exactly seven fields")
    return items


def overall_safety_gate(report: dict[str, Any], timing: dict[str, Any]) -> dict[str, Any]:
    overall = report["overall"]
    recall_pass = float(overall["recall"]) >= MINIMUM_RECALL
    fdr_pass = float(overall["false_alarm_rate"]) <= MAXIMUM_FALSE_ALARM_RATE
    timing_gate = timing.get("competition_large_image_gate") or {}
    timing_applicable = timing_gate.get("applicable") is True
    timing_pass = timing_gate.get("pass") if timing_applicable else None
    detection_pass = recall_pass and fdr_pass
    return {
        "minimum_recall": MINIMUM_RECALL,
        "maximum_false_alarm_rate": MAXIMUM_FALSE_ALARM_RATE,
        "maximum_seconds_per_image_excluding_image_read": MAXIMUM_SECONDS_PER_IMAGE,
        "recall_pass": recall_pass,
        "false_alarm_rate_pass": fdr_pass,
        "timing_gate_applicable": timing_applicable,
        "timing_pass": timing_pass,
        "detection_gate_pass": detection_pass,
        # ``passed`` is the score sheet's overall detection gate.  Runtime is
        # a separate 10000x10000 gate and remains unknown on crop validation.
        "passed": detection_pass,
        "all_applicable_gates_pass": detection_pass and (timing_pass is not False),
        "complete_submission_gate_pass": detection_pass and timing_pass if timing_applicable else None,
    }


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
            item.get("nms_iou", 0.0),
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
            -item["confidence"],
            -item.get("nms_iou", 0.0),
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
            item["overall"]["f1"],
            item["confidence"],
            -item.get("nms_iou", 0.0),
        ),
    )
    return {
        "gate_min_fdr": copy.deepcopy(gate_min_fdr),
        "best_f1": copy.deepcopy(best_f1),
        "max_recall_under_fdr": copy.deepcopy(max_recall_under_fdr),
    }


def _model_device_name(model: Any) -> str | None:
    try:
        parameter = next(model.model.parameters())
        device = str(parameter.device)
    except (AttributeError, StopIteration, TypeError):
        return None
    if device.startswith("cuda"):
        try:
            import torch

            index = parameter.device.index if parameter.device.index is not None else torch.cuda.current_device()
            return f"{device} ({torch.cuda.get_device_name(index)})"
        except Exception:
            return device
    return device


def _synchronize_model_device(model: Any) -> None:
    """Synchronize CUDA when present so wall-clock intervals are not optimistic."""
    try:
        import torch

        parameter = next(model.model.parameters())
        if parameter.is_cuda:
            torch.cuda.synchronize(parameter.device)
    except (AttributeError, StopIteration, TypeError):
        return


def _materialize_detections(result: Any) -> list[Detection]:
    if result.boxes is None or not len(result.boxes):
        return []
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    scores = result.boxes.conf.detach().cpu().numpy()
    labels = result.boxes.cls.detach().cpu().numpy().astype(int)
    return [
        Detection(box=box.astype(np.float32), score=float(score), fine_class=int(label))
        for box, score, label in zip(boxes, scores, labels)
    ]


def warmup_model(
    model: Any,
    *,
    iterations: int,
    image_size: int,
    confidence: float,
    nms_iou: float,
    max_det: int,
    device: str | int | None,
) -> float:
    """Warm up model initialization outside the scored timing interval."""
    if iterations < 0:
        raise ValueError("warmup iterations cannot be negative")
    if iterations == 0:
        return 0.0
    dummy = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    started = time.perf_counter()
    for _ in range(iterations):
        kwargs: dict[str, Any] = {
            "source": [dummy],
            "imgsz": image_size,
            "conf": confidence,
            "iou": nms_iou,
            "max_det": max_det,
            "batch": 1,
            "verbose": False,
        }
        if device is not None:
            kwargs["device"] = device
        model.predict(**kwargs)
    _synchronize_model_device(model)
    return time.perf_counter() - started


def collect_timed_model_predictions(
    model: Any,
    image_paths: list[Path],
    confidence: float,
    image_size: int,
    nms_iou: float,
    *,
    batch: int,
    max_det: int,
    warmup_iterations: int = 0,
    device: str | int | None = None,
    model_load_seconds: float | None = None,
) -> tuple[dict[Path, list[Detection]], dict[str, Any], list[str]]:
    """Run batched inference while keeping image I/O outside official timing."""
    if not image_paths:
        raise ValueError("cannot benchmark an empty image list")
    if batch <= 0 or max_det <= 0 or image_size <= 0:
        raise ValueError("batch, max_det and image_size must be positive")
    import cv2

    predictions: dict[Path, list[Detection]] = {}
    image_read_seconds = 0.0
    post_read_batch_seconds: list[float] = []
    batch_sizes: list[int] = []
    stage_ms = {"preprocess": 0.0, "model": 0.0, "postprocess": 0.0}
    hit_images: list[str] = []
    source_shapes: list[tuple[int, int]] = []
    if warmup_iterations:
        warmup_model(
            model,
            iterations=warmup_iterations,
            image_size=image_size,
            confidence=confidence,
            nms_iou=nms_iou,
            max_det=max_det,
            device=device,
        )
    for offset in range(0, len(image_paths), batch):
        paths = image_paths[offset : offset + batch]
        read_started = time.perf_counter()
        images = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in paths]
        image_read_seconds += time.perf_counter() - read_started
        unreadable = [str(path) for path, image in zip(paths, images) if image is None]
        if unreadable:
            raise RuntimeError(f"failed to read {len(unreadable)} image(s): {', '.join(unreadable[:3])}")
        source_shapes.extend((int(image.shape[0]), int(image.shape[1])) for image in images)

        _synchronize_model_device(model)
        scored_started = time.perf_counter()
        predict_kwargs: dict[str, Any] = {
            "source": images,
            "imgsz": image_size,
            "conf": confidence,
            "iou": nms_iou,
            "max_det": max_det,
            "batch": len(paths),
            "verbose": False,
        }
        if device is not None:
            predict_kwargs["device"] = device
        results = model.predict(**predict_kwargs)
        if len(results) != len(paths):
            raise RuntimeError(f"model returned {len(results)} result(s) for {len(paths)} input image(s)")
        for path, result in zip(paths, results):
            detections = _materialize_detections(result)
            predictions[path] = detections
            if len(detections) >= max_det:
                hit_images.append(str(path))
            speed = getattr(result, "speed", None) or {}
            for stage in stage_ms:
                # Ultralytics calls the model-forward stage ``inference``.
                value = speed.get("inference" if stage == "model" else stage, 0.0)
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    stage_ms[stage] += float(value)
        _synchronize_model_device(model)
        post_read_batch_seconds.append(time.perf_counter() - scored_started)
        batch_sizes.append(len(paths))

    timing = build_timing_report(
        image_count=len(image_paths),
        post_read_batch_seconds=post_read_batch_seconds,
        batch_sizes=batch_sizes,
        image_read_seconds=image_read_seconds,
        stage_ms=stage_ms,
        batch=batch,
        image_size=image_size,
        max_det=max_det,
        nms_iou=nms_iou,
        warmup_iterations=warmup_iterations,
        device=_model_device_name(model),
        model_load_seconds=model_load_seconds,
        source_shapes=source_shapes,
    )
    timing["max_det_hit_count"] = len(hit_images)
    timing["max_det_hit_images"] = hit_images
    timing["prediction_count_at_collection_floor"] = sum(len(items) for items in predictions.values())
    return predictions, timing, hit_images


def collect_model_predictions(
    model: Any,
    image_paths: list[Path],
    confidence: float,
    image_size: int,
    nms_iou: float,
) -> dict[Path, list[Detection]]:
    """Backwards-compatible untimed collector used by older audit utilities."""
    predictions, _, hit_images = collect_timed_model_predictions(
        model,
        image_paths,
        confidence,
        image_size,
        nms_iou,
        batch=1,
        max_det=DEFAULT_MAX_DET,
    )
    ensure_no_max_det_truncation(hit_images, DEFAULT_MAX_DET)
    return predictions


def evaluate_cached_predictions(
    predicted: dict[Path, list[Detection]],
    data: dict,
    split: str,
    confidence: float,
    class_aware: bool,
    *,
    nms_iou: float | None = None,
    ground_truth: dict[Path, tuple[np.ndarray, np.ndarray]] | None = None,
) -> dict:
    image_dir = split_image_dir(data, split)
    image_paths = sorted([p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES])
    if ground_truth is None:
        ground_truth = load_ground_truth(data, split, image_paths=image_paths)
    overall = np.zeros(3, dtype=np.int64)
    per_group = {name: np.zeros(3, dtype=np.int64) for name in COARSE_NAMES}
    for image_path in image_paths:
        classes, boxes = ground_truth[image_path]
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
    report = {
        "split": split,
        "confidence": confidence,
        "class_aware_matching": class_aware,
        "overall": metric_dict(*overall.tolist()),
        "per_group": {name: metric_dict(*values.tolist()) for name, values in per_group.items()},
    }
    if nms_iou is not None:
        report["nms_iou"] = float(nms_iou)
    return report


def load_ground_truth(
    data: dict[str, Any],
    split: str,
    *,
    image_paths: list[Path] | None = None,
) -> dict[Path, tuple[np.ndarray, np.ndarray]]:
    """Read dimensions and deduplicated labels once for an entire grid scan."""
    import cv2

    image_dir = split_image_dir(data, split)
    label_dir = Path(data["path"]) / "labels" / split
    paths = image_paths or sorted(path for path in image_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    ground_truth: dict[Path, tuple[np.ndarray, np.ndarray]] = {}
    for image_path in paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to read evaluation image: {image_path}")
        height, width = image.shape[:2]
        classes, boxes_n = read_yolo_labels(
            image_to_label_path(image_path, image_dir, label_dir),
            deduplicate=True,
        )
        ground_truth[image_path] = (classes, xywhn_to_xyxy(boxes_n, width, height))
    return ground_truth


def enrich_competition_candidate(report: dict[str, Any], timing: dict[str, Any]) -> dict[str, Any]:
    """Attach all seven ranking items and both safety gates to one candidate."""
    enriched = copy.deepcopy(report)
    enriched["competition_ranking_items"] = competition_ranking_items(enriched, timing)
    enriched["overall_safety_gate"] = overall_safety_gate(enriched, timing)
    # The full timing object is attached so a selected operating point can be
    # snapshotted directly by experiment_ledger without a second benchmark.
    enriched["timing"] = copy.deepcopy(timing)
    return enriched


def sha256_file(path: str | Path) -> str | None:
    source = Path(path)
    if not source.is_file():
        return None
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def make_frozen_operating_point(
    point: dict[str, Any],
    *,
    selection: str,
    model: str,
    data_yaml: str,
    image_size: int,
    batch: int,
    max_det: int,
) -> dict[str, Any]:
    if point.get("class_aware_matching") is not True:
        raise ValueError("a formal operating point must use class-aware coarse-group matching")
    return {
        "schema_version": "fhit.competition_operating_point.v1",
        "selection": selection,
        "confidence": float(point["confidence"]),
        "nms_iou": float(point["nms_iou"]),
        "image_size": int(image_size),
        "batch": int(batch),
        "max_det": int(max_det),
        "class_aware_matching": True,
        "iou_thresholds": copy.deepcopy(OFFICIAL_IOU_THRESHOLDS),
        "model": str(Path(model).resolve()) if Path(model).exists() else str(model),
        "model_sha256": sha256_file(model),
        "data_yaml": str(Path(data_yaml).resolve()),
        "validation_snapshot": {
            "overall": copy.deepcopy(point["overall"]),
            "per_group": copy.deepcopy(point["per_group"]),
            "competition_ranking_items": copy.deepcopy(point["competition_ranking_items"]),
            "overall_safety_gate": copy.deepcopy(point["overall_safety_gate"]),
        },
    }


def load_frozen_operating_point(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "fhit.competition_operating_point.v1":
        raise ValueError(f"unsupported operating-point schema in {source}")
    if payload.get("class_aware_matching") is not True:
        raise ValueError("frozen formal operating point must use class-aware matching")
    for key in ("confidence", "nms_iou"):
        value = payload.get(key)
        if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"frozen operating point has invalid {key}")
    for key in ("image_size", "batch", "max_det"):
        value = payload.get(key)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"frozen operating point has invalid {key}")
    if payload.get("iou_thresholds") != OFFICIAL_IOU_THRESHOLDS:
        raise ValueError("frozen operating point does not use the official IoU thresholds")
    return payload


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
    parser.add_argument("--nms-iou", type=float, action="append", default=None, help="Evaluate this NMS IoU. Repeat to scan explicit values.")
    parser.add_argument(
        "--nms-iou-range",
        type=float,
        nargs=3,
        metavar=("START", "STOP", "STEP"),
        help="Evaluate an inclusive NMS-IoU range, for example: --nms-iou-range 0.45 0.75 0.05.",
    )
    parser.add_argument("--batch", type=int, default=None, help=f"Inference batch size (default: {DEFAULT_BATCH}).")
    parser.add_argument(
        "--max-det",
        type=int,
        default=None,
        help=f"Maximum detections per image (default: {DEFAULT_MAX_DET}; evaluation fails if reached).",
    )
    parser.add_argument("--image-size", type=int, default=None, help="Override dataset.image_size for the benchmark.")
    parser.add_argument("--device", default=None, help="Ultralytics device, for example 0 or cpu.")
    parser.add_argument("--warmup-iterations", type=int, default=3, help="Unscored model warm-up calls before timing.")
    parser.add_argument(
        "--allow-max-det-hit",
        action="store_true",
        help="Diagnostic only: keep a report when max_det is reached; formal_protocol_valid will be false.",
    )
    parser.add_argument("--class-aware", action="store_true", help="Require the predicted ship/aircraft/vehicle group to match the GT group.")
    parser.add_argument("--class-agnostic", action="store_true", help="Diagnostic only: allow cross-group matches.")
    parser.add_argument("--use-operating-point", help="Load a frozen fhit.competition_operating_point.v1 JSON and run only that point.")
    parser.add_argument("--freeze-operating-point", help="Write the selected point as an immutable JSON for test/large-image inference.")
    parser.add_argument(
        "--freeze-selection",
        choices=("gate_min_fdr", "best_f1", "max_recall_under_fdr"),
        default="best_f1",
        help="Which selected validation point to freeze (default: best_f1).",
    )
    args = parser.parse_args()
    if args.class_aware and args.class_agnostic:
        parser.error("--class-aware and --class-agnostic cannot be used together")
    if args.confidence and args.confidence_range:
        parser.error("--confidence and --confidence-range cannot be used together")
    if args.nms_iou and args.nms_iou_range:
        parser.error("--nms-iou and --nms-iou-range cannot be used together")
    explicit_grid = args.confidence or args.confidence_range or args.nms_iou or args.nms_iou_range
    if args.use_operating_point and explicit_grid:
        parser.error("--use-operating-point cannot be combined with confidence/NMS scan arguments")
    if args.use_operating_point and any(
        value is not None for value in (args.batch, args.max_det, args.image_size)
    ):
        parser.error("--use-operating-point cannot override --batch, --max-det or --image-size")
    if args.use_operating_point and (args.class_aware or args.class_agnostic):
        parser.error("--use-operating-point already freezes class-aware matching")
    if args.warmup_iterations < 0:
        parser.error("--warmup-iterations cannot be negative")
    config = load_config(args.config)
    data = resolve_data_yaml(config)
    image_dir = split_image_dir(data, args.split)
    image_paths = sorted([p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES])
    if not image_paths:
        parser.error(f"no images found under {image_dir}")

    frozen = None
    if args.use_operating_point:
        try:
            frozen = load_frozen_operating_point(args.use_operating_point)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            parser.error(str(exc))
        frozen_model_sha = frozen.get("model_sha256")
        current_model_sha = sha256_file(args.model)
        if frozen_model_sha and current_model_sha != frozen_model_sha:
            parser.error(
                f"--model SHA256 does not match frozen operating point: "
                f"expected {frozen_model_sha}, got {current_model_sha}"
            )
    evaluation = config.get("evaluation", {})
    try:
        if frozen:
            confidences = [float(frozen["confidence"])]
            nms_values = [float(frozen["nms_iou"])]
        else:
            confidences = confidence_range(*args.confidence_range) if args.confidence_range else (args.confidence or evaluation.get("confidence_grid", [0.50]))
            nms_values = nms_iou_range(*args.nms_iou_range) if args.nms_iou_range else (args.nms_iou or [evaluation.get("nms_iou", 0.50)])
    except ValueError as exc:
        parser.error(str(exc))
    if any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0 for value in confidences):
        parser.error("--confidence must contain finite values between 0 and 1")
    if any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0 for value in nms_values):
        parser.error("--nms-iou must contain finite values between 0 and 1")
    confidences = sorted(set(float(value) for value in confidences))
    nms_values = sorted(set(float(value) for value in nms_values))

    # P0 is class-aware by construction.  The only way to disable it is the
    # explicit diagnostic flag; a stale config cannot silently weaken scoring.
    class_aware = not args.class_agnostic
    image_size = int(
        frozen["image_size"]
        if frozen
        else (args.image_size if args.image_size is not None else config["dataset"]["image_size"])
    )
    batch = int(
        frozen["batch"]
        if frozen
        else (args.batch if args.batch is not None else evaluation.get("batch", DEFAULT_BATCH))
    )
    max_det = int(
        frozen["max_det"]
        if frozen
        else (args.max_det if args.max_det is not None else evaluation.get("max_det", DEFAULT_MAX_DET))
    )
    if image_size <= 0 or batch <= 0 or max_det <= 0:
        parser.error("--image-size, --batch and --max-det must be positive")

    from ultralytics import YOLO
    model_load_started = time.perf_counter()
    model = YOLO(args.model)
    model_load_seconds = time.perf_counter() - model_load_started
    warmup_seconds = warmup_model(
        model,
        iterations=args.warmup_iterations,
        image_size=image_size,
        confidence=min(confidences),
        nms_iou=nms_values[0],
        max_det=max_det,
        device=args.device,
    )
    ground_truth = load_ground_truth(data, args.split, image_paths=image_paths)

    candidates: list[dict[str, Any]] = []
    timing_by_nms: dict[str, dict[str, Any]] = {}
    all_max_det_hits: list[str] = []
    for nms_iou in nms_values:
        print(f"inference nms_iou={nms_iou:.4f} conf_floor={min(confidences):.4f} batch={batch} max_det={max_det}")
        predictions, timing, hit_images = collect_timed_model_predictions(
            model,
            image_paths,
            min(confidences),
            image_size,
            nms_iou,
            batch=batch,
            max_det=max_det,
            warmup_iterations=0,
            device=args.device,
            model_load_seconds=model_load_seconds,
        )
        timing["warmup_iterations"] = args.warmup_iterations
        timing["warmup_total_seconds_excluded"] = warmup_seconds
        timing_by_nms[f"{nms_iou:.10g}"] = timing
        all_max_det_hits.extend(hit_images)
        if hit_images and not args.allow_max_det_hit:
            ensure_no_max_det_truncation(hit_images, max_det)
        for confidence in confidences:
            report = evaluate_cached_predictions(
                predictions,
                data,
                args.split,
                confidence,
                class_aware,
                nms_iou=nms_iou,
                ground_truth=ground_truth,
            )
            candidates.append(enrich_competition_candidate(report, timing))

    operating_points = select_operating_points(candidates)
    for point in operating_points.values():
        point["model"] = str(args.model)
    primary_name = args.freeze_selection
    primary_point = operating_points[primary_name]
    formal_protocol_valid = bool(class_aware and not all_max_det_hits)
    result = {
        "schema_version": "fhit.competition_eval.v3",
        "split": args.split,
        "model": str(args.model),
        "model_sha256": sha256_file(args.model),
        "class_aware_matching": class_aware,
        "primary_operating_point": primary_name,
        # Keep selected as a backwards-compatible alias.  It is a conservative
        # hard-gate point, not the balanced competition optimum.
        "selected": operating_points["gate_min_fdr"],
        "operating_points": operating_points,
        "all_thresholds": candidates,
        "timing": copy.deepcopy(primary_point["timing"]),
        "competition_ranking_items": copy.deepcopy(primary_point["competition_ranking_items"]),
        "overall_safety_gate": copy.deepcopy(primary_point["overall_safety_gate"]),
        "timing_by_nms": timing_by_nms,
        "formal_protocol_valid": formal_protocol_valid,
        "formal_protocol_violations": [
            *([] if class_aware else ["class_aware_matching_is_false"]),
            *([] if not all_max_det_hits else ["max_det_reached"]),
        ],
        "rules": {
            "class_aware_matching": class_aware,
            "iou_thresholds": OFFICIAL_IOU_THRESHOLDS,
            "one_to_one_confidence_ordered_matching": True,
            "duplicate_predictions_count_as_false_positives": True,
            "nms_iou_grid": nms_values,
            "confidence_grid": confidences,
            "minimum_recall": MINIMUM_RECALL,
            "maximum_false_alarm_rate": MAXIMUM_FALSE_ALARM_RATE,
            "maximum_seconds_per_image_excluding_image_read": MAXIMUM_SECONDS_PER_IMAGE,
            "batch": batch,
            "max_det": max_det,
            "max_det_hit_count": len(all_max_det_hits),
            "image_size": image_size,
        },
    }
    json_dump(result, args.out)
    if args.freeze_operating_point:
        if not formal_protocol_valid:
            parser.error("cannot freeze a non-formal operating point (class-aware matching and uncapped output are required)")
        frozen_output = make_frozen_operating_point(
            primary_point,
            selection=primary_name,
            model=args.model,
            data_yaml=config["paths"]["data_yaml"],
            image_size=image_size,
            batch=batch,
            max_det=max_det,
        )
        json_dump(frozen_output, args.freeze_operating_point)
        print(f"frozen {primary_name} to {args.freeze_operating_point}")
    print(f"saved {args.out}")
    print({
        name: {
            "confidence": point["confidence"],
            "nms_iou": point["nms_iou"],
            "overall": point["overall"],
            "overall_safety_gate": point["overall_safety_gate"],
        }
        for name, point in operating_points.items()
    })


if __name__ == "__main__":
    main()
