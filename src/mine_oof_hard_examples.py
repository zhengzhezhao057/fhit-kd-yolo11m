from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .build_hard_example_manifest import build_manifest as build_hard_manifest
from .build_vehicle_negative_manifest import build_vehicle_negative_manifest
from .common import (
    COARSE_NAMES,
    FINE_TO_COARSE,
    image_to_label_path,
    json_dump,
    load_config,
)
from .common import read_yolo_labels, resolve_data_yaml, split_image_dir, xywhn_to_xyxy
from .competition_eval import metric_dict
from .oof_training import OOF_RUN_PREFIX, fold_summary, load_fold_manifest, sha256_file
from .weak_group_diagnostics import (
    ImageGroundTruth,
    classify_false_positives,
    classify_instances,
    collect_predictions,
    crowded_flags,
    edge_flags,
    evaluate_prediction_map,
    save_false_positive_visuals,
    size_bucket,
    summarize_false_positive_rows,
    summarize_instance_rows,
    write_csv,
)


def validate_oof_image_coverage(fold_paths: list[list[Path]], expected_paths: set[Path]) -> None:
    flattened = [path.resolve() for paths in fold_paths for path in paths]
    counts = Counter(flattened)
    duplicates = [path for path, count in counts.items() if count != 1]
    actual = set(flattened)
    if duplicates or actual != {path.resolve() for path in expected_paths}:
        raise RuntimeError(
            "OOF validation folds must cover every Dataset-D0 TRAIN image exactly once: "
            f"duplicates={len(duplicates)}, missing={len(expected_paths - actual)}, extra={len(actual - expected_paths)}"
        )


def select_safe_vehicle_backgrounds(
    rows: list[dict], *, minimum_score: float = 0.35, maximum_gt_iou: float = 0.05,
    edge_fraction: float = 0.02, maximum_per_image: int = 4,
) -> tuple[list[dict], dict[str, int]]:
    """Conservatively filter OOF vehicle FPs before mandatory human review."""
    if not 0.0 <= minimum_score <= 1.0 or not 0.0 <= maximum_gt_iou <= 1.0:
        raise ValueError("score and IoU thresholds must be between 0 and 1")
    if maximum_per_image < 1:
        raise ValueError("maximum_per_image must be >= 1")
    eligible: dict[str, list[dict]] = defaultdict(list)
    excluded: Counter[str] = Counter()
    for row in rows:
        if row["coarse_group"] != "vehicle":
            excluded["other_group"] += 1
            continue
        if row["reason"] != "background":
            excluded[row["reason"]] += 1
            continue
        if float(row["score"]) < minimum_score:
            excluded["below_minimum_score"] += 1
            continue
        if float(row["nearest_gt_iou"]) > maximum_gt_iou:
            excluded["near_any_gt"] += 1
            continue
        width, height = int(row["image_width"]), int(row["image_height"])
        margin_x, margin_y = width * edge_fraction, height * edge_fraction
        if (
            float(row["box_x1"]) <= margin_x or float(row["box_y1"]) <= margin_y
            or float(row["box_x2"]) >= width - margin_x or float(row["box_y2"]) >= height - margin_y
        ):
            excluded["prediction_at_image_edge"] += 1
            continue
        eligible[str(row["image"])].append(row)
    kept: list[dict] = []
    for image in sorted(eligible):
        candidates = sorted(eligible[image], key=lambda row: (-float(row["score"]), int(row["prediction_index"])))
        kept.extend(candidates[:maximum_per_image])
        excluded["per_image_cap"] += max(0, len(candidates) - maximum_per_image)
    return kept, dict(sorted(excluded.items()))


def load_items(paths: list[Path], image_dir: Path, label_dir: Path) -> list[ImageGroundTruth]:
    import cv2

    items: list[ImageGroundTruth] = []
    for path in paths:
        resolved = path.resolve()
        try:
            resolved.relative_to(image_dir)
        except ValueError as error:
            raise RuntimeError(f"OOF validation image is outside Dataset-D0 TRAIN: {resolved}") from error
        image = cv2.imread(str(resolved))
        if image is None:
            raise RuntimeError(f"Cannot read OOF image: {resolved}")
        height, width = image.shape[:2]
        classes, boxes = read_yolo_labels(
            image_to_label_path(resolved, image_dir, label_dir), deduplicate=True
        )
        items.append(ImageGroundTruth(resolved, classes, xywhn_to_xyxy(boxes, width, height), width, height))
    return items


def add_report_counts(destination: dict, report: dict) -> None:
    for group, values in [("overall", report["overall"]), *report["per_group"].items()]:
        accumulator = destination.setdefault(group, np.zeros(3, dtype=np.int64))
        accumulator += np.asarray([values["TP"], values["FP"], values["FN"]], dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine leakage-free hard examples from the three product-grouped OOF detectors.")
    parser.add_argument("--config", default="configs/direction1.yaml")
    parser.add_argument("--folds-dir", default="reports/dataset_d0/oof3")
    parser.add_argument("--out", default="reports/dataset_d0/oof_mining_v1")
    parser.add_argument("--positive-confidence", type=float, default=0.50)
    parser.add_argument("--negative-confidence", type=float, default=0.35)
    parser.add_argument("--confidence-floor", type=float, default=0.01)
    parser.add_argument("--base-nms-iou", type=float, default=0.50)
    parser.add_argument("--loose-nms-iou", type=float, default=0.80)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--max-det", type=int, default=3000)
    parser.add_argument("--max-visuals", type=int, default=80)
    parser.add_argument("--safe-maximum-gt-iou", type=float, default=0.05)
    parser.add_argument("--safe-edge-fraction", type=float, default=0.02)
    parser.add_argument("--safe-maximum-per-image", type=int, default=4)
    args = parser.parse_args()
    if not 0.0 <= args.confidence_floor <= args.negative_confidence <= args.positive_confidence <= 1.0:
        parser.error("require confidence-floor <= negative-confidence <= positive-confidence")
    if not 0.0 <= args.base_nms_iou <= args.loose_nms_iou <= 1.0:
        parser.error("require base-nms-iou <= loose-nms-iou")

    config = load_config(args.config)
    root = Path(config["paths"]["project_root"]).resolve()
    folds_dir = Path(args.folds_dir)
    folds_dir = folds_dir if folds_dir.is_absolute() else root / folds_dir
    output = Path(args.out)
    output = output if output.is_absolute() else root / output
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output} already contains evidence; choose a new --out to preserve it.")
    # No files are written until all three inference folds finish. Therefore
    # an empty directory left by a killed inference is safe to reuse.
    output.mkdir(parents=True, exist_ok=True)

    fold_manifest = load_fold_manifest(folds_dir)
    data = resolve_data_yaml(config)
    train_dir = split_image_dir(data, "train").resolve()
    label_dir = (Path(data["path"]) / "labels" / "train").resolve()
    expected_images = {
        path.resolve() for path in train_dir.rglob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    }
    fold_paths: list[list[Path]] = []
    for fold in range(3):
        row = fold_summary(fold_manifest, fold)
        fold_paths.append([Path(line) for line in Path(row["val_list"]).read_text(encoding="utf-8").splitlines() if line])
    validate_oof_image_coverage(fold_paths, expected_images)

    from ultralytics import YOLO

    all_items: list[ImageGroundTruth] = []
    instance_rows: list[dict] = []
    false_positive_rows: list[dict] = []
    checkpoints: dict[str, dict] = {}
    confidence_grid = [0.10, 0.20, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70]
    aggregate_reports: dict[float, dict] = {confidence: {} for confidence in confidence_grid}
    for fold in range(3):
        checkpoint = root / "runs" / f"{OOF_RUN_PREFIX}{fold}" / "weights" / "last.pt"
        provenance = root / "runs" / f"{OOF_RUN_PREFIX}{fold}" / "oof_provenance.json"
        if not checkpoint.exists() or not provenance.exists():
            raise FileNotFoundError(f"Fold {fold} last.pt/provenance is incomplete.")
        provenance_data = json.loads(provenance.read_text(encoding="utf-8"))
        if (
            int(provenance_data.get("fold", -1)) != fold
            or provenance_data.get("train_inventory_fingerprint") != fold_manifest["train_inventory_fingerprint"]
        ):
            raise RuntimeError(f"Fold {fold} checkpoint provenance does not match folds.json.")
        # last.pt is deliberate: its fixed epoch was not selected by looking
        # at this held-out fold, unlike best.pt, so OOF mining remains strict.
        checkpoints[str(fold)] = {
            "path": str(checkpoint), "sha256": sha256_file(checkpoint), "selection": "fixed_epoch_last_pt"
        }
        items = load_items(fold_paths[fold], train_dir, label_dir)
        all_items.extend(items)
        model = YOLO(checkpoint)
        print(f"OOF MINE fold={fold}: images={len(items)} checkpoint=last.pt", flush=True)
        base = collect_predictions(
            model, items, image_size=640, confidence=args.confidence_floor,
            nms_iou=args.base_nms_iou, batch=args.batch, max_det=args.max_det,
        )
        loose = collect_predictions(
            model, items, image_size=640, confidence=args.confidence_floor,
            nms_iou=args.loose_nms_iou, batch=args.batch, max_det=args.max_det,
        )
        for confidence in confidence_grid:
            add_report_counts(aggregate_reports[confidence], evaluate_prediction_map(base, items, confidence))
        for item in items:
            key = item.image_path.resolve()
            current = [prediction for prediction in base[key] if prediction.score >= args.positive_confidence]
            negative_current = [prediction for prediction in base[key] if prediction.score >= args.negative_confidence]
            classifications = classify_instances(current, base[key], loose[key], item.boxes, item.classes)
            crowded = crowded_flags(item.boxes, item.classes)
            edges = edge_flags(item.boxes, item.width, item.height)
            for gt_index, result in enumerate(classifications):
                fine_class = int(item.classes[gt_index])
                box = item.boxes[gt_index]
                instance_rows.append({
                    "model": "OOF", "fold": fold, "image": str(item.image_path), "gt_index": gt_index,
                    "fine_class": fine_class, "coarse_group": COARSE_NAMES[FINE_TO_COARSE[fine_class]],
                    "size": size_bucket(box, item.width, item.height, 640),
                    "crowded": bool(crowded[gt_index]), "edge": bool(edges[gt_index]),
                    "gt_x": float((box[0] + box[2]) / (2.0 * item.width)),
                    "gt_y": float((box[1] + box[3]) / (2.0 * item.height)),
                    "gt_w": float((box[2] - box[0]) / item.width),
                    "gt_h": float((box[3] - box[1]) / item.height),
                    **result,
                })
            for false_positive in classify_false_positives(negative_current, item.boxes, item.classes):
                box = np.asarray([false_positive[key] for key in ("box_x1", "box_y1", "box_x2", "box_y2")])
                false_positive_rows.append({
                    "model": "OOF", "fold": fold, "image": str(item.image_path),
                    "image_width": item.width, "image_height": item.height,
                    "predicted_size": size_bucket(box, item.width, item.height, 640),
                    **false_positive,
                })
        del model
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    expected_instances = sum(len(item.classes) for item in all_items)
    if len(instance_rows) != expected_instances or len({item.image_path for item in all_items}) != len(expected_images):
        raise RuntimeError("OOF mining output does not cover the full deduplicated Dataset-D0 TRAIN population.")
    instances_csv = output / "oof_instances.csv"
    false_positives_csv = output / "oof_false_positives_c035.csv"
    write_csv(instances_csv, instance_rows)
    write_csv(false_positives_csv, false_positive_rows)

    safe_rows, safe_excluded = select_safe_vehicle_backgrounds(
        false_positive_rows, minimum_score=args.negative_confidence,
        maximum_gt_iou=args.safe_maximum_gt_iou, edge_fraction=args.safe_edge_fraction,
        maximum_per_image=args.safe_maximum_per_image,
    )
    safe_csv = output / "safe_vehicle_background_candidates.csv"
    write_csv(safe_csv, safe_rows)
    hard_manifest = build_hard_manifest(config, instances_csv, "OOF")
    json_dump(hard_manifest, output / "hard_examples_oof.json")
    if safe_rows:
        negative_manifest = build_vehicle_negative_manifest(config, safe_csv, "OOF", args.negative_confidence)
        negative_manifest.update({
            "review_required": True,
            "source": "product_grouped_oof_fixed_epoch_last_pt",
            "safe_maximum_gt_iou": args.safe_maximum_gt_iou,
            "safe_edge_fraction": args.safe_edge_fraction,
            "safe_maximum_per_image": args.safe_maximum_per_image,
        })
        json_dump(negative_manifest, output / "vehicle_background_candidates_review_required.json")
        if args.max_visuals > 0:
            save_false_positive_visuals(output, "OOF_SAFE", safe_rows, all_items, args.max_visuals)

    matrix = []
    for confidence, groups in aggregate_reports.items():
        for group, counts in groups.items():
            matrix.append({
                "confidence": confidence, "group": group,
                **metric_dict(int(counts[0]), int(counts[1]), int(counts[2])),
            })
    write_csv(output / "threshold_matrix.csv", matrix)
    summary = {
        "kind": "dataset_d0_product_grouped_oof_mining",
        "source_split": "train",
        "read_only_dataset_d0": True,
        "checkpoint_policy": "fixed_epoch_last_pt_not_val_selected_best_pt",
        "fold_inventory_fingerprint": fold_manifest["train_inventory_fingerprint"],
        "images": len(expected_images), "instances": len(instance_rows),
        "checkpoints": checkpoints,
        "rules": {
            "positive_confidence": args.positive_confidence,
            "negative_confidence": args.negative_confidence,
            "confidence_floor": args.confidence_floor,
            "base_nms_iou": args.base_nms_iou,
            "loose_nms_iou": args.loose_nms_iou,
            "competition_iou": {"ship": 0.50, "aircraft": 0.50, "vehicle": 0.35},
        },
        "hard_examples": summarize_instance_rows(instance_rows),
        "false_positives_at_negative_confidence": summarize_false_positive_rows(false_positive_rows),
        "safe_vehicle_background_candidates": {
            "boxes": len(safe_rows), "images": len({row["image"] for row in safe_rows}),
            "excluded": safe_excluded, "review_required": True,
        },
    }
    json_dump(summary, output / "summary.json")
    print(
        f"OOF MINING COMPLETE: images={len(expected_images)} instances={len(instance_rows)} "
        f"safe_vehicle_candidates={len(safe_rows)} output={output}", flush=True,
    )


if __name__ == "__main__":
    main()
