"""Audit real server checkpoints before drawing a distillation conclusion."""
from __future__ import annotations

import argparse
import hashlib
import itertools
from pathlib import Path

import numpy as np
import torch

from .common import json_dump, load_config, resolve_data_yaml, split_image_dir
from .competition_eval import box_iou_one_to_many, collect_model_predictions


def parse_models(values: list[str]) -> list[tuple[str, Path]]:
    models: list[tuple[str, Path]] = []
    seen = set()
    for value in values:
        if "=" not in value:
            raise ValueError(f"--model must be NAME=/absolute/or/relative/path.pt, got {value!r}")
        name, raw_path = value.split("=", 1)
        path = Path(raw_path)
        if not name or name in seen or not path.exists():
            raise ValueError(f"Invalid duplicate name or missing checkpoint: {value!r}")
        models.append((name, path)); seen.add(name)
    return models


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parameter_difference(reference: dict[str, torch.Tensor], candidate: dict[str, torch.Tensor], exclude_addons: bool) -> dict:
    keys = sorted(set(reference).intersection(candidate))
    if exclude_addons:
        keys = [key for key in keys if not key.startswith("distill_addons.")]
    max_abs = mean_abs = 0.0
    changed = 0
    compared = 0
    for key in keys:
        left, right = reference[key], candidate[key]
        if left.shape != right.shape or not (left.is_floating_point() or left.is_complex()):
            continue
        delta = (left.detach().float().cpu() - right.detach().float().cpu()).abs()
        compared += delta.numel(); current_max = float(delta.max()) if delta.numel() else 0.0
        max_abs = max(max_abs, current_max); mean_abs += float(delta.sum())
        changed += int(torch.count_nonzero(delta))
    return {"tensors_compared": len(keys), "elements_compared": compared, "changed_elements": changed, "max_abs_difference": max_abs, "mean_abs_difference": mean_abs / max(compared, 1)}


def prediction_difference(reference: dict, candidate: dict, iou_exact: float) -> dict:
    ref_total = cand_total = matched = 0
    iou_values: list[float] = []; score_deltas: list[float] = []; exact_images = 0
    for path, ref_detections in reference.items():
        cand_detections = candidate[path]
        ref_total += len(ref_detections); cand_total += len(cand_detections)
        unmatched = set(range(len(ref_detections))); image_matches = 0
        for detection in cand_detections:
            indices = [index for index in unmatched if ref_detections[index].fine_class == detection.fine_class]
            if not indices:
                continue
            ious = box_iou_one_to_many(detection.box, np.stack([ref_detections[index].box for index in indices]))
            best_position = int(np.argmax(ious)); iou = float(ious[best_position])
            if iou >= iou_exact:
                index = indices[best_position]; unmatched.remove(index); matched += 1; image_matches += 1
                iou_values.append(iou); score_deltas.append(abs(detection.score - ref_detections[index].score))
        if image_matches == len(ref_detections) == len(cand_detections):
            exact_images += 1
    return {
        "reference_detections": ref_total, "candidate_detections": cand_total, "same_class_iou_matched": matched,
        "reference_unmatched": ref_total - matched, "candidate_unmatched": cand_total - matched,
        "exact_images": exact_images, "total_images": len(reference),
        "mean_matched_iou": float(np.mean(iou_values)) if iou_values else None,
        "mean_matched_score_abs_difference": float(np.mean(score_deltas)) if score_deltas else None,
        "iou_exact_threshold": iou_exact,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare named server checkpoints by SHA256, detector parameters, and raw validation predictions.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--model", action="append", required=True, metavar="NAME=PATH", help="Repeat exactly once for C0, F, K and FK.")
    parser.add_argument("--split", default="val", choices=("val", "test"))
    parser.add_argument("--confidence", type=float, default=0.01, help="Raw-prediction collection threshold.")
    parser.add_argument("--iou-exact", type=float, default=0.999, help="IoU needed to call same-class detections identical.")
    parser.add_argument("--skip-predictions", action="store_true", help="Only audit checkpoint hashes and state_dicts.")
    parser.add_argument("--out", default="reports/experiment_audit.json")
    args = parser.parse_args()
    if not 0 <= args.confidence <= 1 or not 0 <= args.iou_exact <= 1:
        parser.error("--confidence and --iou-exact must be in [0, 1]")
    try:
        models = parse_models(args.model)
    except ValueError as exc:
        parser.error(str(exc))
    cfg = load_config(args.config); data = resolve_data_yaml(cfg)
    from ultralytics import YOLO
    states: dict[str, dict] = {}; predictions: dict[str, dict] = {}
    metadata = {}
    for name, path in models:
        yolo = YOLO(path)
        states[name] = yolo.model.state_dict()
        metadata[name] = {"path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size, "state_dict_tensors": len(states[name])}
        if not args.skip_predictions:
            image_dir = split_image_dir(data, args.split)
            images = sorted(path for path in image_dir.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})
            predictions[name] = collect_model_predictions(yolo, images, args.confidence, cfg["dataset"]["image_size"], cfg["evaluation"]["nms_iou"])
    parameter_pairs = {}; prediction_pairs = {}
    for (left_name, _), (right_name, _) in itertools.combinations(models, 2):
        key = f"{left_name}__vs__{right_name}"
        parameter_pairs[key] = {
            "all_parameters": parameter_difference(states[left_name], states[right_name], exclude_addons=False),
            "detector_only": parameter_difference(states[left_name], states[right_name], exclude_addons=True),
        }
        if predictions:
            prediction_pairs[key] = prediction_difference(predictions[left_name], predictions[right_name], args.iou_exact)
    output = {"split": args.split, "raw_prediction_confidence": args.confidence, "models": metadata, "parameter_pairs": parameter_pairs, "prediction_pairs": prediction_pairs}
    json_dump(output, args.out)
    print(f"saved {args.out}; audited {', '.join(name for name, _ in models)}")


if __name__ == "__main__":
    main()
