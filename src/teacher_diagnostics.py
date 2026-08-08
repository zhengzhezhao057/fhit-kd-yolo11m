from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

import torch

from .common import COARSE_NAMES, FINE_TO_COARSE, load_config, resolve_data_yaml


def size_bucket(box_xywhn: torch.Tensor, image_size: int) -> str:
    width_px = float(box_xywhn[2]) * image_size
    height_px = float(box_xywhn[3]) * image_size
    area = width_px * height_px
    if area < 32.0**2:
        return "small"
    if area < 96.0**2:
        return "medium"
    return "large"


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {name: 0.0 for name in ("p10", "p25", "p50", "p75", "p90")}
    tensor = torch.tensor(values, dtype=torch.float32)
    result = torch.quantile(tensor, torch.tensor([0.10, 0.25, 0.50, 0.75, 0.90]))
    return {name: float(value) for name, value in zip(("p10", "p25", "p50", "p75", "p90"), result)}


def analyse_entries(
    entries: Iterable[dict[str, Any]],
    num_classes: int,
    image_size: int,
    class_names: list[str] | None = None,
    temperature_grid: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 6.0),
) -> dict[str, Any]:
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
    class_total = Counter(); class_correct = Counter(); class_confidence: dict[int, list[float]] = {index: [] for index in range(num_classes)}
    class_entropy: dict[int, list[float]] = {index: [] for index in range(num_classes)}
    size_total = Counter(); size_correct = Counter(); coarse_total = Counter(); coarse_correct = Counter()
    all_logits: list[torch.Tensor] = []; all_truths: list[torch.Tensor] = []
    skipped = 0; entry_count = 0
    for entry in entries:
        entry_count += 1
        classes = entry.get("classes")
        logits = entry.get("roi_logits")
        boxes = entry.get("boxes_xywhn")
        if classes is None or logits is None or boxes is None or len(classes) != len(logits) or len(classes) != len(boxes):
            skipped += 1
            continue
        probabilities = logits.float().softmax(dim=1)
        valid_class_rows = (classes.long() >= 0) & (classes.long() < num_classes)
        if bool(valid_class_rows.any()):
            all_logits.append(logits.float()[valid_class_rows].cpu())
            all_truths.append(classes.long()[valid_class_rows].cpu())
        predictions = probabilities.argmax(dim=1)
        confidences = probabilities.max(dim=1).values
        entropies = -(probabilities.clamp_min(1e-9).log() * probabilities).sum(dim=1) / math.log(num_classes)
        for truth, prediction, confidence, entropy, box in zip(classes.long(), predictions.long(), confidences, entropies, boxes.float()):
            true_index = int(truth); predicted_index = int(prediction)
            if not (0 <= true_index < num_classes and 0 <= predicted_index < num_classes):
                skipped += 1
                continue
            correct = true_index == predicted_index
            confusion[true_index, predicted_index] += 1
            class_total[true_index] += 1; class_correct[true_index] += int(correct)
            class_confidence[true_index].append(float(confidence)); class_entropy[true_index].append(float(entropy))
            bucket = size_bucket(box, image_size)
            size_total[bucket] += 1; size_correct[bucket] += int(correct)
            coarse = COARSE_NAMES[FINE_TO_COARSE[true_index]]
            coarse_total[coarse] += 1; coarse_correct[coarse] += int(correct)

    total = sum(class_total.values()); correct = sum(class_correct.values())
    all_confidences = [value for values in class_confidence.values() for value in values]
    all_entropies = [value for values in class_entropy.values() for value in values]
    per_class = {}
    for index in range(num_classes):
        confidences = class_confidence[index]; entropies = class_entropy[index]
        per_class[str(index)] = {
            "name": class_names[index] if class_names and index < len(class_names) else str(index),
            "instances": class_total[index],
            "correct": class_correct[index],
            "accuracy": _safe_ratio(class_correct[index], class_total[index]),
            "confidence_mean": sum(confidences) / len(confidences) if confidences else 0.0,
            "confidence_quantiles": _quantiles(confidences),
            "entropy_mean": sum(entropies) / len(entropies) if entropies else 0.0,
            "entropy_quantiles": _quantiles(entropies),
        }
    temperature_profiles = {}
    if all_logits:
        logits_tensor = torch.cat(all_logits, dim=0)
        truths_tensor = torch.cat(all_truths, dim=0)
        for temperature in temperature_grid:
            if temperature <= 0:
                raise ValueError(f"Temperature must be positive, got {temperature}")
            softened = (logits_tensor / temperature).softmax(dim=1)
            top1 = softened.max(dim=1).values
            target = softened.gather(1, truths_tensor.unsqueeze(1)).squeeze(1)
            entropy = -(softened.clamp_min(1e-9).log() * softened).sum(dim=1) / math.log(num_classes)
            temperature_profiles[f"{temperature:g}"] = {
                "top1_probability_quantiles": _quantiles(top1.tolist()),
                "target_probability_quantiles": _quantiles(target.tolist()),
                "normalized_entropy_quantiles": _quantiles(entropy.tolist()),
            }
    return {
        "entries": entry_count,
        "skipped_entries_or_rois": skipped,
        "instances": total,
        "correct": correct,
        "accuracy": _safe_ratio(correct, total),
        "confidence_quantiles": _quantiles(all_confidences),
        "entropy_quantiles": _quantiles(all_entropies),
        "temperature_profiles": temperature_profiles,
        "per_class": per_class,
        "per_coarse_group": {
            name: {"instances": coarse_total[name], "correct": coarse_correct[name], "accuracy": _safe_ratio(coarse_correct[name], coarse_total[name])}
            for name in COARSE_NAMES
        },
        "per_size": {
            name: {"instances": size_total[name], "correct": size_correct[name], "accuracy": _safe_ratio(size_correct[name], size_total[name])}
            for name in ("small", "medium", "large")
        },
        "confusion_matrix": confusion.tolist(),
    }


def load_cache_entries(cache_dir: Path, limit: int | None = None) -> Iterator[dict[str, Any]]:
    files = sorted(path for path in cache_dir.glob("*.pt") if path.name != "manifest.pt")
    if limit is not None:
        files = files[:limit]
    for path in files:
        value = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(value, dict):
            raise RuntimeError(f"Teacher cache entry is not a mapping: {path}")
        yield value


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit cached DINOv3 teacher logits by fine class, coarse class and object size.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--cache", default=None, help="Defaults to cache/teacher_signals/train under project_root.")
    parser.add_argument("--output", default="reports/direction1/teacher_diagnostics.json")
    parser.add_argument("--limit", type=int, default=None, help="Optional cache-file limit for a quick smoke test.")
    args = parser.parse_args()

    cfg = load_config(args.config); data = resolve_data_yaml(cfg)
    root = Path(cfg["paths"]["project_root"])
    cache_dir = Path(args.cache) if args.cache else root / "cache" / "teacher_signals" / "train"
    if not cache_dir.exists():
        raise FileNotFoundError(f"Teacher cache not found: {cache_dir}")
    class_names_value = data.get("names")
    if isinstance(class_names_value, dict):
        class_names = [str(class_names_value.get(index, class_names_value.get(str(index), index))) for index in range(len(class_names_value))]
    elif isinstance(class_names_value, list):
        class_names = [str(value) for value in class_names_value]
    else:
        class_names = None
    report = analyse_entries(
        load_cache_entries(cache_dir, args.limit),
        int(cfg["dataset"]["nc"]),
        int(cfg["dataset"]["image_size"]),
        class_names,
    )
    report["cache"] = str(cache_dir.resolve())
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Teacher diagnostics saved to {output}")
    print(json.dumps({
        "instances": report["instances"],
        "accuracy": report["accuracy"],
        "temperature_profiles": report["temperature_profiles"],
        "per_coarse_group": report["per_coarse_group"],
        "per_size": report["per_size"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
