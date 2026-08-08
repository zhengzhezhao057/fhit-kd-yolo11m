from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

from .common import json_dump, load_config, resolve_data_yaml, split_image_dir
from .dataset_d0 import inventory_fingerprint, scan_split


def assign_product_folds(inventory: list[dict], folds: int, seed: int) -> dict[str, int]:
    if folds < 2:
        raise ValueError("OOF requires at least two folds.")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in inventory:
        grouped[row["product"]].append(row)
    if len(grouped) < folds:
        raise RuntimeError(f"Only {len(grouped)} product groups are available for {folds} folds.")
    total_classes = np.zeros(25, dtype=np.float64)
    group_vectors = {}
    for product, rows in grouped.items():
        vector = np.zeros(25, dtype=np.float64)
        for row in rows:
            for class_id in row["classes"]:
                vector[int(class_id)] += 1
        group_vectors[product] = vector
        total_classes += vector
    target_classes = total_classes / folds
    target_images = len(inventory) / folds
    rng = np.random.default_rng(seed)
    tie_break = {name: float(rng.random()) for name in grouped}
    rarity = 1.0 / np.maximum(total_classes, 1.0)
    order = sorted(
        grouped,
        key=lambda name: (
            -float((group_vectors[name] * rarity).sum()),
            -len(grouped[name]),
            tie_break[name],
            name,
        ),
    )
    fold_classes = [np.zeros(25, dtype=np.float64) for _ in range(folds)]
    fold_images = [0 for _ in range(folds)]
    assignment = {}
    for position, product in enumerate(order):
        vector = group_vectors[product]; count = len(grouped[product])
        # Seed every fold before optimizing. This is a hard guard against an
        # empty OOF validation fold under highly multi-label product groups.
        if position < folds:
            selected = position
            assignment[product] = selected
            fold_classes[selected] += vector; fold_images[selected] += count
            continue
        scores = []
        for fold in range(folds):
            candidate_classes = np.stack(fold_classes).copy()
            candidate_images = np.asarray(fold_images, dtype=np.float64).copy()
            candidate_classes[fold] += vector; candidate_images[fold] += count
            class_error = np.mean(
                ((candidate_classes - target_classes[None, :]) / np.maximum(target_classes[None, :], 1.0)) ** 2
            )
            image_error = np.mean(((candidate_images - target_images) / max(target_images, 1.0)) ** 2)
            scores.append((class_error + 0.25 * image_error, fold_images[fold], fold))
        selected = min(scores)[-1]
        assignment[product] = selected
        fold_classes[selected] += vector; fold_images[selected] += count
    return assignment


def build_folds(config: dict, output: Path, folds: int, seed: int) -> dict:
    data = resolve_data_yaml(config)
    root = Path(data["path"]).resolve()
    train_images = split_image_dir(data, "train").resolve()
    _, inventory = scan_split(
        root, train_images, root / "labels" / "train", int(config["dataset"]["image_size"]), hash_images=False
    )
    assignment = assign_product_folds(inventory, folds, seed)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for fold in range(folds):
        validation = [row for row in inventory if assignment[row["product"]] == fold]
        training = [row for row in inventory if assignment[row["product"]] != fold]
        train_products = {row["product"] for row in training}
        val_products = {row["product"] for row in validation}
        if train_products.intersection(val_products):
            raise RuntimeError(f"OOF product leakage detected in fold {fold}.")
        if not validation:
            raise RuntimeError(f"OOF fold {fold} is empty; product balancing failed.")
        train_txt = output / f"fold{fold}_train.txt"
        val_txt = output / f"fold{fold}_val.txt"
        train_txt.write_text("\n".join(row["image"] for row in training) + "\n", encoding="utf-8", newline="\n")
        val_txt.write_text("\n".join(row["image"] for row in validation) + "\n", encoding="utf-8", newline="\n")
        fold_yaml = output / f"fold{fold}.yaml"
        fold_yaml.write_text(yaml.safe_dump({
            "path": str(root), "train": str(train_txt.resolve()), "val": str(val_txt.resolve()),
            "nc": int(data["nc"]), "names": data["names"],
        }, allow_unicode=True, sort_keys=False), encoding="utf-8")
        class_counts = Counter(class_id for row in validation for class_id in row["classes"])
        rows.append({
            "fold": fold, "train_images": len(training), "val_images": len(validation),
            "train_products": len(train_products), "val_products": len(val_products),
            "val_per_class": {str(index): class_counts[index] for index in range(25)},
            "train_list": str(train_txt.resolve()), "val_list": str(val_txt.resolve()),
            "data_yaml": str(fold_yaml.resolve()),
        })
    report = {
        "format": 1, "kind": "dataset_d0_product_grouped_oof", "source_split": "train",
        "read_only_source": True, "folds": folds, "seed": seed,
        "train_inventory_fingerprint": inventory_fingerprint(inventory),
        "product_assignments": dict(sorted(assignment.items())), "fold_summaries": rows,
    }
    json_dump(report, output / "folds.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build product-grouped OOF lists without modifying Dataset-D0.")
    parser.add_argument("--config", default="configs/direction1.yaml")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--out", default="reports/dataset_d0/oof3")
    args = parser.parse_args()
    config = load_config(args.config); root = Path(config["paths"]["project_root"])
    output = Path(args.out); output = output if output.is_absolute() else root / output
    report = build_folds(config, output, args.folds, args.seed)
    print(f"saved {output / 'folds.json'}; source Dataset-D0 was not modified")
    for row in report["fold_summaries"]:
        print(row)


if __name__ == "__main__":
    main()
