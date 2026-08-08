from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from .artifact_paths import dataset_root, manifests_dir, oof_dir
from .common import json_dump, read_yolo_labels
from .dataset_d0 import image_files, inventory_fingerprint
from .dataset_registry import load_manifest
from .build_oof_folds import assign_product_folds


def build_scene_inventory(dataset_root: Path, manifest_path: Path) -> tuple[list[dict], dict]:
    """TRAIN inventory with authoritative manifest scene_id per image."""
    root = Path(dataset_root)
    rows = load_manifest(manifest_path)
    by_image = {(row["split"], row["image"]): row for row in rows}
    inventory = []
    summary = Counter()
    for image_path in image_files(root / "images" / "train"):
        relative = image_path.relative_to(root).as_posix()
        key = ("train", image_path.relative_to(root / "images" / "train").as_posix())
        entry = by_image.get(key)
        if entry is None:
            raise RuntimeError(f"TRAIN image missing from manifest: {relative}")
        label_path = root / "labels" / "train" / image_path.relative_to(root / "images" / "train").with_suffix(".txt")
        classes, _ = read_yolo_labels(label_path, deduplicate=True)
        summary["instances"] += len(classes)
        inventory.append({
            "relative_image": relative,
            "image": str(image_path.resolve()),
            "scene_id": entry["scene_id"],
            "classes": [int(value) for value in classes],
        })
    return inventory, {"images": len(inventory), "instances": summary["instances"]}


def build_folds(dataset_root: Path, manifest_path: Path, *, folds: int = 3, seed: int = 20260804, out_dir: Path) -> dict:
    root = Path(dataset_root)
    inventory, summary = build_scene_inventory(root, manifest_path)
    # Re-key inventory as {"product": scene_id, ...} for the existing balancer.
    grouped = [{"product": row["scene_id"], "image": row["image"], "classes": row["classes"]} for row in inventory]
    assignment = assign_product_folds(grouped, folds, seed)
    scene_assignment = {row["scene_id"]: assignment[row["scene_id"]] for row in inventory}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for fold in range(folds):
        validation = [row for row in inventory if scene_assignment[row["scene_id"]] == fold]
        training = [row for row in inventory if scene_assignment[row["scene_id"]] != fold]
        train_scenes = {row["scene_id"] for row in training}
        val_scenes = {row["scene_id"] for row in validation}
        if train_scenes.intersection(val_scenes):
            raise RuntimeError(f"Scene leakage detected in fold {fold}.")
        if not validation:
            raise RuntimeError(f"Fold {fold} is empty.")
        train_txt = out / f"fold{fold}_train.txt"
        val_txt = out / f"fold{fold}_val.txt"
        train_txt.write_text("\n".join(row["image"] for row in training) + "\n", encoding="utf-8", newline="\n")
        val_txt.write_text("\n".join(row["image"] for row in validation) + "\n", encoding="utf-8", newline="\n")
        fold_yaml = out / f"fold{fold}.yaml"
        with (root / "dataset.yaml").open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
        fold_yaml.write_text(yaml.safe_dump({
            "path": str(root), "train": str(train_txt.resolve()), "val": str(val_txt.resolve()),
            "nc": int(data["nc"]), "names": data["names"],
        }, allow_unicode=True, sort_keys=False), encoding="utf-8")
        class_counts = Counter(class_id for row in validation for class_id in row["classes"])
        rows.append({
            "fold": fold, "train_images": len(training), "val_images": len(validation),
            "train_scenes": len(train_scenes), "val_scenes": len(val_scenes),
            "val_per_class": {str(index): class_counts[index] for index in range(25)},
            "train_list": str(train_txt.resolve()), "val_list": str(val_txt.resolve()),
            "data_yaml": str(fold_yaml.resolve()),
        })
    report = {
        "format": 1,
        "kind": "scene811_scene_grouped_oof",
        "dataset_id": "scene811_v2",
        "read_only_source": True,
        "source_split": "train",
        "folds": folds,
        "seed": seed,
        "train_inventory_fingerprint": inventory_fingerprint([
            {"relative_image": row["relative_image"], "image_sha256": row["image"], "label_sha256": "", "scene": row["scene_id"]}
            for row in inventory
        ]),
        "note": "scene_id/cluster_id are authoritative from the source-aware split manifest; near-duplicate review must be incorporated into cluster_id before formal OOF use",
        "fold_summaries": rows,
    }
    json_dump(report, out / "folds.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build scene-grouped OOF folds for Scene811 TRAIN.")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    root = Path(args.dataset_root) if args.dataset_root else dataset_root()
    manifest = Path(args.manifest) if args.manifest else root / "split_manifest.csv"
    out = Path(args.out) if args.out else oof_dir() / "oof3"
    report = build_folds(root, manifest, folds=args.folds, seed=args.seed, out_dir=out)
    print(f"saved {out / 'folds.json'}")
    for row in report["fold_summaries"]:
        print(row)


if __name__ == "__main__":
    main()
