from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from .dataset_d0 import IMAGE_SUFFIXES, file_sha256, image_files


REQUIRED_SOURCE_FIELDS = {"image", "source", "scene_id", "cluster_id"}
SPLITS = ("train", "val", "test")


def load_sources(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = REQUIRED_SOURCE_FIELDS.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Source manifest is missing columns: {sorted(missing)}")
        rows = list(reader)
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        key = Path(row["image"].strip()).name.casefold()
        if key in result:
            raise ValueError(f"Duplicate source-manifest image: {row['image']}")
        source = row["source"].strip().lower()
        if source not in {"official", "added"}:
            raise ValueError(f"Invalid source {source!r} for {row['image']}")
        if not row["scene_id"].strip() or not row["cluster_id"].strip():
            raise ValueError(f"scene_id/cluster_id must not be empty for {row['image']}")
        result[key] = {
            "source": source,
            "scene_id": row["scene_id"].strip(),
            "cluster_id": row["cluster_id"].strip(),
        }
    return result


def label_for(image: Path, images_root: Path, labels_root: Path) -> Path:
    relative = image.relative_to(images_root)
    return (labels_root / relative).with_suffix(".txt")


def read_classes(label: Path, nc: int) -> Counter[int]:
    counts: Counter[int] = Counter()
    if not label.is_file():
        raise FileNotFoundError(f"Missing label for dataset image: {label}")
    for number, raw in enumerate(label.read_text(encoding="utf-8").splitlines(), start=1):
        fields = raw.split()
        if not fields:
            continue
        if len(fields) != 5:
            raise ValueError(f"Invalid YOLO label at {label}:{number}")
        class_id = int(fields[0])
        if not 0 <= class_id < nc:
            raise ValueError(f"Class id {class_id} outside 0..{nc - 1} at {label}:{number}")
        counts[class_id] += 1
    return counts


def inventory(dataset_root: Path, source_manifest: Path, nc: int) -> list[dict]:
    root = dataset_root.resolve()
    images_root, labels_root = root / "images", root / "labels"
    sources = load_sources(source_manifest)
    found: set[str] = set()
    rows: list[dict] = []
    for image in image_files(images_root):
        key = image.name.casefold()
        if key not in sources:
            raise RuntimeError(f"Image is absent from source manifest: {image.name}")
        if key in found:
            raise RuntimeError(f"Duplicate image basename below images/: {image.name}")
        found.add(key)
        label = label_for(image, images_root, labels_root)
        source = sources[key]
        rows.append({
            "image": image.resolve(),
            "label": label.resolve(),
            "source": source["source"],
            "scene_id": source["scene_id"],
            "cluster_id": source["cluster_id"],
            "classes": read_classes(label, nc),
        })
    absent = sorted(set(sources).difference(found))
    if absent:
        raise RuntimeError(f"Source manifest contains {len(absent)} absent images; examples={absent[:10]}")
    return rows


def grouped(rows: list[dict], source: str) -> list[dict]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["source"] == source:
            buckets[row["cluster_id"]].append(row)
    groups = []
    for cluster_id, members in buckets.items():
        counts: Counter[int] = Counter()
        for member in members:
            counts.update(member["classes"])
        groups.append({"cluster_id": cluster_id, "rows": members, "images": len(members), "classes": counts})
    return groups


def assign_official_groups(
    groups: list[dict],
    ratios: dict[str, float],
    seed: int,
    nc: int,
    forced_train: set[str] | None = None,
) -> dict[str, str]:
    if not groups:
        raise RuntimeError("No official groups were found.")
    rng = random.Random(seed)
    total_images = sum(group["images"] for group in groups)
    total_classes: Counter[int] = Counter()
    for group in groups:
        total_classes.update(group["classes"])
    targets_images = {split: total_images * ratios[split] for split in SPLITS}
    targets_classes = {
        split: {class_id: total_classes[class_id] * ratios[split] for class_id in range(nc)}
        for split in SPLITS
    }
    assigned_images = Counter()
    assigned_classes = {split: Counter() for split in SPLITS}
    tie_break = {group["cluster_id"]: rng.random() for group in groups}
    rarity = {
        class_id: 1.0 / max(total_classes[class_id], 1)
        for class_id in range(nc)
    }
    forced_train = forced_train or set()
    forced_groups = [group for group in groups if group["cluster_id"] in forced_train]
    remaining_groups = [group for group in groups if group["cluster_id"] not in forced_train]
    ordered = sorted(
        remaining_groups,
        key=lambda group: (
            -sum(count * rarity[class_id] for class_id, count in group["classes"].items()),
            -group["images"],
            tie_break[group["cluster_id"]],
        ),
    )
    assignment: dict[str, str] = {}
    for group in forced_groups:
        assignment[group["cluster_id"]] = "train"
        assigned_images["train"] += group["images"]
        assigned_classes["train"].update(group["classes"])
    for group in ordered:
        best_split, best_score = None, None
        for split in SPLITS:
            image_after = assigned_images[split] + group["images"]
            image_score = ((image_after - targets_images[split]) / max(targets_images[split], 1.0)) ** 2
            class_score = 0.0
            for class_id, count in group["classes"].items():
                target = targets_classes[split][class_id]
                after = assigned_classes[split][class_id] + count
                class_score += ((after - target) / max(target, 1.0)) ** 2
            overflow = max(0.0, image_after - targets_images[split] * 1.08) / max(targets_images[split], 1.0)
            score = image_score + 0.35 * class_score + 10.0 * overflow
            if best_score is None or score < best_score:
                best_split, best_score = split, score
        assert best_split is not None
        assignment[group["cluster_id"]] = best_split
        assigned_images[best_split] += group["images"]
        assigned_classes[best_split].update(group["classes"])
    if set(assignment.values()) != set(SPLITS):
        raise RuntimeError(f"Official grouping did not populate all splits: {Counter(assignment.values())}")
    return assignment


def exact_duplicate_leaks(rows: list[dict]) -> list[dict]:
    by_hash: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_hash[file_sha256(row["image"])].append(row)
    leaks = []
    for digest, members in by_hash.items():
        splits = sorted({row["split"] for row in members})
        if len(splits) > 1:
            leaks.append({"sha256": digest, "splits": splits, "images": [row["image"].name for row in members]})
    return leaks


def materialize_file(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.resolve() == source.resolve() or file_sha256(destination) == file_sha256(source):
            return
        raise FileExistsError(f"Refusing to overwrite a different materialized file: {destination}")
    if mode == "hardlink":
        os.link(source, destination)
    elif mode == "symlink":
        destination.symlink_to(source)
    elif mode == "copy":
        shutil.copy2(source, destination)
    else:
        raise ValueError(f"Unknown materialization mode: {mode}")


def split_dataset(
    dataset_root: Path,
    source_manifest: Path,
    out_dir: Path,
    seed: int = 42,
    link_mode: str = "hardlink",
    expected_official: int = 4481,
    expected_added: int = 2218,
) -> dict:
    source_yaml = dataset_root / "dataset.yaml"
    with source_yaml.open("r", encoding="utf-8") as stream:
        dataset_config = yaml.safe_load(stream)
    names = dataset_config["names"]
    nc = int(dataset_config.get("nc", len(names)))
    rows = inventory(dataset_root, source_manifest, nc)
    source_counts = Counter(row["source"] for row in rows)
    expected_counts = Counter({"official": expected_official, "added": expected_added})
    if source_counts != expected_counts:
        raise RuntimeError(
            f"Expected official={expected_official}/added={expected_added}, got {dict(source_counts)}"
        )
    ratios = {"train": 0.70, "val": 0.15, "test": 0.15}
    added_clusters = {row["cluster_id"] for row in rows if row["source"] == "added"}
    assignment = assign_official_groups(
        grouped(rows, "official"), ratios, seed, nc, forced_train=added_clusters
    )
    for row in rows:
        row["split"] = "train" if row["source"] == "added" else assignment[row["cluster_id"]]
    cluster_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        cluster_splits[row["cluster_id"]].add(row["split"])
    violations = {key: sorted(value) for key, value in cluster_splits.items() if len(value) > 1}
    if violations:
        raise RuntimeError(f"Cluster leakage detected: {list(violations.items())[:10]}")
    duplicate_leaks = exact_duplicate_leaks(rows)
    if duplicate_leaks:
        raise RuntimeError(
            f"Found {len(duplicate_leaks)} exact-image hashes across splits. Merge their cluster_id values and rerun; "
            f"examples={duplicate_leaks[:3]}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    list_dir = out_dir / "lists"
    list_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        output_image = out_dir / "images" / row["split"] / row["image"].name
        output_label = out_dir / "labels" / row["split"] / row["label"].name
        materialize_file(row["image"], output_image, link_mode)
        materialize_file(row["label"], output_label, link_mode)
        row["output_image"] = output_image.resolve()
        row["output_label"] = output_label.resolve()
    for split in SPLITS:
        paths = sorted(str(row["output_image"]) for row in rows if row["split"] == split)
        (list_dir / f"{split}.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")
    data_yaml = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": nc,
        "names": names,
    }
    (out_dir / "dataset.yaml").write_text(yaml.safe_dump(data_yaml, allow_unicode=True, sort_keys=False), encoding="utf-8")
    manifest_fields = ("split", "source", "scene_id", "cluster_id", "image", "label")
    with (out_dir / "split_manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=manifest_fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item["split"], item["image"].name.casefold())):
            writer.writerow({
                "split": row["split"],
                "source": row["source"],
                "scene_id": row["scene_id"],
                "cluster_id": row["cluster_id"],
                "image": row["output_image"].name,
                "label": row["output_label"].name,
            })
    split_counts = Counter(row["split"] for row in rows)
    source_by_split = {
        split: dict(Counter(row["source"] for row in rows if row["split"] == split))
        for split in SPLITS
    }
    fingerprint_payload = [
        (row["split"], row["source"], row["cluster_id"], row["image"].name)
        for row in sorted(rows, key=lambda item: item["image"].name.casefold())
    ]
    fingerprint = hashlib.sha256(json.dumps(fingerprint_payload, separators=(",", ":")).encode()).hexdigest()
    report = {
        "format": 1,
        "dataset_id": "scene811_v2",
        "dataset_fingerprint": fingerprint,
        "seed": seed,
        "policy": "official-grouped-70/15/15; added-train-only",
        "images": len(rows),
        "split_counts": dict(split_counts),
        "source_by_split": source_by_split,
        "cluster_count": len(cluster_splits),
        "exact_cross_split_duplicates": 0,
        "materialization": link_mode,
    }
    (out_dir / "split_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a source-aware, cluster-disjoint Scene811 v2 split without copying images.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--out", default="artifacts/scene811_v2/split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--link-mode", choices=("hardlink", "symlink", "copy"), default="hardlink")
    args = parser.parse_args()
    report = split_dataset(
        Path(args.dataset_root), Path(args.source_manifest), Path(args.out), args.seed, args.link_mode
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
