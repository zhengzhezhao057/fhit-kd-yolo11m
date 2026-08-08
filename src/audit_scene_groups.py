from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from .common import json_dump
from .artifact_paths import audit_dir, dataset_root, manifests_dir, report_dir
from .dataset_registry import load_manifest

L_NUMBER_RE = re.compile(r"L\d{5,}")
DEFAULT_HAMMING_THRESHOLD = 6


def scene_family(scene_id: str) -> str:
    if L_NUMBER_RE.search(scene_id):
        return "L-number"
    if scene_id.startswith(("AUAU", "AUJP", "LQSP")):
        return scene_id[:4] + "*"
    parts = scene_id.split("_")
    if len(parts) >= 2 and all(token.isdigit() for token in parts[:2]):
        return "_".join(parts[:2]) + "_*"
    return "other:" + scene_id.split("_")[0][:16]


def dhash(image: np.ndarray, size: int = 8) -> int:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    value = 0
    for bit in diff.ravel()[:64]:
        value = (value << 1) | int(bit)
    return value


def chunk_variant_keys(hash_value: int, chunks: int = 4, chunk_bits: int = 16) -> list[tuple[int, int]]:
    keys: list[tuple[int, int]] = []
    mask = (1 << chunk_bits) - 1
    for chunk_index in range(chunks):
        shift = (chunks - 1 - chunk_index) * chunk_bits
        chunk = (hash_value >> shift) & mask
        keys.append((chunk_index, chunk))
        for bit in range(chunk_bits):
            keys.append((chunk_index, chunk ^ (1 << bit)))
    return keys


def near_duplicate_pairs(
    entries: list[dict],
    *,
    threshold: int = DEFAULT_HAMMING_THRESHOLD,
) -> list[dict]:
    """Find visually near-identical images among different scene ids."""
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, entry in enumerate(entries):
        for key in chunk_variant_keys(entry["dhash"]):
            buckets[key].append(index)
    seen: set[tuple[int, int]] = set()
    pairs: list[dict] = []
    for candidates in buckets.values():
        if len(candidates) < 2:
            continue
        for index_a in range(len(candidates)):
            for index_b in range(index_a + 1, len(candidates)):
                left, right = candidates[index_a], candidates[index_b]
                if left == right:
                    continue
                key = (left, right) if left < right else (right, left)
                if key in seen:
                    continue
                seen.add(key)
                distance = (entries[left]["dhash"] ^ entries[right]["dhash"]).bit_count()
                if distance <= threshold and entries[left]["scene_id"] != entries[right]["scene_id"]:
                    pairs.append({
                        "hamming": distance,
                        "image_a": entries[left]["relative_image"],
                        "scene_id_a": entries[left]["scene_id"],
                        "family_a": entries[left]["family"],
                        "image_b": entries[right]["relative_image"],
                        "scene_id_b": entries[right]["scene_id"],
                        "family_b": entries[right]["family"],
                        "same_family": entries[left]["family"] == entries[right]["family"],
                    })
    pairs.sort(key=lambda item: (item["hamming"], item["image_a"], item["image_b"]))
    return pairs


def audit_non_l_scenes(dataset_root: Path, manifest_path: Path, *, threshold: int = DEFAULT_HAMMING_THRESHOLD) -> tuple[dict, list[dict]]:
    root = Path(dataset_root)
    rows = load_manifest(manifest_path)
    family_rows: Counter = Counter()
    scene_rows: Counter = Counter()
    split_rows: Counter = Counter()
    entries: list[dict] = []
    unreadable: list[str] = []
    for row in rows:
        scene_id = row["scene_id"]
        if L_NUMBER_RE.search(scene_id):
            continue
        family = scene_family(scene_id)
        family_rows[family] += 1
        scene_rows[scene_id] += 1
        split_rows[(row["split"], family)] += 1
        image_path = root / "images" / row["split"] / row["image"]
        relative = image_path.relative_to(root).as_posix()
        image = cv2.imread(str(image_path))
        if image is None:
            unreadable.append(relative)
            continue
        entries.append({
            "relative_image": relative,
            "scene_id": scene_id,
            "family": family,
            "split": row["split"],
            "dhash": dhash(image),
        })
    pairs = near_duplicate_pairs(entries, threshold=threshold)
    families = sorted(family_rows, key=lambda name: -family_rows[name])
    family_scenes: dict[str, set[str]] = defaultdict(set)
    for entry in entries:
        family_scenes[entry["family"]].add(entry["scene_id"])
    family_summary = [
        {
            "family": name,
            "images": family_rows[name],
            "scenes": len(family_scenes[name]),
            "singleton_scenes": sum(1 for scene_id in family_scenes[name] if scene_rows[scene_id] == 1),
            "per_split": {split: split_rows[(split, name)] for split in ("train", "val", "test")},
        }
        for name in families
    ]
    report = {
        "format": 1,
        "kind": "scene811_non_l_scene_audit",
        "read_only": True,
        "dataset_root": str(root.resolve()),
        "hamming_threshold": threshold,
        "non_l_images": len(entries),
        "non_l_scene_ids": len({entry["scene_id"] for entry in entries}),
        "unreadable_images": unreadable,
        "near_duplicate_pairs": len(pairs),
        "near_duplicates": pairs[:300],
        "near_duplicate_pairs_full_list": "artifacts/scene811_v1/audit/non_l_near_duplicate_pairs.csv",
        "family_summary": family_summary,
        "interpretation": (
            "Non-L-number files are currently one scene per image. Near-duplicate pairs with different scene ids "
            "require human review before claiming strict scene disjointness."
        ),
    }
    return report, entries, pairs


def write_review_files(entries: list[dict], pairs: list[dict], inventory_out: Path, pairs_out: Path) -> None:
    inventory_out.parent.mkdir(parents=True, exist_ok=True)
    pairs_out.parent.mkdir(parents=True, exist_ok=True)
    with inventory_out.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["relative_image", "scene_id", "family", "split", "dhash_hex"])
        for entry in entries:
            writer.writerow([entry["relative_image"], entry["scene_id"], entry["family"], entry["split"], f"{entry['dhash']:016x}"])
    fieldnames = ["hamming", "image_a", "scene_id_a", "family_a", "image_b", "scene_id_b", "family_b", "same_family"]
    with pairs_out.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pairs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit non-L-number Scene811 files for scene/family and near-duplicate risk.")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--threshold", type=int, default=DEFAULT_HAMMING_THRESHOLD)
    args = parser.parse_args()
    root = Path(args.dataset_root) if args.dataset_root else dataset_root()
    manifest = Path(args.manifest) if args.manifest else root / "split_manifest.csv"
    report, entries, pairs = audit_non_l_scenes(root, manifest, threshold=args.threshold)
    out = Path(args.out) if args.out else manifests_dir() / "non_l_scene_audit.json"
    json_dump(report, out)
    inventory_out = report_dir() / "non_l_inventory.csv"
    pairs_out = audit_dir() / "non_l_near_duplicate_pairs.csv"
    write_review_files(entries, pairs, inventory_out, pairs_out)
    print(f"saved {out}; non-L images={report['non_l_images']} near_duplicate_pairs={report['near_duplicate_pairs']}")
    print(f"review files: {inventory_out} {pairs_out}")


if __name__ == "__main__":
    main()
