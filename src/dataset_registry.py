from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from .common import json_dump
from .dataset_d0 import file_sha256

FINGERPRINT_SCHEMA = 1
REQUIRED_MANIFEST_FIELDS = ("split", "scene_id", "image", "label")


def canonical_digest(payload: dict) -> str:
    """SHA-256 over a stable, compact, UTF-8 JSON serialization."""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def load_manifest(manifest_path: Path) -> list[dict]:
    """Load split_manifest.csv with validation; returns sorted canonical rows."""
    path = Path(manifest_path)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Manifest has no data rows: {path}")
    for field in REQUIRED_MANIFEST_FIELDS:
        if field not in rows[0]:
            raise ValueError(f"Manifest is missing required column {field!r}: {path}")
    for row in rows:
        for field in REQUIRED_MANIFEST_FIELDS:
            row[field] = row[field].strip()
    rows.sort(key=lambda row: (row["split"], row["image"]))
    return rows


def manifest_split_fingerprint(manifest_path: Path) -> str:
    rows = load_manifest(manifest_path)
    payload = {"schema": FINGERPRINT_SCHEMA, "kind": "scene811_split", "rows": [
        {"split": row["split"], "scene_id": row["scene_id"], "image": row["image"]} for row in rows
    ]}
    return canonical_digest(payload)


def class_mapping_fingerprint(names: dict) -> str:
    if not names:
        raise ValueError("Class mapping must not be empty.")
    normalized = {str(int(key)): str(value) for key, value in sorted(names.items(), key=lambda item: int(item[0]))}
    expected = sorted(int(key) for key in normalized)
    if expected != list(range(len(normalized))):
        raise ValueError(f"Class ids must be contiguous 0..n-1, got {expected}")
    payload = {"schema": FINGERPRINT_SCHEMA, "kind": "scene811_class_mapping", "names": normalized}
    return canonical_digest(payload)


def inventory_rows(dataset_root: Path, manifest_path: Path, *, hash_images: bool = True) -> list[dict]:
    """Per-image fingerprint inventory driven by the manifest (authoritative order)."""
    root = Path(dataset_root)
    rows = load_manifest(manifest_path)
    inventory: list[dict] = []
    missing_images: list[str] = []
    missing_labels: list[str] = []
    for row in rows:
        split, image, label = row["split"], row["image"], row["label"]
        image_path = root / "images" / split / image
        label_path = root / "labels" / split / label
        relative_image = image_path.relative_to(root).as_posix()
        if not image_path.is_file():
            missing_images.append(relative_image)
            continue
        if not label_path.is_file():
            missing_labels.append(relative_image)
            continue
        inventory.append({
            "relative_image": relative_image,
            "split": split,
            "scene_id": row["scene_id"],
            "image_sha256": file_sha256(image_path) if hash_images else f"unhashed:{image_path.stat().st_size}",
            "label_sha256": file_sha256(label_path),
        })
    if missing_images or missing_labels:
        raise RuntimeError(
            f"Fingerprint inventory incomplete: missing_images={len(missing_images)} "
            f"missing_labels={len(missing_labels)}"
        )
    return inventory


def dataset_fingerprint(
    *,
    class_mapping: str,
    split: str,
    inventory: list[dict],
    label_fix_manifest_sha256: str | None,
    background_confirmation: str,
    non_l_scene_audit_sha256: str | None,
) -> str:
    payload = {
        "schema": FINGERPRINT_SCHEMA,
        "dataset_id": "scene811_v1",
        "class_mapping_fingerprint": class_mapping,
        "split_fingerprint": split,
        "background_confirmation": background_confirmation,
        "label_fix_manifest_sha256": label_fix_manifest_sha256 or "none",
        "non_l_scene_audit_sha256": non_l_scene_audit_sha256 or "none",
        "inventory": [
            {
                "relative_image": row["relative_image"],
                "split": row["split"],
                "scene_id": row["scene_id"],
                "image_sha256": row["image_sha256"],
                "label_sha256": row["label_sha256"],
            }
            for row in sorted(inventory, key=lambda row: row["relative_image"])
        ],
    }
    return canonical_digest(payload)


def fingerprint_scene811(
    dataset_root: Path,
    manifest_path: Path,
    names: dict,
    *,
    hash_images: bool = True,
    label_fix_manifest_path: Path | None = None,
    background_confirmation: str = "pending_human_review",
    non_l_scene_audit_path: Path | None = None,
) -> dict:
    """Build the Scene811 v1 fingerprint report (design section 6.4)."""
    root = Path(dataset_root)
    class_fp = class_mapping_fingerprint(names)
    split_fp = manifest_split_fingerprint(manifest_path)
    inventory = inventory_rows(root, manifest_path, hash_images=hash_images)
    fix_sha = None
    if label_fix_manifest_path is not None and Path(label_fix_manifest_path).is_file():
        fix_sha = file_sha256(Path(label_fix_manifest_path))
    non_l_sha = None
    if non_l_scene_audit_path is not None and Path(non_l_scene_audit_path).is_file():
        non_l_sha = file_sha256(Path(non_l_scene_audit_path))
    fingerprint = dataset_fingerprint(
        class_mapping=class_fp,
        split=split_fp,
        inventory=inventory,
        label_fix_manifest_sha256=fix_sha,
        background_confirmation=background_confirmation,
        non_l_scene_audit_sha256=non_l_sha,
    )
    report = {
        "format": 1,
        "kind": "scene811_v1_fingerprint",
        "dataset_id": "scene811_v1",
        "dataset_fingerprint": fingerprint,
        "class_mapping_fingerprint": class_fp,
        "split_fingerprint": split_fp,
        "source_dataset": "original_6699",
        "image_count": len(inventory),
        "hash_images": hash_images,
        "label_fix_manifest_sha256": fix_sha,
        "background_confirmation": background_confirmation,
        "non_l_scene_audit_sha256": non_l_sha,
        "inventory": inventory,
    }
    return report


def write_fingerprint_report(report: dict, out_summary: Path, out_inventory: Path | None = None) -> None:
    summary = {key: value for key, value in report.items() if key != "inventory"}
    json_dump(summary, out_summary)
    if out_inventory is not None:
        json_dump({"format": 1, "kind": "scene811_v1_inventory", "inventory": report["inventory"]}, out_inventory)
