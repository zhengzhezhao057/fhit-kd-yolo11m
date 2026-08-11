from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from .build_scene811_v3_dataset import IMAGE_SUFFIXES, canonical_sha256, file_sha256


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def normalize_patch_rows(rows: list[dict[str, str]]) -> list[dict]:
    normalized: list[dict] = []
    for row in rows:
        item: dict = dict(row)
        for key in ("line_number", "duplicate_of_line"):
            item[key] = int(item[key])
        normalized.append(item)
    return normalized


def expected_fingerprints(root: Path) -> dict[str, str]:
    fingerprint = json.loads((root / "dataset_fingerprint.json").read_text("utf-8"))
    split_rows = read_csv(root / "split_manifest.csv")
    source_rows = read_csv(root / "source_manifest.csv")
    patch_rows = normalize_patch_rows(read_csv(root / "patch_manifest.csv"))
    identity_rows = read_csv(root / "reports/official_identity_manifest.csv")
    split_fingerprint = canonical_sha256(
        [
            {
                "split": row["split"],
                "source": row["source"],
                "scene_id": row["scene_id"],
                "cluster_id": row["cluster_id"],
                "image": row["image"],
            }
            for row in split_rows
        ]
    )
    official_identity_fingerprint = canonical_sha256(identity_rows)
    source_fingerprint = canonical_sha256(
        {
            "source_zip_sha256": fingerprint["source_zip_sha256"],
            "official_identity_fingerprint": official_identity_fingerprint,
            "rows": [
                {
                    "image": row["image"],
                    "source": row["source"],
                    "selected": int(row["selected"]),
                    "image_sha256": row["image_sha256"],
                    "input_label_sha256": row["input_label_sha256"],
                }
                for row in source_rows
            ],
        }
    )
    patch_fingerprint = canonical_sha256(patch_rows)
    dataset_fingerprint = canonical_sha256(
        {
            "dataset_id": fingerprint["dataset_id"],
            "source_fingerprint": source_fingerprint,
            "split_fingerprint": split_fingerprint,
            "patch_fingerprint": patch_fingerprint,
            "near_duplicate_review_sha256": fingerprint.get(
                "near_duplicate_review_sha256"
            )
            or "none",
            "semantic_same_scene_review_sha256": fingerprint.get(
                "semantic_same_scene_review_sha256"
            )
            or "none",
            "semantic_same_scene_dino_audit_fingerprint": fingerprint.get(
                "semantic_same_scene_dino_audit_fingerprint"
            )
            or "none",
            "inventory": [
                {
                    "split": row["split"],
                    "image": row["image"],
                    "image_sha256": row["image_sha256"],
                    "label_sha256": row["label_sha256"],
                }
                for row in split_rows
            ],
        }
    )
    return {
        "dataset_fingerprint": dataset_fingerprint,
        "source_fingerprint": source_fingerprint,
        "split_fingerprint": split_fingerprint,
        "patch_fingerprint": patch_fingerprint,
        "official_identity_fingerprint": official_identity_fingerprint,
    }


def portable_yaml_checks(root: Path) -> tuple[list[str], dict[str, object]]:
    errors: list[str] = []
    details: dict[str, object] = {}
    for name in ("dataset.yaml", "data.yaml", "dataset_official.yaml"):
        path = root / name
        if not path.is_file():
            errors.append(f"missing YAML: {name}")
            continue
        data = yaml.safe_load(path.read_text("utf-8"))
        if data.get("path"):
            errors.append(f"non-portable absolute/root path key in {name}")
        for key in ("train", "val", "test"):
            value = data.get(key)
            if not value:
                errors.append(f"missing {key} in {name}")
                continue
            if Path(str(value)).is_absolute():
                errors.append(f"absolute {key} path in {name}: {value}")
            resolved = root / str(value)
            if not resolved.exists():
                errors.append(f"unresolved {key} path in {name}: {value}")
    list_counts: dict[str, int] = {}
    for name in ("train_official.txt", "train_added.txt", "train_mix.txt"):
        path = root / name
        if not path.is_file():
            errors.append(f"missing portable train list: {name}")
            continue
        lines = [line.strip() for line in path.read_text("utf-8").splitlines() if line.strip()]
        list_counts[name] = len(lines)
        for line in lines:
            if not line.startswith("./images/train/"):
                errors.append(f"non-portable line in {name}: {line}")
                break
            if not (path.parent / line[2:]).is_file():
                errors.append(f"missing listed image in {name}: {line}")
                break
    if list_counts.get("train_mix.txt", -1) != list_counts.get(
        "train_official.txt", 0
    ) + list_counts.get("train_added.txt", 0):
        errors.append(f"train list counts do not add up: {list_counts}")
    details["list_counts"] = list_counts
    return errors, details


def verify_scene811_v3(
    root: Path,
    *,
    expected_fingerprint: str | None = None,
    hash_files: bool = True,
    expected_official_images: int = 4481,
) -> dict[str, object]:
    root = Path(root).resolve()
    errors: list[str] = []
    required = (
        "dataset.yaml",
        "dataset_official.yaml",
        "split_manifest.csv",
        "source_manifest.csv",
        "patch_manifest.csv",
        "dataset_fingerprint.json",
        "audit_d0.json",
        "reports/official_identity_manifest.csv",
    )
    for relative in required:
        if not (root / relative).is_file():
            errors.append(f"missing required artifact: {relative}")
    if errors:
        return {
            "format": 1,
            "kind": "scene811_v3_verify",
            "root": str(root),
            "passed": False,
            "errors": errors,
        }

    stored = json.loads((root / "dataset_fingerprint.json").read_text("utf-8"))
    audit = json.loads((root / "audit_d0.json").read_text("utf-8"))
    split_rows = read_csv(root / "split_manifest.csv")
    source_rows = read_csv(root / "source_manifest.csv")
    identity_rows = read_csv(root / "reports/official_identity_manifest.csv")
    if not str(stored.get("dataset_id", "")).startswith("scene811_v3_grouped_clean"):
        errors.append(f"wrong dataset_id: {stored.get('dataset_id')}")
    if not audit.get("training_ready"):
        errors.append("audit_d0.json does not authorize training")
    if audit.get("dataset_fingerprint") != stored.get("dataset_fingerprint"):
        errors.append("D0 fingerprint differs from dataset_fingerprint.json")
    if len(identity_rows) != expected_official_images or any(
        row.get("source") != "official" for row in identity_rows
    ):
        errors.append(
            "official identity manifest count/source mismatch: "
            f"expected={expected_official_images} actual={len(identity_rows)}"
        )
    if file_sha256(root / "reports/official_identity_manifest.csv") != stored.get(
        "official_identity_manifest_sha256"
    ):
        errors.append("official identity manifest SHA-256 mismatch")

    calculated = expected_fingerprints(root)
    for key in (
        "dataset_fingerprint",
        "source_fingerprint",
        "split_fingerprint",
        "patch_fingerprint",
        "official_identity_fingerprint",
    ):
        if calculated[key] != stored.get(key):
            errors.append(
                f"{key} mismatch: stored={stored.get(key)} calculated={calculated[key]}"
            )
    if expected_fingerprint and calculated["dataset_fingerprint"].casefold() != expected_fingerprint.casefold():
        errors.append(
            "expected dataset fingerprint mismatch: "
            f"expected={expected_fingerprint} calculated={calculated['dataset_fingerprint']}"
        )

    scene_splits: dict[str, set[str]] = defaultdict(set)
    cluster_splits: dict[str, set[str]] = defaultdict(set)
    expected_images: set[Path] = set()
    expected_labels: set[Path] = set()
    hash_mismatches: list[str] = []
    for row in split_rows:
        split = row["split"]
        if split not in {"train", "val", "test"}:
            errors.append(f"invalid split in manifest: {split}")
            continue
        if row["source"] == "added" and split != "train":
            errors.append(f"added image outside train: {split}/{row['image']}")
        scene_splits[row["scene_id"]].add(split)
        cluster_splits[row["cluster_id"]].add(split)
        image_path = root / "images" / split / row["image"]
        label_path = root / "labels" / split / row["label"]
        expected_images.add(image_path.resolve())
        expected_labels.add(label_path.resolve())
        if not image_path.is_file():
            errors.append(f"missing image: {image_path.relative_to(root)}")
        elif hash_files and file_sha256(image_path) != row["image_sha256"]:
            hash_mismatches.append(image_path.relative_to(root).as_posix())
        if not label_path.is_file():
            errors.append(f"missing label: {label_path.relative_to(root)}")
        elif file_sha256(label_path) != row["label_sha256"]:
            hash_mismatches.append(label_path.relative_to(root).as_posix())
    if hash_mismatches:
        errors.append(
            f"file SHA-256 mismatches={len(hash_mismatches)} examples={hash_mismatches[:5]}"
        )
    scene_leaks = {key: value for key, value in scene_splits.items() if len(value) > 1}
    cluster_leaks = {key: value for key, value in cluster_splits.items() if len(value) > 1}
    if scene_leaks:
        errors.append(f"scene groups cross splits: {len(scene_leaks)}")
    if cluster_leaks:
        errors.append(f"clusters cross splits: {len(cluster_leaks)}")

    actual_images = {
        path.resolve()
        for split in ("train", "val", "test")
        for path in (root / "images" / split).rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    actual_labels = {
        path.resolve()
        for split in ("train", "val", "test")
        for path in (root / "labels" / split).rglob("*.txt")
        if path.is_file()
    }
    extra_images = actual_images.difference(expected_images)
    extra_labels = actual_labels.difference(expected_labels)
    if extra_images:
        errors.append(f"unmanifested images={len(extra_images)}")
    if extra_labels:
        errors.append(f"orphan labels={len(extra_labels)}")
    if expected_images.difference(actual_images):
        errors.append(f"manifest images absent={len(expected_images.difference(actual_images))}")
    if expected_labels.difference(actual_labels):
        errors.append(f"manifest labels absent={len(expected_labels.difference(actual_labels))}")

    portable_errors, portable = portable_yaml_checks(root)
    errors.extend(portable_errors)
    split_counts = Counter(row["split"] for row in split_rows)
    source_counts = Counter(row["source"] for row in split_rows)
    return {
        "format": 1,
        "kind": "scene811_v3_verify",
        "root": str(root),
        "passed": not errors,
        "hash_files": hash_files,
        "dataset_fingerprint": calculated["dataset_fingerprint"],
        "expected_fingerprint": expected_fingerprint,
        "manifest_rows": len(split_rows),
        "source_manifest_rows": len(source_rows),
        "official_identity_rows": len(identity_rows),
        "split_counts": dict(split_counts),
        "source_counts": dict(source_counts),
        "portable": portable,
        "unmanifested_images": len(extra_images),
        "orphan_labels": len(extra_labels),
        "scene_leaks": len(scene_leaks),
        "cluster_leaks": len(cluster_leaks),
        "hash_mismatches": len(hash_mismatches),
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify a moved/uploaded Scene811 V3 dataset before formal training."
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--expected-fingerprint", default=None)
    parser.add_argument("--expected-official-images", type=int, default=4481)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--no-image-hash",
        action="store_true",
        help="Fast structural check only; formal pre-training verification should hash images.",
    )
    args = parser.parse_args()
    root = Path(args.root)
    report = verify_scene811_v3(
        root,
        expected_fingerprint=args.expected_fingerprint,
        hash_files=not args.no_image_hash,
        expected_official_images=args.expected_official_images,
    )
    out = Path(args.out) if args.out else root / "reports/verify_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
