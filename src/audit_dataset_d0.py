from __future__ import annotations

import argparse
from pathlib import Path

from .common import json_dump, load_config, resolve_data_yaml, split_image_dir
from .dataset_d0 import inventory_fingerprint, scan_split, split_overlap


def audit(config: dict, *, hash_images: bool = True) -> dict:
    data = resolve_data_yaml(config)
    root = Path(data["path"]).resolve()
    summaries, inventories = {}, {}
    for split in ("train", "val", "test"):
        image_dir = split_image_dir(data, split).resolve()
        label_dir = root / "labels" / split
        summary, inventory = scan_split(
            root, image_dir, label_dir, int(config["dataset"]["image_size"]), hash_images=hash_images
        )
        summaries[split] = summary; inventories[split] = inventory
    overlaps = {
        f"{left}_{right}": split_overlap(inventories[left], inventories[right])
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    all_rows = [row for split in ("train", "val", "test") for row in inventories[split]]
    return {
        "format": 1, "kind": "dataset_d0_audit", "read_only": True,
        "dataset_root": str(root), "image_content_hashed": hash_images,
        "d0_fingerprint": inventory_fingerprint(all_rows),
        "splits": summaries, "overlaps": overlaps,
        "interpretation": {
            "image_sha256": (
                "exact image bytes shared across splits"
                if hash_images else "disabled in fast mode; unique inventory markers cannot prove byte duplicates"
            ),
            "scene": "same filename prefix before _cropN shared across splits",
            "product": "same broader product before CCD/PAN tile suffix shared across splits",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit immutable Dataset-D0 without changing images, labels or splits.")
    parser.add_argument("--config", default="configs/direction1.yaml")
    parser.add_argument("--out", default="reports/dataset_d0/audit.json")
    parser.add_argument("--no-image-hash", action="store_true", help="Skip exact image duplicate hashing for a faster structural audit.")
    args = parser.parse_args()
    report = audit(load_config(args.config), hash_images=not args.no_image_hash)
    json_dump(report, args.out)
    print(f"saved {args.out}; D0 fingerprint={report['d0_fingerprint']}")
    for split, summary in report["splits"].items():
        print(split, {key: summary[key] for key in ("images", "instances", "background_images", "duplicate_labels", "invalid_labels")})
    for pair, overlap in report["overlaps"].items():
        print(pair, {key: overlap[key]["shared_groups"] for key in ("image_sha256", "scene", "product")})


if __name__ == "__main__":
    main()
