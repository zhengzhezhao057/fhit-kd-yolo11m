from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from src.dataset_registry import (
    class_mapping_fingerprint,
    dataset_fingerprint,
    inventory_rows,
    load_manifest,
    manifest_split_fingerprint,
)


def make_dataset(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "dataset"
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
    for split, name, content in (
        ("train", "01-PAN-X-L00001-CCD1_1_crop1.jpg", np.full((64, 64, 3), 10, dtype=np.uint8)),
        ("train", "01-PAN-X-L00002-CCD1_1_crop1.jpg", np.full((64, 64, 3), 20, dtype=np.uint8)),
        ("val", "01-PAN-X-L00003-CCD1_1_crop1.jpg", np.full((64, 64, 3), 30, dtype=np.uint8)),
    ):
        import cv2
        cv2.imwrite(str(root / "images" / split / name), content)
        (root / "labels" / split / Path(name).with_suffix(".txt")).write_text("24 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    manifest = tmp_path / "split_manifest.csv"
    manifest.write_text(
        "split,scene_id,image,label,dominant_class_id,dominant_class\n"
        "train,L00001,01-PAN-X-L00001-CCD1_1_crop1.jpg,01-PAN-X-L00001-CCD1_1_crop1.txt,24,FSC\n"
        "train,L00002,01-PAN-X-L00002-CCD1_1_crop1.jpg,01-PAN-X-L00002-CCD1_1_crop1.txt,24,FSC\n"
        "val,L00003,01-PAN-X-L00003-CCD1_1_crop1.jpg,01-PAN-X-L00003-CCD1_1_crop1.txt,24,FSC\n",
        encoding="utf-8",
    )
    return root, manifest


NAMES = {str(index): f"class{index}" for index in range(25)}


def test_class_mapping_fingerprint_is_order_independent() -> None:
    first = class_mapping_fingerprint({0: "a", 1: "b"})
    second = class_mapping_fingerprint({"1": "b", "0": "a"})
    assert first == second
    assert class_mapping_fingerprint({0: "a", 1: "b"}) != class_mapping_fingerprint({0: "a", 1: "c"})


def test_manifest_split_fingerprint_changes_when_membership_changes(tmp_path: Path) -> None:
    root, manifest = make_dataset(tmp_path)
    before = manifest_split_fingerprint(manifest)
    mutated = tmp_path / "mutated.csv"
    mutated.write_text(manifest.read_text(encoding="utf-8").replace("train,L00001", "val,L00001"), encoding="utf-8")
    assert manifest_split_fingerprint(mutated) != before


def test_inventory_rows_driven_by_manifest(tmp_path: Path) -> None:
    root, manifest = make_dataset(tmp_path)
    inventory = inventory_rows(root, manifest, hash_images=True)
    assert len(inventory) == 3
    assert {row["split"] for row in inventory} == {"train", "val"}
    assert all(len(row["image_sha256"]) == 64 for row in inventory)


def test_dataset_fingerprint_is_stable_and_sensitive(tmp_path: Path) -> None:
    root, manifest = make_dataset(tmp_path)
    inventory = inventory_rows(root, manifest, hash_images=True)
    class_fp = class_mapping_fingerprint(NAMES)
    split_fp = manifest_split_fingerprint(manifest)
    kwargs = dict(
        class_mapping=class_fp,
        split=split_fp,
        inventory=inventory,
        label_fix_manifest_sha256=None,
        background_confirmation="pending_human_review",
        non_l_scene_audit_sha256=None,
    )
    first = dataset_fingerprint(**kwargs)
    assert dataset_fingerprint(**kwargs) == first
    altered = [dict(row, label_sha256="0" * 64) for row in inventory]
    assert dataset_fingerprint(**dict(kwargs, inventory=altered)) != first


def test_load_manifest_validates_columns(tmp_path: Path) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_text("split,scene_id,image\nx,y,z\n", encoding="utf-8")
    import pytest
    with pytest.raises(ValueError, match="label"):
        load_manifest(bad)
