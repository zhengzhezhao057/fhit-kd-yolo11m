from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from src.audit_dataset_d0 import audit
from src.build_oof_folds import build_folds
from src.dataset_d0 import source_identities


def make_dataset(tmp_path: Path, products: int = 6) -> tuple[dict, Path]:
    root = tmp_path / "dataset"
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
    names = {index: f"class{index}" for index in range(25)}
    data_yaml = tmp_path / "dataset.yaml"
    data_yaml.write_text(
        "\n".join([
            f"path: {root}", "train: images/train", "val: images/val", "test: images/test",
            "nc: 25", f"names: {json.dumps(names)}",
        ]) + "\n", encoding="utf-8"
    )
    for product in range(products):
        for crop in (1, 2):
            name = f"01-PAN-20260101-X-L{product:05d}-CCD{crop}_1_crop{crop}.jpg"
            image = np.full((64, 64, 3), (product * 10 + crop) % 256, dtype=np.uint8)
            path = root / "images" / "train" / name
            cv2.imwrite(str(path), image)
            class_id = (product + crop) % 25
            (root / "labels" / "train" / path.with_suffix(".txt").name).write_text(
                f"{class_id} 0.5 0.5 0.25 0.25\n", encoding="utf-8"
            )
    # A validation crop from train product L00000 deliberately exercises the
    # product-overlap audit without changing the official split.
    val_name = "01-PAN-20260101-X-L00000-CCD9_1_crop9.jpg"
    cv2.imwrite(str(root / "images" / "val" / val_name), np.full((64, 64, 3), 200, dtype=np.uint8))
    (root / "labels" / "val" / Path(val_name).with_suffix(".txt")).write_text(
        "24 0.5 0.5 0.1 0.1\n", encoding="utf-8"
    )
    test_name = "independent-PAN4_crop1.jpg"
    cv2.imwrite(str(root / "images" / "test" / test_name), np.full((64, 64, 3), 220, dtype=np.uint8))
    (root / "labels" / "test" / Path(test_name).with_suffix(".txt")).write_text(
        "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )
    config = {
        "paths": {"project_root": str(tmp_path), "data_yaml": str(data_yaml)},
        "dataset": {"image_size": 640},
    }
    return config, root


def tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def test_source_identity_groups_sibling_crops_and_ccd_tiles() -> None:
    scene, product = source_identities("01-PAN-X-L00001-CCD3_5_crop7.jpg")
    assert scene == "01-PAN-X-L00001-CCD3_5"
    assert product == "01-PAN-X-L00001"
    assert source_identities("E139_N35-PAN16_crop1.jpg") == ("E139_N35-PAN16", "E139_N35")


def test_d0_audit_reports_product_overlap_without_mutation(tmp_path: Path) -> None:
    config, root = make_dataset(tmp_path)
    before = tree_digest(root)
    report = audit(config, hash_images=True)
    assert report["read_only"] is True
    assert report["splits"]["train"]["images"] == 12
    assert report["splits"]["train"]["instances"] == 12
    assert report["overlaps"]["train_val"]["product"]["shared_groups"] == 1
    assert report["overlaps"]["train_test"]["product"]["shared_groups"] == 0
    assert len(report["d0_fingerprint"]) == 64
    assert tree_digest(root) == before


def test_product_grouped_oof_is_exclusive_complete_and_read_only(tmp_path: Path) -> None:
    config, root = make_dataset(tmp_path, products=9)
    before = tree_digest(root)
    output = tmp_path / "reports" / "oof3"
    report = build_folds(config, output, folds=3, seed=7)
    all_train_images = {str(path.resolve()) for path in (root / "images" / "train").glob("*.jpg")}
    validation_union = set()
    for row in report["fold_summaries"]:
        train_paths = set(Path(row["train_list"]).read_text(encoding="utf-8").splitlines())
        val_paths = set(Path(row["val_list"]).read_text(encoding="utf-8").splitlines())
        assert train_paths and val_paths
        assert not train_paths.intersection(val_paths)
        train_products = {source_identities(path)[1] for path in train_paths}
        val_products = {source_identities(path)[1] for path in val_paths}
        assert not train_products.intersection(val_products)
        validation_union.update(val_paths)
    assert validation_union == all_train_images
    assert report["read_only_source"] is True
    assert tree_digest(root) == before


def test_long_tailed_product_assignment_never_leaves_an_empty_fold(tmp_path: Path) -> None:
    config, root = make_dataset(tmp_path, products=30)
    # Make the first products strongly multi-label and much larger than the
    # others, reproducing the failure mode seen in the real D0 inventory.
    for product in range(3):
        for extra in range(3, 15):
            name = f"01-PAN-20260101-X-L{product:05d}-CCD{extra}_1_crop{extra}.jpg"
            image_path = root / "images" / "train" / name
            cv2.imwrite(str(image_path), np.full((64, 64, 3), 20 + product, dtype=np.uint8))
            label_path = root / "labels" / "train" / image_path.with_suffix(".txt").name
            labels = [f"{class_id} 0.5 0.5 0.2 0.2" for class_id in range(25)]
            label_path.write_text("\n".join(labels) + "\n", encoding="utf-8")
    report = build_folds(config, tmp_path / "long_tail_oof", folds=3, seed=11)
    validation_sizes = [row["val_images"] for row in report["fold_summaries"]]
    assert min(validation_sizes) > 0
    assert max(validation_sizes) < sum(validation_sizes) * 0.60
