from __future__ import annotations

import json
from pathlib import Path

import yaml

from src.build_scene811_oof_folds import build_folds


def make_mini_dataset(tmp_path: Path) -> tuple[Path, Path]:
    """Six scenes x two images with class-skewed labels; returns (root, manifest)."""
    root = tmp_path / "dataset"
    images = root / "images" / "train"
    labels = root / "labels" / "train"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    scenes = [f"S{i:03d}" for i in range(6)]
    manifest_rows = []
    for scene_index, scene in enumerate(scenes):
        for crop in range(2):
            stem = f"{scene}_crop{crop}"
            (images / f"{stem}.jpg").write_bytes(b"\xff\xd8\xff")
            # Skew: later scenes get more instances of the rare class 0.
            extra = [f"0 0.5 0.5 0.1 0.1\n"] * scene_index
            label_text = "".join(["2 0.3 0.4 0.1 0.1\n", "5 0.7 0.8 0.2 0.2\n"] + extra)
            (labels / f"{stem}.txt").write_text(label_text, encoding="utf-8")
            manifest_rows.append({
                "split": "train", "scene_id": scene, "image": f"{stem}.jpg",
                "label": f"{stem}.txt", "dominant_class_id": "2", "dominant_class": "QHS",
            })
    manifest_path = root / "split_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as stream:
        import csv
        writer = csv.DictWriter(stream, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    names = {str(i): f"c{i}" for i in range(25)}
    (root / "dataset.yaml").write_text(
        yaml.safe_dump({"path": str(root), "train": "images/train", "val": "images/val",
                        "test": "images/test", "nc": 25, "names": names}),
        encoding="utf-8",
    )
    return root, manifest_path


def _scene_of(line: str) -> str:
    return Path(line.strip()).stem.rsplit("_crop", 1)[0]


def _val_scene_assignment(out: Path) -> dict[str, int]:
    assignment: dict[str, int] = {}
    for fold in range(3):
        val = [p for p in (out / f"fold{fold}_val.txt").read_text(encoding="utf-8").splitlines() if p]
        for path in val:
            assignment[Path(path).stem] = fold
    return assignment


def test_scene_grouped_folds_are_disjoint_and_complete(tmp_path: Path) -> None:
    root, manifest = make_mini_dataset(tmp_path)
    out = tmp_path / "oof3"
    report = build_folds(root, manifest, folds=3, seed=42, out_dir=out)
    assert report["kind"] == "scene811_scene_grouped_oof"
    assert report["folds"] == 3 and report["source_split"] == "train"
    assert (out / "folds.json").is_file()
    assigned = {}
    for fold in range(3):
        train = [p for p in (out / f"fold{fold}_train.txt").read_text(encoding="utf-8").splitlines() if p]
        val = [p for p in (out / f"fold{fold}_val.txt").read_text(encoding="utf-8").splitlines() if p]
        assert val and train
        train_scenes = {_scene_of(p) for p in train}
        val_scenes = {_scene_of(p) for p in val}
        assert not train_scenes.intersection(val_scenes), f"scene leakage in fold {fold}"
        for path in val:
            stem = Path(path).stem
            assert stem not in assigned, f"image assigned to two folds: {stem}"
            assigned[stem] = fold
        config = yaml.safe_load((out / f"fold{fold}.yaml").read_text(encoding="utf-8"))
        assert config["nc"] == 25 and config["names"]["24"] == "c24"
    assert len(assigned) == 12
    sizes = [report["fold_summaries"][fold]["val_images"] for fold in range(3)]
    assert all(size >= 2 for size in sizes)


def test_same_seed_reproduces_identical_assignments(tmp_path: Path) -> None:
    root, manifest = make_mini_dataset(tmp_path)
    first_out = tmp_path / "a"
    second_out = tmp_path / "b"
    build_folds(root, manifest, folds=3, seed=7, out_dir=first_out)
    build_folds(root, manifest, folds=3, seed=7, out_dir=second_out)
    assert _val_scene_assignment(first_out) == _val_scene_assignment(second_out)


def test_report_records_fingerprint_and_review_gate(tmp_path: Path) -> None:
    root, manifest = make_mini_dataset(tmp_path)
    report = build_folds(root, manifest, folds=3, seed=1, out_dir=tmp_path / "oof3")
    assert len(report["train_inventory_fingerprint"]) == 64
    assert "near-duplicate review" in report["note"]


def test_image_missing_from_manifest_is_rejected(tmp_path: Path) -> None:
    root, manifest = make_mini_dataset(tmp_path)
    stray = root / "images" / "train" / "S999_crop0.jpg"
    stray.write_bytes(b"\xff\xd8\xff")
    import pytest
    with pytest.raises(RuntimeError, match="missing from manifest"):
        build_folds(root, manifest, folds=3, seed=1, out_dir=tmp_path / "oof3")
