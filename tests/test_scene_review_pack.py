from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.build_scene_review_pack import (
    build_empty_label_contact_sheet,
    build_review_html,
    load_empty_label_files,
    prioritize_pairs,
    write_decision_template,
)


def _pair(image_a: str, image_b: str, hamming: int, same_family: bool = True) -> dict:
    return {
        "hamming": str(hamming),
        "image_a": f"images/{image_a}",
        "image_b": f"images/{image_b}",
        "scene_id_a": "s_a", "family_a": "fam_a",
        "scene_id_b": "s_b", "family_b": "fam_b",
        "same_family": str(same_family),
    }


def test_prioritize_pairs_buckets_p0_p1_and_rest() -> None:
    pairs = [
        _pair("train/a.jpg", "train/b.jpg", 0),
        _pair("train/c.jpg", "val/d.jpg", 1),
        _pair("train/e.jpg", "val/f.jpg", 0),
        _pair("train/g.jpg", "train/h.jpg", 1),
        _pair("train/i.jpg", "val/j.jpg", 2),
    ]
    buckets = prioritize_pairs(pairs)
    assert len(buckets["p0"]) == 2
    assert len(buckets["p1"]) == 1
    assert len(buckets["rest"]) == 2
    assert all(int(p["hamming"]) <= 0 for p in buckets["p0"])
    assert all(int(p["hamming"]) == 1 and p["image_a"].split("/")[1] != p["image_b"].split("/")[1] for p in buckets["p1"])


def test_load_empty_label_files_converts_label_to_image_entry() -> None:
    audit = {"empty_label_files": ["labels/val/01-abc_crop1.txt", "labels/test/02-def_crop2.txt"]}
    entries = load_empty_label_files(audit)
    assert entries == [
        {"split": "val", "image": "01-abc_crop1.jpg"},
        {"split": "test", "image": "02-def_crop2.jpg"},
    ]
    assert load_empty_label_files({"empty_label_files": []}) == []


def test_review_html_embeds_thumbnail_data_uri(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    image_dir = root / "images" / "train"
    image_dir.mkdir(parents=True)
    cv2.imwrite(str(image_dir / "sample.jpg"), np.full((64, 64, 3), 200, dtype=np.uint8))
    cv2.imwrite(str(image_dir / "other.jpg"), np.full((64, 64, 3), 120, dtype=np.uint8))
    out = tmp_path / "review.html"
    build_review_html([_pair("train/sample.jpg", "train/other.jpg", 0)], "test pack", out, root)
    html = out.read_text(encoding="utf-8")
    assert "data:image/jpeg;base64," in html
    assert "scene=s_a" in html and "hamming=0" in html


def test_empty_label_contact_sheet_and_decision_template(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    image_dir = root / "images" / "val"
    image_dir.mkdir(parents=True)
    cv2.imwrite(str(image_dir / "x.jpg"), np.full((32, 48, 3), 100, dtype=np.uint8))
    entries = [{"split": "val", "image": "x.jpg"}]
    sheet = build_empty_label_contact_sheet(entries, tmp_path / "sheet.png", root)
    assert sheet.is_file() and sheet.stat().st_size > 0
    pairs = [_pair("val/x.jpg", "val/y.jpg", 0)]
    decision = write_decision_template(pairs, tmp_path / "decisions.csv", "p0")
    rows = decision.read_text(encoding="utf-8-sig").splitlines()
    assert rows[0].startswith("priority,index")
    assert rows[1].startswith("p0,0,")
