from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.audit_scene_groups import dhash, near_duplicate_pairs, scene_family


def test_scene_family_classification() -> None:
    assert scene_family("L00000010061") == "L-number"
    assert scene_family("AUAU20240101") == "AUAU*"
    assert scene_family("AUJP20240101") == "AUJP*"
    assert scene_family("1_3_100_10300") == "1_3_*"
    assert scene_family("2_6_80_12321") == "2_6_*"
    assert scene_family("E103.9_N1.2_abc") == "other:E103.9"


def _entry(relative: str, scene_id: str, family: str, split: str, dhash_value: int) -> dict:
    return {"relative_image": relative, "scene_id": scene_id, "family": family, "split": split, "dhash": dhash_value}


def test_near_duplicate_pairs_finds_same_and_cross_family(tmp_path: Path) -> None:
    import numpy as np
    base = np.random.default_rng(7).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    near = base.copy()
    near[10, 10] = (0, 0, 0)
    entries = [
        _entry("a.jpg", "2_5_100_1", "2_5_*", "train", dhash(base)),
        _entry("b.jpg", "2_5_100_2", "2_5_*", "train", dhash(near)),
        _entry("c.jpg", "7_1_9_3", "7_1_*", "val", dhash(np.zeros((64, 64, 3), dtype=np.uint8))),
    ]
    pairs = near_duplicate_pairs(entries, threshold=6)
    assert len(pairs) == 1
    assert pairs[0]["hamming"] <= 6
    assert pairs[0]["same_family"] is True


def test_different_scene_same_image_detected() -> None:
    image = np.random.default_rng(3).integers(0, 255, (128, 128, 3), dtype=np.uint8)
    entries = [
        _entry("x.jpg", "s1", "1_1_*", "train", dhash(image)),
        _entry("y.jpg", "s2", "1_2_*", "val", dhash(image.copy())),
    ]
    assert len(near_duplicate_pairs(entries, threshold=6)) == 1
