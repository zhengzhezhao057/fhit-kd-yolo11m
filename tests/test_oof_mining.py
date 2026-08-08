from __future__ import annotations

from pathlib import Path

import pytest

from src.mine_oof_hard_examples import select_safe_vehicle_backgrounds, validate_oof_image_coverage


def row(image: str, *, score: float = 0.6, iou: float = 0.0, group: str = "vehicle", reason: str = "background", x1: float = 20) -> dict:
    return {
        "image": image, "coarse_group": group, "reason": reason, "score": score,
        "nearest_gt_iou": iou, "image_width": 100, "image_height": 100,
        "box_x1": x1, "box_y1": 20, "box_x2": 40, "box_y2": 40,
        "prediction_index": 0,
    }


def test_safe_vehicle_background_filter_is_conservative_and_capped() -> None:
    rows = [
        row("a.jpg", score=0.8), row("a.jpg", score=0.7), row("a.jpg", score=0.6),
        row("a.jpg", score=0.5), row("a.jpg", score=0.4),
        row("edge.jpg", x1=1), row("near.jpg", iou=0.06),
        row("low.jpg", score=0.2), row("ship.jpg", group="ship"),
        row("loc.jpg", reason="localization"),
    ]
    kept, excluded = select_safe_vehicle_backgrounds(rows, maximum_per_image=4)
    assert [item["score"] for item in kept] == [0.8, 0.7, 0.6, 0.5]
    assert excluded == {
        "below_minimum_score": 1, "localization": 1, "near_any_gt": 1,
        "other_group": 1, "per_image_cap": 1, "prediction_at_image_edge": 1,
    }


def test_oof_coverage_requires_exactly_once(tmp_path: Path) -> None:
    paths = {tmp_path / "a.jpg", tmp_path / "b.jpg", tmp_path / "c.jpg"}
    validate_oof_image_coverage([[tmp_path / "a.jpg"], [tmp_path / "b.jpg"], [tmp_path / "c.jpg"]], paths)
    with pytest.raises(RuntimeError, match="exactly once"):
        validate_oof_image_coverage([[tmp_path / "a.jpg"], [tmp_path / "a.jpg"], [tmp_path / "c.jpg"]], paths)
