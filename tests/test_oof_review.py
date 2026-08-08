from __future__ import annotations

import numpy as np
import pytest

from src.apply_oof_candidate_review import validate_review
from src.build_oof_review_pack import context_view, diverse_positive_references, render_context_panel


def test_context_panel_contains_two_views() -> None:
    image = np.full((100, 160, 3), 180, dtype=np.uint8)
    box = np.asarray([70, 40, 90, 60], dtype=np.float32)
    view, mapped = context_view(image, box, 3.0, size=128)
    assert view.shape == (128, 128, 3)
    assert np.all(mapped[2:] > mapped[:2])
    panel = render_context_panel(image, box, "candidate", color=(0, 0, 255), panel_size=128)
    assert panel.shape == (170, 256, 3)


def test_tri_state_review_only_approves_confirmed_background() -> None:
    manifest = {"candidates": [
        {"candidate_id": "VBG-0001", "source_row": {"model": "OOF", "score": "0.8"}},
        {"candidate_id": "VBG-0002", "source_row": {"model": "OOF", "score": "0.7"}},
        {"candidate_id": "VBG-0003", "source_row": {"model": "OOF", "score": "0.6"}},
    ]}
    approved, counts = validate_review(manifest, [
        {"candidate_id": "VBG-0001", "status": "confirmed_background", "note": "building"},
        {"candidate_id": "VBG-0002", "status": "ambiguous_ignore", "note": "blur"},
        {"candidate_id": "VBG-0003", "status": "", "note": ""},
    ])
    assert len(approved) == 1 and approved[0]["review_candidate_id"] == "VBG-0001"
    assert counts == {"ambiguous_ignore": 1, "confirmed_background": 1, "missing_from_csv": 0, "unreviewed": 1}


def test_positive_references_include_hard_and_clear_fsc() -> None:
    rows = []
    for index in range(20):
        rows.append({
            "coarse_group": "vehicle", "error_type": "low_confidence" if index < 10 else "detected",
            "prediction_score": str(index / 20), "image": f"L{index:03d}-CCD1_crop1.jpg", "gt_index": "0",
        })
    selected = diverse_positive_references(rows, 8)
    assert len(selected) == 8
    assert any(row["error_type"] == "detected" for row in selected)
    assert any(row["error_type"] != "detected" for row in selected)


def test_review_rejects_unknown_duplicate_and_invalid_status() -> None:
    manifest = {"candidates": [{"candidate_id": "VBG-0001", "source_row": {}}]}
    with pytest.raises(RuntimeError, match="Unknown"):
        validate_review(manifest, [{"candidate_id": "bad", "status": "confirmed_background"}])
    with pytest.raises(RuntimeError, match="Invalid"):
        validate_review(manifest, [{"candidate_id": "VBG-0001", "status": "vehicle"}])
    with pytest.raises(RuntimeError, match="Duplicate"):
        validate_review(manifest, [
            {"candidate_id": "VBG-0001", "status": ""}, {"candidate_id": "VBG-0001", "status": ""},
        ])
