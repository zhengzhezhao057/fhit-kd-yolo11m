import json

import numpy as np
import pytest

from src.common import read_yolo_labels
from src.competition_eval import (
    Detection,
    build_timing_report,
    competition_ranking_items,
    confidence_range,
    enrich_competition_candidate,
    ensure_no_max_det_truncation,
    load_frozen_operating_point,
    make_frozen_operating_point,
    metric_dict,
    nms_iou_range,
    overall_safety_gate,
    score_image,
    select_operating_points,
)


def test_confidence_order_makes_duplicate_a_false_positive():
    gt_boxes = np.array([[0, 0, 100, 100]], dtype=np.float32)
    gt_classes = np.array([24], dtype=np.int64)
    preds = [Detection(np.array([1, 1, 99, 99], dtype=np.float32), 0.9, 24), Detection(np.array([1, 1, 99, 99], dtype=np.float32), 0.8, 24)]
    assert score_image(preds, gt_boxes, gt_classes) == (1, 1, 0)


def test_vehicle_uses_lower_iou_threshold():
    gt_boxes = np.array([[0, 0, 100, 100]], dtype=np.float32)
    prediction = Detection(np.array([45, 0, 145, 100], dtype=np.float32), 0.9, 24)  # IoU about 0.38
    assert score_image([prediction], gt_boxes, np.array([24], dtype=np.int64)) == (1, 0, 0)
    assert score_image([prediction], gt_boxes, np.array([0], dtype=np.int64)) == (0, 1, 1)


def test_class_aware_scoring_handles_prediction_with_zero_ground_truth():
    prediction = Detection(np.array([0, 0, 10, 10], dtype=np.float32), 0.9, 4)
    empty_boxes = np.zeros((0, 4), dtype=np.float32)
    empty_classes = np.zeros((0,), dtype=np.int64)
    assert score_image([prediction], empty_boxes, empty_classes, class_aware=True) == (0, 1, 0)


def test_label_reader_removes_exact_duplicates_and_preserves_order(tmp_path):
    label = tmp_path / "sample.txt"
    label.write_text(
        "24 0.5 0.5 0.1 0.1\n"
        "0 0.2 0.3 0.1 0.2\n"
        "24 0.5 0.5 0.1 0.1\n",
        encoding="utf-8",
    )
    raw_classes, _ = read_yolo_labels(label)
    classes, boxes = read_yolo_labels(label, deduplicate=True)
    assert raw_classes.tolist() == [24, 0, 24]
    assert classes.tolist() == [24, 0]
    assert boxes.shape == (2, 4)


def test_dense_confidence_range_is_inclusive_and_stable():
    assert confidence_range(0.3, 0.35, 0.01) == [0.3, 0.31, 0.32, 0.33, 0.34, 0.35]
    assert nms_iou_range(0.45, 0.55, 0.05) == [0.45, 0.5, 0.55]


def test_operating_point_selection_separates_gate_and_balanced_choices():
    candidates = []
    for confidence, counts in [(0.4, (95, 5, 5)), (0.8, (85, 1, 15))]:
        overall = metric_dict(*counts)
        candidates.append({"confidence": confidence, "overall": overall})
    selected = select_operating_points(candidates)
    assert selected["gate_min_fdr"]["confidence"] == 0.8
    assert selected["best_f1"]["confidence"] == 0.4
    assert selected["max_recall_under_fdr"]["confidence"] == 0.4


def timing_report(*, seconds: float = 1.4, image_count: int = 2, large: bool = False):
    return build_timing_report(
        image_count=image_count,
        post_read_batch_seconds=[seconds],
        batch_sizes=[image_count],
        image_read_seconds=0.2,
        stage_ms={"preprocess": 20.0, "model": 100.0, "postprocess": 10.0},
        batch=image_count,
        image_size=640,
        max_det=10_000,
        nms_iou=0.5,
        warmup_iterations=3,
        device="cuda:0 (test)",
        model_load_seconds=0.4,
        source_shapes=[(10_000, 10_000) if large else (640, 640)] * image_count,
    )


def formal_report():
    return {
        "split": "val",
        "confidence": 0.5,
        "nms_iou": 0.5,
        "class_aware_matching": True,
        "overall": metric_dict(90, 10, 10),
        "per_group": {
            "ship": metric_dict(20, 2, 2),
            "aircraft": metric_dict(60, 4, 4),
            "vehicle": metric_dict(10, 4, 4),
        },
    }


def test_timing_schema_separates_official_image_io_and_end_to_end():
    timing = timing_report()
    assert timing["protocol"] == "competition_no_image_io_v1"
    assert timing["excludes_image_read"] is True
    assert timing["total_seconds"] == pytest.approx(1.4)
    assert timing["mean_ms_per_image"] == pytest.approx(700.0)
    assert timing["image_read"]["total_seconds"] == pytest.approx(0.2)
    assert timing["end_to_end"]["total_seconds"] == pytest.approx(1.6)
    assert set(timing["ultralytics_stage_mean_ms_per_image"]) == {"preprocess", "model", "postprocess"}
    assert timing["model_load_seconds_excluded"] == pytest.approx(0.4)


def test_report_contains_exactly_all_seven_official_ranking_items_and_gate():
    report = formal_report()
    timing = timing_report()
    enriched = enrich_competition_candidate(report, timing)
    items = enriched["competition_ranking_items"]
    assert len(items) == 7
    assert set(items) == {
        "ship_recall",
        "ship_false_alarm_rate",
        "aircraft_recall",
        "aircraft_false_alarm_rate",
        "vehicle_recall",
        "vehicle_false_alarm_rate",
        "total_inference_seconds_excluding_image_read",
    }
    assert items == competition_ranking_items(report, timing)
    gate = overall_safety_gate(report, timing)
    assert gate["recall_pass"] is True
    assert gate["false_alarm_rate_pass"] is True
    assert gate["timing_gate_applicable"] is False
    assert gate["timing_pass"] is None
    assert gate["passed"] is True
    assert gate["complete_submission_gate_pass"] is None


def test_timing_gate_fails_when_mean_exceeds_twenty_seconds_per_image():
    timing = timing_report(seconds=42.0, image_count=2, large=True)
    gate = overall_safety_gate(formal_report(), timing)
    assert timing["seconds_per_image"] == pytest.approx(21.0)
    assert timing["competition_timing_pass"] is False
    assert gate["timing_gate_applicable"] is True
    assert gate["timing_pass"] is False
    assert gate["passed"] is True
    assert gate["complete_submission_gate_pass"] is False


def test_max_det_reaching_cap_fails_closed():
    with pytest.raises(RuntimeError, match="Predictions may be truncated"):
        ensure_no_max_det_truncation(["dense_a.jpg", "dense_b.jpg"], 300)
    ensure_no_max_det_truncation([], 10_000)


def test_frozen_operating_point_round_trip(tmp_path):
    point = enrich_competition_candidate(formal_report(), timing_report())
    frozen = make_frozen_operating_point(
        point,
        selection="best_f1",
        model=str(tmp_path / "model.pt"),
        data_yaml=str(tmp_path / "dataset.yaml"),
        image_size=640,
        batch=8,
        max_det=10_000,
    )
    path = tmp_path / "operating_point.json"
    path.write_text(json.dumps(frozen), encoding="utf-8")
    loaded = load_frozen_operating_point(path)
    assert loaded["confidence"] == 0.5
    assert loaded["nms_iou"] == 0.5
    assert loaded["max_det"] == 10_000
    assert loaded["class_aware_matching"] is True


def test_frozen_point_rejects_non_official_iou_schema(tmp_path):
    point = enrich_competition_candidate(formal_report(), timing_report())
    frozen = make_frozen_operating_point(
        point,
        selection="best_f1",
        model="model.pt",
        data_yaml=str(tmp_path / "dataset.yaml"),
        image_size=640,
        batch=8,
        max_det=10_000,
    )
    frozen["iou_thresholds"]["vehicle"] = 0.5
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(frozen), encoding="utf-8")
    with pytest.raises(ValueError, match="official IoU"):
        load_frozen_operating_point(path)
