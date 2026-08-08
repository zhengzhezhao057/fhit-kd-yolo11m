import numpy as np

from src.common import read_yolo_labels
from src.competition_eval import Detection, confidence_range, metric_dict, score_image, select_operating_points


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


def test_operating_point_selection_separates_gate_and_balanced_choices():
    candidates = []
    for confidence, counts in [(0.4, (95, 5, 5)), (0.8, (85, 1, 15))]:
        overall = metric_dict(*counts)
        candidates.append({"confidence": confidence, "overall": overall})
    selected = select_operating_points(candidates)
    assert selected["gate_min_fdr"]["confidence"] == 0.8
    assert selected["best_f1"]["confidence"] == 0.4
    assert selected["max_recall_under_fdr"]["confidence"] == 0.4
