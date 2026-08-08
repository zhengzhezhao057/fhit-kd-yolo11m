from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.competition_eval import Detection
from src.weak_group_diagnostics import (
    ImageGroundTruth,
    build_parser,
    classify_false_positives,
    classify_instances,
    collect_predictions,
    crowded_flags,
    match_ground_truths,
    parse_model_spec,
    size_bucket,
)


def test_cli_accepts_train_for_leakage_safe_hard_example_mining() -> None:
    args = build_parser().parse_args(["--model", "B0=/tmp/best.pt", "--split", "train"])
    assert args.split == "train"


def detection(box, score: float, fine_class: int) -> Detection:
    return Detection(np.asarray(box, dtype=np.float32), score, fine_class)


def test_match_ground_truths_is_class_aware_and_one_to_one() -> None:
    boxes = np.asarray([[0, 0, 10, 10], [2, 0, 12, 10]], dtype=np.float32)
    classes = np.asarray([0, 0], dtype=np.int64)
    predictions = [detection([0, 0, 12, 10], 0.9, 0)]
    matches, _ = match_ground_truths(predictions, boxes, classes, class_aware=True)
    assert int((matches >= 0).sum()) == 1


def test_error_taxonomy_separates_confidence_nms_class_and_localization() -> None:
    boxes = np.asarray(
        [[0, 0, 10, 10], [20, 0, 30, 10], [40, 0, 50, 10], [60, 0, 70, 10], [80, 0, 90, 10]],
        dtype=np.float32,
    )
    classes = np.asarray([0, 0, 0, 0, 0], dtype=np.int64)
    current = [detection([0, 0, 10, 10], 0.9, 0)]
    low = current + [detection([20, 0, 30, 10], 0.2, 0)]
    loose = low + [
        detection([40, 0, 50, 10], 0.15, 0),
        detection([60, 0, 70, 10], 0.15, 4),
        detection([80, 0, 84, 10], 0.15, 0),
    ]
    rows = classify_instances(current, low, loose, boxes, classes, localization_iou_floor=0.1)
    assert [row["error_type"] for row in rows] == [
        "detected",
        "low_confidence",
        "nms_suppressed",
        "wrong_group",
        "localization",
    ]


def test_same_group_candidate_cannot_be_reported_as_wrong_group() -> None:
    boxes = np.asarray([[0, 0, 10, 10]], dtype=np.float32)
    classes = np.asarray([0], dtype=np.int64)
    same_group_low_iou = [detection([0, 0, 4, 10], 0.2, 1)]
    rows = classify_instances([], [], same_group_low_iou, boxes, classes, localization_iou_floor=0.1)
    assert rows[0]["error_type"] == "localization"


def test_false_positive_taxonomy_separates_duplicate_class_location_and_background() -> None:
    boxes = np.asarray([[0, 0, 10, 10], [20, 0, 30, 10]], dtype=np.float32)
    classes = np.asarray([0, 24], dtype=np.int64)
    predictions = [
        detection([0, 0, 10, 10], 0.9, 0),
        detection([0, 0, 10, 10], 0.8, 0),
        detection([20, 0, 30, 10], 0.7, 4),
        detection([20, 0, 23, 10], 0.6, 24),
        detection([50, 0, 60, 10], 0.5, 24),
    ]
    rows = classify_false_positives(predictions, boxes, classes, localization_iou_floor=0.1)
    assert [row["reason"] for row in rows] == ["duplicate", "wrong_group", "localization", "background"]


def test_size_and_crowding_metadata() -> None:
    assert size_bucket(np.asarray([0, 0, 10, 10]), 640, 640, 640) == "small"
    assert size_bucket(np.asarray([0, 0, 50, 50]), 640, 640, 640) == "medium"
    boxes = np.asarray([[0, 0, 10, 10], [11, 0, 21, 10], [100, 100, 110, 110]], dtype=np.float32)
    classes = np.asarray([0, 0, 0], dtype=np.int64)
    assert crowded_flags(boxes, classes).tolist() == [True, True, False]


def test_model_spec_preserves_paths_containing_equals() -> None:
    name, path = parse_model_spec("FK=/tmp/a=b.pt")
    assert name == "FK"
    assert path == Path("/tmp/a=b.pt")


def test_prediction_collection_explicitly_chunks_path_lists() -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.batch_sizes = []

        def predict(self, *, source, **kwargs):
            self.batch_sizes.append(len(source))
            return [SimpleNamespace(path=path, boxes=None) for path in source]

    items = [
        ImageGroundTruth(Path(f"image_{index}.jpg"), np.zeros(0, dtype=np.int64), np.zeros((0, 4), dtype=np.float32), 640, 640)
        for index in range(5)
    ]
    model = FakeModel()
    predictions = collect_predictions(model, items, image_size=640, confidence=0.01, nms_iou=0.5, batch=2, max_det=3000)
    assert model.batch_sizes == [2, 2, 1]
    assert len(predictions) == 5
