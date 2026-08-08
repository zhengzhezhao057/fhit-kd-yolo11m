from src.select_group_operating_points import combine_group_points, parse_group_caps, select_for_model


def row(group: str, confidence: float, nms_iou: float, tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return {
        "model": "FK",
        "group": group,
        "confidence": confidence,
        "nms_iou": nms_iou,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "false_alarm_rate": 1.0 - precision,
        "f1": 2 * precision * recall / (precision + recall),
    }


def test_selector_can_choose_different_group_operating_points() -> None:
    rows = [
        row("ship", 0.4, 0.65, 90, 8, 10),
        row("ship", 0.5, 0.50, 80, 2, 20),
        row("aircraft", 0.5, 0.50, 98, 1, 2),
        row("aircraft", 0.6, 0.50, 95, 0, 5),
        row("vehicle", 0.4, 0.50, 80, 30, 20),  # over the 0.20 FDR cap
        row("vehicle", 0.5, 0.50, 70, 10, 30),
    ]
    result = select_for_model(rows, "FK", parse_group_caps(None, 0.20))
    assert result["max_recall_under_fdr"]["per_group"]["ship"]["confidence"] == 0.4
    assert result["max_recall_under_fdr"]["per_group"]["vehicle"]["confidence"] == 0.5
    assert result["balanced"]["combined"]["TP"] == sum(
        result["balanced"]["per_group"][group]["TP"] for group in ("ship", "aircraft", "vehicle")
    )


def test_group_specific_fdr_cap_override() -> None:
    caps = parse_group_caps(["ship=0.1", "vehicle=0.25"], 0.2)
    assert caps == {"ship": 0.1, "aircraft": 0.2, "vehicle": 0.25}
