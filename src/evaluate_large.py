from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .common import FINE_TO_COARSE, json_dump, load_config, read_yolo_labels, xywhn_to_xyxy
from .competition_eval import Detection, metric_dict, score_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate large_inference JSON against YOLO labels using the competition rules.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--label-dir", required=True, help="Directory containing <large-image-stem>.txt labels normalized to the full canvas.")
    parser.add_argument("--out", default="reports/large_eval.json")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--class-aware", action="store_true", help="Require the predicted ship/aircraft/vehicle group to match the GT group.")
    parser.add_argument("--class-agnostic", action="store_true", help="Diagnostic only: allow cross-group matches.")
    args = parser.parse_args()
    if args.class_aware and args.class_agnostic:
        parser.error("--class-aware and --class-agnostic cannot be used together")
    import json
    cfg = load_config(args.config); payload = json.loads(Path(args.predictions).read_text(encoding="utf-8")); label_dir = Path(args.label_dir)
    class_aware = False if args.class_agnostic else (True if args.class_aware else bool(cfg["evaluation"].get("class_aware_matching", True)))
    candidates = []
    for confidence in cfg["evaluation"]["confidence_grid"]:
        overall = np.zeros(3, dtype=np.int64)
        for record in payload["images"]:
            label_path = label_dir / f"{Path(record['image']).stem}.txt"
            classes, boxes_n = read_yolo_labels(label_path)
            boxes = xywhn_to_xyxy(boxes_n, record["width"], record["height"])
            detections = [Detection(np.asarray(item["xyxy"], dtype=np.float32), float(item["score"]), int(item["fine_class"])) for item in record["detections"] if item["score"] >= confidence]
            overall += np.asarray(score_image(detections, boxes, classes, class_aware))
        candidates.append({"confidence": confidence, "overall": metric_dict(*overall.tolist())})
    valid = [item for item in candidates if item["overall"]["recall"] >= 0.85]
    selected = min(valid, key=lambda item: item["overall"]["false_alarm_rate"]) if valid else max(candidates, key=lambda item: item["overall"]["recall"])
    output = {"source": args.predictions, "class_aware_matching": class_aware, "selected": selected, "all_thresholds": candidates}
    json_dump(output, args.out)
    print(output["selected"])


if __name__ == "__main__":
    main()
