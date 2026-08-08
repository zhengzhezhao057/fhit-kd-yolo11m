"""Run the native Ultralytics 25-class validation for each named experiment checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

from .audit_experiments import parse_models
from .common import json_dump, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a native YOLO mAP/precision/recall comparison table for named checkpoints.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--model", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--device", default=0)
    parser.add_argument("--out", default="reports/native_val.json")
    args = parser.parse_args()
    try:
        models = parse_models(args.model)
    except ValueError as exc:
        parser.error(str(exc))
    cfg = load_config(args.config); root = Path(cfg["paths"]["project_root"])
    from ultralytics import YOLO
    results = {}
    for name, path in models:
        metrics = YOLO(path).val(data=cfg["paths"]["data_yaml"], imgsz=cfg["dataset"]["image_size"], batch=args.batch or cfg["student"]["batch"], device=args.device, project=str(root / "reports" / "native_val"), name=name, exist_ok=True, plots=True, verbose=False)
        values = {key: float(value) for key, value in metrics.results_dict.items() if isinstance(value, (float, int))}
        results[name] = {"path": str(path.resolve()), "metrics": values, "per_class_map50_95": [float(value) for value in metrics.box.maps]}
        print(name, values)
    json_dump({"models": results, "imgsz": cfg["dataset"]["image_size"], "batch": args.batch or cfg["student"]["batch"]}, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
