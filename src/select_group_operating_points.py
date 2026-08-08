from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .common import COARSE_NAMES, json_dump
from .competition_eval import metric_dict


INTEGER_FIELDS = {"TP", "FP", "FN"}
FLOAT_FIELDS = {"nms_iou", "confidence", "precision", "recall", "false_alarm_rate", "f1"}


def load_matrix(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        for raw in csv.DictReader(stream):
            row = dict(raw)
            for field in INTEGER_FIELDS:
                row[field] = int(float(row[field]))
            for field in FLOAT_FIELDS:
                row[field] = float(row[field])
            rows.append(row)
    if not rows:
        raise ValueError(f"operating-point matrix is empty: {path}")
    return rows


def parse_group_caps(values: list[str] | None, default: float) -> dict[str, float]:
    caps = {group: default for group in COARSE_NAMES}
    for value in values or []:
        if "=" not in value:
            raise ValueError("--fdr-cap must use GROUP=VALUE")
        group, raw_cap = value.split("=", 1)
        if group not in COARSE_NAMES:
            raise ValueError(f"unknown coarse group in --fdr-cap: {group}")
        cap = float(raw_cap)
        if not 0.0 <= cap <= 1.0:
            raise ValueError("FDR caps must be between 0 and 1")
        caps[group] = cap
    return caps


def combine_group_points(points: dict[str, dict]) -> dict:
    totals = [sum(int(points[group][field]) for group in COARSE_NAMES) for field in ("TP", "FP", "FN")]
    return metric_dict(*totals)


def select_for_model(rows: list[dict], model: str, caps: dict[str, float]) -> dict:
    model_rows = [row for row in rows if row["model"] == model and row["group"] in COARSE_NAMES]
    if not model_rows:
        raise ValueError(f"model not found in matrix: {model}")
    output: dict[str, dict] = {}
    for mode in ("balanced", "max_recall_under_fdr"):
        selected: dict[str, dict] = {}
        for group in COARSE_NAMES:
            candidates = [row for row in model_rows if row["group"] == group and row["false_alarm_rate"] <= caps[group]]
            if not candidates:
                raise ValueError(f"{model}/{group} has no operating point under FDR cap {caps[group]}")
            if mode == "balanced":
                winner = max(candidates, key=lambda row: (row["f1"], row["recall"], -row["false_alarm_rate"]))
            else:
                winner = max(candidates, key=lambda row: (row["recall"], row["f1"], -row["false_alarm_rate"]))
            selected[group] = dict(winner)
        output[mode] = {
            "per_group": selected,
            "combined": combine_group_points(selected),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Select independent ship/aircraft/vehicle confidence and NMS operating points.")
    parser.add_argument("--matrix", default="reports/weak_group_diagnostics_v1/threshold_nms_matrix.csv")
    parser.add_argument("--model", action="append", default=None, help="Matrix model name; repeat or omit for every model.")
    parser.add_argument("--max-fdr", type=float, default=0.20)
    parser.add_argument("--fdr-cap", action="append", default=None, help="Optional GROUP=VALUE override.")
    parser.add_argument("--out", default="reports/weak_group_diagnostics_v1/group_operating_points.json")
    args = parser.parse_args()
    if not 0.0 <= args.max_fdr <= 1.0:
        parser.error("--max-fdr must be between 0 and 1")
    try:
        caps = parse_group_caps(args.fdr_cap, args.max_fdr)
        rows = load_matrix(args.matrix)
        available_models = list(dict.fromkeys(row["model"] for row in rows))
        models = args.model or available_models
        unknown = sorted(set(models) - set(available_models))
        if unknown:
            raise ValueError(f"model(s) not found in matrix: {unknown}")
        result = {
            "matrix": str(args.matrix),
            "fdr_caps": caps,
            "models": {model: select_for_model(rows, model, caps) for model in models},
            "note": "Combined metrics sum disjoint class-aware group counts; deployment must apply the selected threshold and NMS per group.",
        }
    except ValueError as error:
        parser.error(str(error))
    json_dump(result, args.out)
    print(f"saved group operating points to {Path(args.out).resolve()}")
    for model, model_result in result["models"].items():
        for mode, selection in model_result.items():
            points = selection["per_group"]
            compact = {group: (points[group]["confidence"], points[group]["nms_iou"]) for group in COARSE_NAMES}
            print(model, mode, compact, selection["combined"])


if __name__ == "__main__":
    main()
