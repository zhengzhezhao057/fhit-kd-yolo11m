from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .artifact_paths import report_dir, run_dir
from .common import json_dump
from .oof_training import completed_epochs

GROUPS = ("ship", "aircraft", "vehicle")
GROUP_KEYS = ("F1", "precision", "recall", "FP", "FN")


def read_diagnostics(experiment_dir: Path) -> dict | None:
    path = experiment_dir / "diagnostics.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_results_tail(experiment_dir: Path) -> dict | None:
    path = experiment_dir / "results.csv"
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return None
    last = rows[-1]
    return {
        "epoch": int(float(last.get("epoch", 0) or 0)),
        "mAP50_95": float(last.get("metrics/mAP50-95(B)", 0.0) or 0.0),
        "mAP50": float(last.get("metrics/mAP50(B)", 0.0) or 0.0),
        "precision": float(last.get("metrics/precision(B)", 0.0) or 0.0),
        "recall": float(last.get("metrics/recall(B)", 0.0) or 0.0),
    }


def summarize_screen(runs_root: Path) -> dict:
    experiments: list[dict] = []
    for experiment_dir in sorted(item for item in runs_root.iterdir() if item.is_dir()):
        diagnostics = read_diagnostics(experiment_dir)
        results = read_results_tail(experiment_dir)
        entry = {
            "experiment": experiment_dir.name,
            "completed_epochs": completed_epochs(experiment_dir / "results.csv") if (experiment_dir / "results.csv").is_file() else 0,
            "results_tail": results,
        }
        if diagnostics is not None:
            per_group = {}
            for group in GROUPS:
                raw = (diagnostics.get("per_group") or {}).get(group, {})
                per_group[group] = {key: raw.get(key) for key in GROUP_KEYS}
            entry["best_epoch"] = diagnostics.get("best_epoch")
            entry["val_mAP50_95"] = diagnostics.get("val_mAP50_95")
            entry["per_group"] = per_group
            entry["hard_panel"] = diagnostics.get("hard_panel")
        experiments.append(entry)
    return {"format": 1, "kind": "scene811_data_centric_summary", "experiments": experiments}


def markdown_summary(summary: dict) -> str:
    lines = ["| experiment | epochs | val mAP50-95 | ship F1 | aircraft F1 | vehicle F1 | vehicle FP |", "|---|---|---|---|---|---|---|"]
    for entry in summary["experiments"]:
        per_group = entry.get("per_group") or {}
        vehicle_fp = (per_group.get("vehicle") or {}).get("FP")
        lines.append(
            "| {experiment} | {epochs} | {mAP} | {ship} | {aircraft} | {vehicle} | {fp} |".format(
                experiment=entry["experiment"],
                epochs=entry["completed_epochs"],
                mAP=entry.get("val_mAP50_95", "-"),
                ship=_fmt(per_group.get("ship", {}).get("F1")),
                aircraft=_fmt(per_group.get("aircraft", {}).get("F1")),
                vehicle=_fmt(per_group.get("vehicle", {}).get("F1")),
                fp=_fmt(vehicle_fp),
            )
        )
    return "\n".join(lines) + "\n"


def _fmt(value) -> str:
    return "-" if value is None else f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate Scene811 replay-screen results into a summary.")
    parser.add_argument("--runs-root", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    runs_root = Path(args.runs_root) if args.runs_root else run_dir("")
    summary = summarize_screen(runs_root)
    out = Path(args.out) if args.out else report_dir() / "data_centric_summary.json"
    json_dump(summary, out)
    md_out = out.with_suffix(".md")
    md_out.write_text(markdown_summary(summary), encoding="utf-8")
    print(f"saved {out} and {md_out}")


if __name__ == "__main__":
    main()
