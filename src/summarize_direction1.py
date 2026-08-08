from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_results(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def as_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return float("nan")


def summarize_run(run_dir: Path) -> dict[str, Any]:
    rows = read_results(run_dir / "results.csv")
    health = read_jsonl(run_dir / "kd_health.jsonl")
    if rows:
        best = max(rows, key=lambda row: as_float(row, "metrics/mAP50-95(B)"))
        native = {
            "best_epoch": int(float(best.get("epoch", 0))) + 1,
            "precision": as_float(best, "metrics/precision(B)"),
            "recall": as_float(best, "metrics/recall(B)"),
            "mAP50": as_float(best, "metrics/mAP50(B)"),
            "mAP50_95": as_float(best, "metrics/mAP50-95(B)"),
        }
    else:
        native = None
    latest_health = health[-1] if health else None
    return {
        "run": run_dir.name,
        "epochs_recorded": len(rows),
        "native_best": native,
        "latest_kd_health": latest_health,
        "best_weight_exists": (run_dir / "weights" / "best.pt").exists(),
        "deploy_weight_exists": (run_dir / "weights" / "best_deploy.pt").exists(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Direction-1 native validation and KD health without rerunning inference.")
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--pattern", default="d1_*")
    parser.add_argument("--output", default="reports/direction1/screen_summary.json")
    args = parser.parse_args()
    runs_dir = Path(args.runs).resolve()
    summaries = [summarize_run(path) for path in sorted(runs_dir.glob(args.pattern)) if path.is_dir()]
    output = Path(args.output)
    if not output.is_absolute():
        output = Path.cwd() / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(summaries)} run summaries to {output}")
    for item in summaries:
        native = item["native_best"] or {}
        print(f"{item['run']}: epochs={item['epochs_recorded']} mAP50-95={native.get('mAP50_95', 'n/a')} recall={native.get('recall', 'n/a')}")


if __name__ == "__main__":
    main()
