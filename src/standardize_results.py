from __future__ import annotations

import argparse
import json
from pathlib import Path


def assignments(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected NAME=PATH, got {value!r}")
        name, path = value.split("=", 1)
        result[name] = Path(path)
    return result


def health_ok(path: Path) -> bool:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return False
    record = json.loads(lines[-1])
    experiment = record.get("experiment")
    branch_ok = True
    if experiment in {"f", "fk"}:
        branch_ok &= int(record.get("feature_nonzero_batches", 0)) > 0 and int(record.get("feature_grad_events", 0)) > 0
    if experiment in {"k", "fk"}:
        branch_ok &= int(record.get("cls_nonzero_batches", 0)) > 0 and int(record.get("roi_grad_events", 0)) > 0
    if experiment in {"g", "gp"}:
        branch_ok &= (
            int(record.get("global_nonzero_batches", 0)) > 0
            and int(record.get("global_grad_events", 0)) > 0
            and float(record.get("global_grad_norm_sum", 0.0)) > 0.0
        )
    if experiment in {"p", "gp"}:
        branch_ok &= (
            int(record.get("prototype_nonzero_batches", 0)) > 0
            and int(record.get("prototype_valid_rois", 0)) > 0
            and int(record.get("prototype_grad_events", 0)) > 0
            and float(record.get("prototype_grad_norm_sum", 0.0)) > 0.0
        )
    if experiment == "gp":
        # GP is an exclusive target router. A healthy run must exercise both
        # routes and must never assign one object to both objectives.
        branch_ok &= (
            int(record.get("global_routed_objects", 0)) > 0
            and int(record.get("prototype_routed_objects", 0)) > 0
            and int(record.get("route_overlap_objects", -1)) == 0
        )
    return bool(record.get("optimizer_verified")) and int(record.get("cache_misses", 1)) == 0 and branch_ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert native/competition/parity/health outputs into result_gate input.")
    parser.add_argument("--native", required=True)
    parser.add_argument("--competition", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--health", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--parity", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--baseline", default="C0")
    parser.add_argument("--operating-point", default="best_f1")
    parser.add_argument("--out", default="reports/standard_metrics.json")
    args = parser.parse_args()
    competition, health, parity = assignments(args.competition), assignments(args.health), assignments(args.parity)
    native = json.loads(Path(args.native).read_text(encoding="utf-8"))["models"]
    models = {}
    for name, entry in native.items():
        if name not in competition:
            raise RuntimeError(f"Missing --competition {name}=PATH")
        comp = json.loads(competition[name].read_text(encoding="utf-8"))
        point = comp["operating_points"][args.operating_point]
        metrics = entry["metrics"]
        model = {
            "native": {"map50_95": float(metrics["metrics/mAP50-95(B)"])},
            "competition": {"overall": point["overall"], "groups": point["per_group"]},
        }
        if name != args.baseline:
            parity_report = json.loads(parity[name].read_text(encoding="utf-8")) if name in parity else {}
            model["evidence"] = {
                "kd_health": name in health and health_ok(health[name]),
                "deploy_parity": bool(parity_report.get("parity", False)),
            }
        models[name] = model
    payload = {"baseline": args.baseline, "models": models}
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {destination} models={list(models)}")


if __name__ == "__main__":
    main()
