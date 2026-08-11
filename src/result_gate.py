from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_GATES = {
    "minimum_recall": 0.85,
    "maximum_fdr": 0.20,
    "maximum_map50_95_drop": 0.002,
    "maximum_aircraft_f1_drop": 0.005,
    "minimum_weak_group_f1_gain": 0.005,
}


def f1(group: dict) -> float:
    if "f1" in group:
        return float(group["f1"])
    recall = float(group["recall"])
    precision = 1.0 - float(group["false_alarm_rate"])
    return 2.0 * precision * recall / max(precision + recall, 1e-12)


def check_model(baseline: dict, candidate: dict, gates: dict | None = None) -> dict:
    limits = {**DEFAULT_GATES, **(gates or {})}
    native_base = float(baseline["native"]["map50_95"])
    native_candidate = float(candidate["native"]["map50_95"])
    overall = candidate["competition"]["overall"]
    base_groups = baseline["competition"]["groups"]
    groups = candidate["competition"]["groups"]
    deltas = {
        "map50_95": native_candidate - native_base,
        "aircraft_f1": f1(groups["aircraft"]) - f1(base_groups["aircraft"]),
        "ship_f1": f1(groups["ship"]) - f1(base_groups["ship"]),
        "vehicle_f1": f1(groups["vehicle"]) - f1(base_groups["vehicle"]),
    }
    checks = {
        "competition_recall": float(overall["recall"]) >= limits["minimum_recall"],
        "competition_fdr": float(overall["false_alarm_rate"]) <= limits["maximum_fdr"],
        "map_noninferiority": deltas["map50_95"] >= -limits["maximum_map50_95_drop"],
        "aircraft_noninferiority": deltas["aircraft_f1"] >= -limits["maximum_aircraft_f1_drop"],
        "weak_group_gain": max(deltas["ship_f1"], deltas["vehicle_f1"]) >= limits["minimum_weak_group_f1_gain"],
        "kd_health": bool(candidate.get("evidence", {}).get("kd_health", False)),
        "deploy_parity": bool(candidate.get("evidence", {}).get("deploy_parity", False)),
    }
    passed = all(checks.values())
    failed = [name for name, state in checks.items() if not state]
    if not checks["kd_health"] or not checks["deploy_parity"]:
        action = "STOP: repair wiring/export evidence before interpreting accuracy."
    elif not checks["competition_recall"] or not checks["competition_fdr"]:
        action = "REJECT: candidate violates competition hard gates."
    elif not checks["map_noninferiority"] or not checks["aircraft_noninferiority"]:
        action = "REJECT: candidate damages localization quality or saturated aircraft performance."
    elif not checks["weak_group_gain"]:
        action = "HOLD: do not long-train; revise failure routing or data curriculum with the same control."
    else:
        action = "ADVANCE: repeat with a preregistered independent seed; only then enter finalist training."
    return {"passed": passed, "checks": checks, "deltas": deltas, "failed": failed, "next_action": action}


def decide(payload: dict) -> dict:
    models = payload.get("models", {})
    baseline_name = payload.get("baseline", "C0")
    if baseline_name not in models:
        return {"status": "blocked", "next_action": f"Run and register baseline model {baseline_name!r} first."}
    candidates = {
        name: check_model(models[baseline_name], model, payload.get("gates"))
        for name, model in models.items()
        if name != baseline_name
    }
    passed = [name for name, result in candidates.items() if result["passed"]]
    if passed:
        best = max(passed, key=lambda name: models[name]["native"]["map50_95"])
        next_action = f"Advance {best} to the next preregistered seed; keep {baseline_name} as the matched control."
    elif candidates:
        next_action = "No candidate passed. Keep C0, inspect failure modes, and do not start long KD training."
    else:
        next_action = "Run the Global-KD short screen next, then the failure-aware hierarchical candidate."
    return {"status": "complete", "baseline": baseline_name, "candidates": candidates, "passed": passed, "next_action": next_action}


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply immutable competition/non-inferiority gates to standardized experiment metrics.")
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--out", default="reports/gate_decision.json")
    args = parser.parse_args()
    payload = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    result = decide(payload)
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
