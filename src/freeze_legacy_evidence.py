from __future__ import annotations

import argparse
import json
from pathlib import Path

from .artifact_paths import manifests_dir, run_dir
from .common import json_dump
from .dataset_d0 import file_sha256

DEFAULT_LEGACY_EVIDENCE = {
    "dataset_generation": "legacy D0 (old train/val/test) is superseded by scene811_v1; metrics are historical context only",
    "historical_metrics": {
        "original_baseline_mAP50_95": 0.79965,
        "v5_highest_native_mAP50_95": 0.79812,
        "aircraft_f1": 0.99,
        "note": "Old VAL/TEST overlap heavily with new TRAIN; these metrics cannot gate Scene811 experiments.",
    },
    "forbidden_initializations": [
        "/root/dinov3-yolo11m-distill/runs/v4_eh_fk/weights/best.pt",
        "/root/dinov3-yolo11m-distill/runs/v4_eh_fk/weights/best_deploy.pt",
        "/root/dinov3-yolo11m-distill/runs/v5_*/weights/*.pt",
        "/root/rsdet/weights/baseline_best.pt",
    ],
    "conclusions": [
        "FK relative to C0 replay showed no stable improvement; DINOv3 distillation benefit is not proven.",
        "FK+BG vehicle F1 improved slightly but vehicle FP did not stably drop; strict gate not passed.",
        "Proven gain so far comes from hard-example replay (C0 replay > legacy V4 main model).",
        "DINOv3/MGD/background auxiliary head are paused until the data-centric path plateaus.",
    ],
    "v5_runs": {
        "C0": {"best": "15612d...9dcc6", "last": "c06223...fe56e", "sha_source": "documented_truncated"},
        "FK": {"best": "2b7648...45c54", "last": "f82f8c...7dc1cd7", "sha_source": "documented_truncated"},
        "FK+BG": {"best": "2f744e...7a99b4", "last": "dbc89e...1dfaf18", "sha_source": "documented_truncated"},
    },
}


def _resolve_full_sha(v5_runs: dict) -> dict:
    """Replace documented truncated hashes with full file SHAs when local weights exist."""
    resolved = {}
    for name, entry in v5_runs.items():
        copy = dict(entry)
        for key in ("best", "last"):
            candidates = [
                run_dir(name) / "weights" / f"{key}.pt",
                Path(r"C:\deeplearning\runs") / name / "weights" / f"{key}.pt",
            ]
            for candidate in candidates:
                if candidate.is_file():
                    copy[key] = file_sha256(candidate)
                    copy["sha_source"] = f"full:{candidate}"
                    break
        resolved[name] = copy
    return resolved


def freeze_legacy(evidence: dict, out: Path) -> dict:
    merged = {**DEFAULT_LEGACY_EVIDENCE, **(evidence or {})}
    merged["v5_runs"] = _resolve_full_sha(merged.get("v5_runs", {}))
    merged["format"] = 1
    merged["kind"] = "scene811_legacy_evidence"
    merged["dataset_id"] = "scene811_v1"
    merged["frozen"] = True
    out = Path(out)
    json_dump(merged, out)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze legacy V5 evidence and SHA records for Scene811 lineage.")
    parser.add_argument("--evidence", default=None, help="Optional JSON overriding default legacy evidence.")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    evidence = None
    if args.evidence:
        with Path(args.evidence).open("r", encoding="utf-8") as stream:
            evidence = json.load(stream)
    out = Path(args.out) if args.out else manifests_dir() / "legacy_evidence.json"
    freeze_legacy(evidence, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
