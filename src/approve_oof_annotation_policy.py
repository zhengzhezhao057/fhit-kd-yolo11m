from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .common import json_dump, load_config


def approve_manifest(candidate: dict, summary: dict, candidate_sha256: str) -> dict:
    safe = summary.get("safe_vehicle_background_candidates", {})
    if candidate.get("format") != 1 or candidate.get("kind") != "vehicle_background" or candidate.get("split") != "train":
        raise RuntimeError("Candidate must be a format=1 TRAIN vehicle_background manifest.")
    if not bool(candidate.get("review_required")):
        raise RuntimeError("Candidate manifest is not marked review_required; annotation-policy approval source is ambiguous.")
    if int(candidate.get("negative_boxes", -1)) != int(safe.get("boxes", -2)):
        raise RuntimeError("Candidate box count differs from the OOF mining summary.")
    approved = json.loads(json.dumps(candidate))
    approved.update({
        "review_required": False,
        "human_reviewed": False,
        "approval_basis": "competition_dataset_unlabeled_regions_are_background",
        "annotation_policy": "dataset_d0_exhaustive_positive_annotations",
        "source_candidate_manifest_sha256": candidate_sha256,
        "oof_checkpoint_policy": summary.get("checkpoint_policy"),
        "oof_fold_inventory_fingerprint": summary.get("fold_inventory_fingerprint"),
    })
    return approved


def main() -> None:
    parser = argparse.ArgumentParser(description="Approve geometrically safe OOF negatives under the competition annotation policy.")
    parser.add_argument("--config", default="configs/direction1.yaml")
    parser.add_argument("--mining-dir", default="reports/dataset_d0/oof_mining_v1")
    parser.add_argument("--out", default="reports/dataset_d0/oof_mining_v1/vehicle_background_annotation_policy.json")
    args = parser.parse_args()
    config = load_config(args.config); root = Path(config["paths"]["project_root"]).resolve()
    mining = Path(args.mining_dir); mining = mining if mining.is_absolute() else root / mining
    source = mining / "vehicle_background_candidates_review_required.json"
    summary_path = mining / "summary.json"
    if not source.exists() or not summary_path.exists():
        raise FileNotFoundError("OOF candidate manifest or mining summary is missing.")
    candidate = json.loads(source.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    approved = approve_manifest(candidate, summary, hashlib.sha256(source.read_bytes()).hexdigest())
    destination = Path(args.out); destination = destination if destination.is_absolute() else root / destination
    json_dump(approved, destination)
    print(
        f"ANNOTATION POLICY APPROVED: boxes={approved['negative_boxes']} images={len(approved['images'])} "
        f"policy={approved['annotation_policy']} output={destination}"
    )


if __name__ == "__main__":
    main()
