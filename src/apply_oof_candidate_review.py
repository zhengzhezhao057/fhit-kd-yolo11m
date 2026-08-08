from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from .build_vehicle_negative_manifest import build_vehicle_negative_manifest
from .build_oof_review_pack import ALLOWED_REVIEW_STATES, read_csv
from .common import json_dump, load_config
from .weak_group_diagnostics import write_csv


def validate_review(manifest: dict, rows: list[dict[str, str]]) -> tuple[list[dict], dict]:
    candidates = {item["candidate_id"]: item for item in manifest["candidates"]}
    seen: set[str] = set(); approved: list[dict] = []; counts: Counter[str] = Counter()
    for row in rows:
        candidate_id = row.get("candidate_id", "").strip()
        if not candidate_id:
            continue
        if candidate_id in seen:
            raise RuntimeError(f"Duplicate review row: {candidate_id}")
        seen.add(candidate_id)
        if candidate_id not in candidates:
            raise RuntimeError(f"Unknown candidate_id in review CSV: {candidate_id}")
        status = row.get("status", "").strip()
        if not status:
            counts["unreviewed"] += 1
            continue
        if status not in ALLOWED_REVIEW_STATES:
            raise RuntimeError(f"Invalid status for {candidate_id}: {status!r}")
        counts[status] += 1
        if status == "confirmed_background":
            source = dict(candidates[candidate_id]["source_row"])
            source["review_candidate_id"] = candidate_id
            source["review_note"] = row.get("note", "")
            approved.append(source)
    counts["missing_from_csv"] = len(candidates) - len(seen)
    return approved, dict(sorted(counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert tri-state human review into confirmed OOF background evidence.")
    parser.add_argument("--config", default="configs/direction1.yaml")
    parser.add_argument("--review-pack", default="reports/dataset_d0/oof_vehicle_review_v1")
    parser.add_argument("--review-csv", required=True)
    parser.add_argument("--out", default="reports/dataset_d0/oof_vehicle_review_approved_v1")
    args = parser.parse_args()
    config = load_config(args.config); root = Path(config["paths"]["project_root"]).resolve()
    pack = Path(args.review_pack); pack = pack if pack.is_absolute() else root / pack
    review_csv = Path(args.review_csv).resolve()
    output = Path(args.out); output = output if output.is_absolute() else root / output
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output} already contains reviewed evidence; choose a new --out.")
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((pack / "review_manifest.json").read_text(encoding="utf-8"))
    approved, counts = validate_review(manifest, read_csv(review_csv))
    approved_csv = output / "confirmed_vehicle_backgrounds.csv"
    write_csv(approved_csv, approved)
    report = {
        "format": 1, "kind": "oof_vehicle_tri_state_review_result", "pack_id": manifest["pack_id"],
        "review_csv": str(review_csv), "review_csv_sha256": hashlib.sha256(review_csv.read_bytes()).hexdigest(),
        "counts": counts, "confirmed_background_boxes": len(approved),
        "unreviewed_and_ambiguous_are_never_negative": True,
    }
    if approved:
        negative = build_vehicle_negative_manifest(config, approved_csv, "OOF", minimum_score=0.35)
        negative.update({
            "human_reviewed": True, "review_required": False, "review_pack_id": manifest["pack_id"],
            "review_result_sha256": report["review_csv_sha256"],
        })
        json_dump(negative, output / "confirmed_vehicle_background_manifest.json")
    json_dump(report, output / "review_result.json")
    print(f"REVIEW APPLIED: confirmed={len(approved)} counts={counts} output={output}")


if __name__ == "__main__":
    main()
