"""Create a hash-only semantic scene-union review from a geometry audit."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--geometry", required=True, type=Path)
    p.add_argument("--split-manifest", required=True, type=Path)
    p.add_argument("--source-zip-sha256", required=True)
    p.add_argument("--dino-audit-cache-fingerprint", required=True)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--merge-existing", type=Path, default=None)
    args = p.parse_args()
    with args.split_manifest.open(encoding="utf-8-sig", newline="") as stream:
        manifest = {row["image"]: row for row in csv.DictReader(stream)}
    with args.geometry.open(encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["strong"].casefold() == "true"]
    pairs = []
    for row in rows:
        left, right = manifest[row["query_image"]], manifest[row["neighbor_image"]]
        pairs.append({
            "left_image_sha256": left["image_sha256"],
            "right_image_sha256": right["image_sha256"],
            "decision": "same_scene_union",
            "reviewer": "sift_ransac_v1_with_manual_spot_check",
            "reviewed_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "rationale": (
                f"DINO candidate verified by SIFT homography: inliers={row['homography_inliers']}, "
                f"inlier_rate={float(row['inlier_rate']):.6f}, "
                f"min_area_overlap={float(row['min_area_overlap']):.6f}."
            ),
            "evidence": {
                "cosine_similarity": float(row["cosine_similarity"]),
                "homography_inliers": int(row["homography_inliers"]),
                "inlier_rate": float(row["inlier_rate"]),
                "min_area_overlap": float(row["min_area_overlap"]),
                "thresholds": {"inliers": 20, "inlier_rate": 0.35, "min_area_overlap": 0.20},
            },
        })
    if args.merge_existing:
        existing = json.loads(args.merge_existing.read_text(encoding="utf-8"))
        if existing.get("source_zip_sha256", "").casefold() != args.source_zip_sha256.casefold():
            raise RuntimeError("Existing review belongs to a different source ZIP")
        pairs = list(existing.get("pairs", [])) + pairs
        deduplicated = {}
        for pair in pairs:
            key = tuple(sorted((pair["left_image_sha256"], pair["right_image_sha256"])))
            deduplicated[key] = pair
        pairs = [deduplicated[key] for key in sorted(deduplicated)]
    payload = {
        "format": 1,
        "kind": "scene811_v3_semantic_same_scene_review",
        "source_zip_sha256": args.source_zip_sha256.casefold(),
        "dino_audit_cache_fingerprint": args.dino_audit_cache_fingerprint.casefold(),
        "method": "DINO candidate retrieval followed by SIFT-RANSAC geometric verification; sampled highest-confidence pairs manually inspected",
        "automatic_dataset_mutation": False,
        "pairs": pairs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(pairs)} reviewed unions to {args.out}")


if __name__ == "__main__":
    main()
