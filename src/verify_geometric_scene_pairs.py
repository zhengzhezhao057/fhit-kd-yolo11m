"""Verify DINO cross-split candidates using local-feature geometry.

This is an audit tool, not an automatic dataset mutator.  It reports whether
two images admit a strong SIFT homography and leaves the final split decision
to an explicit, versioned review manifest.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def _read(path: Path, max_side: int) -> tuple[np.ndarray, float]:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    scale = min(1.0, max_side / max(image.shape))
    if scale < 1.0:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return image, scale


def verify_pair(left: Path, right: Path, max_side: int = 1600) -> dict[str, float | int | bool]:
    a, _ = _read(left, max_side)
    b, _ = _read(right, max_side)
    sift = cv2.SIFT_create(nfeatures=5000)
    ka, da = sift.detectAndCompute(a, None)
    kb, db = sift.detectAndCompute(b, None)
    if da is None or db is None:
        return {"keypoints_a": len(ka), "keypoints_b": len(kb), "good_matches": 0,
                "homography_inliers": 0, "inlier_rate": 0.0, "min_area_overlap": 0.0,
                "strong": False}
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
    good = [m for m, n in pairs if m.distance < 0.75 * n.distance]
    inliers = 0
    overlap = 0.0
    if len(good) >= 8:
        src = np.float32([ka[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kb[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        h, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
        if h is not None and mask is not None:
            inliers = int(mask.sum())
            ha, wa = a.shape
            hb, wb = b.shape
            corners = np.float32([[0, 0], [wa, 0], [wa, ha], [0, ha]]).reshape(-1, 1, 2)
            warped = cv2.perspectiveTransform(corners, h).reshape(-1, 2)
            target = np.float32([[0, 0], [wb, 0], [wb, hb], [0, hb]])
            hull = cv2.convexHull(warped.astype(np.float32))
            inter, _ = cv2.intersectConvexConvex(hull, target)
            area_a = abs(float(cv2.contourArea(hull)))
            area_b = float(wb * hb)
            overlap = float(inter / max(1.0, min(area_a, area_b)))
    rate = inliers / max(1, len(good))
    strong = inliers >= 20 and rate >= 0.35 and overlap >= 0.20
    return {"keypoints_a": len(ka), "keypoints_b": len(kb), "good_matches": len(good),
            "homography_inliers": inliers, "inlier_rate": rate,
            "min_area_overlap": overlap, "strong": strong}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", required=True, type=Path)
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--min-cosine", type=float, default=0.985)
    p.add_argument("--max-side", type=int, default=1600)
    args = p.parse_args()
    rows = list(csv.DictReader(args.candidates.open(encoding="utf-8-sig")))
    unique: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        if float(row["cosine_similarity"]) < args.min_cosine:
            continue
        pair = tuple(sorted((row["query_image"], row["neighbor_image"])))
        unique.setdefault(pair, row)
    output = []
    for index, (pair, row) in enumerate(sorted(unique.items()), 1):
        left = args.dataset / "images" / row["query_split"] / row["query_image"]
        right = args.dataset / "images" / row["neighbor_split"] / row["neighbor_image"]
        result = verify_pair(left, right, args.max_side)
        output.append({**row, **result})
        if index % 50 == 0:
            print(f"verified {index}/{len(unique)}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(output[0]) if output else []
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)
    print(f"wrote {len(output)} pairs to {args.output}")


if __name__ == "__main__":
    main()
