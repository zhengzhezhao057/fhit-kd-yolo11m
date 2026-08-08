from __future__ import annotations

import argparse
import json
from pathlib import Path

from .artifact_paths import replay_dir
from .common import json_dump

FORMAT = 2
DEFAULT_ERROR_WEIGHTS = {
    "no_candidate": 2.0,
    "low_confidence": 1.5,
    "nms_suppressed": 1.2,
    "localization": 1.0,
    "background_fp": 1.5,
}
DEFAULT_MAX_REPEAT = 3
DEFAULT_MAX_REPLAY_FRACTION = 0.20


def _reasons(entry: dict, hard_keys: tuple[str, ...]) -> list[str]:
    reasons = [item["error_type"] for item in entry.get("objects", []) if item.get("error_type") in hard_keys]
    return sorted(set(reasons))


def _score(reasons: list[str], error_weights: dict, background: int = 0) -> float:
    weights = [float(error_weights[reason]) for reason in reasons if reason in error_weights]
    base = max(weights, default=0.0) + 0.5 * max(0, len(weights) - 1)
    return base + background


def build_native_replay_pools(
    hard_manifest: dict,
    background_manifest: dict,
    *,
    dataset_fingerprint: str,
    error_weights: dict | None = None,
    max_repeat: int = DEFAULT_MAX_REPEAT,
    max_replay_fraction: float = DEFAULT_MAX_REPLAY_FRACTION,
) -> dict:
    if hard_manifest.get("format") != 1 or hard_manifest.get("split") != "train":
        raise RuntimeError("Hard-example source must be a format=1 TRAIN manifest.")
    if background_manifest.get("format") != 1 or background_manifest.get("split") != "train":
        raise RuntimeError("Background source must be a format=1 TRAIN manifest.")
    weights = {**DEFAULT_ERROR_WEIGHTS, **(error_weights or {})}
    images: dict[str, dict] = {}
    for relative, entry in hard_manifest.get("images", {}).items():
        reasons = _reasons(entry, ("no_candidate", "low_confidence", "nms_suppressed", "localization"))
        if not reasons:
            continue
        score = _score(reasons, weights)
        pool = "ship_hard_positive" if entry.get("coarse_group") == "ship" else "vehicle_hard_positive"
        images[relative] = {
            "pool": pool,
            "score": round(score, 3),
            "repeat_count": min(max_repeat, max(1, int(score))),
            "reasons": reasons,
        }
    for relative, entry in background_manifest.get("images", {}).items():
        boxes = entry.get("boxes", [])
        if not boxes:
            continue
        target = images.setdefault(relative, {
            "pool": "vehicle_background", "score": 0.0, "repeat_count": 1, "reasons": [],
        })
        target["pool"] = "vehicle_background"
        target["score"] = round(_score([], weights, background=float(weights.get("background_fp", 1.5))), 3)
        target["repeat_count"] = min(max_repeat, max(1, int(target["score"])))
        if "background_fp" not in target["reasons"]:
            target["reasons"].append("background_fp")
        target["background_boxes"] = len(boxes)
    if not images:
        raise RuntimeError("No TRAIN images survived native replay pool selection.")
    return {
        "format": FORMAT,
        "kind": "native_replay",
        "dataset_id": "scene811_v1",
        "dataset_fingerprint": dataset_fingerprint,
        "error_weights": weights,
        "max_repeat_per_image": max_repeat,
        "max_replay_fraction": max_replay_fraction,
        "images": dict(sorted(images.items())),
        "image_count": len(images),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Scene811 native YOLO replay pools (design section 11).")
    parser.add_argument("--hard-manifest", required=True)
    parser.add_argument("--background-manifest", required=True)
    parser.add_argument("--dataset-fingerprint", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    hard = json.loads(Path(args.hard_manifest).read_text(encoding="utf-8"))
    background = json.loads(Path(args.background_manifest).read_text(encoding="utf-8"))
    manifest = build_native_replay_pools(hard, background, dataset_fingerprint=args.dataset_fingerprint)
    out = Path(args.out) if args.out else replay_dir() / "replay_inventory.json"
    json_dump(manifest, out)
    print(f"saved {out}; images={manifest['image_count']}")


if __name__ == "__main__":
    main()
