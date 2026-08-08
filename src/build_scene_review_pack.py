from __future__ import annotations

import argparse
import base64
import csv
import io
import json
from pathlib import Path

import cv2
import numpy as np

from .artifact_paths import audit_dir, dataset_root, manifests_dir
from .common import json_dump

THUMB_SIZE = 256
P0_MAX_HAMMING = 0
P1_MAX_HAMMING = 1
DECISIONS = ("merge_scene", "duplicate_content", "distinct", "uncertain")


def load_empty_label_files(audit_json: dict) -> list[dict]:
    """Convert audit empty_label_files (labels/...txt) to image entries."""
    entries = []
    for relative in audit_json.get("empty_label_files", []):
        parts = Path(relative).parts
        split = parts[1]
        image = Path(*parts[2:]).with_suffix(".jpg").as_posix()
        entries.append({"split": split, "image": image})
    return entries


def prioritize_pairs(pairs: list[dict]) -> dict:
    p0, p1, rest = [], [], []
    for pair in pairs:
        hamming = int(pair["hamming"])
        split_a = pair["image_a"].split("/")[1]
        split_b = pair["image_b"].split("/")[1]
        cross_split = split_a != split_b
        if hamming <= P0_MAX_HAMMING:
            p0.append(pair)
        elif cross_split and hamming <= P1_MAX_HAMMING:
            p1.append(pair)
        else:
            rest.append(pair)
    return {"p0": p0, "p1": p1, "rest": rest}


def _thumb_data_uri(image_path: Path, size: int = THUMB_SIZE) -> str | None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    scale = min(size / image.shape[1], size / image.shape[0], 1.0)
    if scale < 1.0:
        image = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 72])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode("ascii")


def _pair_card(pair: dict, dataset_root: Path, index: int) -> str:
    rows = []
    for side in ("a", "b"):
        image_rel = pair[f"image_{side}"]
        split = image_rel.split("/")[1]
        image_path = dataset_root / image_rel
        uri = _thumb_data_uri(image_path)
        image_html = f'<img src="{uri}" alt="{image_rel}">' if uri else '<div class="missing">unreadable</div>'
        rows.append(
            f'<div class="side"><div class="img">{image_html}</div>'
            f'<div class="meta"><b>{image_rel}</b><br>scene={pair[f"scene_id_{side}"]} '
            f'family={pair[f"family_{side}"]} split={split}</div></div>'
        )
    cross = pair["image_a"].split("/")[1] != pair["image_b"].split("/")[1]
    return (
        f'<div class="pair" id="pair{index}">'
        f'<div class="pairhead">#{index} hamming={pair["hamming"]} '
        f'cross_split={str(cross).lower()} same_family={str(pair["same_family"]).lower()}</div>'
        f'<div class="sides">{"".join(rows)}</div>'
        f'<div class="notes"><textarea rows="2" placeholder="decision: merge_scene | duplicate_content | distinct | uncertain; note..."></textarea></div>'
        f'</div>'
    )


def build_review_html(pairs: list[dict], title: str, out_path: Path, dataset_root: Path) -> Path:
    cards = [_pair_card(pair, dataset_root, index) for index, pair in enumerate(pairs)]
    html = f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8"><title>{title}</title>
<style>
body {{ font-family: sans-serif; margin: 16px; }}
.pair {{ border: 1px solid #ccc; border-radius: 8px; padding: 12px; margin: 14px 0; }}
.pairhead {{ font-weight: bold; margin-bottom: 8px; }}
.sides {{ display: flex; gap: 12px; flex-wrap: wrap; }}
.side {{ flex: 1 1 300px; max-width: 420px; }}
.img img {{ width: 100%; border: 1px solid #ddd; }}
.meta {{ font-size: 12px; margin-top: 6px; word-break: break-all; }}
.notes textarea {{ width: 100%; }}
.missing {{ color: red; }}
</style></head>
<body><h1>{title}</h1><p>pairs: {len(pairs)}</p>{''.join(cards)}</body></html>
"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


def build_empty_label_contact_sheet(entries: list[dict], out_path: Path, dataset_root: Path) -> Path:
    if not entries:
        raise RuntimeError("No empty-label files to review.")
    cols = 2
    rows = (len(entries) + cols - 1) // cols
    cell_w, cell_h = 460, 420
    sheet = np.full((rows * cell_h, cols * cell_w, 3), 245, dtype=np.uint8)
    for index, entry in enumerate(entries):
        image_path = dataset_root / "images" / entry["split"] / entry["image"]
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        max_h = cell_h - 70
        scale = min(cell_w / image.shape[1], max_h / image.shape[0])
        resized = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        col, row = index % cols, index // cols
        ox, oy = col * cell_w, row * cell_h
        px = ox + (cell_w - resized.shape[1]) // 2
        py = oy + 40 + (max_h - resized.shape[0]) // 2
        sheet[py:py + resized.shape[0], px:px + resized.shape[1]] = resized
        cv2.putText(sheet, f"[{index}] {entry['split']} | {entry['image']}", (ox + 8, oy + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return out_path


def write_decision_template(pairs: list[dict], out_path: Path, priority: str) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["priority", "index", "image_a", "image_b", "hamming", "split_a", "split_b", "same_family", "decision", "note"])
        for index, pair in enumerate(pairs):
            writer.writerow([
                priority, index, pair["image_a"], pair["image_b"], pair["hamming"],
                pair["image_a"].split("/")[1], pair["image_b"].split("/")[1],
                pair["same_family"], "", "",
            ])
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build human review materials for Scene811 empty labels and near-duplicate pairs.")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--audit", default=None)
    parser.add_argument("--pairs-csv", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--all", action="store_true", help="Include all 9044 pairs in the review pack instead of P0+P1.")
    args = parser.parse_args()
    root = Path(args.dataset_root) if args.dataset_root else dataset_root()
    audit_path = Path(args.audit) if args.audit else manifests_dir() / "audit_scene811.json"
    pairs_path = Path(args.pairs_csv) if args.pairs_csv else audit_dir() / "non_l_near_duplicate_pairs.csv"
    out_dir = Path(args.out) if args.out else audit_dir() / "scene_review_pack"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    pairs = list(csv.DictReader(pairs_path.open(encoding="utf-8-sig")))
    prioritized = prioritize_pairs(pairs)
    if args.all:
        p0 = pairs
        p1: list[dict] = []
    else:
        p0 = prioritized["p0"]
        p1 = prioritized["p1"]
    empty = load_empty_label_files(audit)
    contact = build_empty_label_contact_sheet(empty, out_dir / "empty_label_review.png", root)
    p0_html = build_review_html(p0, "Scene811 近重复 P0（hamming=0）", out_dir / "near_duplicates_p0.html", root)
    files = [contact, p0_html]
    if p1:
        p1_html = build_review_html(p1, "Scene811 近重复 P1（跨 split 且 hamming<=1）", out_dir / "near_duplicates_p1.html", root)
        files.append(p1_html)
    decisions = [write_decision_template(p0, out_dir / "review_decisions_p0.csv", "p0")]
    if p1:
        decisions.append(write_decision_template(p1, out_dir / "review_decisions_p1.csv", "p1"))
    report = {
        "format": 1,
        "kind": "scene811_review_pack",
        "dataset_root": str(root.resolve()),
        "empty_label_images": [{"split": e["split"], "image": e["image"]} for e in empty],
        "near_duplicate_p0": len(p0),
        "near_duplicate_p1": len(p1),
        "near_duplicate_total": len(pairs),
        "outputs": [str(path.relative_to(out_dir.parent.parent.parent.parent.parent)) for path in files + decisions],
    }
    json_dump(report, out_dir / "review_pack_manifest.json")
    print(f"review pack saved to {out_dir}")
    print(f"  empty labels: {len(empty)} -> {contact.name}")
    print(f"  P0 pairs (hamming=0): {len(p0)} -> {p0_html.name}")
    if p1:
        print(f"  P1 pairs (cross-split hamming<=1): {len(p1)} -> {p1_html.name}")
    print(f"  decision templates: {[d.name for d in decisions]}")


if __name__ == "__main__":
    main()
