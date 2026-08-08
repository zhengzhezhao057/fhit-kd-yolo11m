from __future__ import annotations

import argparse
from pathlib import Path

from .artifact_paths import dataset_root, manifests_dir
from .common import json_dump
from .dataset_d0 import file_sha256
from .dataset_registry import load_manifest


def find_duplicate_rows(label_path: Path) -> list[dict]:
    """Exact duplicate rows with 1-based line numbers, preserving first occurrence."""
    lines = label_path.read_text(encoding="utf-8").splitlines()
    positions: dict[str, list[int]] = {}
    for index, line in enumerate(lines, start=1):
        text = line.strip()
        if text:
            positions.setdefault(text, []).append(index)
    return [{"row": text, "line_numbers": positions[text]} for text in sorted(positions) if len(positions[text]) > 1]


def deduplicate_file(label_path: Path) -> tuple[str, list[str]]:
    lines = label_path.read_text(encoding="utf-8").splitlines()
    seen: set[str] = set()
    kept: list[str] = []
    for line in lines:
        text = line.strip()
        if text and text not in seen:
            seen.add(text)
            kept.append(text)
    content = "".join(f"{line}\n" for line in kept)
    return content, kept


def fix_duplicate_labels(dataset_root: Path, manifest_path: Path, *, apply: bool = False, backup_dir: Path | None = None) -> dict:
    root = Path(dataset_root)
    rows = load_manifest(manifest_path)
    files: dict[str, dict] = {}
    total_removed = 0
    seen_pairs: set[tuple[str, str]] = set()
    for row in rows:
        key = (row["split"], row["label"])
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        label_path = root / "labels" / row["split"] / row["label"]
        if not label_path.is_file():
            continue
        duplicates = find_duplicate_rows(label_path)
        if not duplicates:
            continue
        relative = label_path.relative_to(root).as_posix()
        before = file_sha256(label_path)
        removed_rows = [{"row": item["row"], "line_numbers": item["line_numbers"]} for item in duplicates]
        removed = sum(len(item["line_numbers"]) - 1 for item in duplicates)
        total_removed += removed
        entry = {
            "split": row["split"],
            "label": row["label"],
            "before_sha256": before,
            "removed_rows": removed_rows,
            "removed_row_count": removed,
        }
        if apply:
            content, _ = deduplicate_file(label_path)
            label_path.write_text(content, encoding="utf-8")
            entry["after_sha256"] = file_sha256(label_path)
            if backup_dir is not None:
                backup = Path(backup_dir) / (row["split"] + "__" + row["label"].replace("\\", "__").replace("/", "__"))
                backup.parent.mkdir(parents=True, exist_ok=True)
                backup.write_bytes(label_path.read_bytes())
                entry["backed_up_to"] = str(backup)
        files[relative] = entry
    report = {
        "format": 1,
        "kind": "scene811_label_duplicate_fix",
        "dataset_id": "scene811_v1",
        "applied": apply,
        "total_removed_rows": total_removed,
        "files": files,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Find and optionally remove exact duplicate label rows in Scene811 v1.")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--apply", action="store_true", help="Deduplicate label files in place (backup first).")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    root = Path(args.dataset_root) if args.dataset_root else dataset_root()
    manifest = Path(args.manifest) if args.manifest else root / "split_manifest.csv"
    backup_dir = manifests_dir() / "label_backups" if args.apply else None
    report = fix_duplicate_labels(root, manifest, apply=args.apply, backup_dir=backup_dir)
    out = Path(args.out) if args.out else manifests_dir() / "label_fix_manifest.json"
    json_dump(report, out)
    print(f"saved {out}; applied={report['applied']} files={len(report['files'])} removed_rows={report['total_removed_rows']}")


if __name__ == "__main__":
    main()
