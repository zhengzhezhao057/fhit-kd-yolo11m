from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


GROUPS = {
    "ship (0-3)": set(range(0, 4)),
    "aircraft (4-23)": set(range(4, 24)),
    "vehicle (24)": {24},
}


def label_classes(path: Path) -> list[int]:
    classes: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if fields:
            classes.append(int(float(fields[0])))
    return classes


def find_image(folder: Path, stem: str) -> Path | None:
    for suffix in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
        candidate = folder / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a balanced contact sheet from Ultralytics prediction outputs.")
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--per-group", type=int, default=5)
    parser.add_argument("--cell-width", type=int, default=420)
    parser.add_argument("--cell-height", type=int, default=450)
    args = parser.parse_args()

    prediction_dir = Path(args.prediction_dir)
    label_dir = prediction_dir / "labels"
    selected: list[tuple[str, Path, int]] = []
    used: set[str] = set()
    for group_name, group_classes in GROUPS.items():
        candidates: list[tuple[int, Path]] = []
        for label_path in label_dir.glob("*.txt"):
            classes = label_classes(label_path)
            count = sum(class_id in group_classes for class_id in classes)
            if count:
                candidates.append((count, label_path))
        for count, label_path in sorted(candidates, key=lambda item: (-item[0], item[1].name)):
            if label_path.stem in used:
                continue
            image_path = find_image(prediction_dir, label_path.stem)
            if image_path is None:
                continue
            selected.append((group_name, image_path, count))
            used.add(label_path.stem)
            if sum(item[0] == group_name for item in selected) >= args.per_group:
                break

    rows, cols = len(GROUPS), args.per_group
    sheet = np.full((rows * args.cell_height, cols * args.cell_width, 3), 245, dtype=np.uint8)
    row_for_group = {name: index for index, name in enumerate(GROUPS)}
    col_counts = {name: 0 for name in GROUPS}
    for group_name, image_path, count in selected:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        max_h = args.cell_height - 54
        scale = min(args.cell_width / image.shape[1], max_h / image.shape[0])
        resized = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        row, col = row_for_group[group_name], col_counts[group_name]
        col_counts[group_name] += 1
        ox, oy = col * args.cell_width, row * args.cell_height
        px = ox + (args.cell_width - resized.shape[1]) // 2
        py = oy + 28 + (max_h - resized.shape[0]) // 2
        sheet[py:py + resized.shape[0], px:px + resized.shape[1]] = resized
        cv2.putText(sheet, f"{group_name} | predicted={count}", (ox + 6, oy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (20, 20, 20), 1, cv2.LINE_AA)
        short_name = image_path.stem[:50]
        cv2.putText(sheet, short_name, (ox + 6, oy + args.cell_height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (40, 40, 40), 1, cv2.LINE_AA)

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"saved {output}; selected={len(selected)}")


if __name__ == "__main__":
    main()
