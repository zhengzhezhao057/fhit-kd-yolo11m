from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


CLASS_NAMES = {
    0: "HM",
    1: "LQS",
    2: "QHS",
    3: "MS",
    4: "A1_SU-35",
    5: "A2_C-130",
    6: "A3_C-17",
    7: "A4_C-5",
    8: "A5_F-16",
    9: "A6_TU-160",
    10: "A7_E-3",
    11: "A8_B-52",
    12: "A9_P-3C",
    13: "A10_B-1B",
    14: "A11_E-8",
    15: "A12_TU-22",
    16: "A13_F-15",
    17: "A14_KC-135",
    18: "A15_F-22",
    19: "A16_FA-18",
    20: "A17_TU-95",
    21: "A18_KC-10",
    22: "A19_SU-34",
    23: "A20_SU-24",
    24: "FSC",
}


def color_for_class(class_id: int) -> tuple[int, int, int]:
    if class_id <= 3:
        return 255, 0, 255  # magenta: ship
    if class_id <= 23:
        return 0, 165, 255  # orange: aircraft
    return 0, 255, 0  # green: vehicle


def draw_detection(image: np.ndarray, item: dict, thickness: int) -> None:
    x1, y1, x2, y2 = (int(round(value)) for value in item["xyxy"])
    class_id = int(item["fine_class"])
    color = color_for_class(class_id)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    label = f"F:{CLASS_NAMES.get(class_id, str(class_id))} {item['score']:.2f}"
    font_scale = max(0.55, thickness * 0.22)
    text_thickness = max(1, thickness // 3)
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
    ty = max(th + baseline + 2, y1)
    cv2.rectangle(image, (x1, ty - th - baseline - 4), (x1 + tw + 6, ty + 2), color, -1)
    cv2.putText(
        image,
        label,
        (x1 + 3, ty - baseline - 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        text_thickness,
        cv2.LINE_AA,
    )


def make_contact_sheet(image: np.ndarray, detections: list[dict], output: Path, cell: int = 320, columns: int = 5) -> None:
    if not detections:
        return
    rows = math.ceil(len(detections) / columns)
    sheet = np.full((rows * cell, columns * cell, 3), 32, dtype=np.uint8)
    height, width = image.shape[:2]
    for index, item in enumerate(detections):
        x1, y1, x2, y2 = (int(round(value)) for value in item["xyxy"])
        box_w, box_h = max(1, x2 - x1), max(1, y2 - y1)
        margin = max(48, int(max(box_w, box_h) * 1.5))
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        radius = max(margin, box_w, box_h)
        sx1, sy1 = max(0, cx - radius), max(0, cy - radius)
        sx2, sy2 = min(width, cx + radius), min(height, cy + radius)
        crop = image[sy1:sy2, sx1:sx2].copy()
        local = dict(item)
        local["xyxy"] = [x1 - sx1, y1 - sy1, x2 - sx1, y2 - sy1]
        draw_detection(crop, local, max(2, round(max(crop.shape[:2]) / 220)))
        scale = min(cell / crop.shape[1], (cell - 26) / crop.shape[0])
        resized = cv2.resize(crop, (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale))))
        row, col = divmod(index, columns)
        oy, ox = row * cell, col * cell
        px = ox + (cell - resized.shape[1]) // 2
        py = oy + 24 + (cell - 24 - resized.shape[0]) // 2
        sheet[py:py + resized.shape[0], px:px + resized.shape[1]] = resized
        cv2.putText(sheet, f"#{index + 1} ({cx},{cy})", (ox + 6, oy + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.imwrite(str(output), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])


def main() -> None:
    parser = argparse.ArgumentParser(description="Render full-image detections emitted by src.large_inference.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overview-size", type=int, default=2000)
    args = parser.parse_args()

    payload = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for record in payload["images"]:
        source = Path(record["image"])
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Cannot read {source}")
        detections = sorted(record["detections"], key=lambda item: item["score"], reverse=True)
        annotated = image.copy()
        thickness = max(4, round(max(annotated.shape[:2]) / 1600))
        for item in detections:
            draw_detection(annotated, item, thickness)
        full_path = output_dir / f"{source.stem}_main_model_full.jpg"
        cv2.imwrite(str(full_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 94])
        scale = min(1.0, args.overview_size / max(annotated.shape[:2]))
        overview = cv2.resize(annotated, (round(annotated.shape[1] * scale), round(annotated.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        overview_path = output_dir / f"{source.stem}_main_model_overview.jpg"
        cv2.imwrite(str(overview_path), overview, [cv2.IMWRITE_JPEG_QUALITY, 94])
        contact_path = output_dir / f"{source.stem}_main_model_detections.jpg"
        make_contact_sheet(image, detections, contact_path)
        print(f"saved {full_path}")
        print(f"saved {overview_path}")
        print(f"saved {contact_path}; detections={len(detections)}")


if __name__ == "__main__":
    main()
