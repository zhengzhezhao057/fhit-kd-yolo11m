from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from .dataset_d0 import IMAGE_SUFFIXES, image_files, source_identities


L_NUMBER = re.compile(r"L\d{11}", re.IGNORECASE)


def read_name_list(path: Path) -> set[str]:
    names: set[str] = set()
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        value = raw.strip()
        if not value or value.startswith("#"):
            continue
        names.add(Path(value).name.casefold())
    if not names:
        raise ValueError(f"Image list is empty: {path}")
    return names


def derived_scene_id(image: Path) -> str:
    match = L_NUMBER.search(image.stem)
    if match:
        return match.group(0).upper()
    scene, product = source_identities(image)
    return product or scene


def build_manifest(dataset_root: Path, official_names: set[str]) -> list[dict[str, str]]:
    root = dataset_root.resolve()
    images_root = root / "images"
    images = image_files(images_root)
    if not images:
        raise RuntimeError(f"No images found below {images_root}")
    basenames: dict[str, Path] = {}
    for image in images:
        key = image.name.casefold()
        if key in basenames:
            raise RuntimeError(
                f"Duplicate image basename {image.name!r}; use unique filenames before source mapping."
            )
        basenames[key] = image
    unknown = sorted(official_names.difference(basenames))
    if unknown:
        raise RuntimeError(
            f"Official list contains {len(unknown)} names absent from the dataset; examples={unknown[:10]}"
        )
    rows: list[dict[str, str]] = []
    for key, image in sorted(basenames.items(), key=lambda item: item[1].name.casefold()):
        source = "official" if key in official_names else "added"
        scene_id = derived_scene_id(image)
        rows.append({
            "image": image.name,
            "source": source,
            "scene_id": scene_id,
            # Users may replace this with a reviewed pHash/overlap cluster. The
            # splitter treats cluster_id as the authoritative indivisible unit.
            "cluster_id": f"{source}:{scene_id}",
        })
    return rows


def write_manifest(rows: list[dict[str, str]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("image", "source", "scene_id", "cluster_id"))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an explicit official/added source manifest for the 6699-image dataset."
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument(
        "--official-list",
        required=True,
        help="UTF-8 text file containing one official image filename/path per line. All remaining images are added data.",
    )
    parser.add_argument("--out", default="artifacts/scene811_v2/source_manifest.csv")
    args = parser.parse_args()
    official = read_name_list(Path(args.official_list))
    rows = build_manifest(Path(args.dataset_root), official)
    write_manifest(rows, Path(args.out))
    counts = {name: sum(row["source"] == name for row in rows) for name in ("official", "added")}
    print(f"SOURCE MANIFEST: images={len(rows)} official={counts['official']} added={counts['added']} out={args.out}")
    if counts != {"official": 4481, "added": 2218}:
        raise RuntimeError(f"Expected official=4481 and added=2218, got {counts}; fix the official list before splitting.")


if __name__ == "__main__":
    main()
