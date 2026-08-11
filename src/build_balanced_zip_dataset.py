from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import shutil
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import yaml
import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SPLITS = ("train", "val", "test")
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


@dataclass
class Record:
    name: str
    label_name: str
    source: str
    split: str
    scene_id: str
    cluster_id: str
    image_entry: str
    label_entry: str
    archive: str
    classes: Counter[int]
    selected: bool = True
    selection_reason: str = ""
    image_sha256: str = ""
    label_sha256: str = ""
    dhash: int | None = None
    phash: int | None = None


def _entry_split(name: str, kind: str) -> str | None:
    match = re.search(rf"(?:^|/){kind}/(train|val|test)/[^/]+$", name, flags=re.IGNORECASE)
    return match.group(1).lower() if match else None


def _image_entries(archive: zipfile.ZipFile) -> dict[str, tuple[str, str]]:
    result: dict[str, tuple[str, str]] = {}
    for info in archive.infolist():
        suffix = PurePosixPath(info.filename).suffix.lower()
        split = _entry_split(info.filename, "images")
        if split is None or suffix not in IMAGE_SUFFIXES:
            continue
        basename = PurePosixPath(info.filename).name
        key = basename.casefold()
        if key in result:
            raise RuntimeError(f"Duplicate image basename in archive: {basename}")
        result[key] = (info.filename, split)
    return result


def _label_entries(archive: zipfile.ZipFile) -> dict[tuple[str, str], str]:
    result: dict[tuple[str, str], str] = {}
    for info in archive.infolist():
        split = _entry_split(info.filename, "labels")
        if split is None or PurePosixPath(info.filename).suffix.lower() != ".txt":
            continue
        basename = PurePosixPath(info.filename).name
        if basename.casefold() == "classes.txt":
            continue
        result[(split, basename.casefold())] = info.filename
    return result


def _read_label(archive: zipfile.ZipFile, entry: str, nc: int = 25) -> tuple[Counter[int], bytes]:
    raw = archive.read(entry)
    text = raw.decode("utf-8-sig")
    counts: Counter[int] = Counter()
    for line_number, line in enumerate(text.splitlines(), start=1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 5:
            raise ValueError(f"Invalid YOLO row at {entry}:{line_number}: {line!r}")
        class_id = int(fields[0])
        values = [float(value) for value in fields[1:]]
        if not 0 <= class_id < nc:
            raise ValueError(f"Class id outside 0..{nc - 1} at {entry}:{line_number}")
        x, y, width, height = values
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
            raise ValueError(f"Invalid normalized box at {entry}:{line_number}: {values}")
        counts[class_id] += 1
    return counts, raw


def strict_official_group(filename: str) -> str:
    """Recover the strongest source scene available from each official filename family."""
    stem = Path(filename).stem
    pan = re.match(r"^(\d{2}-PAN-\d{8}-\d+-\d+-L\d+)-CCD", stem, flags=re.IGNORECASE)
    if pan:
        return f"satellite:{pan.group(1)}"
    l1a = re.match(
        r"^([EW]\d+(?:\.\d+)?_[NS]\d+(?:\.\d+)?_\d{8}_L1A\d+)-PAN\d+",
        stem,
        flags=re.IGNORECASE,
    )
    if l1a:
        return f"l1a:{l1a.group(1)}"
    fsc = re.search(r"-([NS]\d+(?:\.\d+)?)-([EW]\d+(?:\.\d+)?)", stem, flags=re.IGNORECASE)
    if stem.lower().startswith("fsc_") and fsc:
        return f"fsc:{fsc.group(1).upper()}:{fsc.group(2).upper()}"
    if stem.upper().startswith("MAR20_"):
        return f"mar20:{stem}"
    match = re.search(r"^(.*?)(?:-CCD\d+_\d+)?_crop\d+$", stem, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    match = re.search(r"^(.*?)_crop\d+$", stem, flags=re.IGNORECASE)
    return match.group(1) if match else stem


def added_group_key(filename: str) -> str:
    """Conservative source/sequence key recovered from the added filenames."""
    stem = Path(filename).stem
    numeric = re.fullmatch(r"(\d+_\d+_\d+)_\d+", stem)
    if numeric:
        return f"numeric:{numeric.group(1)}"
    sequence = re.fullmatch(r"([A-Za-z]+)(\d{2})\d+", stem)
    if sequence:
        return f"sequence:{sequence.group(1).upper()}{sequence.group(2)}"
    return f"single:{stem}"


def evenly_spaced(items: list[Record], limit: int) -> list[Record]:
    ordered = sorted(items, key=lambda record: record.name.casefold())
    if limit <= 0:
        return []
    if len(ordered) <= limit:
        return ordered
    if limit == 1:
        return [ordered[len(ordered) // 2]]
    indices = [round(index * (len(ordered) - 1) / (limit - 1)) for index in range(limit)]
    return [ordered[index] for index in indices]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _dhash(data: bytes) -> int:
    with Image.open(io.BytesIO(data)) as image:
        gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        pixels = np.asarray(gray, dtype=np.uint8).reshape(-1)
    value = 0
    for row in range(8):
        offset = row * 9
        for column in range(8):
            value = (value << 1) | int(pixels[offset + column] > pixels[offset + column + 1])
    return value


def _phash(data: bytes) -> int:
    """Dependency-light 64-bit perceptual hash used for strong MAR20 clustering."""
    with Image.open(io.BytesIO(data)) as image:
        array = np.asarray(
            image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64
        )
    size = 32
    coordinates = np.arange(size)
    basis = np.cos(
        (2 * coordinates[None, :] + 1) * coordinates[:, None] * math.pi / (2 * size)
    )
    basis[0] *= 1 / math.sqrt(2)
    basis *= math.sqrt(2 / size)
    values = (basis @ array @ basis.T)[:8, :8].flatten()
    median = float(np.median(values[1:]))
    result = 0
    for value in values:
        result = (result << 1) | int(value > median)
    return result


class _DisjointSet:
    def __init__(self, keys: list[str]) -> None:
        self.parent = {key: key for key in keys}

    def find(self, key: str) -> str:
        root = key
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[key] != key:
            parent = self.parent[key]
            self.parent[key] = root
            key = parent
        return root

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        keep, merge = sorted((left_root, right_root))
        self.parent[merge] = keep


def _assign_official(
    records: list[Record], ratios: dict[str, float], seed: int = 20260809
) -> None:
    import random

    groups: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        groups[record.cluster_id].append(record)
    total_images = len(records)
    total_classes: Counter[int] = Counter()
    for record in records:
        total_classes.update(record.classes)
    target_images = {split: ratios[split] * total_images for split in SPLITS}
    target_classes = {
        split: {class_id: ratios[split] * total_classes[class_id] for class_id in CLASS_NAMES}
        for split in SPLITS
    }
    summaries = []
    for cluster_id, members in groups.items():
        counts: Counter[int] = Counter()
        for member in members:
            counts.update(member.classes)
        summaries.append((cluster_id, members, counts))
    rarity = {class_id: 1 / max(total_classes[class_id], 1) for class_id in CLASS_NAMES}
    rng = random.Random(seed)
    tie_break = {cluster_id: rng.random() for cluster_id in groups}
    summaries.sort(
        key=lambda item: (
            -sum(count * rarity[class_id] for class_id, count in item[2].items()),
            -len(item[1]),
            tie_break[item[0]],
        )
    )
    assigned_images = Counter()
    assigned_classes = {split: Counter() for split in SPLITS}
    assignment: dict[str, str] = {}
    split_tie = {split: rng.random() for split in SPLITS}
    for cluster_id, members, counts in summaries:
        candidates = []
        for split in SPLITS:
            before_image = (assigned_images[split] - target_images[split]) / max(
                target_images[split], 1
            )
            after_image = (
                assigned_images[split] + len(members) - target_images[split]
            ) / max(target_images[split], 1)
            delta = after_image**2 - before_image**2
            for class_id, count in counts.items():
                target = target_classes[split][class_id]
                before = (assigned_classes[split][class_id] - target) / max(target, 1)
                after = (assigned_classes[split][class_id] + count - target) / max(target, 1)
                delta += 0.30 * (after**2 - before**2)
            overflow = max(
                0.0, assigned_images[split] + len(members) - 1.05 * target_images[split]
            )
            delta += 20 * (overflow / max(target_images[split], 1)) ** 2
            candidates.append((delta, split_tie[split], split))
        split = min(candidates)[2]
        assignment[cluster_id] = split
        assigned_images[split] += len(members)
        assigned_classes[split].update(counts)
    if set(assignment.values()) != set(SPLITS):
        raise RuntimeError(
            f"Strict grouped assignment failed to populate all splits: {Counter(assignment.values())}"
        )
    for record in records:
        record.split = assignment[record.cluster_id]


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(data)


def _near_duplicate_rows(records: list[Record], threshold: int) -> list[dict[str, object]]:
    train = [record for record in records if record.split == "train"]
    evaluation = [record for record in records if record.split in {"val", "test"}]
    findings: list[dict[str, object]] = []
    for eval_record in evaluation:
        assert eval_record.dhash is not None
        for train_record in train:
            assert train_record.dhash is not None
            distance = (eval_record.dhash ^ train_record.dhash).bit_count()
            if distance <= threshold:
                findings.append({
                    "distance": distance,
                    "evaluation_split": eval_record.split,
                    "evaluation_image": eval_record.name,
                    "evaluation_scene": eval_record.scene_id,
                    "train_image": train_record.name,
                    "train_source": train_record.source,
                    "train_scene": train_record.scene_id,
                })
    findings.sort(key=lambda row: (int(row["distance"]), str(row["evaluation_image"]), str(row["train_image"])))
    return findings


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_dataset(
    official_zip_path: Path,
    combined_zip_path: Path,
    out_dir: Path,
    *,
    fsc_max_images_per_sequence: int = 15,
    near_duplicate_distance: int = 2,
) -> dict:
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(official_zip_path) as official_zip, zipfile.ZipFile(combined_zip_path) as combined_zip:
        official_images = _image_entries(official_zip)
        official_labels = _label_entries(official_zip)
        combined_images = _image_entries(combined_zip)
        combined_labels = _label_entries(combined_zip)

        if len(official_images) != 4481:
            raise RuntimeError(f"Expected 4481 official images, found {len(official_images)}")
        records: list[Record] = []
        official_records: list[Record] = []

        for key, (image_entry, original_split) in sorted(official_images.items()):
            name = PurePosixPath(image_entry).name
            label_name = f"{Path(name).stem}.txt"
            label_entry = official_labels.get((original_split, label_name.casefold()))
            if label_entry is None:
                raise FileNotFoundError(f"Missing official label for {original_split}/{name}")
            classes, _ = _read_label(official_zip, label_entry)
            scene_id = strict_official_group(name)
            image_data = official_zip.read(image_entry)
            official_records.append(Record(
                name=name,
                label_name=label_name,
                source="official",
                split="unassigned",
                scene_id=scene_id,
                cluster_id=f"official:{scene_id}",
                image_entry=image_entry,
                label_entry=label_entry,
                archive="official",
                classes=classes,
                selection_reason="official_strict_group_resplit",
                phash=_phash(image_data) if name.upper().startswith("MAR20_") else None,
            ))

        # Merge strict filename groups, plus only extremely close MAR20 perceptual hashes.
        dsu = _DisjointSet(sorted({record.cluster_id for record in official_records}))
        mar20 = [record for record in official_records if record.phash is not None]
        for index, left in enumerate(mar20):
            assert left.phash is not None
            for right in mar20[index + 1:]:
                assert right.phash is not None
                if (left.phash ^ right.phash).bit_count() <= 2:
                    dsu.union(left.cluster_id, right.cluster_id)
        for record in official_records:
            record.cluster_id = dsu.find(record.cluster_id)
        _assign_official(official_records, {"train": 0.70, "val": 0.15, "test": 0.15})
        records.extend(official_records)

        added_records: list[Record] = []
        for key, (image_entry, old_split) in sorted(combined_images.items()):
            if key in official_images:
                continue
            name = PurePosixPath(image_entry).name
            label_name = f"{Path(name).stem}.txt"
            label_entry = combined_labels.get((old_split, label_name.casefold()))
            if label_entry is None:
                raise FileNotFoundError(f"Missing added label for {old_split}/{name}")
            classes, _ = _read_label(combined_zip, label_entry)
            group = added_group_key(name)
            added_records.append(Record(
                name=name,
                label_name=label_name,
                source="added",
                split="train",
                scene_id=group,
                cluster_id=f"added:{group}",
                image_entry=image_entry,
                label_entry=label_entry,
                archive="combined",
                classes=classes,
                selection_reason="added_non_fsc_kept",
            ))

        # The only aggressively over-represented addition is FSC. Preserve every official
        # sample, but uniformly sample pure-FSC frames inside each recovered sequence.
        fsc_groups: dict[str, list[Record]] = defaultdict(list)
        for record in added_records:
            if not record.classes:
                record.selected = False
                record.selection_reason = "excluded_empty_added_pending_relabel"
                continue
            if record.classes.get(24, 0) and set(record.classes) == {24}:
                fsc_groups[record.cluster_id].append(record)
        for group_records in fsc_groups.values():
            selected_names = {
                record.name.casefold()
                for record in evenly_spaced(group_records, fsc_max_images_per_sequence)
            }
            for record in group_records:
                if record.name.casefold() in selected_names:
                    record.selection_reason = "fsc_sequence_uniform_sample"
                else:
                    record.selected = False
                    record.selection_reason = "excluded_fsc_sequence_cap"

        selected_records = records + [record for record in added_records if record.selected]
        excluded_records = [record for record in added_records if not record.selected]
        selected_names: set[str] = set()
        exact_hashes: dict[str, list[Record]] = defaultdict(list)

        for record in selected_records:
            key = record.name.casefold()
            if key in selected_names:
                raise RuntimeError(f"Duplicate selected basename: {record.name}")
            selected_names.add(key)
            archive = official_zip if record.archive == "official" else combined_zip
            image_data = archive.read(record.image_entry)
            label_data = archive.read(record.label_entry)
            record.image_sha256 = _sha256(image_data)
            record.label_sha256 = _sha256(label_data)
            record.dhash = _dhash(image_data)
            exact_hashes[record.image_sha256].append(record)
            _write_bytes(out_dir / "images" / record.split / record.name, image_data)
            _write_bytes(out_dir / "labels" / record.split / record.label_name, label_data)

    exact_duplicate_rows: list[dict] = []
    cross_split_exact = 0
    for digest, members in exact_hashes.items():
        if len(members) < 2:
            continue
        splits = sorted({member.split for member in members})
        if len(splits) > 1:
            cross_split_exact += 1
        exact_duplicate_rows.append({
            "sha256": digest,
            "splits": ";".join(splits),
            "images": ";".join(member.name for member in members),
        })
    if cross_split_exact:
        raise RuntimeError(f"Found {cross_split_exact} exact-image hashes across splits")

    near_duplicates = _near_duplicate_rows(selected_records, near_duplicate_distance)
    manifest_rows = []
    for record in sorted(selected_records, key=lambda item: (item.split, item.name.casefold())):
        manifest_rows.append({
            "split": record.split,
            "source": record.source,
            "scene_id": record.scene_id,
            "cluster_id": record.cluster_id,
            "image": record.name,
            "label": record.label_name,
            "classes": ";".join(f"{key}:{value}" for key, value in sorted(record.classes.items())),
            "image_sha256": record.image_sha256,
            "label_sha256": record.label_sha256,
            "selection_reason": record.selection_reason,
        })
    _write_csv(
        reports_dir / "split_manifest.csv",
        ["split", "source", "scene_id", "cluster_id", "image", "label", "classes", "image_sha256", "label_sha256", "selection_reason"],
        manifest_rows,
    )
    _write_csv(
        reports_dir / "excluded_added.csv",
        ["source", "scene_id", "cluster_id", "image", "label", "classes", "reason"],
        [{
            "source": record.source,
            "scene_id": record.scene_id,
            "cluster_id": record.cluster_id,
            "image": record.name,
            "label": record.label_name,
            "classes": ";".join(f"{key}:{value}" for key, value in sorted(record.classes.items())),
            "reason": record.selection_reason,
        } for record in sorted(excluded_records, key=lambda item: item.name.casefold())],
    )
    _write_csv(
        reports_dir / "exact_duplicate_report.csv",
        ["sha256", "splits", "images"],
        exact_duplicate_rows,
    )
    _write_csv(
        reports_dir / "near_duplicate_report.csv",
        ["distance", "evaluation_split", "evaluation_image", "evaluation_scene", "train_image", "train_source", "train_scene"],
        near_duplicates,
    )

    split_images = Counter(record.split for record in selected_records)
    split_sources = {
        split: dict(Counter(record.source for record in selected_records if record.split == split))
        for split in SPLITS
    }
    class_rows = []
    for class_id, name in CLASS_NAMES.items():
        row = {"class_id": class_id, "class_name": name}
        total = 0
        for split in SPLITS:
            count = sum(record.classes[class_id] for record in selected_records if record.split == split)
            row[split] = count
            total += count
        row["total"] = total
        class_rows.append(row)
    _write_csv(
        reports_dir / "class_distribution.csv",
        ["class_id", "class_name", "train", "val", "test", "total"],
        class_rows,
    )

    scene_rows = []
    buckets: dict[tuple[str, str, str], list[Record]] = defaultdict(list)
    for record in selected_records:
        buckets[(record.split, record.source, record.cluster_id)].append(record)
    for (split, source, cluster_id), members in sorted(buckets.items()):
        counts: Counter[int] = Counter()
        for member in members:
            counts.update(member.classes)
        scene_rows.append({
            "split": split,
            "source": source,
            "cluster_id": cluster_id,
            "image_count": len(members),
            "instance_count": sum(counts.values()),
            "classes": ";".join(f"{key}:{value}" for key, value in sorted(counts.items())),
        })
    _write_csv(
        reports_dir / "scene_distribution.csv",
        ["split", "source", "cluster_id", "image_count", "instance_count", "classes"],
        scene_rows,
    )

    yaml_data = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 25,
        "names": CLASS_NAMES,
    }
    yaml_text = yaml.safe_dump(yaml_data, allow_unicode=True, sort_keys=False)
    (out_dir / "dataset.yaml").write_text(yaml_text, encoding="utf-8")
    (out_dir / "data.yaml").write_text(yaml_text, encoding="utf-8")
    (out_dir / "classes.txt").write_text("\n".join(CLASS_NAMES.values()) + "\n", encoding="utf-8")

    fingerprint_payload = [
        (row["split"], row["source"], row["cluster_id"], row["image"], row["image_sha256"], row["label_sha256"])
        for row in manifest_rows
    ]
    dataset_fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    class_totals = {str(row["class_id"]): row["total"] for row in class_rows}
    report = {
        "format": 1,
        "dataset_id": "scene811_official_strict701515_added_balanced_v1",
        "dataset_fingerprint": dataset_fingerprint,
        "policy": {
            "official": "preserve all 4481 images; strict filename groups plus strong MAR20 pHash components; grouped 70/15/15",
            "added": "train-only; retain non-FSC; uniformly sample pure-FSC sequences",
            "empty_added": "exclude pending relabel; never train an unverified empty annotation as background",
            "fsc_max_images_per_sequence": fsc_max_images_per_sequence,
            "near_duplicate_dhash_distance": near_duplicate_distance,
        },
        "images": len(selected_records),
        "excluded_added_images": len(excluded_records),
        "split_images": dict(split_images),
        "source_by_split": split_sources,
        "class_totals": class_totals,
        "official_scene_leaks": 0,
        "cluster_cross_split_leaks": 0,
        "exact_cross_split_duplicates": 0,
        "near_duplicate_candidates_for_review": len(near_duplicates),
    }
    (reports_dir / "dataset_fingerprint.json").write_text(
        json.dumps({"dataset_id": report["dataset_id"], "dataset_fingerprint": dataset_fingerprint}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (reports_dir / "leakage_report.json").write_text(
        json.dumps({
            "official_scene_leaks": 0,
            "added_in_val": split_sources["val"].get("added", 0),
            "added_in_test": split_sources["test"].get("added", 0),
            "cluster_cross_split_leaks": 0,
            "exact_cross_split_duplicates": 0,
            "near_duplicate_candidates_for_review": len(near_duplicates),
            "near_duplicate_report": "near_duplicate_report.csv",
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (reports_dir / "build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "README.md").write_text(
        "# Scene811 strict official 70:15:15 + balanced added training data\n\n"
        "- All 4,481 official images are preserved.\n"
        "- Official images are re-split by strict satellite/L1A/FSC-coordinate groups; strong MAR20 pHash matches are joined.\n"
        "- Official validation and test splits remain free of added data.\n"
        "- Added images are training-only.\n"
        f"- Pure-FSC sequences are uniformly sampled to at most {fsc_max_images_per_sequence} images per sequence.\n"
        "- See `reports/build_report.json` and `reports/split_manifest.csv` for provenance and hashes.\n"
        "- `near_duplicate_report.csv` contains conservative dHash candidates for review; it is not an automatic deletion list.\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the balanced Scene811 dataset directly from the official and combined ZIP files.")
    parser.add_argument("--official-zip", required=True)
    parser.add_argument("--combined-zip", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fsc-max-images-per-sequence", type=int, default=15)
    parser.add_argument("--near-duplicate-distance", type=int, default=2)
    args = parser.parse_args()
    report = build_dataset(
        Path(args.official_zip),
        Path(args.combined_zip),
        Path(args.out),
        fsc_max_images_per_sequence=args.fsc_max_images_per_sequence,
        near_duplicate_distance=args.near_duplicate_distance,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
