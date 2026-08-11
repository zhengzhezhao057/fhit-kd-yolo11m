from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import numpy as np
import yaml
from PIL import Image

from .build_balanced_zip_dataset import CLASS_NAMES, IMAGE_SUFFIXES, strict_official_group
from .common import COARSE_NAMES, FINE_TO_COARSE
from .weak_group_diagnostics import crowded_flags, edge_flags, size_bucket


SPLITS = ("train", "val", "test")
DEFAULT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
DEFAULT_SEED = 20260810
DEFAULT_EXPECTED_ZIP_SHA256 = "f66212d1693baa92c6342ddac003775671a9c99e38fb6d26eee2cacd28d63bc5"
SUSPECT_UNLABELED_ADDED = "4_8_96_10345.jpg"


@dataclass
class V3Record:
    name: str
    label_name: str
    source: str
    source_family: str
    original_split: str
    split: str
    scene_id: str
    cluster_id: str
    image_entry: str
    label_entry: str
    classes: Counter[int]
    balance_features: Counter[str]
    image_sha256: str
    input_label_sha256: str
    output_label_sha256: str
    output_label_bytes: bytes = field(repr=False)
    dhash: int = 0
    phash: int | None = None
    selected: bool = True
    selection_reason: str = ""


class DisjointSet:
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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_sha256(payload: object) -> str:
    blob = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return bytes_sha256(blob)


def _entry_split(name: str, kind: str) -> str | None:
    match = re.search(
        rf"(?:^|/){kind}/(train|val|test)/[^/]+$", name, flags=re.IGNORECASE
    )
    return match.group(1).lower() if match else None


def active_image_entries(archive: zipfile.ZipFile) -> dict[str, tuple[str, str]]:
    result: dict[str, tuple[str, str]] = {}
    for info in archive.infolist():
        split = _entry_split(info.filename, "images")
        suffix = PurePosixPath(info.filename).suffix.lower()
        if split is None or suffix not in IMAGE_SUFFIXES:
            continue
        name = PurePosixPath(info.filename).name
        key = name.casefold()
        if key in result:
            raise RuntimeError(f"Duplicate active image basename: {name}")
        result[key] = (info.filename, split)
    return result


def active_label_entries(archive: zipfile.ZipFile) -> dict[tuple[str, str], str]:
    result: dict[tuple[str, str], str] = {}
    for info in archive.infolist():
        split = _entry_split(info.filename, "labels")
        if split is None or PurePosixPath(info.filename).suffix.lower() != ".txt":
            continue
        name = PurePosixPath(info.filename).name
        if name.casefold() == "classes.txt":
            continue
        key = (split, name.casefold())
        if key in result:
            raise RuntimeError(f"Duplicate active label path for {split}/{name}")
        result[key] = info.filename
    return result


def quarantine_inventory(archive: zipfile.ZipFile) -> dict[str, int]:
    images = labels = 0
    for info in archive.infolist():
        lowered = info.filename.casefold()
        if "/curation_quarantine/" not in lowered:
            continue
        suffix = PurePosixPath(info.filename).suffix.lower()
        if "/images/" in lowered and suffix in IMAGE_SUFFIXES:
            images += 1
        elif "/labels/" in lowered and suffix == ".txt":
            labels += 1
    return {"images": images, "labels": labels}


def load_official_manifest(
    path: Path, *, expected_official_images: int = 4481
) -> dict[str, dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Official manifest has no rows: {path}")
    required = {"image", "image_sha256"}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Official manifest is missing columns: {sorted(missing)}")
    if "source" in rows[0]:
        rows = [row for row in rows if row.get("source", "").strip().casefold() == "official"]
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        name = PurePosixPath(row["image"].strip().replace("\\", "/")).name
        key = name.casefold()
        if key in result:
            raise RuntimeError(f"Duplicate official basename in manifest: {name}")
        normalized = {field: (value or "").strip() for field, value in row.items()}
        normalized["image"] = name
        result[key] = normalized
    if len(result) != expected_official_images:
        raise RuntimeError(
            f"Expected {expected_official_images} official manifest images, found {len(result)}"
        )
    return result


def source_family_for_official(scene_id: str, name: str) -> str:
    if scene_id.startswith(("satellite:", "pan_strip:")):
        return "official_pan_l_product"
    if scene_id.startswith(("l1a:", "geo_site:")):
        return "official_coordinate_l1a"
    if scene_id.startswith("fsc:"):
        return "official_fsc_location"
    if name.upper().startswith("MAR20_"):
        return "official_mar20"
    return "official_other"


def conservative_official_scene(name: str) -> str:
    """Group acquisitions by physical site/strip, not only by product ID.

    Adjacent satellite products from the same daily strip and repeated L1A
    acquisitions within the same one-degree site can overlap substantially.
    Treating them as indivisible is intentionally conservative for evaluation.
    """
    stem = Path(name).stem
    pan = re.match(r"^(\d{2}-PAN-\d{8}-\d+)-\d+-L\d+", stem, flags=re.IGNORECASE)
    if pan:
        return f"pan_strip:{pan.group(1).upper()}"
    geo = re.match(
        r"^([EW])(\d+(?:\.\d+)?)_([NS])(\d+(?:\.\d+)?)_\d{8}_L1A\d+",
        stem,
        flags=re.IGNORECASE,
    )
    if geo:
        longitude = float(geo.group(2)) * (-1 if geo.group(1).upper() == "W" else 1)
        latitude = float(geo.group(4)) * (-1 if geo.group(3).upper() == "S" else 1)
        return f"geo_site:{round(longitude):+04d}:{round(latitude):+03d}"
    return strict_official_group(name)


def added_scene_group(name: str) -> tuple[str, str]:
    stem = Path(name).stem
    numeric = re.fullmatch(r"(\d+_\d+_\d+)_\d+", stem)
    if numeric:
        return f"added_numeric:{numeric.group(1)}", "added_numeric_ship"
    sequence = re.fullmatch(r"((?:AU|RU)(?:AU|RU)\d{2})\d+", stem, flags=re.IGNORECASE)
    if sequence:
        family = sequence.group(1).upper()
        return f"added_sequence:{family}", "added_au_ru_launcher"
    if re.fullmatch(r"P\d+(?:\s*\(\d+\))?", stem, flags=re.IGNORECASE):
        normalized = re.sub(r"\s*\(\d+\)$", "", stem).upper()
        return f"added_p:{normalized}", "added_p_ship"
    return f"added_single:{stem}", "added_other"


def _dhash(image_data: bytes) -> int:
    with Image.open(io.BytesIO(image_data)) as image:
        gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        pixels = np.asarray(gray, dtype=np.uint8).reshape(-1)
    value = 0
    for row in range(8):
        offset = row * 9
        for column in range(8):
            value = (value << 1) | int(
                pixels[offset + column] > pixels[offset + column + 1]
            )
    return value


def _phash(image_data: bytes) -> int:
    with Image.open(io.BytesIO(image_data)) as image:
        array = np.asarray(
            image.convert("L").resize((32, 32), Image.Resampling.LANCZOS),
            dtype=np.float64,
        )
    size = 32
    coordinates = np.arange(size)
    basis = np.cos(
        (2 * coordinates[None, :] + 1)
        * coordinates[:, None]
        * math.pi
        / (2 * size)
    )
    basis[0] *= 1 / math.sqrt(2)
    basis *= math.sqrt(2 / size)
    values = (basis @ array @ basis.T)[:8, :8].flatten()
    median = float(np.median(values[1:]))
    result = 0
    for value in values:
        result = (result << 1) | int(value > median)
    return result


def parse_and_patch_label(
    raw: bytes, entry: str, *, nc: int = 25
) -> tuple[Counter[int], np.ndarray, np.ndarray, bytes, list[dict[str, object]]]:
    text = raw.decode("utf-8-sig")
    output_lines: list[str] = []
    seen: dict[str, int] = {}
    patches: list[dict[str, object]] = []
    classes: list[int] = []
    boxes: list[list[float]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        fields = stripped.split()
        if len(fields) != 5:
            raise ValueError(f"Invalid YOLO row at {entry}:{line_number}: {line!r}")
        try:
            class_id = int(fields[0])
            box = [float(value) for value in fields[1:]]
        except ValueError as error:
            raise ValueError(
                f"Non-numeric YOLO row at {entry}:{line_number}: {line!r}"
            ) from error
        x, y, width, height = box
        if not 0 <= class_id < nc:
            raise ValueError(f"Class id outside 0..{nc - 1} at {entry}:{line_number}")
        if not (
            np.isfinite(box).all()
            and 0 <= x <= 1
            and 0 <= y <= 1
            and 0 < width <= 1
            and 0 < height <= 1
        ):
            raise ValueError(f"Invalid normalized box at {entry}:{line_number}: {box}")
        canonical = " ".join(fields)
        if canonical in seen:
            patches.append(
                {
                    "line_number": line_number,
                    "duplicate_of_line": seen[canonical],
                    "row": canonical,
                }
            )
            continue
        seen[canonical] = line_number
        output_lines.append(canonical)
        classes.append(class_id)
        boxes.append(box)
    output = ("\n".join(output_lines) + ("\n" if output_lines else "")).encode("utf-8")
    return (
        Counter(classes),
        np.asarray(classes, dtype=np.int64),
        np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
        output,
        patches,
    )


def image_balance_features(
    image_data: bytes,
    classes: np.ndarray,
    boxes: np.ndarray,
    source_family: str,
    *,
    image_size: int,
) -> Counter[str]:
    with Image.open(io.BytesIO(image_data)) as image:
        width, height = image.size
    features: Counter[str] = Counter({"images": 1, f"family:{source_family}": 1})
    if not len(classes):
        features["background_images"] = 1
        return features
    xyxy = np.empty_like(boxes)
    xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * width
    xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * height
    xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * width
    xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * height
    crowded = crowded_flags(xyxy, classes)
    edge = edge_flags(xyxy, width, height)
    for index, (class_id, box) in enumerate(zip(classes, xyxy)):
        fine = int(class_id)
        features[f"fine:{fine}"] += 1
        features[f"coarse:{COARSE_NAMES[FINE_TO_COARSE[fine]]}"] += 1
        features[f"size:{size_bucket(box, width, height, image_size)}"] += 1
        features["crowded_instances"] += int(crowded[index])
        features["edge_instances"] += int(edge[index])
    return features


def _feature_weight(name: str) -> float:
    if name == "images":
        return 3.0
    if name.startswith("size:"):
        return 2.0
    if name in {"crowded_instances", "edge_instances"}:
        return 2.0
    if name.startswith("coarse:"):
        return 1.4
    if name.startswith("fine:"):
        return 0.8
    if name.startswith("family:"):
        return 1.0
    if name == "background_images":
        return 1.0
    return 0.5


def _split_cost(
    split: str,
    counts: Counter[str],
    totals: Counter[str],
    ratios: dict[str, float],
) -> float:
    ratio = ratios[split]
    cost = 0.0
    for feature, total in totals.items():
        if total <= 0:
            continue
        target = ratio * total
        denominator = max(target, 3.0 if feature != "images" else 1.0)
        error = (counts[feature] - target) / denominator
        cost += _feature_weight(feature) * error * error
    image_target = ratio * totals["images"]
    overflow = max(0.0, counts["images"] - (ratio + 0.025) * totals["images"])
    cost += 50.0 * (overflow / max(image_target, 1.0)) ** 2
    return cost


def assign_official_groups(
    records: list[V3Record],
    *,
    ratios: dict[str, float] | None = None,
    seed: int = DEFAULT_SEED,
) -> dict[str, object]:
    import random

    ratios = dict(ratios or DEFAULT_RATIOS)
    if set(ratios) != set(SPLITS) or not math.isclose(sum(ratios.values()), 1.0):
        raise ValueError(f"Invalid split ratios: {ratios}")
    groups: dict[str, list[V3Record]] = defaultdict(list)
    for record in records:
        if record.source != "official":
            raise ValueError("assign_official_groups only accepts official records")
        groups[record.cluster_id].append(record)
    if len(groups) < len(SPLITS):
        raise RuntimeError(f"Need at least three official groups, found {len(groups)}")

    totals: Counter[str] = Counter()
    group_features: dict[str, Counter[str]] = {}
    for cluster_id, members in groups.items():
        features: Counter[str] = Counter()
        for member in members:
            features.update(member.balance_features)
        group_features[cluster_id] = features
        totals.update(features)

    rng = random.Random(seed)
    tie_break = {cluster_id: rng.random() for cluster_id in groups}
    rarity = {
        feature: 1.0 / math.sqrt(max(total, 1)) for feature, total in totals.items()
    }
    ordered = sorted(
        groups,
        key=lambda cluster_id: (
            -sum(
                value * rarity[feature] * _feature_weight(feature)
                for feature, value in group_features[cluster_id].items()
                if feature != "images"
            ),
            -group_features[cluster_id]["images"],
            tie_break[cluster_id],
            cluster_id,
        ),
    )
    assigned: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    assigned_groups: dict[str, list[str]] = {split: [] for split in SPLITS}
    assignment: dict[str, str] = {}
    split_tie = {split: rng.random() for split in SPLITS}
    for cluster_id in ordered:
        features = group_features[cluster_id]
        candidates: list[tuple[float, float, str]] = []
        for split in SPLITS:
            before = _split_cost(split, assigned[split], totals, ratios)
            after_counts = assigned[split] + features
            after = _split_cost(split, after_counts, totals, ratios)
            candidates.append((after - before, split_tie[split], split))
        split = min(candidates)[2]
        assignment[cluster_id] = split
        assigned[split].update(features)
        assigned_groups[split].append(cluster_id)

    if any(not assigned_groups[split] for split in SPLITS):
        raise RuntimeError(
            f"Grouped assignment failed to populate all splits: "
            f"{ {split: len(value) for split, value in assigned_groups.items()} }"
        )

    # Deterministic single-group local search. It improves size/crowd/source balance
    # without breaking scene integrity or using model predictions.
    moves = 0
    for _ in range(20):
        changed = False
        for cluster_id in sorted(groups):
            source = assignment[cluster_id]
            if len(assigned_groups[source]) <= 1:
                continue
            features = group_features[cluster_id]
            best: tuple[float, str] | None = None
            for target in SPLITS:
                if target == source:
                    continue
                before = _split_cost(source, assigned[source], totals, ratios) + _split_cost(
                    target, assigned[target], totals, ratios
                )
                source_after = assigned[source] - features
                target_after = assigned[target] + features
                after = _split_cost(source, source_after, totals, ratios) + _split_cost(
                    target, target_after, totals, ratios
                )
                delta = after - before
                candidate = (delta, target)
                if best is None or candidate < best:
                    best = candidate
            if best is not None and best[0] < -1e-10:
                target = best[1]
                assigned[source].subtract(features)
                assigned[target].update(features)
                assigned_groups[source].remove(cluster_id)
                assigned_groups[target].append(cluster_id)
                assignment[cluster_id] = target
                moves += 1
                changed = True
        if not changed:
            break

    for record in records:
        record.split = assignment[record.cluster_id]
    objective = sum(_split_cost(split, assigned[split], totals, ratios) for split in SPLITS)
    return {
        "seed": seed,
        "ratios": ratios,
        "groups": len(groups),
        "local_search_moves": moves,
        "objective": objective,
        "images": {split: assigned[split]["images"] for split in SPLITS},
    }


def merge_mar20_components(
    records: list[V3Record], *, phash_distance: int = 2
) -> dict[str, int]:
    official = [record for record in records if record.source == "official"]
    dsu = DisjointSet(sorted({record.cluster_id for record in official}))
    by_exact_hash: dict[str, list[V3Record]] = defaultdict(list)
    for record in official:
        by_exact_hash[record.image_sha256].append(record)
    exact_unions = 0
    for members in by_exact_hash.values():
        for member in members[1:]:
            dsu.union(members[0].cluster_id, member.cluster_id)
            exact_unions += 1
    mar20 = [record for record in official if record.phash is not None]
    phash_unions = 0
    if phash_distance >= 0:
        for index, left in enumerate(mar20):
            assert left.phash is not None
            for right in mar20[index + 1 :]:
                assert right.phash is not None
                if (left.phash ^ right.phash).bit_count() <= phash_distance:
                    if dsu.find(left.cluster_id) != dsu.find(right.cluster_id):
                        phash_unions += 1
                    dsu.union(left.cluster_id, right.cluster_id)
    for record in official:
        record.cluster_id = dsu.find(record.cluster_id)
    return {
        "mar20_images": len(mar20),
        "exact_unions": exact_unions,
        "phash_unions": phash_unions,
        "components": len({record.cluster_id for record in official}),
    }


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(data)


def _classes_text(classes: Counter[int]) -> str:
    return ";".join(f"{class_id}:{count}" for class_id, count in sorted(classes.items()))


def _features_text(features: Counter[str]) -> str:
    return ";".join(
        f"{feature}:{count}" for feature, count in sorted(features.items()) if count
    )


def cross_split_near_duplicates(
    records: list[V3Record], *, maximum_distance: int
) -> list[dict[str, object]]:
    if maximum_distance < 0:
        return []
    rows: list[dict[str, object]] = []
    by_split = {split: [record for record in records if record.split == split] for split in SPLITS}
    for left_split, right_split in (("train", "val"), ("train", "test"), ("val", "test")):
        for left in by_split[left_split]:
            for right in by_split[right_split]:
                distance = (left.dhash ^ right.dhash).bit_count()
                if distance <= maximum_distance:
                    rows.append(
                        {
                            "distance": distance,
                            "left_split": left_split,
                            "left_image": left.name,
                            "left_source": left.source,
                            "left_scene": left.scene_id,
                            "left_image_sha256": left.image_sha256,
                            "right_split": right_split,
                            "right_image": right.name,
                            "right_source": right.source,
                            "right_scene": right.scene_id,
                            "right_image_sha256": right.image_sha256,
                        }
                    )
    rows.sort(
        key=lambda row: (
            int(row["distance"]),
            str(row["left_image"]),
            str(row["right_image"]),
        )
    )
    return rows


def load_near_duplicate_review(
    path: Path | None, *, source_zip_sha256: str
) -> tuple[dict[frozenset[str], dict], str | None]:
    """Load an auditable false-positive review; it never silently merges scenes."""
    if path is None:
        return {}, None
    review_path = Path(path)
    payload = json.loads(review_path.read_text(encoding="utf-8"))
    recorded_source = str(payload.get("source_zip_sha256", "")).casefold()
    if recorded_source != source_zip_sha256.casefold():
        raise RuntimeError(
            "Near-duplicate review belongs to a different source ZIP: "
            f"review={recorded_source or 'missing'} current={source_zip_sha256}"
        )
    result: dict[frozenset[str], dict] = {}
    for row in payload.get("pairs", []):
        key = frozenset(
            {
                str(row["left_image_sha256"]).casefold(),
                str(row["right_image_sha256"]).casefold(),
            }
        )
        if len(key) != 2:
            raise ValueError(f"Invalid review pair hashes: {row}")
        decision = str(row.get("decision", ""))
        if decision != "different_scene_false_positive":
            raise ValueError(
                "Only 'different_scene_false_positive' may resolve a candidate. "
                "A true near duplicate must be joined in the scene grouping, not allowlisted."
            )
        if key in result:
            raise RuntimeError(f"Duplicate near-duplicate review pair: {sorted(key)}")
        result[key] = row
    return result, file_sha256(review_path)


def apply_near_duplicate_reviews(
    rows: list[dict[str, object]], review: dict[frozenset[str], dict]
) -> list[dict[str, object]]:
    """Annotate raw candidates and return only candidates still blocking D0."""
    unresolved: list[dict[str, object]] = []
    for row in rows:
        key = frozenset(
            {
                str(row["left_image_sha256"]).casefold(),
                str(row["right_image_sha256"]).casefold(),
            }
        )
        decision = review.get(key)
        if decision is None:
            row["review_decision"] = "unresolved"
            row["reviewer"] = ""
            row["reviewed_at_utc"] = ""
            row["review_rationale"] = ""
            unresolved.append(row)
        else:
            row["review_decision"] = decision["decision"]
            row["reviewer"] = decision.get("reviewer", "")
            row["reviewed_at_utc"] = decision.get("reviewed_at_utc", "")
            row["review_rationale"] = decision.get("rationale", "")
    return unresolved


def load_semantic_same_scene_review(
    path: Path | None, *, source_zip_sha256: str
) -> tuple[list[dict[str, str]], str | None, str | None]:
    """Load hash-only, human-confirmed same-scene unions.

    This is deliberately separate from perceptual-hash allowlisting: an audit
    candidate is never merged automatically. Only an immutable review record
    bound to both the source ZIP and DINO audit cache may join two groups.
    """
    if path is None:
        return [], None, None
    review_path = Path(path)
    payload = json.loads(review_path.read_text(encoding="utf-8"))
    if payload.get("kind") != "scene811_v3_semantic_same_scene_review":
        raise ValueError(f"Unexpected semantic review kind: {payload.get('kind')}")
    recorded_source = str(payload.get("source_zip_sha256", "")).casefold()
    if recorded_source != source_zip_sha256.casefold():
        raise RuntimeError(
            "Semantic same-scene review belongs to a different source ZIP: "
            f"review={recorded_source or 'missing'} current={source_zip_sha256}"
        )
    dino_audit_fingerprint = str(
        payload.get("dino_audit_cache_fingerprint", "")
    ).casefold()
    if not re.fullmatch(r"[0-9a-f]{64}", dino_audit_fingerprint):
        raise ValueError("Semantic review must bind a valid DINO audit cache fingerprint")
    result: list[dict[str, str]] = []
    seen: set[frozenset[str]] = set()
    for row in payload.get("pairs", []):
        left = str(row.get("left_image_sha256", "")).casefold()
        right = str(row.get("right_image_sha256", "")).casefold()
        if not re.fullmatch(r"[0-9a-f]{64}", left) or not re.fullmatch(
            r"[0-9a-f]{64}", right
        ):
            raise ValueError(f"Invalid semantic review pair hashes: {row}")
        key = frozenset({left, right})
        if len(key) != 2:
            raise ValueError(f"Semantic review must join two distinct images: {row}")
        if key in seen:
            raise RuntimeError(f"Duplicate semantic same-scene pair: {sorted(key)}")
        if row.get("decision") != "same_scene_union":
            raise ValueError("Only a human 'same_scene_union' decision may merge groups")
        for required in ("reviewer", "reviewed_at_utc", "rationale"):
            if not str(row.get(required, "")).strip():
                raise ValueError(f"Semantic review pair is missing {required}: {row}")
        seen.add(key)
        result.append({key: str(value) for key, value in row.items()})
    if not result:
        raise ValueError("Semantic same-scene review contains no reviewed pairs")
    return result, file_sha256(review_path), dino_audit_fingerprint


def apply_semantic_same_scene_unions(
    records: list[V3Record], review_pairs: list[dict[str, str]]
) -> list[dict[str, str]]:
    """Apply only declared SHA-pairs before grouped split assignment."""
    official = [record for record in records if record.source == "official"]
    dsu = DisjointSet(sorted({record.cluster_id for record in official}))
    by_hash: dict[str, list[V3Record]] = defaultdict(list)
    for record in official:
        by_hash[record.image_sha256.casefold()].append(record)
    applied: list[dict[str, str]] = []
    for row in review_pairs:
        left_hash = row["left_image_sha256"].casefold()
        right_hash = row["right_image_sha256"].casefold()
        left_matches = by_hash.get(left_hash, [])
        right_matches = by_hash.get(right_hash, [])
        if len(left_matches) != 1 or len(right_matches) != 1:
            raise RuntimeError(
                "Semantic same-scene review hashes must each resolve to exactly one "
                f"official image: left={len(left_matches)} right={len(right_matches)}"
            )
        left, right = left_matches[0], right_matches[0]
        before_left, before_right = left.cluster_id, right.cluster_id
        dsu.union(before_left, before_right)
        applied.append(
            {
                "left_image": left.name,
                "right_image": right.name,
                "left_image_sha256": left.image_sha256,
                "right_image_sha256": right.image_sha256,
                "left_cluster_before": before_left,
                "right_cluster_before": before_right,
                "decision": row["decision"],
                "reviewer": row["reviewer"],
                "reviewed_at_utc": row["reviewed_at_utc"],
                "rationale": row["rationale"],
            }
        )
    for record in official:
        record.cluster_id = dsu.find(record.cluster_id)
    for item in applied:
        item["cluster_after"] = dsu.find(item["left_cluster_before"])
    return applied


def _distribution(records: list[V3Record]) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for split in SPLITS:
        members = [record for record in records if record.split == split]
        features: Counter[str] = Counter()
        classes: Counter[int] = Counter()
        sources = Counter(record.source for record in members)
        families = Counter(record.source_family for record in members)
        for record in members:
            features.update(record.balance_features)
            classes.update(record.classes)
        instances = sum(classes.values())
        result[split] = {
            "images": len(members),
            "instances": instances,
            "sources": dict(sorted(sources.items())),
            "source_families": dict(sorted(families.items())),
            "fine_classes": {str(index): classes[index] for index in CLASS_NAMES},
            "coarse_groups": {
                name: features[f"coarse:{name}"] for name in COARSE_NAMES
            },
            "sizes": {
                name: features[f"size:{name}"] for name in ("small", "medium", "large")
            },
            "small_fraction": features["size:small"] / max(instances, 1),
            "crowded_instances": features["crowded_instances"],
            "crowded_fraction": features["crowded_instances"] / max(instances, 1),
            "edge_instances": features["edge_instances"],
            "edge_fraction": features["edge_instances"] / max(instances, 1),
            "background_images": features["background_images"],
        }
    return result


def _maximum_fraction_gap(distribution: dict[str, dict], key: str) -> float:
    values = [float(distribution[split][key]) for split in SPLITS]
    return max(values) - min(values)


def build_scene811_v3(
    latest_zip_path: Path,
    official_manifest_path: Path,
    out_dir: Path,
    *,
    seed: int = DEFAULT_SEED,
    ratios: dict[str, float] | None = None,
    image_size: int = 640,
    expected_official_images: int = 4481,
    expected_zip_sha256: str | None = DEFAULT_EXPECTED_ZIP_SHA256,
    mar20_phash_distance: int = 2,
    strong_near_duplicate_distance: int = 2,
    near_duplicate_review_path: Path | None = None,
    semantic_same_scene_review_path: Path | None = None,
    dataset_id: str = "scene811_v3_grouped_clean",
) -> dict[str, object]:
    latest_zip_path = Path(latest_zip_path)
    official_manifest_path = Path(official_manifest_path)
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    source_zip_sha256 = file_sha256(latest_zip_path)
    if expected_zip_sha256 and source_zip_sha256.casefold() != expected_zip_sha256.casefold():
        raise RuntimeError(
            f"Latest ZIP SHA-256 mismatch: expected {expected_zip_sha256}, "
            f"got {source_zip_sha256}"
        )
    official_manifest_sha256 = file_sha256(official_manifest_path)
    official_evidence = load_official_manifest(
        official_manifest_path, expected_official_images=expected_official_images
    )
    patch_rows: list[dict[str, object]] = []
    records: list[V3Record] = []

    with zipfile.ZipFile(latest_zip_path) as archive:
        image_entries = active_image_entries(archive)
        label_entries = active_label_entries(archive)
        quarantine = quarantine_inventory(archive)
        missing_official = sorted(set(official_evidence).difference(image_entries))
        if missing_official:
            raise RuntimeError(
                f"Latest ZIP is missing {len(missing_official)} official images; "
                f"examples={missing_official[:5]}"
            )
        for key, (image_entry, original_split) in sorted(image_entries.items()):
            name = PurePosixPath(image_entry).name
            label_name = f"{Path(name).stem}.txt"
            label_entry = label_entries.get((original_split, label_name.casefold()))
            if label_entry is None:
                raise FileNotFoundError(
                    f"Missing active label for {original_split}/{name}"
                )
            image_data = archive.read(image_entry)
            raw_label = archive.read(label_entry)
            image_sha = bytes_sha256(image_data)
            classes, class_array, boxes, output_label, duplicate_patches = parse_and_patch_label(
                raw_label, label_entry
            )
            if key in official_evidence:
                evidence = official_evidence[key]
                expected_image_sha = evidence["image_sha256"].casefold()
                if image_sha.casefold() != expected_image_sha:
                    raise RuntimeError(
                        f"Official image hash mismatch for {name}: "
                        f"manifest={expected_image_sha} latest_zip={image_sha}"
                    )
                scene_id = conservative_official_scene(name)
                source = "official"
                source_family = source_family_for_official(scene_id, name)
                cluster_id = f"official:{scene_id}"
                split = "unassigned"
                reason = "official_grouped_70_15_15"
            else:
                scene_id, source_family = added_scene_group(name)
                source = "added"
                cluster_id = scene_id
                split = "train"
                reason = "added_active_train_only"
            selected = True
            if source == "added" and name.casefold() == SUSPECT_UNLABELED_ADDED.casefold():
                selected = False
                reason = "excluded_known_unlabeled_ship_pending_relabel"
            elif source == "added" and not classes:
                raise RuntimeError(
                    f"Unreviewed empty added label is forbidden: {original_split}/{name}"
                )
            features = image_balance_features(
                image_data,
                class_array,
                boxes,
                source_family,
                image_size=image_size,
            )
            record = V3Record(
                name=name,
                label_name=label_name,
                source=source,
                source_family=source_family,
                original_split=original_split,
                split=split,
                scene_id=scene_id,
                cluster_id=cluster_id,
                image_entry=image_entry,
                label_entry=label_entry,
                classes=classes,
                balance_features=features,
                image_sha256=image_sha,
                input_label_sha256=bytes_sha256(raw_label),
                output_label_sha256=bytes_sha256(output_label),
                output_label_bytes=output_label,
                dhash=_dhash(image_data),
                phash=_phash(image_data) if name.upper().startswith("MAR20_") else None,
                selected=selected,
                selection_reason=reason,
            )
            records.append(record)
            for patch in duplicate_patches:
                patch_rows.append(
                    {
                        "image": name,
                        "label": label_name,
                        "source": source,
                        "original_split": original_split,
                        "patch_type": "remove_exact_duplicate_label_row",
                        "line_number": patch["line_number"],
                        "duplicate_of_line": patch["duplicate_of_line"],
                        "row": patch["row"],
                        "input_label_sha256": bytes_sha256(raw_label),
                        "output_label_sha256": bytes_sha256(output_label),
                    }
                )

        official_records = [record for record in records if record.source == "official"]
        if len(official_records) != expected_official_images:
            raise RuntimeError(
                f"Expected {expected_official_images} active official images, "
                f"found {len(official_records)}"
            )
        merge_report = merge_mar20_components(
            official_records, phash_distance=mar20_phash_distance
        )
        semantic_review, semantic_review_sha256, semantic_dino_audit_fingerprint = (
            load_semantic_same_scene_review(
                semantic_same_scene_review_path,
                source_zip_sha256=source_zip_sha256,
            )
        )
        semantic_union_rows = apply_semantic_same_scene_unions(
            official_records, semantic_review
        )
        balance_report = assign_official_groups(
            official_records, ratios=ratios, seed=seed
        )

        selected_records = [record for record in records if record.selected]
        for record in selected_records:
            _write_bytes(
                out_dir / "images" / record.split / record.name,
                archive.read(record.image_entry),
            )
            _write_bytes(
                out_dir / "labels" / record.split / record.label_name,
                record.output_label_bytes,
            )

    selected_records = [record for record in records if record.selected]
    selected_names = [record.name.casefold() for record in selected_records]
    if len(selected_names) != len(set(selected_names)):
        raise RuntimeError("Selected dataset contains duplicate basenames")

    split_rows = [
        {
            "split": record.split,
            "source": record.source,
            "source_family": record.source_family,
            "scene_id": record.scene_id,
            "cluster_id": record.cluster_id,
            "image": record.name,
            "label": record.label_name,
            "classes": _classes_text(record.classes),
            "balance_features": _features_text(record.balance_features),
            "image_sha256": record.image_sha256,
            "label_sha256": record.output_label_sha256,
            "selection_reason": record.selection_reason,
        }
        for record in sorted(
            selected_records, key=lambda item: (item.split, item.name.casefold())
        )
    ]
    split_fields = [
        "split",
        "source",
        "source_family",
        "scene_id",
        "cluster_id",
        "image",
        "label",
        "classes",
        "balance_features",
        "image_sha256",
        "label_sha256",
        "selection_reason",
    ]
    _write_csv(out_dir / "split_manifest.csv", split_fields, split_rows)

    source_rows = [
        {
            "image": record.name,
            "label": record.label_name,
            "source": record.source,
            "source_family": record.source_family,
            "original_split": record.original_split,
            "original_image_entry": record.image_entry,
            "original_label_entry": record.label_entry,
            "image_sha256": record.image_sha256,
            "input_label_sha256": record.input_label_sha256,
            "selected": int(record.selected),
            "selection_reason": record.selection_reason,
        }
        for record in sorted(records, key=lambda item: item.name.casefold())
    ]
    source_fields = [
        "image",
        "label",
        "source",
        "source_family",
        "original_split",
        "original_image_entry",
        "original_label_entry",
        "image_sha256",
        "input_label_sha256",
        "selected",
        "selection_reason",
    ]
    _write_csv(out_dir / "source_manifest.csv", source_fields, source_rows)
    patch_fields = [
        "image",
        "label",
        "source",
        "original_split",
        "patch_type",
        "line_number",
        "duplicate_of_line",
        "row",
        "input_label_sha256",
        "output_label_sha256",
    ]
    _write_csv(out_dir / "patch_manifest.csv", patch_fields, patch_rows)
    semantic_union_fields = [
        "left_image",
        "right_image",
        "left_image_sha256",
        "right_image_sha256",
        "left_cluster_before",
        "right_cluster_before",
        "cluster_after",
        "decision",
        "reviewer",
        "reviewed_at_utc",
        "rationale",
    ]
    _write_csv(
        reports_dir / "semantic_same_scene_unions.csv",
        semantic_union_fields,
        semantic_union_rows,
    )

    # Deliberately omit `path`: Ultralytics resolves relative entries from the
    # YAML file's parent, so the dataset remains portable after upload/move.
    yaml_data = {
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 25,
        "names": CLASS_NAMES,
    }
    yaml_text = yaml.safe_dump(yaml_data, allow_unicode=True, sort_keys=False)
    (out_dir / "dataset.yaml").write_text(yaml_text, encoding="utf-8")
    (out_dir / "data.yaml").write_text(yaml_text, encoding="utf-8")
    (out_dir / "classes.txt").write_text(
        "\n".join(CLASS_NAMES.values()) + "\n", encoding="utf-8"
    )

    official_train = sorted(
        record.name
        for record in selected_records
        if record.split == "train" and record.source == "official"
    )
    added_train = sorted(
        record.name
        for record in selected_records
        if record.split == "train" and record.source == "added"
    )

    def write_portable_image_list(path: Path, names: list[str]) -> None:
        # BaseDataset.get_img_files resolves only lines beginning with "./"
        # relative to the list file. Keep these files at the dataset root.
        path.write_text(
            "".join(f"./images/train/{name}\n" for name in names), encoding="utf-8"
        )

    write_portable_image_list(out_dir / "train_official.txt", official_train)
    write_portable_image_list(out_dir / "train_added.txt", added_train)
    write_portable_image_list(
        out_dir / "train_mix.txt", sorted(official_train + added_train)
    )
    official_yaml = dict(yaml_data)
    official_yaml["train"] = "train_official.txt"
    (out_dir / "dataset_official.yaml").write_text(
        yaml.safe_dump(official_yaml, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    official_identity_rows = [
        {
            "source": "official",
            "image": record.name,
            "image_sha256": record.image_sha256,
            "scene_id": record.scene_id,
        }
        for record in sorted(
            (item for item in selected_records if item.source == "official"),
            key=lambda item: item.name.casefold(),
        )
    ]
    official_identity_path = reports_dir / "official_identity_manifest.csv"
    _write_csv(
        official_identity_path,
        ["source", "image", "image_sha256", "scene_id"],
        official_identity_rows,
    )
    official_identity_manifest_sha256 = file_sha256(official_identity_path)
    official_identity_fingerprint = canonical_sha256(official_identity_rows)

    exact_hash_splits: dict[str, set[str]] = defaultdict(set)
    exact_hash_images: dict[str, list[str]] = defaultdict(list)
    for record in selected_records:
        exact_hash_splits[record.image_sha256].add(record.split)
        exact_hash_images[record.image_sha256].append(record.name)
    exact_cross_rows = [
        {
            "image_sha256": digest,
            "splits": ";".join(sorted(exact_hash_splits[digest])),
            "images": ";".join(sorted(exact_hash_images[digest])),
        }
        for digest in sorted(exact_hash_splits)
        if len(exact_hash_splits[digest]) > 1
    ]
    _write_csv(
        reports_dir / "exact_cross_split_duplicates.csv",
        ["image_sha256", "splits", "images"],
        exact_cross_rows,
    )
    near_rows = cross_split_near_duplicates(
        selected_records, maximum_distance=strong_near_duplicate_distance
    )
    near_review, near_review_sha256 = load_near_duplicate_review(
        near_duplicate_review_path, source_zip_sha256=source_zip_sha256
    )
    unresolved_near_rows = apply_near_duplicate_reviews(near_rows, near_review)
    _write_csv(
        reports_dir / "strong_near_duplicate_report.csv",
        [
            "distance",
            "left_split",
            "left_image",
            "left_source",
            "left_scene",
            "left_image_sha256",
            "right_split",
            "right_image",
            "right_source",
            "right_scene",
            "right_image_sha256",
            "review_decision",
            "reviewer",
            "reviewed_at_utc",
            "review_rationale",
        ],
        near_rows,
    )

    scene_splits: dict[str, set[str]] = defaultdict(set)
    cluster_splits: dict[str, set[str]] = defaultdict(set)
    for record in selected_records:
        if record.source == "official":
            scene_splits[record.scene_id].add(record.split)
            cluster_splits[record.cluster_id].add(record.split)
    scene_leaks = {
        scene: sorted(splits) for scene, splits in scene_splits.items() if len(splits) > 1
    }
    cluster_leaks = {
        cluster: sorted(splits)
        for cluster, splits in cluster_splits.items()
        if len(splits) > 1
    }
    official_distribution = _distribution(
        [record for record in selected_records if record.source == "official"]
    )
    full_distribution = _distribution(selected_records)
    distribution_report = {
        "format": 1,
        "official_only": official_distribution,
        "full_training_view": full_distribution,
        "official_fraction_gaps": {
            "small": _maximum_fraction_gap(official_distribution, "small_fraction"),
            "crowded": _maximum_fraction_gap(
                official_distribution, "crowded_fraction"
            ),
            "edge": _maximum_fraction_gap(official_distribution, "edge_fraction"),
        },
    }
    (reports_dir / "distribution_report.json").write_text(
        json.dumps(distribution_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    split_fingerprint = canonical_sha256(
        [
            {
                "split": row["split"],
                "source": row["source"],
                "scene_id": row["scene_id"],
                "cluster_id": row["cluster_id"],
                "image": row["image"],
            }
            for row in split_rows
        ]
    )
    source_fingerprint = canonical_sha256(
        {
            "source_zip_sha256": source_zip_sha256,
            "official_identity_fingerprint": official_identity_fingerprint,
            "rows": [
                {
                    "image": row["image"],
                    "source": row["source"],
                    "selected": row["selected"],
                    "image_sha256": row["image_sha256"],
                    "input_label_sha256": row["input_label_sha256"],
                }
                for row in source_rows
            ],
        }
    )
    patch_fingerprint = canonical_sha256(patch_rows)
    dataset_fingerprint = canonical_sha256(
        {
            "dataset_id": dataset_id,
            "source_fingerprint": source_fingerprint,
            "split_fingerprint": split_fingerprint,
            "patch_fingerprint": patch_fingerprint,
            "near_duplicate_review_sha256": near_review_sha256 or "none",
            "semantic_same_scene_review_sha256": semantic_review_sha256 or "none",
            "semantic_same_scene_dino_audit_fingerprint": (
                semantic_dino_audit_fingerprint or "none"
            ),
            "inventory": [
                {
                    "split": row["split"],
                    "image": row["image"],
                    "image_sha256": row["image_sha256"],
                    "label_sha256": row["label_sha256"],
                }
                for row in split_rows
            ],
        }
    )
    fingerprint_report = {
        "format": 1,
        "dataset_id": dataset_id,
        "dataset_fingerprint": dataset_fingerprint,
        "source_fingerprint": source_fingerprint,
        "split_fingerprint": split_fingerprint,
        "patch_fingerprint": patch_fingerprint,
        "near_duplicate_review_sha256": near_review_sha256,
        "semantic_same_scene_review_sha256": semantic_review_sha256,
        "semantic_same_scene_dino_audit_fingerprint": (
            semantic_dino_audit_fingerprint
        ),
        "source_zip_sha256": source_zip_sha256,
        "official_manifest_sha256": official_manifest_sha256,
        "official_identity_manifest": "reports/official_identity_manifest.csv",
        "official_identity_manifest_sha256": official_identity_manifest_sha256,
        "official_identity_fingerprint": official_identity_fingerprint,
        "seed": seed,
        "ratios": ratios or DEFAULT_RATIOS,
        "image_count": len(selected_records),
    }
    (out_dir / "dataset_fingerprint.json").write_text(
        json.dumps(fingerprint_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    split_source_counts = {
        split: dict(
            Counter(
                record.source for record in selected_records if record.split == split
            )
        )
        for split in SPLITS
    }
    checks = {
        "official_images_equal_expected": len(
            [record for record in selected_records if record.source == "official"]
        )
        == expected_official_images,
        "added_images_in_val_test_zero": sum(
            1
            for record in selected_records
            if record.source == "added" and record.split in {"val", "test"}
        )
        == 0,
        "missing_image_label_pairs_zero": all(
            (out_dir / "images" / record.split / record.name).is_file()
            and (out_dir / "labels" / record.split / record.label_name).is_file()
            for record in selected_records
        ),
        "invalid_yolo_rows_zero": True,
        "official_scene_cross_split_zero": not scene_leaks,
        "cluster_cross_split_zero": not cluster_leaks,
        "exact_cross_split_duplicates_zero": not exact_cross_rows,
        "strong_near_duplicate_leaks_zero": not unresolved_near_rows,
        "unreviewed_empty_added_labels_zero": not any(
            record.source == "added" and record.selected and not record.classes
            for record in records
        ),
        "known_unlabeled_ship_excluded": any(
            record.name.casefold() == SUSPECT_UNLABELED_ADDED.casefold()
            and not record.selected
            for record in records
        ),
        "official_identity_manifest_count_correct": len(official_identity_rows)
        == expected_official_images,
        "portable_yaml_has_no_absolute_path": "path" not in yaml_data
        and "path" not in official_yaml,
        "portable_train_lists_complete": len(official_train) + len(added_train)
        == sum(record.split == "train" for record in selected_records),
        "semantic_same_scene_unions_applied": len(semantic_union_rows)
        == len(semantic_review),
        "semantic_same_scene_unions_single_cluster": all(
            next(
                record.cluster_id
                for record in selected_records
                if record.image_sha256 == row["left_image_sha256"]
            )
            == next(
                record.cluster_id
                for record in selected_records
                if record.image_sha256 == row["right_image_sha256"]
            )
            for row in semantic_union_rows
        ),
    }
    d0 = {
        "format": 1,
        "kind": "scene811_v3_d0_audit",
        "dataset_id": dataset_id,
        "dataset_fingerprint": dataset_fingerprint,
        "training_ready": all(checks.values()),
        "checks": checks,
        "active_images": len(records),
        "selected_images": len(selected_records),
        "excluded_active_images": len(records) - len(selected_records),
        "official_images": sum(
            record.source == "official" for record in selected_records
        ),
        "added_images": sum(record.source == "added" for record in selected_records),
        "split_images": dict(Counter(record.split for record in selected_records)),
        "source_by_split": split_source_counts,
        "quarantine_ignored": quarantine,
        "label_patch_rows": len(patch_rows),
        "official_identity_manifest_rows": len(official_identity_rows),
        "official_identity_manifest_sha256": official_identity_manifest_sha256,
        "scene_leaks": scene_leaks,
        "cluster_leaks": cluster_leaks,
        "exact_cross_split_duplicates": len(exact_cross_rows),
        "strong_near_duplicate_candidates": len(near_rows),
        "strong_near_duplicate_reviewed_distinct": len(near_rows)
        - len(unresolved_near_rows),
        "strong_near_duplicate_unresolved": len(unresolved_near_rows),
        "semantic_same_scene_review_sha256": semantic_review_sha256,
        "semantic_same_scene_dino_audit_fingerprint": (
            semantic_dino_audit_fingerprint
        ),
        "semantic_same_scene_unions": len(semantic_union_rows),
        "distribution_soft_targets": {
            "small_gap_preferred_max": 0.015,
            "small_gap_explain_above": 0.03,
            "crowded_edge_gap_preferred_max": 0.05,
            "crowded_edge_gap_explain_above": 0.08,
            "observed_official_fraction_gaps": distribution_report[
                "official_fraction_gaps"
            ],
        },
    }
    (out_dir / "audit_d0.json").write_text(
        json.dumps(d0, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    build_report = {
        "format": 1,
        "dataset_id": dataset_id,
        "dataset_fingerprint": dataset_fingerprint,
        "training_ready": d0["training_ready"],
        "policy": {
            "official": "all official images identified by immutable manifest and split as indivisible scene/near-duplicate groups",
            "added": "latest active added images are train-only",
            "quarantine": "ignored by default and never restored automatically",
            "patches": "derived view removes exact duplicate label rows; source ZIP remains read-only",
            "known_unlabeled_ship": f"{SUSPECT_UNLABELED_ADDED} excluded until relabeled",
            "assignment_inputs": "labels, object size, crowded/edge flags and source family only; no model predictions",
        },
        "source_zip": str(latest_zip_path.resolve()),
        "official_manifest": str(official_manifest_path.resolve()),
        "merge_report": merge_report,
        "semantic_same_scene_report": {
            "review_sha256": semantic_review_sha256,
            "dino_audit_cache_fingerprint": semantic_dino_audit_fingerprint,
            "declared_pairs": len(semantic_review),
            "applied_unions": len(semantic_union_rows),
            "evidence": "reports/semantic_same_scene_unions.csv",
        },
        "balance_report": balance_report,
        "d0_audit": "audit_d0.json",
    }
    (reports_dir / "build_report.json").write_text(
        json.dumps(build_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "README.md").write_text(
        "# Scene811 V3 grouped clean dataset\n\n"
        "This is a reproducible derived dataset. The source ZIP is immutable.\n\n"
        "- Official images: all 4,481, assigned by indivisible scene groups.\n"
        "- Added images: active, reviewed view only, train-only.\n"
        "- Quarantine: ignored by default.\n"
        f"- Known unlabeled ship `{SUSPECT_UNLABELED_ADDED}`: excluded.\n"
        "- Exact duplicate label rows: removed in the derived view and recorded.\n"
        "- Training is allowed only when `audit_d0.json` says `training_ready: true`.\n"
        "- `test` is locked after creation and must not be used for tuning.\n",
        encoding="utf-8",
    )
    return {"build_report": build_report, "audit_d0": d0}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the immutable Scene811 V3 grouped-clean training view."
    )
    parser.add_argument("--latest-zip", required=True)
    parser.add_argument(
        "--official-manifest",
        required=True,
        help="Audited manifest containing exactly 4,481 official image basenames and SHA-256 values.",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--dataset-id", default="scene811_v3_grouped_clean")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--expected-official-images", type=int, default=4481)
    parser.add_argument(
        "--expected-zip-sha256",
        default=DEFAULT_EXPECTED_ZIP_SHA256,
        help="Use an empty string only for synthetic/development archives.",
    )
    parser.add_argument("--mar20-phash-distance", type=int, default=2)
    parser.add_argument("--strong-near-duplicate-distance", type=int, default=2)
    parser.add_argument(
        "--near-duplicate-review",
        default=None,
        help="Optional immutable JSON review for visually distinct hash collisions.",
    )
    parser.add_argument(
        "--semantic-same-scene-review",
        default=None,
        help=(
            "Optional immutable hash-only human review that joins DINO-confirmed "
            "same-scene pairs before grouped assignment."
        ),
    )
    args = parser.parse_args()
    report = build_scene811_v3(
        Path(args.latest_zip),
        Path(args.official_manifest),
        Path(args.out),
        seed=args.seed,
        image_size=args.image_size,
        expected_official_images=args.expected_official_images,
        expected_zip_sha256=args.expected_zip_sha256 or None,
        mar20_phash_distance=args.mar20_phash_distance,
        strong_near_duplicate_distance=args.strong_near_duplicate_distance,
        near_duplicate_review_path=(
            Path(args.near_duplicate_review) if args.near_duplicate_review else None
        ),
        semantic_same_scene_review_path=(
            Path(args.semantic_same_scene_review)
            if args.semantic_same_scene_review
            else None
        ),
        dataset_id=args.dataset_id,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["audit_d0"]["training_ready"]:
        raise SystemExit("D0 audit did not pass; formal training is forbidden.")


if __name__ == "__main__":
    main()
