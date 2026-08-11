from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

from src.build_scene811_v3_dataset import (
    SUSPECT_UNLABELED_ADDED,
    added_scene_group,
    apply_near_duplicate_reviews,
    build_scene811_v3,
    conservative_official_scene,
    strict_official_group,
)
from src.verify_scene811_v3_dataset import verify_scene811_v3


def image_bytes(index: int) -> bytes:
    image = Image.new("RGB", (96, 80), (index * 17 % 255, index * 29 % 255, index * 43 % 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle((5 + index, 8, 25 + index, 28), outline=(255, 255, 255), width=2)
    draw.line((0, index * 7 % 80, 95, (index * 11 + 13) % 80), fill=(0, 0, 0), width=2)
    stream = io.BytesIO()
    image.save(stream, format="JPEG", quality=94)
    return stream.getvalue()


def write_fixture_archive(path: Path) -> tuple[list[tuple[str, bytes]], str]:
    official_names = [
        "01-PAN-20240420-113-325-L00000010001-CCD1_1_crop1.jpg",
        "01-PAN-20240420-113-325-L00000010001-CCD1_1_crop2.jpg",
        "01-PAN-20240421-113-325-L00000010002-CCD1_1_crop1.jpg",
        "01-PAN-20240422-113-325-L00000010003-CCD1_1_crop1.jpg",
        "01-PAN-20240423-113-325-L00000010004-CCD1_1_crop1.jpg",
        "01-PAN-20240424-113-325-L00000010005-CCD1_1_crop1.jpg",
        "01-PAN-20240425-113-325-L00000010006-CCD1_1_crop1.jpg",
        "01-PAN-20240426-113-325-L00000010007-CCD1_1_crop1.jpg",
        "fsc_TG-N24.15-E120.73-lv20-Google_crop0001.jpg",
    ]
    official: list[tuple[str, bytes]] = []
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index, name in enumerate(official_names):
            split = ("train", "val", "test")[index % 3]
            data = image_bytes(index + 1)
            official.append((name, data))
            archive.writestr(f"fixture/images/{split}/{name}", data)
            label_name = Path(name).with_suffix(".txt").name
            class_id = 24 if name.startswith("fsc_") else index % 4
            row = f"{class_id} 0.5 0.5 0.2 0.2"
            label = f"{row}\n{row}\n" if index == 2 else f"{row}\n"
            archive.writestr(f"fixture/labels/{split}/{label_name}", label)

        added_name = "P0080 (2).bmp"
        stream = io.BytesIO()
        Image.open(io.BytesIO(image_bytes(20))).save(stream, format="BMP")
        archive.writestr(f"fixture/images/val/{added_name}", stream.getvalue())
        archive.writestr("fixture/labels/val/P0080 (2).txt", "0 0.5 0.5 0.25 0.25\n")

        suspect_data = image_bytes(21)
        archive.writestr(f"fixture/images/train/{SUSPECT_UNLABELED_ADDED}", suspect_data)
        archive.writestr(
            f"fixture/labels/train/{Path(SUSPECT_UNLABELED_ADDED).with_suffix('.txt').name}",
            "",
        )
        archive.writestr(
            "fixture/curation_quarantine/example/images/AUAU010001.jpg", image_bytes(30)
        )
        archive.writestr(
            "fixture/curation_quarantine/example/labels/AUAU010001.txt",
            "24 0.5 0.5 0.2 0.2\n",
        )
    return official, hashlib.sha256(path.read_bytes()).hexdigest()


def write_official_manifest(path: Path, official: list[tuple[str, bytes]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["split", "source", "scene_id", "image", "image_sha256"]
        )
        writer.writeheader()
        for name, data in official:
            writer.writerow(
                {
                    "split": "legacy",
                    "source": "official",
                    "scene_id": strict_official_group(name),
                    "image": name,
                    "image_sha256": hashlib.sha256(data).hexdigest(),
                }
            )


def test_scene_group_rules_cover_official_and_added_families() -> None:
    assert strict_official_group(
        "01-PAN-20240420-113-325-L00000010882-CCD3_5_crop4.jpg"
    ) == "satellite:01-PAN-20240420-113-325-L00000010882"
    assert strict_official_group(
        "E103.9_N1.2_20200419_L1A0004749016-PAN20_crop1.jpg"
    ) == "l1a:E103.9_N1.2_20200419_L1A0004749016"
    assert strict_official_group(
        "fsc_AGZ-N24.15-E120.73-lv20-Bing_crop0002.jpg"
    ) == "fsc:N24.15:E120.73"
    assert conservative_official_scene(
        "02-PAN-20250325-081-360-L00000000455-CCD15_6_crop1.jpg"
    ) == conservative_official_scene(
        "02-PAN-20250325-081-361-L00000000456-CCD15_1_crop2.jpg"
    )
    assert conservative_official_scene(
        "E139.6_N35.3_20160202_L1A0001384774-PAN19_crop2.jpg"
    ) == conservative_official_scene(
        "E139.7_N35.2_20150922_L1A0001053752-PAN23_crop5.jpg"
    )
    assert added_scene_group("4_8_96_10346.jpg") == (
        "added_numeric:4_8_96",
        "added_numeric_ship",
    )
    assert added_scene_group("RURU020571.jpg") == (
        "added_sequence:RURU02",
        "added_au_ru_launcher",
    )
    assert added_scene_group("P0080 (2).bmp") == ("added_p:P0080", "added_p_ship")


def test_near_duplicate_candidate_blocks_until_manually_reviewed() -> None:
    candidate = {
        "left_image_sha256": "a" * 64,
        "right_image_sha256": "b" * 64,
    }
    rows = [dict(candidate)]
    assert apply_near_duplicate_reviews(rows, {}) == rows
    assert rows[0]["review_decision"] == "unresolved"

    key = frozenset({"a" * 64, "b" * 64})
    review = {
        key: {
            "decision": "different_scene_false_positive",
            "reviewer": "manual_visual",
            "reviewed_at_utc": "2026-08-10T16:03:22Z",
            "rationale": "different products and visibly different targets",
        }
    }
    reviewed_rows = [dict(candidate)]
    assert apply_near_duplicate_reviews(reviewed_rows, review) == []
    assert reviewed_rows[0]["review_decision"] == "different_scene_false_positive"
    assert reviewed_rows[0]["reviewer"] == "manual_visual"


def test_v3_build_is_deterministic_grouped_and_training_ready(tmp_path: Path) -> None:
    archive_path = tmp_path / "latest.zip"
    official, archive_sha = write_fixture_archive(archive_path)
    manifest_path = tmp_path / "official.csv"
    write_official_manifest(manifest_path, official)

    first = build_scene811_v3(
        archive_path,
        manifest_path,
        tmp_path / "out1",
        expected_official_images=9,
        expected_zip_sha256=archive_sha,
        mar20_phash_distance=-1,
        strong_near_duplicate_distance=-1,
    )
    second = build_scene811_v3(
        archive_path,
        tmp_path / "out1/reports/official_identity_manifest.csv",
        tmp_path / "out2",
        expected_official_images=9,
        expected_zip_sha256=archive_sha,
        mar20_phash_distance=-1,
        strong_near_duplicate_distance=-1,
    )

    assert first["audit_d0"]["training_ready"] is True
    assert first["audit_d0"]["selected_images"] == 10
    assert first["audit_d0"]["official_images"] == 9
    assert first["audit_d0"]["added_images"] == 1
    assert first["audit_d0"]["source_by_split"]["val"] == {"official": 1}
    assert first["audit_d0"]["source_by_split"]["test"] == {"official": 1}
    assert first["audit_d0"]["quarantine_ignored"] == {"images": 1, "labels": 1}
    assert first["audit_d0"]["label_patch_rows"] == 1
    assert first["audit_d0"]["official_identity_manifest_rows"] == 9
    assert first["audit_d0"]["checks"]["portable_yaml_has_no_absolute_path"] is True

    fingerprint1 = json.loads((tmp_path / "out1/dataset_fingerprint.json").read_text("utf-8"))
    fingerprint2 = json.loads((tmp_path / "out2/dataset_fingerprint.json").read_text("utf-8"))
    assert fingerprint1["dataset_fingerprint"] == fingerprint2["dataset_fingerprint"]
    assert fingerprint1["split_fingerprint"] == fingerprint2["split_fingerprint"]

    with (tmp_path / "out1/split_manifest.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    official_rows = [row for row in rows if row["source"] == "official"]
    scene_splits: dict[str, set[str]] = {}
    for row in official_rows:
        scene_splits.setdefault(row["scene_id"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in scene_splits.values())
    added_rows = [row for row in rows if row["source"] == "added"]
    assert {row["split"] for row in added_rows} == {"train"}

    patched = (
        tmp_path
        / "out1"
        / "labels"
        / next(row["split"] for row in rows if row["image"].endswith("10002-CCD1_1_crop1.jpg"))
        / "01-PAN-20240421-113-325-L00000010002-CCD1_1_crop1.txt"
    )
    assert len(patched.read_text("utf-8").splitlines()) == 1

    portable = tmp_path / "renamed_dataset"
    (tmp_path / "out1").rename(portable)
    mix_yaml = yaml.safe_load((portable / "dataset.yaml").read_text("utf-8"))
    official_yaml = yaml.safe_load(
        (portable / "dataset_official.yaml").read_text("utf-8")
    )
    assert "path" not in mix_yaml
    assert "path" not in official_yaml
    assert (portable / mix_yaml["train"]).is_dir()
    assert (portable / mix_yaml["val"]).is_dir()
    assert (portable / mix_yaml["test"]).is_dir()
    official_list = portable / official_yaml["train"]
    official_lines = official_list.read_text("utf-8").splitlines()
    assert len(official_lines) == 7
    assert all(line.startswith("./images/train/") for line in official_lines)
    assert all((official_list.parent / line[2:]).is_file() for line in official_lines)
    assert len((portable / "train_added.txt").read_text("utf-8").splitlines()) == 1
    with (portable / "reports/official_identity_manifest.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        assert len(list(csv.DictReader(stream))) == 9

    verified = verify_scene811_v3(
        portable,
        expected_fingerprint=fingerprint1["dataset_fingerprint"],
        hash_files=True,
        expected_official_images=9,
    )
    assert verified["passed"] is True
    tampered_label = next((portable / "labels/train").glob("*.txt"))
    original_label = tampered_label.read_bytes()
    tampered_label.write_bytes(original_label + b"\n")
    rejected = verify_scene811_v3(
        portable,
        expected_fingerprint=fingerprint1["dataset_fingerprint"],
        hash_files=True,
        expected_official_images=9,
    )
    assert rejected["passed"] is False
    assert rejected["hash_mismatches"] == 1
