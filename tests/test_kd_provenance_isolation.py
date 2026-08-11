from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch
import yaml

from src.distillation import TeacherSignalStore
from src.common import resolve_data_yaml, split_image_dir
from src.provenance import (
    artifact_namespace,
    cache_inventory,
    cache_manifest,
    file_sha256,
    prepare_run_lineage,
    require_teacher_compatible,
    resolve_dataset_identity,
    split_fingerprint_from_rows,
    student_runs_root,
    teacher_cache_dir,
    teacher_provenance,
    teacher_run_dir,
    validate_cache_manifest,
    verify_cache_sample,
    write_cache_inventory,
)
from src.prototype_bank import build_prototype_bank, validate_prototype_bank


def make_config(tmp_path: Path, *, strict: bool = True) -> tuple[dict, Path]:
    root = tmp_path / "dataset"
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
    rows: list[dict[str, str]] = []
    for split, name, payload in (
        ("train", "train.jpg", b"train-image"),
        ("val", "val.jpg", b"val-image"),
        ("test", "test.jpg", b"test-image"),
    ):
        image = root / "images" / split / name
        label = root / "labels" / split / Path(name).with_suffix(".txt")
        image.write_bytes(payload)
        label.write_text("24 0.5 0.5 0.1 0.1\n", encoding="utf-8")
        rows.append(
            {
                "split": split,
                "source": "official",
                "source_family": "fixture",
                "scene_id": f"scene:{split}",
                "cluster_id": f"cluster:{split}",
                "image": name,
                "label": label.name,
                "classes": "24:1",
                "balance_features": "vehicle:1",
                "image_sha256": file_sha256(image),
                "label_sha256": file_sha256(label),
                "selection_reason": "fixture",
            }
        )
    rows.sort(key=lambda row: (row["split"], row["image"].casefold()))
    manifest = root / "split_manifest.csv"
    with manifest.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    split_fp = split_fingerprint_from_rows(rows)
    dataset_fp = "a" * 64
    report = {
        "format": 1,
        "dataset_id": "scene811_v3_grouped_clean",
        "dataset_fingerprint": dataset_fp,
        "split_fingerprint": split_fp,
    }
    (root / "dataset_fingerprint.json").write_text(
        json.dumps(report), encoding="utf-8"
    )
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "nc": 25,
                "names": {index: f"c{index}" for index in range(25)},
            }
        ),
        encoding="utf-8",
    )
    dino_weights = tmp_path / "dino.pth"
    dino_weights.write_bytes(b"dino-fixture")
    dino_repo = tmp_path / "dino-repo"
    dino_repo.mkdir()
    baseline = tmp_path / "baseline.pt"
    baseline.write_bytes(b"baseline")
    project = tmp_path / "project"
    project.mkdir()
    cfg = {
        "paths": {
            "data_yaml": str(data_yaml),
            "baseline_weights": str(baseline),
            "dino_weights": str(dino_weights),
            "dino_repo": str(dino_repo),
            "project_root": str(project),
        },
        "dataset": {
            "id": "scene811_v3_grouped_clean" if strict else "scene811_v2",
            "nc": 25,
            "image_size": 640,
            "class_groups": {"ship": [0, 1, 2, 3], "aircraft": list(range(4, 24)), "vehicle": [24]},
        },
        "teacher": {
            "batch": 2,
            "accumulate": 4,
            "lr": 0.001,
            "num_workers": 0,
            "feature_channels": 256,
            "roi_size": 7,
        },
        "student": {"seed": 0},
    }
    if strict:
        cfg["dataset"].update(
            dataset_fingerprint=dataset_fp,
            split_fingerprint=split_fp,
        )
    else:
        (root / "dataset_fingerprint.json").unlink()
    return cfg, root


def make_teacher_checkpoint(cfg: dict, identity: dict, path: Path) -> dict:
    checkpoint = {"model": {}, "provenance": teacher_provenance(cfg, identity)}
    torch.save(checkpoint, path)
    return checkpoint


def test_v3_identity_and_artifacts_are_fingerprint_namespaced(tmp_path: Path) -> None:
    cfg, _ = make_config(tmp_path)
    identity = resolve_dataset_identity(cfg)
    namespace = "scene811_v3_grouped_clean__" + "a" * 12
    assert identity["strict"] is True
    assert artifact_namespace(identity) == namespace
    assert teacher_run_dir(cfg, identity) == Path(cfg["paths"]["project_root"]) / "runs" / namespace / "teacher"
    assert teacher_cache_dir(cfg, "train", identity) == Path(cfg["paths"]["project_root"]) / "cache" / "teacher_signals" / namespace / "train"
    assert student_runs_root(cfg, identity).name == namespace


def test_portable_v3_yaml_without_path_uses_yaml_parent(tmp_path: Path) -> None:
    cfg, root = make_config(tmp_path)
    portable = root / "dataset.yaml"
    portable.write_text(
        yaml.safe_dump(
            {
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "nc": 25,
                "names": {index: f"c{index}" for index in range(25)},
            }
        ),
        encoding="utf-8",
    )
    cfg["paths"]["data_yaml"] = str(portable)
    identity = resolve_dataset_identity(cfg)
    assert Path(identity["dataset_root"]) == root.resolve()
    assert artifact_namespace(identity) == "scene811_v3_grouped_clean__" + "a" * 12
    data = resolve_data_yaml(cfg)
    assert Path(data["path"]) == root.resolve()
    assert split_image_dir(data, "train") == root / "images" / "train"


def test_teacher_rejects_official_txt_list_dataset(tmp_path: Path) -> None:
    cfg, root = make_config(tmp_path)
    listing = root / "train_official.txt"
    listing.write_text("images/train/train.jpg\n", encoding="utf-8")
    official_yaml = root / "dataset_official.yaml"
    official_yaml.write_text(
        yaml.safe_dump(
            {
                "train": listing.name,
                "val": "images/val",
                "test": "images/test",
                "nc": 25,
                "names": {index: f"c{index}" for index in range(25)},
            }
        ),
        encoding="utf-8",
    )
    cfg["paths"]["data_yaml"] = str(official_yaml)
    with pytest.raises(RuntimeError, match="dataset_official.yaml"):
        split_image_dir(resolve_data_yaml(cfg), "train")


def test_legacy_paths_remain_backward_compatible(tmp_path: Path) -> None:
    cfg, _ = make_config(tmp_path, strict=False)
    identity = resolve_dataset_identity(cfg)
    project = Path(cfg["paths"]["project_root"])
    assert identity["strict"] is False
    assert teacher_run_dir(cfg, identity) == project / "runs" / "teacher"
    assert teacher_cache_dir(cfg, "train", identity) == project / "cache" / "teacher_signals" / "train"
    assert student_runs_root(cfg, identity) == project / "runs"


def test_legacy_v1_fingerprint_report_does_not_enable_v3_schema(tmp_path: Path) -> None:
    cfg, root = make_config(tmp_path, strict=False)
    (root / "dataset_fingerprint.json").write_text(
        json.dumps(
            {
                "dataset_id": "scene811_v1",
                "dataset_fingerprint": "1" * 64,
                "split_fingerprint": "2" * 64,
            }
        ),
        encoding="utf-8",
    )
    assert resolve_dataset_identity(cfg)["strict"] is False


def test_configured_fingerprint_mismatch_is_fatal(tmp_path: Path) -> None:
    cfg, _ = make_config(tmp_path)
    cfg["dataset"]["dataset_fingerprint"] = "b" * 64
    with pytest.raises(RuntimeError, match="differs"):
        resolve_dataset_identity(cfg)


def test_manifest_membership_mutation_is_fatal(tmp_path: Path) -> None:
    cfg, root = make_config(tmp_path)
    manifest = root / "split_manifest.csv"
    manifest.write_text(
        manifest.read_text(encoding="utf-8-sig").replace("scene:train", "scene:changed"),
        encoding="utf-8-sig",
    )
    with pytest.raises(RuntimeError, match="does not reproduce"):
        resolve_dataset_identity(cfg)


def create_cache_fixture(tmp_path: Path) -> tuple[dict, dict, Path, dict]:
    cfg, root = make_config(tmp_path)
    identity = resolve_dataset_identity(cfg)
    teacher_path = tmp_path / "teacher.pt"
    checkpoint = make_teacher_checkpoint(cfg, identity, teacher_path)
    cache_dir = teacher_cache_dir(cfg, "train", identity)
    cache_dir.mkdir(parents=True)
    records = cache_inventory(cfg, identity, "train")
    inventory_sha = write_cache_inventory(records, cache_dir / "inventory.jsonl")
    manifest = cache_manifest(
        cfg, identity, "train", teacher_path, checkpoint, inventory_sha, len(records)
    )
    (cache_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    record = records[0]
    torch.save(
        {
            "path": str((root / record["relative_image"]).resolve()),
            "image_sha256": record["image_sha256"],
            "label_sha256": record["label_sha256"],
            "cache_manifest_fingerprint": manifest["compatibility_fingerprint"],
            "boxes_xywhn": torch.zeros((1, 4)),
            "classes": torch.tensor([24]),
            # A valid teacher cache fixture must pass the production prototype
            # confidence, margin and coarse-class gates.
            "roi_logits": torch.nn.functional.one_hot(
                torch.tensor([24]), num_classes=25
            ).float() * 20.0,
            "roi_embeddings": torch.ones((1, 512)),
            "p3": torch.zeros((1, 1, 1)),
            "p4": torch.zeros((1, 1, 1)),
            "p5": torch.zeros((1, 1, 1)),
        },
        cache_dir / f"{record['key']}.pt",
    )
    return cfg, identity, cache_dir, manifest


def test_v3_rejects_legacy_cache_manifest(tmp_path: Path) -> None:
    cfg, _ = make_config(tmp_path)
    with pytest.raises(RuntimeError, match="format=3"):
        validate_cache_manifest(
            cfg,
            {"format": 2, "split": "train", "image_size": 640, "feature_channels": 256, "num_classes": 25},
            "train",
        )


def test_v3_teacher_resume_rejects_checkpoint_without_provenance(tmp_path: Path) -> None:
    cfg, _ = make_config(tmp_path)
    identity = resolve_dataset_identity(cfg)
    with pytest.raises(RuntimeError, match="has no provenance"):
        require_teacher_compatible(
            {"model": {}}, teacher_provenance(cfg, identity), strict=True
        )


def test_cache_manifest_carries_full_provenance_and_sample_verifies(tmp_path: Path) -> None:
    cfg, identity, cache_dir, manifest = create_cache_fixture(tmp_path)
    validate_cache_manifest(cfg, manifest, "train", identity)
    assert manifest["format"] == 3
    assert manifest["dataset_fingerprint"] == "a" * 64
    assert manifest["split_fingerprint"] == identity["split_fingerprint"]
    assert manifest["inventory_fingerprint"] == identity["inventory_fingerprint"]
    assert len(manifest["teacher_sha256"]) == 64
    assert len(manifest["dino"]["weights_sha256"]) == 64
    assert len(manifest["preprocess"]["fingerprint"]) == 64
    assert "project_git" in manifest
    assert verify_cache_sample(cfg, cache_dir, manifest, samples=1) == 1


def test_prototype_bank_build_and_verify_binds_v3_cache_provenance(tmp_path: Path) -> None:
    cfg, _, cache_dir, manifest = create_cache_fixture(tmp_path)
    cfg["distillation"] = {"prototype_min_count": 2}
    destination = tmp_path / "prototype_bank.pt"
    bank = build_prototype_bank(
        cfg, cache_dir, destination, min_count=2, verify_samples=1
    )
    assert destination.is_file()
    assert bank["roi_count"] == 1
    assert bank["embedding_dim"] == 512
    assert bank["provenance"]["cache_compatibility_fingerprint"] == manifest["compatibility_fingerprint"]
    validate_prototype_bank(cfg, manifest, bank)
    bank["provenance"]["teacher_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="fingerprint is invalid"):
        validate_prototype_bank(cfg, manifest, bank)


def test_cache_sample_detects_label_mutation(tmp_path: Path) -> None:
    cfg, _, cache_dir, manifest = create_cache_fixture(tmp_path)
    root = Path(yaml.safe_load(Path(cfg["paths"]["data_yaml"]).read_text())["path"])
    (root / "labels" / "train" / "train.txt").write_text(
        "24 0.4 0.4 0.2 0.2\n", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="changed after cache creation"):
        verify_cache_sample(cfg, cache_dir, manifest, samples=1)


def test_signal_store_rejects_entry_from_other_manifest(tmp_path: Path) -> None:
    cfg, _, cache_dir, manifest = create_cache_fixture(tmp_path)
    record = json.loads((cache_dir / "inventory.jsonl").read_text().splitlines()[0])
    entry_path = cache_dir / f"{record['key']}.pt"
    entry = torch.load(entry_path, map_location="cpu", weights_only=False)
    entry["cache_manifest_fingerprint"] = "0" * 64
    torch.save(entry, entry_path)
    store = TeacherSignalStore(
        cache_dir, manifest_fingerprint=manifest["compatibility_fingerprint"]
    )
    with pytest.raises(RuntimeError, match="another dataset/teacher manifest"):
        store.get(entry["path"])


def test_v3_run_lineage_forbids_cross_namespace_resume(tmp_path: Path) -> None:
    cfg, _ = make_config(tmp_path)
    identity = resolve_dataset_identity(cfg)
    run_dir = student_runs_root(cfg, identity) / "fk_seed0"
    baseline = Path(cfg["paths"]["baseline_weights"])
    prepare_run_lineage(
        cfg,
        run_dir,
        experiment="fk",
        initial_checkpoint=baseline,
        resume_checkpoint=None,
        cache_manifest_data=None,
        identity=identity,
    )
    foreign = tmp_path / "foreign.pt"
    foreign.write_bytes(b"foreign")
    with pytest.raises(RuntimeError, match="namespaced last.pt"):
        prepare_run_lineage(
            cfg,
            run_dir,
            experiment="fk",
            initial_checkpoint=baseline,
            resume_checkpoint=foreign,
            cache_manifest_data=None,
            identity=identity,
        )
