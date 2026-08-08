from __future__ import annotations

from collections import Counter
import csv
from pathlib import Path

import torch

from src.distillation import DistillationLoss
from src.result_gate import check_model, decide
from src.source_aware_split import assign_official_groups, split_dataset


def model(map95: float, ship: float, aircraft: float, vehicle: float, *, health: bool = True) -> dict:
    return {
        "native": {"map50_95": map95},
        "competition": {
            "overall": {"recall": 0.95, "false_alarm_rate": 0.04},
            "groups": {
                "ship": {"f1": ship},
                "aircraft": {"f1": aircraft},
                "vehicle": {"f1": vehicle},
            },
        },
        "evidence": {"kd_health": health, "deploy_parity": health},
    }


def test_result_gate_advances_only_noninferior_weak_group_gain() -> None:
    baseline = model(0.800, 0.91, 0.99, 0.78)
    candidate = model(0.801, 0.916, 0.988, 0.79)
    result = check_model(baseline, candidate)
    assert result["passed"]
    assert result["checks"]["weak_group_gain"]
    rejected = check_model(baseline, model(0.790, 0.92, 0.98, 0.80))
    assert not rejected["passed"]
    assert not rejected["checks"]["map_noninferiority"]


def test_result_gate_blocks_missing_health_evidence() -> None:
    payload = {
        "baseline": "C0",
        "models": {
            "C0": model(0.800, 0.91, 0.99, 0.78),
            "KD": model(0.802, 0.92, 0.99, 0.80, health=False),
        },
    }
    result = decide(payload)
    assert not result["passed"]
    assert result["candidates"]["KD"]["next_action"].startswith("STOP")


def test_official_split_keeps_clusters_intact_and_forces_added_overlap_to_train() -> None:
    groups = []
    for index in range(30):
        groups.append({
            "cluster_id": f"g{index}",
            "rows": [],
            "images": 2 if index % 3 == 0 else 1,
            "classes": Counter({index % 3: index % 4 + 1}),
        })
    assignment = assign_official_groups(
        groups, {"train": 0.7, "val": 0.15, "test": 0.15}, 42, 3, forced_train={"g4"}
    )
    assert assignment["g4"] == "train"
    assert set(assignment) == {group["cluster_id"] for group in groups}
    assert set(assignment.values()) == {"train", "val", "test"}


def test_hierarchical_kl_is_finite_and_zero_for_identical_logits() -> None:
    loss = object.__new__(DistillationLoss)
    loss.cfg = {
        "dataset": {"class_groups": {"ship": [0, 1], "aircraft": [2, 3], "vehicle": [4]}},
        "distillation": {"hierarchical_kl": {"enabled": True, "coarse_weight": 0.6, "within_group_weight": 0.4}},
    }
    logits = torch.tensor([[3.0, 1.0, -1.0, -2.0, -3.0], [-2.0, -1.0, 2.0, 1.0, 0.0]])
    values = loss._hierarchical_kl(logits, logits.clone(), torch.tensor([0, 2]), temperature=6.0)
    assert torch.isfinite(values).all()
    assert torch.allclose(values, torch.zeros_like(values), atol=1e-6)


def test_source_aware_split_materializes_directory_dataset(tmp_path: Path) -> None:
    dataset = tmp_path / "source"
    rows = []
    for index in range(17):
        old_split = ("train", "val", "test")[index % 3]
        image_dir = dataset / "images" / old_split
        label_dir = dataset / "labels" / old_split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        name = f"image_{index}.jpg"
        (image_dir / name).write_bytes(f"unique-image-{index}".encode())
        (label_dir / f"image_{index}.txt").write_text(f"{index % 3} 0.5 0.5 0.1 0.1\n", encoding="utf-8")
        source = "official" if index < 15 else "added"
        rows.append({
            "image": name,
            "source": source,
            "scene_id": f"scene_{index}",
            "cluster_id": f"cluster_{index}",
        })
    (dataset / "dataset.yaml").write_text(
        "nc: 3\nnames: {0: ship, 1: aircraft, 2: vehicle}\n", encoding="utf-8"
    )
    source_manifest = tmp_path / "source_manifest.csv"
    with source_manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("image", "source", "scene_id", "cluster_id"))
        writer.writeheader(); writer.writerows(rows)
    output = tmp_path / "scene811_v2"
    report = split_dataset(
        dataset, source_manifest, output, seed=42, link_mode="hardlink",
        expected_official=15, expected_added=2,
    )
    assert report["source_by_split"]["val"] == {"official": report["split_counts"]["val"]}
    assert report["source_by_split"]["test"] == {"official": report["split_counts"]["test"]}
    assert sum(1 for _ in (output / "images" / "train").glob("*.jpg")) == report["split_counts"]["train"]
    assert (output / "dataset.yaml").is_file()
