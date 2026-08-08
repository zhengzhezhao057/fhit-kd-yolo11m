from __future__ import annotations

import json
from pathlib import Path

import yaml

from src.build_data_centric_matrix import ROUND1, build_matrix


def make_base(tmp_path: Path) -> Path:
    base = tmp_path / "baseline_seed0.yaml"
    base.write_text(
        yaml.safe_dump({
            "model": "yolo11m.pt",
            "data": "configs/datasets/scene811.server.yaml",
            "imgsz": 640, "batch": 16, "optimizer": "AdamW", "epochs": 50, "seed": 0,
        }),
        encoding="utf-8",
    )
    return base


def test_round1_matrix_contains_five_experiments(tmp_path: Path) -> None:
    base = make_base(tmp_path)
    out = tmp_path / "round1"
    matrix = build_matrix(base, 1, out)
    assert matrix["experiments"] == list(ROUND1)
    assert (out / "matrix.json").is_file()
    for name in ROUND1:
        assert (out / f"{name}.yaml").is_file()


def test_round1_budgets_match_design_table(tmp_path: Path) -> None:
    base = make_base(tmp_path)
    out = tmp_path / "round1"
    build_matrix(base, 1, out)
    for name, spec in ROUND1.items():
        config = yaml.safe_load((out / f"{name}.yaml").read_text(encoding="utf-8"))
        assert config["replay"]["hard_positive_fraction"] == spec["hard_positive"]
        assert config["replay"]["hard_negative_fraction"] == spec["hard_negative"]
        assert config["screen"]["epochs"] == 8
        assert config["screen"]["imgsz"] == 640


def test_round4_resolution_configs(tmp_path: Path) -> None:
    base = make_base(tmp_path)
    out = tmp_path / "round4"
    matrix = build_matrix(base, 4, out)
    assert matrix["experiments"] == ["res_640", "res_832", "res_1024"]
    config = yaml.safe_load((out / "res_1024.yaml").read_text(encoding="utf-8"))
    assert config["imgsz"] == 1024 and config["screen"]["imgsz"] == 1024
