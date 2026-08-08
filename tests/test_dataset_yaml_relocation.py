from __future__ import annotations

import json
from pathlib import Path

import yaml

from src.prepare_dataset_yaml import build_dataset_yaml, prepare_configs


def make_source_yaml(tmp_path: Path, root: Path) -> Path:
    source = tmp_path / "dataset.yaml"
    source.write_text(
        "\n".join([
            f"path: {root}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "nc: 25",
            f"names: {json.dumps({str(i): f'c{i}' for i in range(25)})}",
        ]) + "\n",
        encoding="utf-8",
    )
    return source


def test_build_dataset_yaml_rewrites_path_only(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    source = make_source_yaml(tmp_path, root)
    target = tmp_path / "out" / "scene811.local.yaml"
    output = build_dataset_yaml(source, r"C:\data\scene811", target)
    assert output["path"] == "C:/data/scene811"
    assert output["nc"] == 25 and output["train"] == "images/train"
    with target.open("r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream)
    assert loaded["path"] == "C:/data/scene811"
    assert loaded["names"][24] == "c24"


def test_local_and_server_configs_are_independent(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    source = make_source_yaml(tmp_path, root)
    local_out = tmp_path / "scene811.local.yaml"
    server_out = tmp_path / "scene811.server.yaml"
    report = prepare_configs(
        source_yaml=source, local_out=local_out, server_out=server_out,
        local_path=Path(r"E:\deeplearning\dataset_6699_scene811"),
        server_path=Path("/root/rsdet/datasets/scene811_v1"),
    )
    local = yaml.safe_load(local_out.read_text(encoding="utf-8"))
    server = yaml.safe_load(server_out.read_text(encoding="utf-8"))
    assert local["path"] == "E:/deeplearning/dataset_6699_scene811"
    assert server["path"] == "/root/rsdet/datasets/scene811_v1"
    assert local["names"] == server["names"]
    assert report["local_yaml"] and report["server_yaml"]


def test_missing_names_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("path: /x\ntrain: images/train\nval: images/val\ntest: images/test\nnc: 25\n", encoding="utf-8")
    import pytest
    with pytest.raises(ValueError, match="names"):
        build_dataset_yaml(bad, "/data", tmp_path / "out.yaml")
