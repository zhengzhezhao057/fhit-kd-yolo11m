from __future__ import annotations

from pathlib import Path

import pytest

from src.artifact_paths import DATASET_ID, project_root, run_dir


def test_project_root_is_repo_root() -> None:
    root = project_root()
    assert (root / "src" / "artifact_paths.py").is_file()
    assert (root / "docs" / "CLIENT_DELIVERY.md").is_file()


def test_run_dir_is_namespaced() -> None:
    path = run_dir("baseline_yolo11m_seed0")
    assert path == project_root() / "runs" / DATASET_ID / "baseline_yolo11m_seed0"


def test_run_dir_rejects_unsafe_names() -> None:
    with pytest.raises(ValueError):
        run_dir("../escape")
    with pytest.raises(ValueError):
        run_dir("a/b")
    with pytest.raises(ValueError):
        run_dir("")
