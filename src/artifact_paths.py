from __future__ import annotations

import os
from pathlib import Path


DEFAULT_DATASET_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "scene811_v2" / "split"
DEFAULT_SERVER_DATASET_DIR = Path("/root/fhit-kd-yolo11m/artifacts/scene811_v2/split")
DATASET_ID = "scene811_v2"


def project_root() -> Path:
    """Repository root, independent of the current working directory."""
    return Path(__file__).resolve().parents[1]


def dataset_root() -> Path:
    """Local Scene811 dataset root; overridable with SCENE811_DATASET_ROOT."""
    override = os.environ.get("SCENE811_DATASET_ROOT")
    root = Path(override).resolve() if override else DEFAULT_DATASET_DIR
    if not root.exists():
        raise FileNotFoundError(f"Scene811 dataset root does not exist: {root}")
    return root


def server_dataset_root() -> Path:
    return DEFAULT_SERVER_DATASET_DIR


def artifact_dir(dataset_id: str = DATASET_ID) -> Path:
    return project_root() / "artifacts" / dataset_id


def manifests_dir(dataset_id: str = DATASET_ID) -> Path:
    return artifact_dir(dataset_id) / "manifests"


def audit_dir(dataset_id: str = DATASET_ID) -> Path:
    return artifact_dir(dataset_id) / "audit"


def oof_dir(dataset_id: str = DATASET_ID) -> Path:
    return artifact_dir(dataset_id) / "oof"


def replay_dir(dataset_id: str = DATASET_ID) -> Path:
    return artifact_dir(dataset_id) / "replay"


def log_dir(dataset_id: str = DATASET_ID) -> Path:
    return artifact_dir(dataset_id) / "logs"


def pipeline_state_path(dataset_id: str = DATASET_ID) -> Path:
    return artifact_dir(dataset_id) / "pipeline_state.json"


def run_dir(experiment: str, dataset_id: str = DATASET_ID) -> Path:
    """Namespaced training output: runs/<dataset_id>/<experiment>/."""
    if not experiment or experiment != Path(experiment).name:
        raise ValueError(f"Unsafe experiment name: {experiment!r}")
    return project_root() / "runs" / dataset_id / experiment


def report_dir(dataset_id: str = DATASET_ID) -> Path:
    return project_root() / "reports" / dataset_id


def config_dir() -> Path:
    return project_root() / "configs" / "datasets"


def local_dataset_yaml() -> Path:
    return config_dir() / "scene811.local.yaml"


def server_dataset_yaml() -> Path:
    return config_dir() / "scene811.server.yaml"
