from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.pipeline import STAGES, empty_state, initialize, prerequisite_blocked, run_stage


def test_initialize_repairs_interrupted_running_stage(tmp_path: Path) -> None:
    state = empty_state()
    path = tmp_path / "pipeline_state.json"
    state["stages"]["freeze_legacy_v5"]["status"] = "running"
    state["stages"]["freeze_legacy_v5"]["started_at"] = "2026-08-04T00:00:00+00:00"
    path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    initialized = initialize(path)
    assert initialized["stages"]["freeze_legacy_v5"]["status"] == "failed"
    assert "interrupted" in initialized["stages"]["freeze_legacy_v5"]["note"]


def test_stage_requires_prerequisites(tmp_path: Path) -> None:
    state = empty_state()
    assert prerequisite_blocked(state, "audit_scene811") == "freeze_legacy_v5"
    state["stages"]["freeze_legacy_v5"]["status"] = "completed"
    assert prerequisite_blocked(state, "audit_scene811") == "prepare_scene811_yaml"
    for name in STAGES[:STAGES.index("audit_scene811")]:
        state["stages"][name]["status"] = "completed"
    assert prerequisite_blocked(state, "audit_scene811") is None


def test_unimplemented_stages_are_blocked(tmp_path: Path, monkeypatch) -> None:
    state = empty_state()
    state_path = tmp_path / "pipeline_state.json"
    state_path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    run_stage(state, "train_scene811_baselines", state_path=state_path)
    assert state["stages"]["train_scene811_baselines"]["status"] == "blocked"
    assert "not implemented" in state["stages"]["train_scene811_baselines"]["note"]
    assert json.loads(state_path.read_text(encoding="utf-8"))["stages"]["train_scene811_baselines"]["status"] == "blocked"


def test_rerun_completed_stage_requires_force(tmp_path: Path, monkeypatch) -> None:
    state = empty_state()
    state_path = tmp_path / "pipeline_state.json"
    state["stages"]["freeze_legacy_v5"]["status"] = "completed"
    state_path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr("src.pipeline._dispatch", lambda stage: [])
    with pytest.raises(RuntimeError, match="already completed"):
        run_stage(state, "freeze_legacy_v5", state_path=state_path)
    run_stage(state, "freeze_legacy_v5", force=True, state_path=state_path)
    assert state["stages"]["freeze_legacy_v5"]["status"] == "completed"
