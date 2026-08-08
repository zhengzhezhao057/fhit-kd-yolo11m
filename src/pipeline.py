from __future__ import annotations

import argparse
import contextlib
import datetime
import io
import json
import subprocess
import sys
from pathlib import Path

from .artifact_paths import (
    audit_dir,
    dataset_root,
    log_dir,
    manifests_dir,
    pipeline_state_path,
    project_root,
    report_dir,
)
from .common import json_dump
from .dataset_d0 import file_sha256

STATE_FORMAT = 1

STAGES = [
    "freeze_legacy_v5",
    "prepare_scene811_yaml",
    "audit_scene811",
    "audit_non_l_scenes",
    "freeze_scene811_v1",
    "train_scene811_baselines",
    "evaluate_scene811_baselines",
    "build_scene811_oof_folds",
    "train_scene811_oof",
    "mine_scene811_oof",
    "build_native_replay_pools",
    "run_replay_screen",
    "evaluate_replay_screen",
    "run_replay_composition_screen",
    "run_augmentation_screen",
    "run_resolution_screen",
    "train_finalists",
    "freeze_thresholds",
    "evaluate_sealed_test",
    "final_report",
    "large_image_stage",
]

IMPLEMENTED = {
    "freeze_legacy_v5",
    "prepare_scene811_yaml",
    "audit_scene811",
    "audit_non_l_scenes",
    "freeze_scene811_v1",
}


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(project_root()), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def empty_state() -> dict:
    return {
        "format": STATE_FORMAT,
        "dataset_id": "scene811_v1",
        "created_at": utc_now(),
        "stages": {name: {"status": "pending", "note": ""} for name in STAGES},
    }


def load_state(path: Path | None = None) -> dict:
    state_path = path or pipeline_state_path()
    if not state_path.is_file():
        return empty_state()
    return json.loads(state_path.read_text(encoding="utf-8"))


def save_state(state: dict, path: Path | None = None) -> None:
    json_dump(state, path or pipeline_state_path())


def initialize(path: Path | None = None) -> dict:
    state_path = path or pipeline_state_path()
    state = load_state(state_path)
    changed = False
    for name in STAGES:
        entry = state["stages"].setdefault(name, {"status": "pending", "note": ""})
        if entry["status"] == "running":
            entry["status"] = "failed"
            entry["note"] = "interrupted before completion"
            changed = True
    if not changed and state_path.is_file() and state.get("format") == STATE_FORMAT:
        return state
    state["format"] = STATE_FORMAT
    state["dataset_id"] = "scene811_v1"
    save_state(state, state_path)
    return state


def status(state: dict) -> None:
    print(f"dataset_id={state['dataset_id']} created_at={state.get('created_at')}")
    for name in STAGES:
        entry = state["stages"][name]
        print(f"{name:38s} {entry['status']:10s} {entry.get('note', '')}")


def prerequisite_blocked(state: dict, stage: str) -> str | None:
    index = STAGES.index(stage)
    for previous in STAGES[:index]:
        status_value = state["stages"][previous]["status"]
        if status_value != "completed":
            return previous
    return None


def run_stage(state: dict, stage: str, *, force: bool = False, state_path: Path | None = None) -> dict:
    if stage not in STAGES:
        raise ValueError(f"Unknown stage {stage!r}")
    if stage not in IMPLEMENTED:
        entry = state["stages"][stage]
        entry["status"] = "blocked"
        entry["note"] = "not implemented yet; defined in DATASET_UPDATE_FRAMEWORK_DESIGN.md section 16"
        save_state(state, state_path)
        return state
    blocker = prerequisite_blocked(state, stage)
    if blocker is not None:
        raise RuntimeError(f"Stage {stage!r} requires completed stage {blocker!r} first.")
    entry = state["stages"][stage]
    if entry["status"] == "completed" and not force:
        raise RuntimeError(f"Stage {stage!r} is already completed; use --force to rerun.")
    entry["status"] = "running"
    entry["started_at"] = utc_now()
    entry["command"] = list(sys.argv)
    entry["git_commit"] = git_commit()
    entry["note"] = ""
    save_state(state, state_path)
    log_file = log_dir() / f"{stage}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    try:
        with log_file.open("w", encoding="utf-8") as log_stream, contextlib.redirect_stdout(buffer):
            outputs = _dispatch(stage)
        entry["status"] = "completed"
        entry["finished_at"] = utc_now()
        entry["log"] = str(log_file.relative_to(project_root()))
        entry["outputs"] = {str(path.relative_to(project_root())): file_sha256(path) for path in outputs}
        entry["note"] = f"outputs={len(outputs)}"
    except Exception as error:
        entry["status"] = "failed"
        entry["finished_at"] = utc_now()
        entry["log"] = str(log_file.relative_to(project_root()))
        entry["note"] = f"{type(error).__name__}: {error}"
        log_file.write_text(buffer.getvalue() + f"\nERROR: {type(error).__name__}: {error}\n", encoding="utf-8")
        save_state(state, state_path)
        raise
    log_file.write_text(buffer.getvalue(), encoding="utf-8")
    save_state(state, state_path)
    return state


def _dispatch(stage: str) -> list[Path]:
    root = dataset_root()
    manifest = root / "split_manifest.csv"
    if stage == "freeze_legacy_v5":
        from .freeze_legacy_evidence import freeze_legacy
        out = manifests_dir() / "legacy_evidence.json"
        freeze_legacy(None, out)
        return [out]
    if stage == "prepare_scene811_yaml":
        from .prepare_dataset_yaml import prepare_configs
        from .artifact_paths import config_dir, local_dataset_yaml, server_dataset_yaml, server_dataset_root
        local_out, server_out = local_dataset_yaml(), server_dataset_yaml()
        report = prepare_configs(
            source_yaml=root / "dataset.yaml",
            local_out=local_out,
            server_out=server_out,
            local_path=root,
            server_path=server_dataset_root(),
        )
        manifest_path = config_dir() / "scene811_yaml_manifest.json"
        json_dump(report, manifest_path)
        return [local_out, server_out, manifest_path]
    if stage == "audit_scene811":
        from .audit_dataset import audit_scene811
        out = manifests_dir() / "audit_scene811.json"
        report = audit_scene811(root, manifest, hash_images=True)
        json_dump(report, out)
        return [out]
    if stage == "audit_non_l_scenes":
        from .audit_scene_groups import audit_non_l_scenes, write_review_files
        report, entries, pairs = audit_non_l_scenes(root, manifest)
        out = manifests_dir() / "non_l_scene_audit.json"
        json_dump(report, out)
        inventory_out = report_dir() / "non_l_inventory.csv"
        pairs_out = audit_dir() / "non_l_near_duplicate_pairs.csv"
        write_review_files(entries, pairs, inventory_out, pairs_out)
        return [out, inventory_out, pairs_out]
    if stage == "freeze_scene811_v1":
        from .dataset_registry import fingerprint_scene811, write_fingerprint_report
        import yaml
        with (root / "dataset.yaml").open("r", encoding="utf-8") as stream:
            names = yaml.safe_load(stream)["names"]
        report = fingerprint_scene811(
            root, manifest, names,
            hash_images=True,
            label_fix_manifest_path=manifests_dir() / "label_fix_manifest.json",
            background_confirmation="pending_human_review",
            non_l_scene_audit_path=manifests_dir() / "non_l_scene_audit.json",
        )
        summary_out = manifests_dir() / "dataset_fingerprint.json"
        inventory_out = report_dir() / "fingerprint_inventory.json"
        write_fingerprint_report(report, summary_out, inventory_out)
        return [summary_out, inventory_out]
    raise RuntimeError(f"No handler for implemented stage {stage!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scene811 v1 unified pipeline (design section 16).")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("init")
    subparsers.add_parser("status")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("stage")
    run_parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.command == "init":
        state = initialize()
        print(f"initialized {pipeline_state_path()}")
    elif args.command == "status":
        status(load_state())
    else:
        state = load_state()
        run_stage(state, args.stage, force=args.force)
        status(load_state())


if __name__ == "__main__":
    main()
