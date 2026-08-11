from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src import experiment_ledger as ledger


def json_write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


def make_inputs(tmp_path: Path) -> dict[str, Path]:
    dataset = json_write(
        tmp_path / "dataset_fingerprint.json",
        {"dataset_id": "scene811_v3_grouped_clean", "dataset_fingerprint": "a" * 64, "image_count": 8000},
    )
    config = json_write(tmp_path / "resolved_config.json", {"student": {"epochs": 12}, "seed": 42})
    initial = tmp_path / "yolo11m.pt"
    initial.write_bytes(b"initial checkpoint")
    return {"dataset": dataset, "config": config, "initial": initial}


@pytest.fixture
def deterministic_probes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ledger,
        "git_state",
        lambda repo: {
            "root": str(repo),
            "commit": "b" * 40,
            "branch": "main",
            "remote_origin": "https://example.invalid/repo.git",
            "dirty": False,
            "status_sha256": ledger.canonical_digest({}),
            "status_porcelain": [],
        },
    )
    monkeypatch.setattr(
        ledger,
        "environment_state",
        lambda: {
            "hostname": "test-host",
            "python": {"version": "3.11", "executable": "/python"},
            "packages": {"torch": "test"},
            "torch": {"cuda_available": True, "gpus": [{"name": "RTX 4090"}]},
        },
    )


def initialize(tmp_path: Path) -> tuple[Path, Path, dict]:
    inputs = make_inputs(tmp_path)
    run_dir = tmp_path / "runs" / "B0_seed42"
    registry = tmp_path / "runs" / "registry.jsonl"
    manifest = ledger.initialize_run(
        run_dir=run_dir,
        experiment="B0",
        dataset_report=inputs["dataset"],
        config=inputs["config"],
        seed=42,
        initial_checkpoint=inputs["initial"],
        command="python -m src.train_baseline --seed 42",
        registry=registry,
        repo=tmp_path,
        run_id="B0-seed42",
    )
    return run_dir, registry, manifest


def test_init_writes_immutable_manifest_and_registry(tmp_path: Path, deterministic_probes: None) -> None:
    run_dir, registry, manifest = initialize(tmp_path)
    assert manifest["record"]["dataset"]["fingerprint"] == "a" * 64
    assert manifest["record"]["seed"] == 42
    assert manifest["record"]["repository"]["commit"] == "b" * 40
    assert len(manifest["record"]["initial_checkpoint"]["sha256"]) == 64
    assert ledger.read_envelope(ledger.manifest_path(run_dir), expected_kind="run_manifest") == manifest
    events = ledger.read_chain(registry)
    assert [event["event"]["payload"]["action"] for event in events] == ["init"]
    with pytest.raises(FileExistsError, match="already exists"):
        initialize(tmp_path)


def test_manifest_tamper_is_detected_before_recording(tmp_path: Path, deterministic_probes: None) -> None:
    run_dir, _, _ = initialize(tmp_path)
    path = ledger.manifest_path(run_dir)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["record"]["seed"] = 7
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        ledger.record_epoch(run_dir=run_dir, epoch=1, metrics={"loss": 1.0})


def test_epoch_and_resume_ledgers_are_hash_chained(tmp_path: Path, deterministic_probes: None) -> None:
    run_dir, registry, _ = initialize(tmp_path)
    first = ledger.record_epoch(
        run_dir=run_dir,
        epoch=1,
        metrics={"train/box_loss": 0.8, "metrics/mAP50-95(B)": 0.7},
        kd_health={"optimizer_verified": True, "cache_misses": 0},
        elapsed_seconds=60,
        gpu_memory_gib=6.5,
    )
    second = ledger.record_epoch(run_dir=run_dir, epoch=2, metrics={"train/box_loss": 0.7})
    assert second["event"]["previous_event_sha256"] == first["event_sha256"]
    with pytest.raises(ValueError, match="already recorded"):
        ledger.record_epoch(run_dir=run_dir, epoch=2, metrics={"loss": 0.1})
    checkpoint = run_dir / "weights" / "last.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"resume")
    ledger.record_resume(run_dir=run_dir, checkpoint=checkpoint, command="python -m src.train_ablation --resume last.pt")
    assert len(ledger.read_chain(ledger.evidence_dir(run_dir) / ledger.RESUME_LEDGER_NAME)) == 1
    assert [event["event"]["payload"]["action"] for event in ledger.read_chain(registry)] == ["init", "resume"]


def make_final_evidence(tmp_path: Path, run_dir: Path) -> dict[str, Path]:
    weights = run_dir / "weights"
    weights.mkdir(parents=True, exist_ok=True)
    best, last, deploy = weights / "best.pt", weights / "last.pt", weights / "best_deploy.pt"
    best.write_bytes(b"best")
    last.write_bytes(b"last")
    deploy.write_bytes(b"deploy")
    results = run_dir / "results.csv"
    with results.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", "train/box_loss", "metrics/mAP50-95(B)"])
        writer.writeheader()
        writer.writerow({"epoch": 1, "train/box_loss": 0.8, "metrics/mAP50-95(B)": 0.75})
        writer.writerow({"epoch": 2, "train/box_loss": 0.7, "metrics/mAP50-95(B)": 0.80})
    kd = run_dir / "kd_health.jsonl"
    kd.write_text(
        "\n".join(
            json.dumps(value)
            for value in (
                {"epoch": 1, "experiment": "fk", "optimizer_verified": True, "cache_misses": 0, "feature_nonzero_batches": 10, "cls_nonzero_batches": 10, "feature_grad_events": 10, "roi_grad_events": 10, "kd_mean": 1.0},
                {"epoch": 2, "experiment": "fk", "optimizer_verified": True, "cache_misses": 0, "feature_nonzero_batches": 10, "cls_nonzero_batches": 10, "feature_grad_events": 10, "roi_grad_events": 10, "kd_mean": 0.5},
            )
        )
        + "\n",
        encoding="utf-8",
    )
    native = json_write(
        tmp_path / "native.json",
        {"models": {"B0": {"metrics": {"metrics/precision(B)": 0.95, "metrics/recall(B)": 0.94, "metrics/mAP50(B)": 0.97, "metrics/mAP50-95(B)": 0.80}, "per_class_map50_95": [0.8] * 25}}},
    )
    group = lambda tp, fp, fn, recall, fdr, f1: {"TP": tp, "FP": fp, "FN": fn, "recall": recall, "false_alarm_rate": fdr, "f1": f1}
    competition = json_write(
        tmp_path / "competition.json",
        {
            "split": "val",
            "class_aware_matching": True,
            "operating_points": {
                "best_f1": {
                    "confidence": 0.5,
                    "overall": group(2300, 80, 100, 0.958, 0.034, 0.962),
                    "per_group": {
                        "ship": group(450, 30, 50, 0.9, 0.0625, 0.918),
                        "aircraft": group(1700, 20, 10, 0.994, 0.012, 0.991),
                        "vehicle": group(150, 30, 40, 0.789, 0.167, 0.811),
                    },
                }
            },
        },
    )
    diagnostic_model = {
        "overall": {"instances": 2400, "detected_rate": 0.95},
        "per_group": {"vehicle": {"instances": 190, "detected_rate": 0.789}},
        "per_size": {
            "small": {"instances": 100, "detected_rate": 0.7},
            "medium": {"instances": 1600, "detected_rate": 0.95},
            "large": {"instances": 700, "detected_rate": 0.98},
        },
        "attributes": {"crowded": {"instances": 1000, "detected_rate": 0.9}, "edge": {"instances": 300, "detected_rate": 0.85}},
        "false_positives": {"overall": {"false_positives": 80}, "per_group": {"vehicle": {"false_positives": 30}}},
    }
    diagnostics = json_write(tmp_path / "diagnostics.json", {"summaries": {"B0": diagnostic_model}})
    timing = json_write(
        tmp_path / "timing.json",
        {
            "protocol": "competition_no_image_io_v1",
            "excludes_image_read": True,
            "interval_start": "after_all_image_reads_complete",
            "interval_end": "after_result_output_complete",
            "image_count": 455,
            "total_seconds": 9.1,
            "warmup_iterations": 10,
            "repetitions": 3,
            "batch": 8,
            "image_size": 640,
            "device": "NVIDIA GeForce RTX 4090",
        },
    )
    parity = json_write(
        tmp_path / "parity.json",
        {
            "full_detector_sha256": "d" * 64,
            "deploy_detector_sha256": "d" * 64,
            "parity": True,
        },
    )
    return {
        "best": best,
        "last": last,
        "deploy": deploy,
        "results": results,
        "kd": kd,
        "native": native,
        "competition": competition,
        "diagnostics": diagnostics,
        "timing": timing,
        "parity": parity,
    }


def test_complete_snapshots_and_normalizes_all_paper_evidence(tmp_path: Path, deterministic_probes: None) -> None:
    run_dir, registry, _ = initialize(tmp_path)
    files = make_final_evidence(tmp_path, run_dir)
    completed = ledger.complete_run(
        run_dir=run_dir,
        status="completed",
        best_checkpoint=files["best"],
        last_checkpoint=files["last"],
        deploy_checkpoint=files["deploy"],
        results_csv=files["results"],
        kd_health=files["kd"],
        native=files["native"],
        competition=files["competition"],
        diagnostics=files["diagnostics"],
        timing=files["timing"],
        parity=files["parity"],
        model_key="B0",
        elapsed_seconds=123.4,
        paper_ready=True,
    )
    record = completed["record"]
    assert record["evidence"]["native"]["map50_95"] == 0.80
    assert record["evidence"]["competition"]["per_group"]["vehicle"]["f1"] == 0.811
    assert record["evidence"]["diagnostics"]["per_size"]["small"]["detected_rate"] == 0.7
    assert record["evidence"]["diagnostics"]["attributes"]["crowded"]["instances"] == 1000
    assert record["evidence"]["kd_health"]["health_pass"] is True
    assert record["evidence"]["timing"]["mean_ms_per_image"] == 20.0
    assert record["evidence"]["deploy_parity"]["parity"] is True
    assert record["paper_ready"] is True
    assert len(record["evidence"]["epoch_metrics"]) == 2
    assert all(Path(value["snapshot"]["path"]).is_file() for value in record["artifacts"].values())
    assert record["checkpoints"]["best"]["sha256"] == ledger.file_sha256(files["best"])
    assert [event["event"]["payload"]["action"] for event in ledger.read_chain(registry)] == ["init", "complete"]
    with pytest.raises(RuntimeError, match="already finalized"):
        ledger.record_epoch(run_dir=run_dir, epoch=3, metrics={"loss": 0.1})


def test_summary_creates_json_csv_and_markdown(tmp_path: Path, deterministic_probes: None) -> None:
    run_dir, registry, _ = initialize(tmp_path)
    files = make_final_evidence(tmp_path, run_dir)
    ledger.complete_run(
        run_dir=run_dir,
        status="completed",
        best_checkpoint=files["best"],
        last_checkpoint=files["last"],
        results_csv=files["results"],
        native=files["native"],
        competition=files["competition"],
        diagnostics=files["diagnostics"],
        timing=files["timing"],
        model_key="B0",
        paper_ready=True,
    )
    output = tmp_path / "paper"
    report = ledger.summarize_registry(registry, output)
    assert report["event_count"] == 2
    assert report["runs"][0]["native_map50_95"] == 0.80
    assert report["runs"][0]["vehicle_f1"] == 0.811
    assert report["runs"][0]["competition_seven_ranking_ready"] is True
    assert (output / "experiment_summary.json").is_file()
    assert "Vehicle F1" in (output / "experiment_summary.md").read_text(encoding="utf-8")
    assert "dataset_fingerprint" in (output / "experiment_summary.csv").read_text(encoding="utf-8-sig")


def test_registry_tamper_breaks_summary(tmp_path: Path, deterministic_probes: None) -> None:
    _, registry, _ = initialize(tmp_path)
    lines = registry.read_text(encoding="utf-8").splitlines()
    event = json.loads(lines[0])
    event["event"]["payload"]["seed"] = 999
    registry.write_text(json.dumps(event) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        ledger.summarize_registry(registry, tmp_path / "summary")


def test_timing_requires_competition_boundary_and_paper_ready_is_strict(
    tmp_path: Path, deterministic_probes: None
) -> None:
    with pytest.raises(ValueError, match="excludes_image_read"):
        ledger.normalize_timing({"image_count": 10, "total_seconds": 1.0, "excludes_image_read": False})
    run_dir, _, _ = initialize(tmp_path)
    files = make_final_evidence(tmp_path, run_dir)
    with pytest.raises(ValueError, match="missing evidence"):
        ledger.complete_run(
            run_dir=run_dir,
            status="completed",
            best_checkpoint=files["best"],
            last_checkpoint=files["last"],
            paper_ready=True,
        )
