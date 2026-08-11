"""Tamper-evident experiment provenance and paper-evidence ledger.

The ledger deliberately lives beside, rather than inside, Ultralytics.  It can
therefore record baseline and distillation runs with the same schema without
changing their optimizers or checkpoints.  A run manifest and its completion
record are write-once JSON envelopes.  Epoch and resume events are append-only,
hash-chained JSONL files.  A global registry links all runs for later tables.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
MANIFEST_NAME = "run_manifest.json"
COMPLETION_NAME = "completion.json"
EVIDENCE_DIR_NAME = "evidence"
EPOCH_LEDGER_NAME = "epoch_events.jsonl"
RESUME_LEDGER_NAME = "resume_events.jsonl"
DEFAULT_REGISTRY = Path("runs") / "experiment_registry.jsonl"
PACKAGES = (
    "torch",
    "torchvision",
    "ultralytics",
    "opencv-python",
    "numpy",
    "PyYAML",
    "torchmetrics",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: str | Path) -> dict[str, Any]:
    target = Path(path).resolve()
    if not target.is_file():
        raise FileNotFoundError(target)
    stat = target.stat()
    return {
        "path": str(target),
        "sha256": file_sha256(target),
        "bytes": stat.st_size,
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def envelope(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "record": record,
        "record_sha256": canonical_digest(record),
    }


def read_envelope(path: str | Path, *, expected_kind: str | None = None) -> dict[str, Any]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION or not isinstance(payload.get("record"), dict):
        raise RuntimeError(f"Unsupported or malformed evidence envelope: {source}")
    actual = canonical_digest(payload["record"])
    if actual != payload.get("record_sha256"):
        raise RuntimeError(f"Evidence integrity failure (record hash mismatch): {source}")
    if expected_kind is not None and payload["record"].get("kind") != expected_kind:
        raise RuntimeError(f"Expected {expected_kind!r}, found {payload['record'].get('kind')!r}: {source}")
    return payload


def write_once_envelope(path: Path, record: dict[str, Any]) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"Immutable evidence already exists: {path}")
    value = envelope(record)
    atomic_json(path, value)
    return value


def evidence_dir(run_dir: str | Path) -> Path:
    return Path(run_dir).resolve() / EVIDENCE_DIR_NAME


def manifest_path(run_dir: str | Path) -> Path:
    return evidence_dir(run_dir) / MANIFEST_NAME


def completion_path(run_dir: str | Path) -> Path:
    return evidence_dir(run_dir) / COMPLETION_NAME


def load_structured(path: str | Path) -> Any:
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    if source.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - requirements include PyYAML
        raise RuntimeError("PyYAML is required to record a YAML config.") from exc
    return yaml.safe_load(text)


def find_dataset_fingerprint(value: Any) -> str | None:
    """Find a single explicit dataset fingerprint in a freeze report."""
    found: set[str] = set()

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            for key, child in node.items():
                if (
                    key in {"dataset_fingerprint", "fingerprint"}
                    and isinstance(child, str)
                    and len(child) == 64
                    and all(character in "0123456789abcdefABCDEF" for character in child)
                ):
                    found.add(child.lower())
                else:
                    visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(value)
    if len(found) > 1:
        raise ValueError(f"Dataset report contains multiple fingerprints: {sorted(found)}")
    return next(iter(found), None)


def command_text(command: str | None, command_file: str | None) -> tuple[str, dict[str, Any] | None]:
    if bool(command) == bool(command_file):
        raise ValueError("Provide exactly one of --command or --command-file.")
    if command_file:
        source = Path(command_file)
        text = source.read_text(encoding="utf-8").strip()
        return text, file_record(source)
    assert command is not None
    return command.strip(), None


def run_command(command: list[str], *, cwd: Path | None = None) -> str | None:
    try:
        result = subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def git_state(repo: Path) -> dict[str, Any]:
    commit = run_command(["git", "rev-parse", "HEAD"], cwd=repo)
    if not commit:
        raise RuntimeError(f"Not a readable Git worktree: {repo}")
    status = run_command(["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=repo) or ""
    branch = run_command(["git", "branch", "--show-current"], cwd=repo) or None
    remote = run_command(["git", "config", "--get", "remote.origin.url"], cwd=repo)
    return {
        "root": str(repo.resolve()),
        "commit": commit,
        "branch": branch,
        "remote_origin": remote,
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "status_porcelain": status.splitlines(),
    }


def disk_state(path: str | Path) -> dict[str, Any]:
    target = Path(path).resolve()
    usage = shutil.disk_usage(target)
    return {
        "path": str(target),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
    }


def environment_state() -> dict[str, Any]:
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    torch_info: dict[str, Any] = {}
    try:
        import torch

        torch_info = {
            "version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            "cuda_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "gpus": [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
                    "capability": list(torch.cuda.get_device_capability(index)),
                }
                for index in range(torch.cuda.device_count() if torch.cuda.is_available() else 0)
            ],
        }
    except Exception as exc:  # importing a mismatched CUDA build must not hide provenance
        torch_info = {"probe_error": f"{type(exc).__name__}: {exc}"}
    nvidia = run_command([
        "nvidia-smi",
        "--query-gpu=name,uuid,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ])
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": {"version": platform.python_version(), "executable": str(Path(sys.executable).resolve())},
        "packages": packages,
        "torch": torch_info,
        "nvidia_smi": nvidia.splitlines() if nvidia else [],
        "reproducibility_environment": {
            key: os.environ.get(key)
            for key in ("CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG", "PYTHONHASHSEED")
        },
    }


@contextmanager
def registry_lock(path: Path, timeout_seconds: float = 30.0) -> Iterable[None]:
    lock_path = path.with_suffix(path.suffix + ".lock")
    deadline = time.monotonic() + timeout_seconds
    descriptor = None
    while descriptor is None:
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(descriptor, f"pid={os.getpid()} utc={utc_now()}\n".encode("utf-8"))
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for ledger lock: {lock_path}")
            time.sleep(0.05)
    try:
        yield
    finally:
        os.close(descriptor)
        lock_path.unlink(missing_ok=True)


def read_chain(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    previous = "0" * 64
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        event = json.loads(line)
        core = event.get("event")
        if event.get("schema_version") != SCHEMA_VERSION or not isinstance(core, dict):
            raise RuntimeError(f"Malformed ledger event at {path}:{line_number}")
        if core.get("sequence") != len(events) + 1 or core.get("previous_event_sha256") != previous:
            raise RuntimeError(f"Broken ledger sequence/hash chain at {path}:{line_number}")
        actual = canonical_digest(core)
        if event.get("event_sha256") != actual:
            raise RuntimeError(f"Ledger event hash mismatch at {path}:{line_number}")
        previous = actual
        events.append(event)
    return events


def append_chain(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with registry_lock(path):
        events = read_chain(path)
        previous = events[-1]["event_sha256"] if events else "0" * 64
        core = {
            "sequence": len(events) + 1,
            "previous_event_sha256": previous,
            "payload": payload,
        }
        event = {"schema_version": SCHEMA_VERSION, "event": core, "event_sha256": canonical_digest(core)}
        with path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(event, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        return event


def append_registry(registry: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return append_chain(registry.resolve(), {"kind": "experiment_registry_event", **payload})


def registry_from_manifest(manifest: dict[str, Any]) -> Path:
    return Path(manifest["record"]["registry_path"])


def ensure_open_run(run_dir: str | Path) -> dict[str, Any]:
    manifest = read_envelope(manifest_path(run_dir), expected_kind="run_manifest")
    if completion_path(run_dir).exists():
        read_envelope(completion_path(run_dir), expected_kind="run_completion")
        raise RuntimeError(f"Run is already finalized and immutable: {Path(run_dir).resolve()}")
    return manifest


def initialize_run(
    *,
    run_dir: str | Path,
    experiment: str,
    dataset_report: str | Path,
    config: str | Path,
    seed: int,
    initial_checkpoint: str | Path,
    command: str,
    command_source: dict[str, Any] | None = None,
    registry: str | Path = DEFAULT_REGISTRY,
    repo: str | Path | None = None,
    run_id: str | None = None,
    resume_checkpoint: str | Path | None = None,
    parent_manifest: str | Path | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    destination = Path(run_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    manifest_target = manifest_path(destination)
    if manifest_target.exists() or completion_path(destination).exists():
        raise FileExistsError(f"Run evidence already exists: {evidence_dir(destination)}")
    repository = Path(repo).resolve() if repo else Path(__file__).resolve().parents[1]
    dataset_source = Path(dataset_report).resolve()
    dataset_value = load_structured(dataset_source)
    fingerprint = find_dataset_fingerprint(dataset_value)
    if not fingerprint:
        raise ValueError(f"No dataset_fingerprint found in {dataset_source}")
    config_source = Path(config).resolve()
    config_value = load_structured(config_source)
    if not isinstance(config_value, dict):
        raise ValueError("Resolved training config must be a mapping.")
    parent: dict[str, Any] | None = None
    if parent_manifest is not None:
        parent_envelope = read_envelope(parent_manifest, expected_kind="run_manifest")
        parent = {
            "manifest_path": str(Path(parent_manifest).resolve()),
            "manifest_sha256": parent_envelope["record_sha256"],
            "run_id": parent_envelope["record"]["run_id"],
        }
    if parent is not None and resume_checkpoint is None:
        raise ValueError("--parent-manifest requires --resume-checkpoint.")
    identifier = run_id or f"{experiment}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    record = {
        "kind": "run_manifest",
        "schema_version": SCHEMA_VERSION,
        "run_id": identifier,
        "experiment": experiment,
        "run_dir": str(destination),
        "created_utc": utc_now(),
        "dataset": {
            "fingerprint": fingerprint,
            "report": file_record(dataset_source),
            "dataset_id": dataset_value.get("dataset_id") if isinstance(dataset_value, dict) else None,
        },
        "repository": git_state(repository),
        "config": {
            "source": file_record(config_source),
            "resolved": config_value,
            "resolved_sha256": canonical_digest(config_value),
        },
        "invocation": {
            "training_command": command,
            "command_source": command_source,
            "working_directory": str(Path.cwd().resolve()),
        },
        "seed": int(seed),
        "initial_checkpoint": file_record(initial_checkpoint),
        "resume_lineage": {
            "mode": "resume" if resume_checkpoint is not None else "fresh",
            "checkpoint": file_record(resume_checkpoint) if resume_checkpoint is not None else None,
            "parent": parent,
        },
        "environment": environment_state(),
        "resource_start": {"disk": disk_state(destination)},
        "registry_path": str(Path(registry).resolve()),
        "notes": notes,
        "tracking_contract": {
            "epoch_metrics": EPOCH_LEDGER_NAME,
            "resume_events": RESUME_LEDGER_NAME,
            "completion": COMPLETION_NAME,
            "required_final_evidence": [
                "best checkpoint SHA-256",
                "last checkpoint SHA-256",
                "native validation",
                "competition evaluation",
                "per-group/size/crowded/edge diagnostics",
                "elapsed time and final disk state",
            ],
        },
    }
    written = write_once_envelope(manifest_target, record)
    append_registry(Path(registry), {
        "action": "init",
        "utc": record["created_utc"],
        "run_id": identifier,
        "experiment": experiment,
        "run_dir": str(destination),
        "manifest_path": str(manifest_target),
        "manifest_sha256": written["record_sha256"],
        "dataset_fingerprint": fingerprint,
        "git_commit": record["repository"]["commit"],
        "config_sha256": record["config"]["resolved_sha256"],
        "seed": int(seed),
    })
    return written


def read_metrics_argument(metrics_json: str | None, metrics_file: str | None) -> dict[str, Any]:
    if bool(metrics_json) == bool(metrics_file):
        raise ValueError("Provide exactly one of --metrics-json or --metrics-file.")
    value = json.loads(metrics_json) if metrics_json else load_structured(metrics_file)  # type: ignore[arg-type]
    if not isinstance(value, dict):
        raise ValueError("Epoch metrics must be a JSON/YAML object.")
    return value


def last_jsonl_record(path: str | Path) -> dict[str, Any]:
    lines = [line for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"JSONL file has no records: {path}")
    value = json.loads(lines[-1])
    if not isinstance(value, dict):
        raise ValueError(f"Last JSONL record is not an object: {path}")
    return value


def record_epoch(
    *,
    run_dir: str | Path,
    epoch: int,
    metrics: dict[str, Any],
    kd_health: dict[str, Any] | None = None,
    elapsed_seconds: float | None = None,
    gpu_memory_gib: float | None = None,
) -> dict[str, Any]:
    manifest = ensure_open_run(run_dir)
    ledger = evidence_dir(run_dir) / EPOCH_LEDGER_NAME
    previous_events = read_chain(ledger)
    prior_epochs = [int(event["event"]["payload"]["epoch"]) for event in previous_events]
    if int(epoch) in prior_epochs:
        raise ValueError(f"Epoch {epoch} is already recorded for run {manifest['record']['run_id']}")
    if prior_epochs and int(epoch) < max(prior_epochs):
        raise ValueError(f"Epoch numbers must increase monotonically; latest={max(prior_epochs)}, got={epoch}")
    payload = {
        "kind": "epoch_metrics",
        "utc": utc_now(),
        "run_id": manifest["record"]["run_id"],
        "epoch": int(epoch),
        "metrics": metrics,
        "kd_health": kd_health,
        "elapsed_seconds": elapsed_seconds,
        "gpu_memory_gib": gpu_memory_gib,
        "disk": disk_state(run_dir),
    }
    return append_chain(ledger, payload)


def record_resume(
    *, run_dir: str | Path, checkpoint: str | Path, command: str, notes: str | None = None
) -> dict[str, Any]:
    manifest = ensure_open_run(run_dir)
    payload = {
        "kind": "resume",
        "utc": utc_now(),
        "run_id": manifest["record"]["run_id"],
        "checkpoint": file_record(checkpoint),
        "command": command,
        "environment": environment_state(),
        "disk": disk_state(run_dir),
        "notes": notes,
    }
    event = append_chain(evidence_dir(run_dir) / RESUME_LEDGER_NAME, payload)
    append_registry(registry_from_manifest(manifest), {
        "action": "resume",
        "utc": payload["utc"],
        "run_id": manifest["record"]["run_id"],
        "checkpoint_sha256": payload["checkpoint"]["sha256"],
        "resume_event_sha256": event["event_sha256"],
    })
    return event


def parse_results_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    parsed: list[dict[str, Any]] = []
    for row in rows:
        values: dict[str, Any] = {}
        for raw_key, raw_value in row.items():
            key = (raw_key or "").strip()
            value = (raw_value or "").strip()
            if not key:
                continue
            try:
                values[key] = float(value)
            except ValueError:
                values[key] = value
        parsed.append(values)
    return parsed


def choose_model(payload: Any, model_key: str | None) -> tuple[str | None, Any]:
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), dict):
        return model_key, payload
    models = payload["models"]
    if model_key in models:
        return model_key, models[model_key]
    if len(models) == 1:
        only = next(iter(models))
        return only, models[only]
    raise ValueError(f"Report contains models {sorted(models)}; pass --model-key.")


def normalize_native(payload: Any, model_key: str | None) -> dict[str, Any]:
    selected_key, selected = choose_model(payload, model_key)
    if not isinstance(selected, dict):
        raise ValueError("Native report model entry must be an object.")
    metrics = selected.get("metrics", selected)
    aliases = {
        "precision": ("metrics/precision(B)", "precision", "P"),
        "recall": ("metrics/recall(B)", "recall", "R"),
        "map50": ("metrics/mAP50(B)", "mAP50", "map50"),
        "map50_95": ("metrics/mAP50-95(B)", "mAP50-95", "map50_95"),
        "fitness": ("fitness",),
    }
    normalized: dict[str, Any] = {"model_key": selected_key}
    for name, keys in aliases.items():
        normalized[name] = next((float(metrics[key]) for key in keys if key in metrics), None)
    if "per_class_map50_95" in selected:
        normalized["per_class_map50_95"] = selected["per_class_map50_95"]
    return normalized


def normalize_competition(payload: Any, operating_point: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Competition report must be an object.")
    if "operating_points" in payload:
        if operating_point not in payload["operating_points"]:
            raise ValueError(f"Operating point {operating_point!r} not present in competition report.")
        point = payload["operating_points"][operating_point]
    else:
        point = payload
    return {
        "split": payload.get("split"),
        "operating_point": operating_point,
        "confidence": point.get("confidence", payload.get("confidence")),
        "class_aware_matching": payload.get("class_aware_matching"),
        "overall": point.get("overall"),
        "per_group": point.get("per_group", point.get("groups")),
    }


def normalize_diagnostics(payload: Any, model_key: str | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Diagnostics report must be an object.")
    summaries = payload.get("summaries")
    if isinstance(summaries, dict):
        if model_key in summaries:
            selected_key = model_key
        elif len(summaries) == 1:
            selected_key = next(iter(summaries))
        else:
            raise ValueError(f"Diagnostics contains models {sorted(summaries)}; pass --model-key.")
        selected = summaries[selected_key]
    else:
        selected_key, selected = model_key, payload
    if not isinstance(selected, dict):
        raise ValueError("Diagnostics model entry must be an object.")
    return {
        "model_key": selected_key,
        "overall": selected.get("overall"),
        "per_group": selected.get("per_group"),
        "per_size": selected.get("per_size"),
        "attributes": selected.get("attributes"),
        "false_positives": selected.get("false_positives"),
    }


def normalize_timing(payload: Any) -> dict[str, Any]:
    """Validate the competition timing boundary: image I/O is excluded.

    Timing starts only after all image bytes have been read and ends after the
    result object/file has been produced.  Preprocess, inference, NMS, merging,
    and result serialization therefore remain inside the measured interval.
    """
    if not isinstance(payload, dict):
        raise ValueError("Timing report must be an object.")
    if payload.get("excludes_image_read") is not True:
        raise ValueError("Timing report must explicitly set excludes_image_read=true.")
    required = ("image_count", "total_seconds")
    missing = [key for key in required if not isinstance(payload.get(key), (int, float))]
    if missing:
        raise ValueError(f"Timing report is missing numeric fields: {missing}")
    image_count = int(payload["image_count"])
    total_seconds = float(payload["total_seconds"])
    if image_count <= 0 or total_seconds < 0:
        raise ValueError("Timing image_count must be positive and total_seconds non-negative.")
    return {
        "protocol": payload.get("protocol", "competition_no_image_io_v1"),
        "excludes_image_read": True,
        "interval_start": payload.get("interval_start", "after_all_image_reads_complete"),
        "interval_end": payload.get("interval_end", "after_result_output_complete"),
        "included_stages": payload.get("included_stages", ["preprocess", "inference", "postprocess", "result_output"]),
        "image_count": image_count,
        "total_seconds": total_seconds,
        "mean_ms_per_image": float(payload.get("mean_ms_per_image", total_seconds * 1000.0 / image_count)),
        "p50_ms_per_image": payload.get("p50_ms_per_image"),
        "p95_ms_per_image": payload.get("p95_ms_per_image"),
        "warmup_iterations": payload.get("warmup_iterations"),
        "repetitions": payload.get("repetitions"),
        "batch": payload.get("batch"),
        "image_size": payload.get("image_size"),
        "device": payload.get("device"),
    }


def summarize_kd_health(path: Path) -> dict[str, Any]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError(f"KD health JSONL has no records: {path}")
    numeric = lambda key: [float(record[key]) for record in records if isinstance(record.get(key), (int, float))]
    summary: dict[str, Any] = {
        "records": len(records),
        "experiments": sorted({str(record.get("experiment")) for record in records if record.get("experiment") is not None}),
        "epochs": sorted({int(record["epoch"]) for record in records if isinstance(record.get("epoch"), (int, float))}),
        "all_optimizer_verified": all(bool(record.get("optimizer_verified")) for record in records),
        "total_cache_misses": int(sum(numeric("cache_misses"))),
        "feature_nonzero_records": sum(int(record.get("feature_nonzero_batches", 0)) > 0 for record in records),
        "cls_nonzero_records": sum(int(record.get("cls_nonzero_batches", 0)) > 0 for record in records),
        "feature_gradient_records": sum(int(record.get("feature_grad_events", 0)) > 0 for record in records),
        "cls_gradient_records": sum(int(record.get("roi_grad_events", 0)) > 0 for record in records),
        "global_nonzero_records": sum(int(record.get("global_nonzero_batches", 0)) > 0 for record in records),
        "prototype_nonzero_records": sum(int(record.get("prototype_nonzero_batches", 0)) > 0 for record in records),
        "global_gradient_records": sum(int(record.get("global_grad_events", 0)) > 0 for record in records),
        "prototype_gradient_records": sum(int(record.get("prototype_grad_events", 0)) > 0 for record in records),
        "prototype_roi_records": sum(int(record.get("prototype_valid_rois", 0)) > 0 for record in records),
        "gp_routing_records": sum(
            int(record.get("global_routed_objects", 0)) > 0
            and int(record.get("prototype_routed_objects", 0)) > 0
            and int(record.get("route_overlap_objects", -1)) == 0
            for record in records
        ),
    }
    for key in (
        "kd_mean",
        "feature_raw_mean",
        "cls_raw_mean",
        "det_feature_grad_cosine_mean",
        "det_cls_grad_cosine_mean",
        "feature_cls_grad_cosine_mean",
        "feature_calibrated_shared_grad_ratio",
        "cls_calibrated_shared_grad_ratio",
    ):
        values = numeric(key)
        if values:
            summary[key] = {"min": min(values), "mean": sum(values) / len(values), "max": max(values)}
    experiment_names = {name.lower() for name in summary["experiments"]}
    feature_required = any(name in {"f", "fk"} or "feature" in name for name in experiment_names)
    cls_required = any(name in {"k", "fk"} or "logit" in name for name in experiment_names)
    global_required = any(name in {"g", "gp", "gpl", "gplb"} for name in experiment_names)
    prototype_required = any(name in {"p", "gp", "gpl", "gplb"} or "prototype" in name for name in experiment_names)
    gp_required = any(name in {"gp", "gpl", "gplb"} for name in experiment_names)
    finite = all(
        math.isfinite(float(value))
        for record in records
        for value in record.values()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    )
    feature_ok = not feature_required or (
        summary["feature_nonzero_records"] == len(records) and summary["feature_gradient_records"] == len(records)
    )
    cls_ok = not cls_required or (
        summary["cls_nonzero_records"] == len(records) and summary["cls_gradient_records"] == len(records)
    )
    global_ok = not global_required or (
        summary["global_nonzero_records"] == len(records)
        and summary["global_gradient_records"] == len(records)
    )
    prototype_ok = not prototype_required or (
        summary["prototype_nonzero_records"] == len(records)
        and summary["prototype_gradient_records"] == len(records)
        and summary["prototype_roi_records"] == len(records)
    )
    gp_ok = not gp_required or summary["gp_routing_records"] == len(records)
    explicit_failures = [
        str(record.get("health_failure"))
        for record in records
        if record.get("health_failure") not in (None, "", False)
    ]
    summary.update({
        "finite_numeric_values": finite,
        "feature_branch_required": feature_required,
        "feature_branch_pass": feature_ok,
        "cls_branch_required": cls_required,
        "cls_branch_pass": cls_ok,
        "global_branch_required": global_required,
        "global_branch_pass": global_ok,
        "prototype_branch_required": prototype_required,
        "prototype_branch_pass": prototype_ok,
        "gp_routing_required": gp_required,
        "gp_routing_pass": gp_ok,
        "explicit_health_failures": explicit_failures,
    })
    summary["health_pass"] = bool(
        summary["all_optimizer_verified"]
        and summary["total_cache_misses"] == 0
        and finite
        and feature_ok
        and cls_ok
        and global_ok
        and prototype_ok
        and gp_ok
        and not explicit_failures
    )
    return summary


def normalize_deploy_parity(payload: Any) -> dict[str, Any]:
    """Validate the detector-tensor identity proof for a stripped KD checkpoint."""
    if not isinstance(payload, dict):
        raise ValueError("Deployment parity report must be an object.")
    required = ("full_detector_sha256", "deploy_detector_sha256", "parity")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Deployment parity report is missing fields: {missing}")
    full = str(payload["full_detector_sha256"])
    deploy = str(payload["deploy_detector_sha256"])
    passed = payload.get("parity") is True and full == deploy
    if not passed:
        raise ValueError("Deployment checkpoint detector tensors differ from the full KD checkpoint.")
    return {
        "parity": True,
        "full_detector_sha256": full,
        "deploy_detector_sha256": deploy,
    }


def validate_formal_evidence(parsed: dict[str, Any], *, require_kd_health: bool) -> None:
    native = parsed.get("native", {})
    if native.get("map50_95") is None:
        raise ValueError("Paper-ready native evidence has no mAP50-95.")
    competition = parsed.get("competition", {})
    if competition.get("class_aware_matching") is not True:
        raise ValueError("Paper-ready competition evidence must use class_aware_matching=true.")
    groups = competition.get("per_group") or {}
    for group in ("ship", "aircraft", "vehicle"):
        if group not in groups:
            raise ValueError(f"Paper-ready competition evidence is missing group {group!r}.")
        for metric in ("recall", "false_alarm_rate"):
            if groups[group].get(metric) is None:
                raise ValueError(f"Paper-ready competition evidence is missing {group}.{metric}.")
    diagnostics = parsed.get("diagnostics", {})
    for key in ("per_group", "per_size", "attributes"):
        if not isinstance(diagnostics.get(key), dict):
            raise ValueError(f"Paper-ready diagnostics is missing {key}.")
    for size in ("small", "medium", "large"):
        if size not in diagnostics["per_size"]:
            raise ValueError(f"Paper-ready diagnostics is missing size bucket {size!r}.")
    for attribute in ("crowded", "edge"):
        if attribute not in diagnostics["attributes"]:
            raise ValueError(f"Paper-ready diagnostics is missing attribute {attribute!r}.")
    if parsed.get("timing", {}).get("excludes_image_read") is not True:
        raise ValueError("Paper-ready timing must exclude image reading.")
    if require_kd_health and not parsed.get("kd_health", {}).get("health_pass"):
        raise ValueError("Paper-ready KD evidence did not pass the health gate.")


def snapshot_file(source: str | Path, snapshots: Path, role: str) -> dict[str, Any]:
    original = Path(source).resolve()
    if not original.is_file():
        raise FileNotFoundError(original)
    safe_role = "".join(character if character.isalnum() or character in "-_" else "_" for character in role)
    destination = snapshots / f"{safe_role}{original.suffix.lower()}"
    snapshots.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Evidence snapshot role already exists: {destination}")
    shutil.copy2(original, destination)
    before, after = file_record(original), file_record(destination)
    if before["sha256"] != after["sha256"]:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"Evidence snapshot hash mismatch: {original} -> {destination}")
    return {"role": role, "source": before, "snapshot": after}


def complete_run(
    *,
    run_dir: str | Path,
    status: str,
    best_checkpoint: str | Path | None = None,
    last_checkpoint: str | Path | None = None,
    deploy_checkpoint: str | Path | None = None,
    results_csv: str | Path | None = None,
    kd_health: str | Path | None = None,
    native: str | Path | None = None,
    competition: str | Path | None = None,
    diagnostics: str | Path | None = None,
    timing: str | Path | None = None,
    parity: str | Path | None = None,
    model_key: str | None = None,
    operating_point: str = "best_f1",
    elapsed_seconds: float | None = None,
    failure: str | None = None,
    notes: str | None = None,
    paper_ready: bool = False,
    require_kd_health: bool = False,
) -> dict[str, Any]:
    if status not in {"completed", "failed"}:
        raise ValueError("status must be 'completed' or 'failed'.")
    if status == "completed" and (best_checkpoint is None or last_checkpoint is None):
        raise ValueError("Completed runs require --best-checkpoint and --last-checkpoint.")
    if paper_ready:
        required = {
            "results_csv": results_csv,
            "native": native,
            "competition": competition,
            "diagnostics": diagnostics,
            "timing": timing,
        }
        missing = [name for name, value in required.items() if value is None]
        if require_kd_health and kd_health is None:
            missing.append("kd_health")
        if deploy_checkpoint is not None and parity is None:
            missing.append("parity")
        if missing:
            raise ValueError(f"Paper-ready completion is missing evidence: {missing}")
    manifest = ensure_open_run(run_dir)
    if paper_ready and manifest["record"]["repository"].get("dirty"):
        raise ValueError("Paper-ready completion is forbidden for a run initialized from a dirty Git worktree.")
    target = completion_path(run_dir)
    snapshots = evidence_dir(run_dir) / "snapshots"
    parsed: dict[str, Any] = {}
    if results_csv is not None:
        parsed["epoch_metrics"] = parse_results_csv(Path(results_csv))
    epoch_ledger = evidence_dir(run_dir) / EPOCH_LEDGER_NAME
    epoch_events = read_chain(epoch_ledger)
    parsed["recorded_epoch_events"] = [event["event"]["payload"] for event in epoch_events]
    if kd_health is not None:
        parsed["kd_health"] = summarize_kd_health(Path(kd_health))
    if native is not None:
        parsed["native"] = normalize_native(load_structured(native), model_key)
    if competition is not None:
        parsed["competition"] = normalize_competition(load_structured(competition), operating_point)
    if diagnostics is not None:
        parsed["diagnostics"] = normalize_diagnostics(load_structured(diagnostics), model_key)
    if timing is not None:
        parsed["timing"] = normalize_timing(load_structured(timing))
    if parity is not None:
        parsed["deploy_parity"] = normalize_deploy_parity(load_structured(parity))
    if paper_ready:
        validate_formal_evidence(parsed, require_kd_health=require_kd_health)
    artifacts: dict[str, Any] = {}
    for role, source in (
        ("results_csv", results_csv),
        ("kd_health", kd_health),
        ("native", native),
        ("competition", competition),
        ("diagnostics", diagnostics),
        ("timing", timing),
        ("parity", parity),
    ):
        if source is not None:
            artifacts[role] = snapshot_file(source, snapshots, role)
    checkpoint_records = {
        "best": file_record(best_checkpoint) if best_checkpoint is not None else None,
        "last": file_record(last_checkpoint) if last_checkpoint is not None else None,
        "deploy": file_record(deploy_checkpoint) if deploy_checkpoint is not None else None,
    }
    created = datetime.fromisoformat(manifest["record"]["created_utc"].replace("Z", "+00:00"))
    completed = datetime.now(timezone.utc)
    wall_seconds = (completed - created).total_seconds()
    completeness = {
        "epoch_metrics": bool(parsed.get("epoch_metrics") or parsed.get("recorded_epoch_events")),
        "native": "native" in parsed,
        "competition": "competition" in parsed,
        "diagnostics_group_size_crowded_edge": "diagnostics" in parsed,
        "competition_timing_excluding_image_io": "timing" in parsed,
        "kd_health": "kd_health" in parsed,
        "deploy_parity": "deploy_parity" in parsed,
    }
    record = {
        "kind": "run_completion",
        "schema_version": SCHEMA_VERSION,
        "run_id": manifest["record"]["run_id"],
        "manifest_sha256": manifest["record_sha256"],
        "status": status,
        "completed_utc": completed.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "elapsed_seconds": float(elapsed_seconds) if elapsed_seconds is not None else wall_seconds,
        "observed_wall_seconds": wall_seconds,
        "checkpoints": checkpoint_records,
        "artifacts": artifacts,
        "evidence": parsed,
        "evidence_completeness": completeness,
        "paper_ready": bool(
            status == "completed"
            and all(completeness[key] for key in (
                "epoch_metrics", "native", "competition", "diagnostics_group_size_crowded_edge",
                "competition_timing_excluding_image_io",
            ))
            and (not require_kd_health or completeness["kd_health"])
            and (deploy_checkpoint is None or completeness["deploy_parity"])
        ),
        "resume_events": [event["event"]["payload"] for event in read_chain(evidence_dir(run_dir) / RESUME_LEDGER_NAME)],
        "resource_end": {"disk": disk_state(run_dir)},
        "failure": failure,
        "notes": notes,
    }
    written = write_once_envelope(target, record)
    summary = registry_summary(manifest["record"], record)
    append_registry(registry_from_manifest(manifest), {
        "action": "complete",
        "utc": record["completed_utc"],
        "run_id": record["run_id"],
        "completion_path": str(target),
        "completion_sha256": written["record_sha256"],
        "summary": summary,
    })
    return written


def nested_get(value: Any, *keys: str) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def registry_summary(manifest: dict[str, Any], completion: dict[str, Any] | None) -> dict[str, Any]:
    evidence = completion.get("evidence", {}) if completion else {}
    native = evidence.get("native", {}) or {}
    competition = evidence.get("competition", {}) or {}
    groups = competition.get("per_group", {}) or {}
    diagnostics = evidence.get("diagnostics", {}) or {}
    timing = evidence.get("timing", {}) or {}
    checkpoints = completion.get("checkpoints", {}) if completion else {}
    row: dict[str, Any] = {
        "run_id": manifest["run_id"],
        "experiment": manifest["experiment"],
        "status": completion.get("status") if completion else "running",
        "dataset_fingerprint": manifest["dataset"]["fingerprint"],
        "git_commit": manifest["repository"]["commit"],
        "git_dirty": manifest["repository"]["dirty"],
        "config_sha256": manifest["config"]["resolved_sha256"],
        "seed": manifest["seed"],
        "initial_checkpoint_sha256": manifest["initial_checkpoint"]["sha256"],
        "best_checkpoint_sha256": nested_get(checkpoints, "best", "sha256"),
        "deploy_checkpoint_sha256": nested_get(checkpoints, "deploy", "sha256"),
        "elapsed_seconds": completion.get("elapsed_seconds") if completion else None,
        "native_precision": native.get("precision"),
        "native_recall": native.get("recall"),
        "native_map50": native.get("map50"),
        "native_map50_95": native.get("map50_95"),
        "competition_confidence": competition.get("confidence"),
        "overall_recall": nested_get(competition, "overall", "recall"),
        "overall_fdr": nested_get(competition, "overall", "false_alarm_rate"),
        "overall_f1": nested_get(competition, "overall", "f1"),
        "kd_health_pass": nested_get(evidence, "kd_health", "health_pass"),
        "paper_ready": completion.get("paper_ready") if completion else False,
        "inference_total_seconds_excluding_io": timing.get("total_seconds"),
        "inference_mean_ms_per_image": timing.get("mean_ms_per_image"),
        "diagnostics": diagnostics,
    }
    for group in ("ship", "aircraft", "vehicle"):
        row[f"{group}_recall"] = nested_get(groups, group, "recall")
        row[f"{group}_fdr"] = nested_get(groups, group, "false_alarm_rate")
        row[f"{group}_f1"] = nested_get(groups, group, "f1")
    row["competition_seven_ranking_ready"] = bool(
        timing.get("excludes_image_read") is True
        and timing.get("total_seconds") is not None
        and all(row.get(f"{group}_{metric}") is not None for group in ("ship", "aircraft", "vehicle") for metric in ("recall", "fdr"))
    )
    return row


def summarize_registry(registry: str | Path, out_dir: str | Path) -> dict[str, Any]:
    registry_path = Path(registry).resolve()
    events = read_chain(registry_path)
    runs: dict[str, dict[str, Any]] = {}
    for event in events:
        payload = event["event"]["payload"]
        run_id = payload.get("run_id")
        if not run_id:
            continue
        entry = runs.setdefault(run_id, {})
        if payload.get("action") == "init":
            manifest = read_envelope(payload["manifest_path"], expected_kind="run_manifest")
            if manifest["record_sha256"] != payload["manifest_sha256"]:
                raise RuntimeError(f"Registry/manifest hash mismatch for {run_id}")
            entry["manifest"] = manifest["record"]
        elif payload.get("action") == "complete":
            completion_file = Path(payload["completion_path"])
            if completion_file.exists():
                completed = read_envelope(completion_file, expected_kind="run_completion")
                if completed["record_sha256"] != payload["completion_sha256"]:
                    raise RuntimeError(f"Registry/completion hash mismatch for {run_id}")
                entry["completion"] = completed["record"]
            entry["registry_summary"] = payload.get("summary")
    rows: list[dict[str, Any]] = []
    for run_id, entry in sorted(runs.items()):
        manifest = entry.get("manifest")
        if manifest:
            rows.append(registry_summary(manifest, entry.get("completion")))
        elif entry.get("registry_summary"):
            rows.append(entry["registry_summary"])
    output = Path(out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": "experiment_registry_summary",
        "generated_utc": utc_now(),
        "registry_path": str(registry_path),
        "registry_sha256": file_sha256(registry_path),
        "event_count": len(events),
        "runs": rows,
    }
    atomic_json(output / "experiment_summary.json", report)
    flat_rows = [{key: value for key, value in row.items() if key != "diagnostics"} for row in rows]
    columns = sorted({key for row in flat_rows for key in row})
    with (output / "experiment_summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(flat_rows)
    markdown = [
        "# Experiment evidence summary",
        "",
        f"Generated: `{report['generated_utc']}`  ",
        f"Registry SHA-256: `{report['registry_sha256']}`",
        "",
        "| Experiment | Seed | Status | mAP50-95 | Recall | FDR | Ship F1 | Aircraft F1 | Vehicle F1 | Inference ms/image* | 7-rank ready | KD health |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    def show(value: Any) -> str:
        return "" if value is None else (f"{value:.4f}" if isinstance(value, float) else str(value))
    for row in rows:
        markdown.append(
            "| " + " | ".join(show(row.get(key)) for key in (
                "experiment", "seed", "status", "native_map50_95", "overall_recall", "overall_fdr",
                "ship_f1", "aircraft_f1", "vehicle_f1", "inference_mean_ms_per_image",
                "competition_seven_ranking_ready", "kd_health_pass",
            )) + " |"
        )
    markdown.extend(["", "\\* Competition timing excludes image reading; preprocess through result output remains included."])
    (output / "experiment_summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Immutable run manifests, metric evidence, and paper-ready experiment summaries.")
    subparsers = parser.add_subparsers(dest="action", required=True)

    init = subparsers.add_parser("init", help="Create a write-once run manifest before training.")
    init.add_argument("--run-dir", required=True)
    init.add_argument("--experiment", required=True)
    init.add_argument("--dataset-report", required=True)
    init.add_argument("--config", required=True)
    init.add_argument("--seed", type=int, required=True)
    init.add_argument("--initial-checkpoint", required=True)
    init_command = init.add_mutually_exclusive_group(required=True)
    init_command.add_argument("--command")
    init_command.add_argument("--command-file")
    init.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    init.add_argument("--repo")
    init.add_argument("--run-id")
    init.add_argument("--resume-checkpoint")
    init.add_argument("--parent-manifest")
    init.add_argument("--notes")

    epoch = subparsers.add_parser("record-epoch", help="Append one hash-chained epoch metric event.")
    epoch.add_argument("--run-dir", required=True)
    epoch.add_argument("--epoch", type=int, required=True)
    epoch_metrics = epoch.add_mutually_exclusive_group(required=True)
    epoch_metrics.add_argument("--metrics-json")
    epoch_metrics.add_argument("--metrics-file")
    epoch.add_argument("--kd-health-file", help="JSON or JSONL; for JSONL the last record is used.")
    epoch.add_argument("--elapsed-seconds", type=float)
    epoch.add_argument("--gpu-memory-gib", type=float)

    resume = subparsers.add_parser("resume", help="Record an exact-resume checkpoint and command before continuing.")
    resume.add_argument("--run-dir", required=True)
    resume.add_argument("--checkpoint", required=True)
    resume_command = resume.add_mutually_exclusive_group(required=True)
    resume_command.add_argument("--command")
    resume_command.add_argument("--command-file")
    resume.add_argument("--notes")

    complete = subparsers.add_parser("complete", help="Snapshot final evidence and write a terminal completion record.")
    complete.add_argument("--run-dir", required=True)
    complete.add_argument("--status", choices=("completed", "failed"), default="completed")
    complete.add_argument("--best-checkpoint")
    complete.add_argument("--last-checkpoint")
    complete.add_argument("--deploy-checkpoint")
    complete.add_argument("--results-csv")
    complete.add_argument("--kd-health")
    complete.add_argument("--native")
    complete.add_argument("--competition")
    complete.add_argument("--diagnostics")
    complete.add_argument("--timing", help="JSON timing report with excludes_image_read=true.")
    complete.add_argument("--parity", help="JSON detector-tensor parity report for a stripped KD deploy checkpoint.")
    complete.add_argument("--model-key")
    complete.add_argument("--operating-point", default="best_f1")
    complete.add_argument("--elapsed-seconds", type=float)
    complete.add_argument("--failure")
    complete.add_argument("--notes")
    complete.add_argument("--paper-ready", action="store_true", help="Require all formal paper/competition evidence before finalizing.")
    complete.add_argument("--require-kd-health", action="store_true", help="With --paper-ready, also require a KD health JSONL.")

    summary = subparsers.add_parser("summarize", help="Verify the registry and create JSON/CSV/Markdown result tables.")
    summary.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    summary.add_argument("--out-dir", default="reports/experiment_ledger")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.action == "init":
        command, source = command_text(args.command, args.command_file)
        result = initialize_run(
            run_dir=args.run_dir,
            experiment=args.experiment,
            dataset_report=args.dataset_report,
            config=args.config,
            seed=args.seed,
            initial_checkpoint=args.initial_checkpoint,
            command=command,
            command_source=source,
            registry=args.registry,
            repo=args.repo,
            run_id=args.run_id,
            resume_checkpoint=args.resume_checkpoint,
            parent_manifest=args.parent_manifest,
            notes=args.notes,
        )
        print(f"initialized run_id={result['record']['run_id']} manifest_sha256={result['record_sha256']}")
    elif args.action == "record-epoch":
        metrics = read_metrics_argument(args.metrics_json, args.metrics_file)
        kd_health = None
        if args.kd_health_file:
            source = Path(args.kd_health_file)
            kd_health = last_jsonl_record(source) if source.suffix.lower() == ".jsonl" else load_structured(source)
        event = record_epoch(
            run_dir=args.run_dir,
            epoch=args.epoch,
            metrics=metrics,
            kd_health=kd_health,
            elapsed_seconds=args.elapsed_seconds,
            gpu_memory_gib=args.gpu_memory_gib,
        )
        print(f"recorded epoch={args.epoch} event_sha256={event['event_sha256']}")
    elif args.action == "resume":
        command, _ = command_text(args.command, args.command_file)
        event = record_resume(run_dir=args.run_dir, checkpoint=args.checkpoint, command=command, notes=args.notes)
        print(f"recorded resume event_sha256={event['event_sha256']}")
    elif args.action == "complete":
        result = complete_run(
            run_dir=args.run_dir,
            status=args.status,
            best_checkpoint=args.best_checkpoint,
            last_checkpoint=args.last_checkpoint,
            deploy_checkpoint=args.deploy_checkpoint,
            results_csv=args.results_csv,
            kd_health=args.kd_health,
            native=args.native,
            competition=args.competition,
            diagnostics=args.diagnostics,
            timing=args.timing,
            parity=args.parity,
            model_key=args.model_key,
            operating_point=args.operating_point,
            elapsed_seconds=args.elapsed_seconds,
            failure=args.failure,
            notes=args.notes,
            paper_ready=args.paper_ready,
            require_kd_health=args.require_kd_health,
        )
        print(f"finalized run_id={result['record']['run_id']} completion_sha256={result['record_sha256']}")
    elif args.action == "summarize":
        report = summarize_registry(args.registry, args.out_dir)
        print(f"summarized runs={len(report['runs'])} events={report['event_count']} to {Path(args.out_dir).resolve()}")


if __name__ == "__main__":
    main()
