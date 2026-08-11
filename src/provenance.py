from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


HEX64 = re.compile(r"^[0-9a-f]{64}$")
SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")
PROVENANCE_FORMAT = 1
CACHE_FORMAT = 3


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hex64(value: Any, field: str) -> str:
    normalized = str(value or "").strip().lower()
    if not HEX64.fullmatch(normalized):
        raise RuntimeError(f"{field} must be a lowercase 64-character SHA-256, got {value!r}.")
    return normalized


def _safe_dataset_id(value: Any) -> str:
    dataset_id = str(value or "").strip()
    if not dataset_id or not SAFE_ID.fullmatch(dataset_id):
        raise RuntimeError(f"Unsafe or missing dataset.id: {value!r}")
    return dataset_id


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a YAML mapping: {path}")
    return value


def dataset_root_from_config(cfg: dict[str, Any]) -> Path:
    data_yaml = Path(cfg["paths"]["data_yaml"]).resolve()
    data = _load_yaml(data_yaml)
    configured_root = data.get("path")
    # Ultralytics accepts a portable dataset.yaml without ``path``; in that
    # case split paths are resolved from the YAML directory itself.
    root = Path(str(configured_root)) if configured_root not in {None, ""} else data_yaml.parent
    if not root.is_absolute():
        root = data_yaml.parent / root
    return root.resolve()


def _read_split_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {
        "split",
        "source",
        "scene_id",
        "cluster_id",
        "image",
        "label",
        "image_sha256",
        "label_sha256",
    }
    if not rows:
        raise RuntimeError(f"Dataset split manifest is empty: {path}")
    missing = required.difference(rows[0])
    if missing:
        raise RuntimeError(f"Dataset split manifest lacks {sorted(missing)}: {path}")
    normalized: list[dict[str, str]] = []
    for row in rows:
        item = {key: str(value or "").strip() for key, value in row.items()}
        item["image_sha256"] = _hex64(item["image_sha256"], "image_sha256")
        item["label_sha256"] = _hex64(item["label_sha256"], "label_sha256")
        if item["split"] not in {"train", "val", "test"}:
            raise RuntimeError(f"Invalid split {item['split']!r} in {path}")
        normalized.append(item)
    normalized.sort(key=lambda row: (row["split"], row["image"].casefold()))
    return normalized


def split_fingerprint_from_rows(rows: Iterable[dict[str, str]]) -> str:
    return canonical_sha256(
        [
            {
                "split": row["split"],
                "source": row["source"],
                "scene_id": row["scene_id"],
                "cluster_id": row["cluster_id"],
                "image": row["image"],
            }
            for row in rows
        ]
    )


def inventory_fingerprint_from_rows(
    rows: Iterable[dict[str, str]], split: str | None = None
) -> str:
    inventory = [
        {
            "split": row["split"],
            "image": row["image"],
            "label": row["label"],
            "image_sha256": row["image_sha256"],
            "label_sha256": row["label_sha256"],
        }
        for row in rows
        if split is None or row["split"] == split
    ]
    if not inventory:
        raise RuntimeError(f"Dataset inventory contains no rows for split={split!r}.")
    return canonical_sha256(inventory)


def _fingerprint_report_path(cfg: dict[str, Any], root: Path) -> Path:
    dataset_cfg = cfg.get("dataset", {})
    configured = dataset_cfg.get("fingerprint_file") or cfg.get("paths", {}).get(
        "dataset_fingerprint"
    )
    path = Path(configured) if configured else root / "dataset_fingerprint.json"
    if not path.is_absolute():
        path = Path(cfg["paths"]["project_root"]) / path
    return path.resolve()


def resolve_dataset_identity(cfg: dict[str, Any]) -> dict[str, Any]:
    """Resolve and validate the immutable dataset identity used by teacher/KD.

    Legacy V2 configurations without either fingerprint remain usable.  The
    moment a config contains a fingerprint, or its dataset package contains the
    V3 fingerprint report, both configured values and the on-disk manifest are
    mandatory and are checked exactly.
    """

    dataset_cfg = cfg.get("dataset", {})
    configured_dataset = dataset_cfg.get("dataset_fingerprint")
    configured_split = dataset_cfg.get("split_fingerprint")
    if bool(configured_dataset) != bool(configured_split):
        raise RuntimeError(
            "dataset.dataset_fingerprint and dataset.split_fingerprint must be supplied together."
        )
    root = dataset_root_from_config(cfg)
    report_path = _fingerprint_report_path(cfg, root)
    dataset_id = _safe_dataset_id(dataset_cfg.get("id", "legacy_v2"))
    preview: dict[str, Any] | None = None
    if report_path.is_file():
        try:
            preview = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            if configured_dataset or dataset_id.startswith("scene811_v3"):
                raise RuntimeError(
                    f"Invalid dataset fingerprint report {report_path}: {error}"
                ) from error
    report_id = str((preview or {}).get("dataset_id", ""))
    strict = bool(configured_dataset) or dataset_id.startswith("scene811_v3") or report_id.startswith("scene811_v3")
    if not strict:
        return {
            "format": PROVENANCE_FORMAT,
            "strict": False,
            "dataset_id": dataset_id,
            "dataset_root": str(root),
            "dataset_fingerprint": None,
            "split_fingerprint": None,
            "inventory_fingerprint": None,
            "train_inventory_fingerprint": None,
            "manifest_sha256": None,
            "fingerprint_report_sha256": None,
        }
    if not report_path.is_file():
        raise FileNotFoundError(
            f"Fingerprint-enforced dataset is missing {report_path}. Refuse to use an unversioned dataset."
        )
    report = preview
    if not isinstance(report, dict):
        raise RuntimeError(f"Invalid dataset fingerprint report mapping: {report_path}")
    report_dataset = _hex64(report.get("dataset_fingerprint"), "report.dataset_fingerprint")
    report_split = _hex64(report.get("split_fingerprint"), "report.split_fingerprint")
    report_id = _safe_dataset_id(report.get("dataset_id"))
    if dataset_cfg.get("id") and dataset_id != report_id:
        raise RuntimeError(
            f"Configured dataset.id={dataset_id!r} differs from fingerprint report {report_id!r}."
        )
    dataset_id = report_id
    if configured_dataset and _hex64(configured_dataset, "dataset.dataset_fingerprint") != report_dataset:
        raise RuntimeError("Configured dataset_fingerprint differs from the dataset package report.")
    if configured_split and _hex64(configured_split, "dataset.split_fingerprint") != report_split:
        raise RuntimeError("Configured split_fingerprint differs from the dataset package report.")
    manifest_path = root / "split_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Fingerprint-enforced dataset is missing its split manifest: {manifest_path}"
        )
    rows = _read_split_manifest(manifest_path)
    computed_split = split_fingerprint_from_rows(rows)
    if computed_split != report_split:
        raise RuntimeError(
            "split_manifest.csv does not reproduce split_fingerprint; the dataset package was modified."
        )
    return {
        "format": PROVENANCE_FORMAT,
        "strict": True,
        "dataset_id": dataset_id,
        "dataset_root": str(root),
        "dataset_fingerprint": report_dataset,
        "split_fingerprint": report_split,
        "inventory_fingerprint": inventory_fingerprint_from_rows(rows),
        "train_inventory_fingerprint": inventory_fingerprint_from_rows(rows, "train"),
        "split_inventory_fingerprints": {
            split: inventory_fingerprint_from_rows(rows, split)
            for split in ("train", "val", "test")
            if any(row["split"] == split for row in rows)
        },
        "manifest_sha256": file_sha256(manifest_path),
        "fingerprint_report_sha256": file_sha256(report_path),
        "fingerprint_report": str(report_path),
        "split_manifest": str(manifest_path),
        "image_count": len(rows),
    }


def artifact_namespace(identity: dict[str, Any]) -> str | None:
    if not identity.get("strict"):
        return None
    return f"{identity['dataset_id']}__{identity['dataset_fingerprint'][:12]}"


def portable_dataset_identity(identity: dict[str, Any]) -> dict[str, Any]:
    """Content identity only; excludes machine-specific absolute paths."""

    fields = (
        "format",
        "strict",
        "dataset_id",
        "dataset_fingerprint",
        "split_fingerprint",
        "inventory_fingerprint",
        "train_inventory_fingerprint",
        "split_inventory_fingerprints",
        "manifest_sha256",
        "fingerprint_report_sha256",
        "image_count",
    )
    return {key: identity.get(key) for key in fields}


def teacher_run_dir(cfg: dict[str, Any], identity: dict[str, Any] | None = None) -> Path:
    identity = identity or resolve_dataset_identity(cfg)
    root = Path(cfg["paths"]["project_root"])
    namespace = artifact_namespace(identity)
    return root / "runs" / namespace / "teacher" if namespace else root / "runs" / "teacher"


def student_runs_root(cfg: dict[str, Any], identity: dict[str, Any] | None = None) -> Path:
    identity = identity or resolve_dataset_identity(cfg)
    root = Path(cfg["paths"]["project_root"])
    namespace = artifact_namespace(identity)
    return root / "runs" / namespace if namespace else root / "runs"


def teacher_cache_dir(
    cfg: dict[str, Any], split: str, identity: dict[str, Any] | None = None
) -> Path:
    identity = identity or resolve_dataset_identity(cfg)
    root = Path(cfg["paths"]["project_root"])
    namespace = artifact_namespace(identity)
    base = root / "cache" / "teacher_signals"
    return base / namespace / split if namespace else base / split


def git_identity(path: str | Path) -> dict[str, Any]:
    root = Path(path).resolve()
    result: dict[str, Any] = {"path": str(root), "commit": None, "dirty": None}
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        result.update(commit=commit, dirty=bool(status.strip()))
    except (OSError, subprocess.SubprocessError):
        pass
    return result


def preprocessing_contract(cfg: dict[str, Any]) -> dict[str, Any]:
    contract = {
        "name": "deterministic_square_letterbox",
        "image_size": int(cfg["dataset"]["image_size"]),
        "resize_interpolation": "cv2.INTER_LINEAR",
        "pad_color_bgr": [114, 114, 114],
        "rounding": "round(w*ratio),round(h*ratio);left/top=floor(remaining/2)",
        "color_conversion": "BGR_to_RGB",
        "tensor_layout": "CHW_float32",
        "normalization": "divide_by_255",
        "augmentations": "none",
    }
    return {**contract, "fingerprint": canonical_sha256(contract)}


def dino_identity(cfg: dict[str, Any]) -> dict[str, Any]:
    weights = Path(cfg["paths"]["dino_weights"]).resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"DINOv3 weights missing: {weights}")
    return {
        "weights_sha256": file_sha256(weights),
        "weights_name": weights.name,
        "repo_git": git_identity(cfg["paths"]["dino_repo"]),
        "architecture": "dinov3_vitl16",
    }


def dino_compatibility(identity: dict[str, Any]) -> dict[str, Any]:
    repo = identity.get("repo_git") or {}
    return {
        "weights_sha256": identity.get("weights_sha256"),
        "architecture": identity.get("architecture"),
        "repo_commit": repo.get("commit"),
    }


def teacher_provenance(
    cfg: dict[str, Any], identity: dict[str, Any] | None = None
) -> dict[str, Any]:
    identity = identity or resolve_dataset_identity(cfg)
    tcfg = cfg["teacher"]
    core = {
        "dataset": portable_dataset_identity(identity),
        "dino": dino_compatibility(dino_identity(cfg)),
        "preprocess": preprocessing_contract(cfg),
        "teacher_model": {
            "feature_channels": int(tcfg["feature_channels"]),
            "roi_size": int(tcfg["roi_size"]),
            "num_classes": int(cfg["dataset"]["nc"]),
        },
        "optimization": {
            "batch": int(tcfg["batch"]),
            "accumulate": int(tcfg["accumulate"]),
            "lr": float(tcfg["lr"]),
            "seed": int(cfg["student"].get("seed", 0)),
        },
    }
    return {
        "format": PROVENANCE_FORMAT,
        "kind": "dinov3_roi_teacher",
        **core,
        "core_fingerprint": canonical_sha256(core),
        "project_git": git_identity(cfg["paths"]["project_root"]),
    }


def require_teacher_compatible(
    checkpoint: dict[str, Any], expected: dict[str, Any], *, strict: bool
) -> None:
    saved = checkpoint.get("provenance")
    if not strict and saved is None:
        return
    if not isinstance(saved, dict):
        raise RuntimeError(
            "Fingerprint-enforced teacher checkpoint has no provenance and may belong to V2. Start a new V3 teacher run."
        )
    if saved.get("core_fingerprint") != expected.get("core_fingerprint"):
        raise RuntimeError(
            "Teacher checkpoint provenance differs from the selected dataset/DINO/preprocess/configuration."
        )


def inventory_rows_for_split(
    identity: dict[str, Any], split: str
) -> list[dict[str, str]]:
    if not identity.get("strict"):
        return []
    rows = _read_split_manifest(Path(identity["split_manifest"]))
    return [row for row in rows if row["split"] == split]


def relative_image_path(row: dict[str, str]) -> str:
    return (Path("images") / row["split"] / row["image"]).as_posix()


def cache_inventory(
    cfg: dict[str, Any], identity: dict[str, Any], split: str
) -> list[dict[str, str]]:
    from .common import stable_image_key

    root = Path(identity["dataset_root"])
    if identity.get("strict"):
        rows = inventory_rows_for_split(identity, split)
        return [
            {
                "key": stable_image_key(root / "images" / split / row["image"]),
                "relative_image": relative_image_path(row),
                "relative_label": (Path("labels") / split / row["label"]).as_posix(),
                "image_sha256": row["image_sha256"],
                "label_sha256": row["label_sha256"],
            }
            for row in rows
        ]
    # Legacy fallback records hashes too, without imposing a new dataset contract.
    data = _load_yaml(Path(cfg["paths"]["data_yaml"]))
    image_dir = root / data[split]
    label_dir = root / "labels" / split
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    records: list[dict[str, str]] = []
    for image in sorted(path for path in image_dir.rglob("*") if path.suffix.lower() in suffixes):
        relative = image.relative_to(image_dir)
        label = label_dir / relative.with_suffix(".txt")
        records.append(
            {
                "key": stable_image_key(image),
                "relative_image": (Path("images") / split / relative).as_posix(),
                "relative_label": (Path("labels") / split / relative.with_suffix(".txt")).as_posix(),
                "image_sha256": file_sha256(image),
                "label_sha256": file_sha256(label) if label.is_file() else "missing",
            }
        )
    return records


def write_cache_inventory(records: list[dict[str, str]], path: Path) -> str:
    text = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for record in records
    )
    path.write_text(text, encoding="utf-8")
    return file_sha256(path)


def read_cache_inventory(path: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"Invalid cache inventory line {number} in {path}: {error}") from error
        records.append(value)
    return records


def cache_manifest(
    cfg: dict[str, Any],
    identity: dict[str, Any],
    split: str,
    teacher_path: Path,
    teacher_checkpoint: dict[str, Any],
    inventory_sha256: str,
    inventory_count: int,
) -> dict[str, Any]:
    teacher_saved = teacher_checkpoint.get("provenance")
    manifest = {
        "format": CACHE_FORMAT,
        "kind": "dinov3_teacher_signals",
        "split": split,
        "image_size": int(cfg["dataset"]["image_size"]),
        "feature_channels": int(cfg["teacher"]["feature_channels"]),
        "num_classes": int(cfg["dataset"]["nc"]),
        "roi_embedding_dim": 512,
        "dataset": identity,
        "dataset_fingerprint": identity.get("dataset_fingerprint"),
        "split_fingerprint": identity.get("split_fingerprint"),
        "inventory_fingerprint": identity.get("inventory_fingerprint"),
        "split_inventory_fingerprint": identity.get(
            "split_inventory_fingerprints", {}
        ).get(split),
        "inventory_file": "inventory.jsonl",
        "inventory_sha256": inventory_sha256,
        "inventory_count": int(inventory_count),
        "teacher_sha256": file_sha256(teacher_path),
        "teacher_provenance_fingerprint": (
            teacher_saved.get("core_fingerprint") if isinstance(teacher_saved, dict) else None
        ),
        "dino": dino_identity(cfg),
        "preprocess": preprocessing_contract(cfg),
        "project_git": git_identity(cfg["paths"]["project_root"]),
    }
    manifest["compatibility_fingerprint"] = cache_compatibility_fingerprint(manifest)
    return manifest


def cache_compatibility_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "format",
        "kind",
        "split",
        "image_size",
        "feature_channels",
        "num_classes",
        "roi_embedding_dim",
        "dataset_fingerprint",
        "split_fingerprint",
        "inventory_fingerprint",
        "split_inventory_fingerprint",
        "inventory_sha256",
        "inventory_count",
        "teacher_sha256",
        "teacher_provenance_fingerprint",
        "preprocess",
    )
    payload = {key: manifest.get(key) for key in fields}
    payload["dino"] = dino_compatibility(manifest.get("dino") or {})
    return payload


def cache_compatibility_fingerprint(manifest: dict[str, Any]) -> str:
    return canonical_sha256(cache_compatibility_payload(manifest))


def validate_cache_manifest(
    cfg: dict[str, Any],
    manifest: dict[str, Any],
    split: str,
    identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    identity = identity or resolve_dataset_identity(cfg)
    if identity.get("strict"):
        if manifest.get("format") != CACHE_FORMAT:
            raise RuntimeError(
                "V3 requires cache format=3 provenance; a V2/legacy cache is forbidden."
            )
        expected = {
            "split": split,
            "image_size": int(cfg["dataset"]["image_size"]),
            "feature_channels": int(cfg["teacher"]["feature_channels"]),
            "num_classes": int(cfg["dataset"]["nc"]),
            "dataset_fingerprint": identity["dataset_fingerprint"],
            "split_fingerprint": identity["split_fingerprint"],
            "inventory_fingerprint": identity["inventory_fingerprint"],
            "split_inventory_fingerprint": identity.get(
                "split_inventory_fingerprints", {}
            ).get(split),
        }
    else:
        expected = {
            "split": split,
            "image_size": int(cfg["dataset"]["image_size"]),
            "feature_channels": int(cfg["teacher"]["feature_channels"]),
            "num_classes": int(cfg["dataset"]["nc"]),
        }
        if manifest.get("format") not in {2, CACHE_FORMAT}:
            raise RuntimeError(f"Unsupported teacher cache format: {manifest.get('format')!r}")
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Teacher cache provenance is incompatible: {mismatches}")
    if manifest.get("format") == CACHE_FORMAT:
        current_dino = dino_compatibility(dino_identity(cfg))
        cached_dino = dino_compatibility(manifest.get("dino") or {})
        if current_dino != cached_dino:
            raise RuntimeError(
                f"Teacher cache DINO provenance differs from current weights/repository: "
                f"cached={cached_dino}, current={current_dino}"
            )
        stored = manifest.get("compatibility_fingerprint")
        computed = cache_compatibility_fingerprint(manifest)
        if stored != computed:
            raise RuntimeError("Teacher cache manifest compatibility fingerprint is invalid.")
    return identity


def verify_cache_sample(
    cfg: dict[str, Any],
    cache_dir: Path,
    manifest: dict[str, Any],
    *,
    samples: int = 64,
) -> int:
    """Deterministically sample cache index, disk files and cache item hashes."""

    if manifest.get("format") != CACHE_FORMAT:
        return 0
    inventory_path = cache_dir / str(manifest.get("inventory_file", "inventory.jsonl"))
    if not inventory_path.is_file():
        raise RuntimeError(f"Teacher cache inventory missing: {inventory_path}")
    if file_sha256(inventory_path) != manifest.get("inventory_sha256"):
        raise RuntimeError("Teacher cache inventory checksum differs from manifest.")
    records = read_cache_inventory(inventory_path)
    if len(records) != int(manifest.get("inventory_count", -1)):
        raise RuntimeError("Teacher cache inventory count differs from manifest.")
    root = dataset_root_from_config(cfg)
    if not records:
        raise RuntimeError("Teacher cache inventory is empty.")
    # Evenly spaced deterministic samples cover the full ordered index.
    count = min(max(int(samples), 1), len(records))
    indices = sorted({round(i * (len(records) - 1) / max(count - 1, 1)) for i in range(count)})
    checked = 0
    for index in indices:
        record = records[index]
        image = root / record["relative_image"]
        label = root / record["relative_label"]
        if not image.is_file() or not label.is_file():
            raise RuntimeError(f"Cache inventory sample is missing image/label: {image}")
        actual_image = file_sha256(image)
        actual_label = file_sha256(label)
        if actual_image != record.get("image_sha256") or actual_label != record.get("label_sha256"):
            raise RuntimeError(f"Dataset content changed after cache creation: {record['relative_image']}")
        entry_path = cache_dir / f"{record['key']}.pt"
        if not entry_path.is_file():
            raise RuntimeError(f"Teacher cache entry missing: {entry_path}")
        import torch

        entry = torch.load(entry_path, map_location="cpu", weights_only=False)
        if entry.get("image_sha256") != actual_image or entry.get("label_sha256") != actual_label:
            raise RuntimeError(f"Teacher cache entry hashes do not match dataset: {entry_path}")
        if entry.get("cache_manifest_fingerprint") != manifest.get("compatibility_fingerprint"):
            raise RuntimeError(f"Teacher cache entry belongs to another manifest: {entry_path}")
        embedding_dim = manifest.get("roi_embedding_dim")
        if embedding_dim is not None:
            embeddings = entry.get("roi_embeddings")
            if (
                embeddings is None
                or getattr(embeddings, "ndim", 0) != 2
                or int(embeddings.shape[1]) != int(embedding_dim)
                or int(embeddings.shape[0]) != int(entry["classes"].shape[0])
            ):
                raise RuntimeError(
                    f"Teacher cache entry has invalid/missing penultimate RoI embeddings: {entry_path}"
                )
        checked += 1
    return checked


def _checkpoint_record(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": file_sha256(path)}


def prepare_run_lineage(
    cfg: dict[str, Any],
    run_dir: Path,
    *,
    experiment: str,
    initial_checkpoint: Path,
    resume_checkpoint: Path | None,
    cache_manifest_data: dict[str, Any] | None,
    identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    identity = identity or resolve_dataset_identity(cfg)
    path = run_dir / "run_provenance.json"
    core = {
        "dataset": portable_dataset_identity(identity),
        "namespace": artifact_namespace(identity),
        "experiment": experiment,
        "run_name": run_dir.name,
        "initial_checkpoint": _checkpoint_record(initial_checkpoint),
        "cache_compatibility_fingerprint": (
            cache_manifest_data.get("compatibility_fingerprint") if cache_manifest_data else None
        ),
    }
    if cfg.get("runtime", {}).get("prototype_bank") is not None:
        # Content identity only: exact resumes remain relocatable across servers.
        core["prototype_bank"] = cfg["runtime"]["prototype_bank"]
    core_fingerprint = canonical_sha256(core)
    now = datetime.now(timezone.utc).isoformat()
    if resume_checkpoint is None:
        if path.exists():
            raise FileExistsError(f"Fresh run already has provenance: {path}")
        value = {
            "format": PROVENANCE_FORMAT,
            "kind": "student_run_lineage",
            **core,
            "core_fingerprint": core_fingerprint,
            "created_at": now,
            "project_git": git_identity(cfg["paths"]["project_root"]),
            "resume_events": [],
        }
        run_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        return value
    expected_parent = run_dir / "weights" / "last.pt"
    if identity.get("strict") and resume_checkpoint.resolve() != expected_parent.resolve():
        raise RuntimeError(
            f"V3 exact resume must use its namespaced last.pt ({expected_parent}), got {resume_checkpoint}."
        )
    if not path.is_file():
        if identity.get("strict"):
            raise RuntimeError(
                f"Fingerprint-enforced resume has no run lineage {path}; cross-dataset resume is forbidden."
            )
        return {"format": 0, "kind": "legacy_resume_without_lineage"}
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("core_fingerprint") != core_fingerprint:
        raise RuntimeError(
            "Run provenance differs from current dataset/baseline/cache/objective; exact resume is forbidden."
        )
    value.setdefault("resume_events", []).append(
        {"resumed_at": now, "checkpoint": _checkpoint_record(resume_checkpoint)}
    )
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    return value


def inject_discovered_fingerprints(cfg: dict[str, Any]) -> dict[str, Any]:
    """Populate config fingerprints from a colocated V3 package report."""

    root = dataset_root_from_config(cfg)
    report_path = root / "dataset_fingerprint.json"
    if not report_path.is_file():
        return cfg
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not str(report.get("dataset_id", "")).startswith("scene811_v3"):
        # Legacy V1/V2 packages used a different split-fingerprint schema and
        # retain their historical non-namespaced behavior.
        return cfg
    dataset_cfg = cfg.setdefault("dataset", {})
    dataset_cfg["id"] = _safe_dataset_id(report.get("dataset_id"))
    dataset_cfg["dataset_fingerprint"] = _hex64(
        report.get("dataset_fingerprint"), "report.dataset_fingerprint"
    )
    dataset_cfg["split_fingerprint"] = _hex64(
        report.get("split_fingerprint"), "report.split_fingerprint"
    )
    dataset_cfg["fingerprint_file"] = str(report_path)
    # Resolve once so a malformed/mismatched package is rejected while config is generated.
    resolve_dataset_identity(cfg)
    return cfg
