from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .common import COARSE_NAMES, FINE_TO_COARSE, load_config
from .provenance import (
    git_identity,
    portable_dataset_identity,
    read_cache_inventory,
    resolve_dataset_identity,
    teacher_cache_dir,
    validate_cache_manifest,
    verify_cache_sample,
)


BANK_FORMAT = 1


def roi_size_bucket(box_xywhn: torch.Tensor, image_size: int) -> str:
    area = float(box_xywhn[2]) * image_size * float(box_xywhn[3]) * image_size
    if area < 32.0**2:
        return "small"
    if area < 96.0**2:
        return "medium"
    return "large"


def _relative_train_image(relative_image: str) -> str:
    path = Path(relative_image)
    parts = list(path.parts)
    lowered = [part.casefold() for part in parts]
    for index in range(len(parts) - 1):
        if lowered[index] == "images" and lowered[index + 1] == "train":
            return Path(*parts[index + 2 :]).as_posix()
    raise RuntimeError(f"Prototype bank received a non-TRAIN cache item: {relative_image}")


def _prototype_keys(fine_class: int, size: str) -> tuple[str, str, str]:
    coarse = COARSE_NAMES[FINE_TO_COARSE[int(fine_class)]]
    return f"fine:{int(fine_class)}", f"coarse:{coarse}", f"size:{size}"


def _bank_fingerprint(value: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    metadata = {
        "format": value["format"],
        "kind": value["kind"],
        "embedding_dim": value["embedding_dim"],
        "min_count": value["min_count"],
        "provenance": value["provenance"],
        "image_to_scene": value["image_to_scene"],
        "source_roi_count": value.get("source_roi_count"),
        "roi_count": value.get("roi_count"),
        "gate_skips": value.get("gate_skips", {}),
    }
    digest.update(
        json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    for key in sorted(value["global_stats"]):
        item = value["global_stats"][key]
        digest.update(key.encode("utf-8"))
        digest.update(str(int(item["count"])).encode("ascii"))
        digest.update(item["sum"].detach().cpu().float().contiguous().numpy().tobytes())
    for scene in sorted(value["scene_stats"]):
        for key in sorted(value["scene_stats"][scene]):
            item = value["scene_stats"][scene][key]
            digest.update(scene.encode("utf-8"))
            digest.update(key.encode("utf-8"))
            digest.update(str(int(item["count"])).encode("ascii"))
            digest.update(item["sum"].detach().cpu().float().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _manifest_scene_map(identity: dict[str, Any]) -> dict[str, str]:
    import csv

    path = Path(identity["split_manifest"])
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    mapping: dict[str, str] = {}
    for row in rows:
        if row["split"] != "train":
            continue
        relative = Path(row["image"]).as_posix()
        if relative in mapping:
            raise RuntimeError(f"Duplicate TRAIN image in split manifest: {relative}")
        mapping[relative] = row["scene_id"]
    if not mapping:
        raise RuntimeError("V3 split manifest has no TRAIN scene identities.")
    return mapping


def _scene_for_relative(mapping: dict[str, str], relative: str) -> str | None:
    normalized = Path(relative).as_posix()
    if normalized in mapping:
        return mapping[normalized]
    # Current V3 manifests use basenames, while future packages may preserve
    # subdirectories. A basename fallback is safe only when it is unique.
    matches = {
        scene for image, scene in mapping.items() if Path(image).name == Path(normalized).name
    }
    return next(iter(matches)) if len(matches) == 1 else None


def _accumulate(
    store: dict[str, dict[str, Any]], key: str, embedding: torch.Tensor
) -> None:
    if key not in store:
        store[key] = {"sum": torch.zeros_like(embedding, dtype=torch.float64), "count": 0}
    store[key]["sum"].add_(embedding.to(dtype=torch.float64))
    store[key]["count"] += 1


def build_prototype_bank(
    cfg: dict[str, Any],
    cache_dir: Path,
    out: Path,
    *,
    min_count: int = 4,
    verify_samples: int = 64,
) -> dict[str, Any]:
    identity = resolve_dataset_identity(cfg)
    if not identity.get("strict"):
        raise RuntimeError("P/GP prototype banks require a fingerprint-enforced V3 dataset.")
    if min_count < 2:
        raise ValueError("Prototype min_count must be >=2 for leave-one-scene-out targets.")
    manifest_path = Path(cache_dir) / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Teacher cache manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_cache_manifest(cfg, manifest, "train", identity)
    if manifest.get("format") != 3 or int(manifest.get("roi_embedding_dim", 0)) <= 0:
        raise RuntimeError(
            "Prototype bank requires the new format=3 cache with penultimate roi_embeddings; rebuild the teacher cache."
        )
    verify_cache_sample(cfg, Path(cache_dir), manifest, samples=verify_samples)
    inventory_path = Path(cache_dir) / str(manifest["inventory_file"])
    inventory = read_cache_inventory(inventory_path)
    scenes = _manifest_scene_map(identity)
    global_stats: dict[str, dict[str, Any]] = {}
    scene_stats: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    image_to_scene: dict[str, str] = {}
    embedding_dim = int(manifest["roi_embedding_dim"])
    roi_count = 0
    source_roi_count = 0
    gate_skips: dict[str, int] = defaultdict(int)
    dcfg = cfg.get("distillation", {})
    confidence_floor = float(dcfg.get("prototype_teacher_confidence_floor", 0.70))
    margin_floor = float(dcfg.get("prototype_teacher_margin_floor", 0.10))
    require_coarse = bool(dcfg.get("prototype_require_coarse_correct", True))
    for record in inventory:
        relative = _relative_train_image(record["relative_image"])
        scene = _scene_for_relative(scenes, relative)
        if scene is None:
            raise RuntimeError(f"Cache image is absent from V3 split manifest: {relative}")
        image_to_scene[relative] = scene
        entry = torch.load(
            Path(cache_dir) / f"{record['key']}.pt", map_location="cpu", weights_only=False
        )
        embeddings = entry.get("roi_embeddings")
        classes = entry.get("classes")
        boxes = entry.get("boxes_xywhn")
        logits = entry.get("roi_logits")
        if (
            not isinstance(embeddings, torch.Tensor)
            or embeddings.ndim != 2
            or embeddings.shape[1] != embedding_dim
            or not isinstance(classes, torch.Tensor)
            or not isinstance(boxes, torch.Tensor)
            or not isinstance(logits, torch.Tensor)
            or len(embeddings) != len(classes)
            or len(embeddings) != len(boxes)
            or len(embeddings) != len(logits)
        ):
            raise RuntimeError(f"Invalid RoI embedding cache entry for prototype bank: {relative}")
        embeddings = F.normalize(embeddings.float(), dim=1)
        for embedding, fine_class, box, logit in zip(embeddings, classes, boxes, logits):
            source_roi_count += 1
            fine = int(fine_class)
            if fine not in FINE_TO_COARSE:
                raise RuntimeError(f"Invalid fine class {fine} in cache entry {relative}")
            probability = logit.float().softmax(dim=0)
            top_values, top_indices = probability.topk(k=2)
            if float(top_values[0]) < confidence_floor:
                gate_skips["teacher_low_confidence"] += 1
                continue
            if float(top_values[0] - top_values[1]) < margin_floor:
                gate_skips["teacher_low_margin"] += 1
                continue
            if require_coarse and FINE_TO_COARSE[int(top_indices[0])] != FINE_TO_COARSE[fine]:
                gate_skips["teacher_wrong_coarse"] += 1
                continue
            size = roi_size_bucket(box.float(), int(cfg["dataset"]["image_size"]))
            for key in _prototype_keys(fine, size):
                _accumulate(global_stats, key, embedding)
                _accumulate(scene_stats[scene], key, embedding)
            roi_count += 1
    if roi_count <= 0:
        raise RuntimeError("Prototype bank cannot be built from an empty teacher cache.")
    for item in global_stats.values():
        item["sum"] = item["sum"].float()
    for by_key in scene_stats.values():
        for item in by_key.values():
            item["sum"] = item["sum"].float()
    provenance = {
        "dataset": portable_dataset_identity(identity),
        "cache_compatibility_fingerprint": manifest["compatibility_fingerprint"],
        "teacher_sha256": manifest["teacher_sha256"],
        "teacher_provenance_fingerprint": manifest.get("teacher_provenance_fingerprint"),
        "dino": manifest["dino"],
        "preprocess": manifest["preprocess"],
        "project_git": git_identity(cfg["paths"]["project_root"]),
        "prototype_gate": {
            "teacher_confidence_floor": confidence_floor,
            "teacher_margin_floor": margin_floor,
            "require_coarse_correct": require_coarse,
        },
    }
    bank: dict[str, Any] = {
        "format": BANK_FORMAT,
        "kind": "fhit_leave_one_scene_out_prototypes",
        "embedding_dim": embedding_dim,
        "min_count": int(min_count),
        "provenance": provenance,
        "image_to_scene": dict(sorted(image_to_scene.items())),
        "global_stats": dict(sorted(global_stats.items())),
        "scene_stats": {scene: dict(sorted(items.items())) for scene, items in sorted(scene_stats.items())},
        "source_roi_count": source_roi_count,
        "roi_count": roi_count,
        "gate_skips": dict(sorted(gate_skips.items())),
    }
    bank["bank_fingerprint"] = _bank_fingerprint(bank)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".tmp")
    torch.save(bank, temporary)
    temporary.replace(out)
    return bank


def validate_prototype_bank(
    cfg: dict[str, Any], cache_manifest: dict[str, Any], bank: dict[str, Any]
) -> None:
    identity = resolve_dataset_identity(cfg)
    if not identity.get("strict"):
        raise RuntimeError("P/GP prototype banks require a fingerprint-enforced V3 dataset.")
    if bank.get("format") != BANK_FORMAT or bank.get("kind") != "fhit_leave_one_scene_out_prototypes":
        raise RuntimeError("Unsupported or invalid prototype bank format.")
    if bank.get("bank_fingerprint") != _bank_fingerprint(bank):
        raise RuntimeError("Prototype bank fingerprint is invalid; the bank was modified or truncated.")
    expected = {
        "dataset": portable_dataset_identity(identity),
        "cache_compatibility_fingerprint": cache_manifest.get("compatibility_fingerprint"),
        "teacher_sha256": cache_manifest.get("teacher_sha256"),
        "teacher_provenance_fingerprint": cache_manifest.get("teacher_provenance_fingerprint"),
        "dino": cache_manifest.get("dino"),
        "preprocess": cache_manifest.get("preprocess"),
        "prototype_gate": {
            "teacher_confidence_floor": float(cfg.get("distillation", {}).get("prototype_teacher_confidence_floor", 0.70)),
            "teacher_margin_floor": float(cfg.get("distillation", {}).get("prototype_teacher_margin_floor", 0.10)),
            "require_coarse_correct": bool(cfg.get("distillation", {}).get("prototype_require_coarse_correct", True)),
        },
    }
    actual = {key: bank.get("provenance", {}).get(key) for key in expected}
    if actual != expected:
        mismatches = {key: (actual[key], value) for key, value in expected.items() if actual[key] != value}
        raise RuntimeError(f"Prototype bank provenance is incompatible: {mismatches}")
    bank_commit = (bank.get("provenance", {}).get("project_git") or {}).get("commit")
    current_commit = git_identity(cfg["paths"]["project_root"]).get("commit")
    if bank_commit != current_commit:
        raise RuntimeError(
            f"Prototype bank project Git commit differs: bank={bank_commit!r}, current={current_commit!r}."
        )
    dimension = int(cache_manifest.get("roi_embedding_dim", 0))
    if dimension <= 0 or int(bank.get("embedding_dim", 0)) != dimension:
        raise RuntimeError("Prototype bank embedding dimension differs from the teacher cache.")
    if int(bank.get("min_count", 0)) < 2:
        raise RuntimeError("Prototype bank min_count is unsafe for leave-one-scene-out use.")
    configured_min_count = cfg.get("distillation", {}).get("prototype_min_count")
    if configured_min_count is not None and int(bank["min_count"]) != int(configured_min_count):
        raise RuntimeError(
            f"Prototype bank min_count={bank['min_count']} differs from configuration={configured_min_count}."
        )


class PrototypeBank:
    def __init__(self, value: dict[str, Any]):
        self.value = value
        self.embedding_dim = int(value["embedding_dim"])
        self.min_count = int(value["min_count"])
        self.global_stats = value["global_stats"]
        self.scene_stats = value["scene_stats"]
        self.image_to_scene = value["image_to_scene"]

    @classmethod
    def load(
        cls, path: str | Path, cfg: dict[str, Any], cache_manifest: dict[str, Any]
    ) -> "PrototypeBank":
        value = torch.load(Path(path), map_location="cpu", weights_only=False)
        if not isinstance(value, dict):
            raise RuntimeError(f"Prototype bank is not a mapping: {path}")
        validate_prototype_bank(cfg, cache_manifest, value)
        return cls(value)

    def scene_for_image(self, image_path: str | Path) -> str:
        parts = list(Path(image_path).parts)
        lowered = [part.casefold() for part in parts]
        for index in range(len(parts) - 1):
            if lowered[index] == "images" and lowered[index + 1] == "train":
                relative = Path(*parts[index + 2 :]).as_posix()
                break
        else:
            raise RuntimeError(f"Prototype lookup received a non-TRAIN image: {image_path}")
        scene = self.image_to_scene.get(relative)
        if scene is None:
            raise RuntimeError(f"Prototype bank has no scene identity for {relative}")
        return scene

    def lookup(
        self, scene: str, key: str, device: torch.device
    ) -> tuple[torch.Tensor | None, int]:
        global_item = self.global_stats.get(key)
        if global_item is None:
            return None, 0
        scene_item = self.scene_stats.get(scene, {}).get(key)
        excluded_count = int(scene_item["count"]) if scene_item else 0
        count = int(global_item["count"]) - excluded_count
        if count < self.min_count:
            return None, count
        total = global_item["sum"].float()
        if scene_item:
            total = total - scene_item["sum"].float()
        prototype = F.normalize((total / float(count)).to(device=device), dim=0)
        return prototype, count

    def target_keys(self, fine_class: int, size: str) -> tuple[str, str, str]:
        return _prototype_keys(fine_class, size)

    def fine_candidates(
        self, scene: str, device: torch.device
    ) -> tuple[torch.Tensor, list[str]]:
        values, keys = [], []
        for key in sorted(item for item in self.global_stats if item.startswith("fine:")):
            value, _count = self.lookup(scene, key, device)
            if value is not None:
                values.append(value)
                keys.append(key)
        if not values:
            return torch.empty((0, self.embedding_dim), device=device), []
        return torch.stack(values), keys


def _load_manifest(cache_dir: Path) -> dict[str, Any]:
    path = cache_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build or verify V3 leave-one-scene-out DINO RoI prototype banks."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("build", "verify"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--config", default="configs/fhit_v2.yaml")
        sub.add_argument("--cache", default=None)
        sub.add_argument("--bank", required=True)
    subparsers.choices["build"].add_argument("--min-count", type=int, default=4)
    subparsers.choices["build"].add_argument("--verify-samples", type=int, default=64)
    args = parser.parse_args()
    cfg = load_config(args.config)
    identity = resolve_dataset_identity(cfg)
    cache_dir = Path(args.cache) if args.cache else teacher_cache_dir(cfg, "train", identity)
    if args.command == "build":
        bank = build_prototype_bank(
            cfg,
            cache_dir,
            Path(args.bank),
            min_count=args.min_count,
            verify_samples=args.verify_samples,
        )
        print(
            f"prototype bank written: {args.bank}; rois={bank['roi_count']} "
            f"fingerprint={bank['bank_fingerprint']}"
        )
    else:
        manifest = _load_manifest(cache_dir)
        validate_cache_manifest(cfg, manifest, "train", identity)
        bank = torch.load(Path(args.bank), map_location="cpu", weights_only=False)
        validate_prototype_bank(cfg, manifest, bank)
        print(
            f"PASS: prototype bank matches dataset/cache/teacher; "
            f"rois={bank['roi_count']} fingerprint={bank['bank_fingerprint']}"
        )


if __name__ == "__main__":
    main()
