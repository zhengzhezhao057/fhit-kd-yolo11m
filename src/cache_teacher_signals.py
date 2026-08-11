from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .common import load_config, stable_image_key, resolve_data_yaml
from .provenance import (
    cache_inventory,
    cache_manifest,
    file_sha256,
    require_teacher_compatible,
    resolve_dataset_identity,
    teacher_cache_dir,
    teacher_provenance,
    teacher_run_dir,
    validate_cache_manifest,
    verify_cache_sample,
    write_cache_inventory,
)
from .teacher import DINOFeatureTeacher
from .train_teacher import YoloRoiDataset, collate, xywhn_to_rois


def sha256_file(path: Path) -> str:
    """Backward-compatible local alias."""
    return file_sha256(path)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Cache deterministic DINOv3 teacher P3/P4/P5 features and ROI logits.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--teacher", default=None)
    parser.add_argument("--resume", action="store_true", help="Continue a compatible interrupted cache job; existing samples are verified and skipped.")
    args = parser.parse_args()
    cfg = load_config(args.config); data = resolve_data_yaml(cfg)
    identity = resolve_dataset_identity(cfg)
    tcfg = cfg["teacher"]; image_size = cfg["dataset"]["image_size"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_path = Path(args.teacher) if args.teacher else teacher_run_dir(cfg, identity) / "best.pt"
    checkpoint = torch.load(teacher_path, map_location="cpu", weights_only=False)
    require_teacher_compatible(
        checkpoint,
        teacher_provenance(cfg, identity),
        strict=bool(identity.get("strict")),
    )
    model = DINOFeatureTeacher(cfg["paths"]["dino_repo"], cfg["paths"]["dino_weights"], tcfg["feature_channels"], tcfg["roi_size"], cfg["dataset"]["nc"])
    model.load_state_dict(checkpoint["model"], strict=True); model.to(device).eval()
    loader = DataLoader(YoloRoiDataset(data, args.split, image_size), batch_size=1, shuffle=False, num_workers=tcfg["num_workers"], pin_memory=True, collate_fn=collate)
    destination = teacher_cache_dir(cfg, args.split, identity); destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "manifest.json"
    inventory_path = destination / "inventory.jsonl"
    records = cache_inventory(cfg, identity, args.split)
    requested_inventory = destination / ".inventory.requested.jsonl"
    requested_inventory_sha = write_cache_inventory(records, requested_inventory)
    manifest = cache_manifest(
        cfg,
        identity,
        args.split,
        teacher_path,
        checkpoint,
        requested_inventory_sha,
        len(records),
    )
    inventory_by_key = {record["key"]: record for record in records}
    if len(inventory_by_key) != len(records):
        raise RuntimeError("Cache inventory contains duplicate stable image keys.")
    existing_signals = any(destination.glob("*.pt"))
    if (existing_signals or manifest_path.exists() or inventory_path.exists()) and not args.resume:
        requested_inventory.unlink(missing_ok=True)
        raise FileExistsError(f"{destination} already contains signals. Use --resume to continue the compatible cache job.")
    if args.resume:
        if not manifest_path.exists():
            requested_inventory.unlink(missing_ok=True)
            raise RuntimeError(f"Cannot safely resume cache without {manifest_path}. Use a new cache directory after preserving the old one.")
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_cache_manifest(cfg, previous, args.split, identity)
        legacy_resume = not identity.get("strict") and previous.get("format") == 2
        if legacy_resume:
            if previous.get("teacher_sha256") != file_sha256(teacher_path):
                requested_inventory.unlink(missing_ok=True)
                raise RuntimeError("Legacy cache teacher checkpoint differs; use a new cache directory.")
        else:
            if previous.get("compatibility_fingerprint") != manifest.get("compatibility_fingerprint"):
                requested_inventory.unlink(missing_ok=True)
                raise RuntimeError("Cache manifest differs from the requested teacher/configuration. Do not mix teacher signals; use a new cache directory.")
            if not inventory_path.is_file() or file_sha256(inventory_path) != requested_inventory_sha:
                requested_inventory.unlink(missing_ok=True)
                raise RuntimeError("Cache inventory differs from the selected dataset. Do not resume this cache.")
        requested_inventory.unlink(missing_ok=True)
        manifest = previous
    else:
        requested_inventory.replace(inventory_path)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for batch in tqdm(loader, desc=f"cache {args.split}"):
        path = Path(batch["paths"][0]); key = stable_image_key(path)
        out = destination / f"{key}.pt"
        record = inventory_by_key.get(key)
        if record is None:
            raise RuntimeError(f"Dataset loader image is absent from immutable cache inventory: {path}")
        root = Path(identity["dataset_root"])
        label_path = root / record["relative_label"]
        actual_image_sha = file_sha256(path)
        actual_label_sha = file_sha256(label_path) if label_path.is_file() else "missing"
        if actual_image_sha != record["image_sha256"] or actual_label_sha != record["label_sha256"]:
            raise RuntimeError(f"Dataset changed after fingerprinting: {path}")
        if out.exists():
            # A corrupt/interrupted write must never be silently accepted as a completed cache entry.
            try:
                cached = torch.load(out, map_location="cpu", weights_only=False)
                if str(Path(cached["path"]).resolve()) != str(path.resolve()):
                    raise RuntimeError("path mismatch")
                if manifest.get("format") == 3:
                    if cached.get("image_sha256") != actual_image_sha or cached.get("label_sha256") != actual_label_sha:
                        raise RuntimeError("image/label fingerprint mismatch")
                    if cached.get("cache_manifest_fingerprint") != manifest.get("compatibility_fingerprint"):
                        raise RuntimeError("cache manifest fingerprint mismatch")
            except Exception as exc:
                raise RuntimeError(f"Invalid existing cache entry {out}: {exc}") from exc
            continue
        images = batch["images"].to(device, non_blocking=True); boxes = batch["boxes"].to(device); indices = batch["batch_indices"].to(device)
        rois = xywhn_to_rois(boxes, indices, image_size)
        output = model(images, rois); p3, p4, p5 = output["features"]
        value = {
            "path": str(path.resolve()),
            "relative_image": record["relative_image"],
            "image_sha256": actual_image_sha,
            "label_sha256": actual_label_sha,
            "cache_manifest_fingerprint": manifest.get("compatibility_fingerprint"),
            "boxes_xywhn": boxes.cpu().float(),
            "classes": batch["classes"].cpu().long(),
            "roi_logits": output["roi_logits"].cpu().half(),
            "roi_embeddings": output["roi_embeddings"].cpu().half(),
            "p3": p3[0].cpu().half(), "p4": p4[0].cpu().half(), "p5": p5[0].cpu().half(),
        }
        temporary = out.with_suffix(out.suffix + ".tmp")
        torch.save(value, temporary)
        temporary.replace(out)
    checked = verify_cache_sample(cfg, destination, manifest, samples=64)
    print(
        f"cache written to {destination}; resume-safe manifest={manifest_path}; "
        f"provenance_samples_verified={checked}"
    )


if __name__ == "__main__":
    main()
