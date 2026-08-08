from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .common import load_config, stable_image_key, resolve_data_yaml
from .teacher import DINOFeatureTeacher
from .train_teacher import YoloRoiDataset, collate, xywhn_to_rois


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Cache deterministic DINOv3 teacher P3/P4/P5 features and ROI logits.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--teacher", default=None)
    parser.add_argument("--resume", action="store_true", help="Continue a compatible interrupted cache job; existing samples are verified and skipped.")
    args = parser.parse_args()
    cfg = load_config(args.config); data = resolve_data_yaml(cfg); root = Path(cfg["paths"]["project_root"])
    tcfg = cfg["teacher"]; image_size = cfg["dataset"]["image_size"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_path = Path(args.teacher) if args.teacher else root / "runs" / "teacher" / "best.pt"
    checkpoint = torch.load(teacher_path, map_location="cpu", weights_only=False)
    model = DINOFeatureTeacher(cfg["paths"]["dino_repo"], cfg["paths"]["dino_weights"], tcfg["feature_channels"], tcfg["roi_size"], cfg["dataset"]["nc"])
    model.load_state_dict(checkpoint["model"], strict=True); model.to(device).eval()
    loader = DataLoader(YoloRoiDataset(data, args.split, image_size), batch_size=1, shuffle=False, num_workers=tcfg["num_workers"], pin_memory=True, collate_fn=collate)
    destination = root / "cache" / "teacher_signals" / args.split; destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "manifest.json"
    manifest = {
        "format": 2, "split": args.split, "teacher_sha256": sha256_file(teacher_path),
        "image_size": image_size, "feature_channels": tcfg["feature_channels"], "num_classes": cfg["dataset"]["nc"],
    }
    existing_signals = any(destination.glob("*.pt"))
    if existing_signals and not args.resume:
        raise FileExistsError(f"{destination} already contains signals. Use --resume to continue the compatible cache job.")
    if args.resume:
        if not manifest_path.exists():
            raise RuntimeError(f"Cannot safely resume cache without {manifest_path}. Use a new cache directory after preserving the old one.")
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous != manifest:
            raise RuntimeError("Cache manifest differs from the requested teacher/configuration. Do not mix teacher signals; use a new cache directory.")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for batch in tqdm(loader, desc=f"cache {args.split}"):
        path = Path(batch["paths"][0]); out = destination / f"{stable_image_key(path)}.pt"
        if out.exists():
            # A corrupt/interrupted write must never be silently accepted as a completed cache entry.
            try:
                cached = torch.load(out, map_location="cpu", weights_only=False)
                if str(Path(cached["path"]).resolve()) != str(path.resolve()):
                    raise RuntimeError("path mismatch")
            except Exception as exc:
                raise RuntimeError(f"Invalid existing cache entry {out}: {exc}") from exc
            continue
        images = batch["images"].to(device, non_blocking=True); boxes = batch["boxes"].to(device); indices = batch["batch_indices"].to(device)
        rois = xywhn_to_rois(boxes, indices, image_size)
        output = model(images, rois); p3, p4, p5 = output["features"]
        torch.save({
            "path": str(path.resolve()),
            "boxes_xywhn": boxes.cpu().float(),
            "classes": batch["classes"].cpu().long(),
            "roi_logits": output["roi_logits"].cpu().half(),
            "p3": p3[0].cpu().half(), "p4": p4[0].cpu().half(), "p5": p5[0].cpu().half(),
        }, out)
    print(f"cache written to {destination}; resume-safe manifest={manifest_path}")


if __name__ == "__main__":
    main()
