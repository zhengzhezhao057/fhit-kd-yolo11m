from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .common import image_to_label_path, letterbox, load_config, read_yolo_labels, resolve_data_yaml, split_image_dir, transform_boxes_to_letterbox
from .teacher import DINOFeatureTeacher


class YoloRoiDataset(Dataset):
    """Deterministic images and GT boxes for training the DINO object-level teacher."""
    def __init__(self, data: dict, split: str, size: int):
        self.image_dir = split_image_dir(data, split)
        self.label_dir = Path(data["path"]) / "labels" / split
        self.size = size
        self.images = sorted([p for p in self.image_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}])

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> dict:
        path = self.images[index]
        image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Cannot read {path}")
        h, w = image_bgr.shape[:2]
        image_bgr, ratio, left, top = letterbox(image_bgr, self.size)
        classes, boxes = read_yolo_labels(image_to_label_path(path, self.image_dir, self.label_dir))
        boxes = transform_boxes_to_letterbox(boxes, w, h, ratio, left, top, self.size)
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        return {"image": image, "classes": torch.from_numpy(classes), "boxes": torch.from_numpy(boxes), "path": str(path)}


def collate(samples: list[dict]) -> dict:
    images = torch.stack([s["image"] for s in samples])
    classes, boxes, batch_indices = [], [], []
    for index, sample in enumerate(samples):
        if len(sample["classes"]):
            classes.append(sample["classes"])
            boxes.append(sample["boxes"])
            batch_indices.append(torch.full((len(sample["classes"]),), index, dtype=torch.long))
    return {
        "images": images,
        "classes": torch.cat(classes) if classes else torch.zeros(0, dtype=torch.long),
        "boxes": torch.cat(boxes) if boxes else torch.zeros(0, 4),
        "batch_indices": torch.cat(batch_indices) if batch_indices else torch.zeros(0, dtype=torch.long),
        "paths": [s["path"] for s in samples],
    }


def xywhn_to_rois(boxes: Tensor, batch_indices: Tensor, image_size: int) -> Tensor:
    if boxes.numel() == 0:
        return boxes.new_zeros((0, 5))
    xyxy = boxes.clone()
    xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * image_size
    xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * image_size
    xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * image_size
    xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * image_size
    return torch.cat([batch_indices.float().unsqueeze(1), xyxy], dim=1)


@torch.no_grad()
def validate(model: DINOFeatureTeacher, loader: DataLoader, device: torch.device, image_size: int) -> float:
    model.eval(); correct = total = 0
    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        classes = batch["classes"].to(device)
        if not len(classes):
            continue
        rois = xywhn_to_rois(batch["boxes"].to(device), batch["batch_indices"].to(device), image_size)
        logits = model(images, rois)["roi_logits"]
        correct += int((logits.argmax(1) == classes).sum())
        total += len(classes)
    return correct / max(total, 1)


def capture_rng_state(loader_generator: torch.Generator) -> dict:
    """Everything needed to make the *next* epoch match an uninterrupted run."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "loader_generator": loader_generator.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict, loader_generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    loader_generator.set_state(state["loader_generator"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the DINOv3-SAT object-level feature teacher.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--resume", nargs="?", const="auto", default=None, help="Resume the full epoch checkpoint. Omit a value to use runs/teacher/last.pt.")
    args = parser.parse_args()
    cfg = load_config(args.config); data = resolve_data_yaml(cfg)
    root = Path(cfg["paths"]["project_root"]); output = root / "runs" / "teacher"
    output.mkdir(parents=True, exist_ok=True)
    image_size = cfg["dataset"]["image_size"]
    tcfg = cfg["teacher"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = int(cfg["student"].get("seed", 0))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    loader_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(YoloRoiDataset(data, "train", image_size), batch_size=tcfg["batch"], shuffle=True, generator=loader_generator, num_workers=tcfg["num_workers"], pin_memory=True, collate_fn=collate)
    val_loader = DataLoader(YoloRoiDataset(data, "val", image_size), batch_size=tcfg["batch"], shuffle=False, num_workers=tcfg["num_workers"], pin_memory=True, collate_fn=collate)
    model = DINOFeatureTeacher(cfg["paths"]["dino_repo"], cfg["paths"]["dino_weights"], tcfg["feature_channels"], tcfg["roi_size"], cfg["dataset"]["nc"]).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=tcfg["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs or tcfg["epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    target_epochs = args.epochs or tcfg["epochs"]
    start_epoch, best_acc = 0, -1.0
    if args.resume:
        resume_path = output / "last.pt" if args.resume == "auto" else Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Teacher resume checkpoint not found: {resume_path}")
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        required = {"model", "optimizer", "scheduler", "scaler", "epoch", "best_acc", "rng_state"}
        missing = required.difference(checkpoint)
        if missing:
            raise RuntimeError(f"{resume_path} is not a full resumable teacher checkpoint; missing {sorted(missing)}")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"]); scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"]); restore_rng_state(checkpoint["rng_state"], loader_generator)
        start_epoch, best_acc = int(checkpoint["epoch"]), float(checkpoint["best_acc"])
        if start_epoch >= target_epochs:
            raise RuntimeError(f"Teacher checkpoint already completed {start_epoch}/{target_epochs} epochs. Start a new run instead.")
        print(f"Resuming teacher exactly at epoch {start_epoch + 1}/{target_epochs} from {resume_path}")
    elif (output / "last.pt").exists():
        raise FileExistsError(f"{output / 'last.pt'} already exists. Use --resume to continue it, or choose a new project root.")
    for epoch in range(start_epoch, target_epochs):
        model.train(); model.backbone.eval(); optimizer.zero_grad(set_to_none=True)
        running = seen = 0.0
        for step, batch in enumerate(tqdm(train_loader, desc=f"teacher {epoch + 1}"), start=1):
            if not len(batch["classes"]):
                continue
            images = batch["images"].to(device, non_blocking=True)
            classes = batch["classes"].to(device); boxes = batch["boxes"].to(device); indices = batch["batch_indices"].to(device)
            rois = xywhn_to_rois(boxes, indices, image_size)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images, rois)["roi_logits"]
                loss = torch.nn.functional.cross_entropy(logits, classes) / tcfg["accumulate"]
            scaler.scale(loss).backward()
            if step % tcfg["accumulate"] == 0:
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            running += float(loss.detach()) * tcfg["accumulate"]; seen += 1
        scheduler.step()
        accuracy = validate(model, val_loader, device, image_size)
        print(f"epoch={epoch + 1} train_ce={running / max(seen, 1):.4f} val_roi_acc={accuracy:.4f}")
        is_best = accuracy > best_acc
        if is_best:
            best_acc = accuracy
        state = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(), "rng_state": capture_rng_state(loader_generator), "epoch": epoch + 1,
            "best_acc": best_acc, "val_roi_acc": accuracy, "target_epochs": target_epochs, "config": cfg,
        }
        torch.save(state, output / "last.pt")
        if is_best:
            torch.save(state, output / "best.pt")
    print(f"saved teacher to {output / 'best.pt'}")


if __name__ == "__main__":
    main()
