from __future__ import annotations

from collections import OrderedDict
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torchvision.ops import roi_align

from .common import stable_image_key


def allow_retained_backward_through_compiled_ops() -> bool:
    """Keep AOTAutograd intermediates needed by KD's repeated gradient probes.

    Torchvision's deterministic CUDA RoIAlign fallback is lazily compiled even
    when Ultralytics ``compile=False``.  AOTAutograd normally donates saved
    buffers to its first backward, which makes the retained graph unusable by
    the subsequent KD calibration/probe and final optimizer backward.  Turning
    donation off preserves the graph without changing the loss or gradients.
    """
    try:
        from torch._functorch import config as functorch_config
    except (ImportError, AttributeError):
        return False
    if not hasattr(functorch_config, "donated_buffer"):
        return False
    functorch_config.donated_buffer = False
    return not bool(functorch_config.donated_buffer)


def _xywhn_to_xyxy(boxes: Tensor, width: int, height: int) -> Tensor:
    out = boxes.clone()
    out[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * width
    out[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * height
    out[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * width
    out[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * height
    return out


def _iou_xywhn(one: Tensor, many: Tensor) -> Tensor:
    if many.numel() == 0:
        return one.new_zeros((0,))
    a = _xywhn_to_xyxy(one.unsqueeze(0), 1, 1)[0]
    b = _xywhn_to_xyxy(many, 1, 1)
    x1 = torch.maximum(a[0], b[:, 0]); y1 = torch.maximum(a[1], b[:, 1])
    x2 = torch.minimum(a[2], b[:, 2]); y2 = torch.minimum(a[3], b[:, 3])
    inter = (x2 - x1).clamp_min(0) * (y2 - y1).clamp_min(0)
    area_a = (a[2] - a[0]).clamp_min(0) * (a[3] - a[1]).clamp_min(0)
    area_b = (b[:, 2] - b[:, 0]).clamp_min(0) * (b[:, 3] - b[:, 1]).clamp_min(0)
    return inter / (area_a + area_b - inter).clamp_min(1e-9)


class TeacherSignalStore:
    def __init__(self, cache_dir: str | Path, capacity: int = 64):
        self.cache_dir = Path(cache_dir)
        self.capacity = capacity
        self.items: OrderedDict[str, dict] = OrderedDict()
        self.requests = 0
        self.memory_hits = 0
        self.disk_hits = 0
        self.misses = 0

    def get(self, image_path: str) -> dict | None:
        self.requests += 1
        key = stable_image_key(image_path)
        if key in self.items:
            self.memory_hits += 1
            self.items.move_to_end(key)
            return self.items[key]
        file = self.cache_dir / f"{key}.pt"
        if not file.exists():
            self.misses += 1
            return None
        self.disk_hits += 1
        value = torch.load(file, map_location="cpu", weights_only=False)
        required = {"path", "boxes_xywhn", "classes", "roi_logits", "p3", "p4", "p5"}
        missing = required.difference(value)
        if missing:
            raise RuntimeError(f"KD health failure: cache entry {file} is missing {sorted(missing)}")
        if stable_image_key(value["path"]) != key:
            raise RuntimeError(f"KD health failure: cache entry {file} belongs to {value['path']}, not {image_path}")
        self.items[key] = value
        if len(self.items) > self.capacity:
            self.items.popitem(last=False)
        return value


def _train_relative_image(image_path: str | Path) -> str:
    """Return the portable path below images/train for a dataset image."""
    parts = list(Path(image_path).parts)
    lowered = [part.lower() for part in parts]
    for index in range(len(parts) - 1):
        if lowered[index] == "images" and lowered[index + 1] == "train":
            return Path(*parts[index + 2:]).as_posix()
    raise RuntimeError(f"KD hard-example manifest received a non-TRAIN image: {image_path}")


class HardExampleStore:
    """Read-only, relocation-safe TRAIN error labels used to redistribute KD."""

    def __init__(self, manifest_path: str | Path | None):
        self.path = Path(manifest_path).resolve() if manifest_path else None
        self.images: dict[str, dict] = {}
        if self.path is None:
            return
        if not self.path.exists():
            raise FileNotFoundError(f"KD hard-example manifest missing: {self.path}")
        try:
            manifest = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Invalid KD hard-example manifest {self.path}: {error}") from error
        if manifest.get("format") != 1 or manifest.get("split") != "train":
            raise RuntimeError(
                f"KD hard-example manifest must be format=1 and split='train', got "
                f"format={manifest.get('format')!r}, split={manifest.get('split')!r}."
            )
        self.images = manifest.get("images", {})
        if not isinstance(self.images, dict) or not self.images:
            raise RuntimeError(f"KD hard-example manifest {self.path} contains no images.")

    @property
    def enabled(self) -> bool:
        return self.path is not None

    def object(
        self,
        image_path: str | Path,
        teacher_index: int,
        fine_class: int,
        teacher_box_xywhn: Tensor | None = None,
    ) -> dict | None:
        if not self.enabled:
            return None
        entry = self.images.get(_train_relative_image(image_path))
        if entry is None:
            raise RuntimeError(f"KD hard-example manifest has no TRAIN entry for {image_path}")
        objects = entry.get("objects", [])
        direct = next((item for item in objects if int(item["gt_index"]) == teacher_index), None)
        if direct is not None and int(direct["fine_class"]) == fine_class:
            if teacher_box_xywhn is None or "box_xywhn" not in direct:
                return direct
            direct_iou = _iou_xywhn(
                teacher_box_xywhn.detach().cpu().float(),
                torch.tensor([direct["box_xywhn"]], dtype=torch.float32),
            )
            if len(direct_iou) and float(direct_iou[0]) >= 0.999:
                return direct
        # Duplicate-label removal can shift every following index. Match in
        # the deterministic teacher letterbox coordinate system before using
        # the unique-class fallback; never assign an ambiguous hard weight.
        candidates = [item for item in objects if int(item["fine_class"]) == fine_class]
        if teacher_box_xywhn is not None and candidates and all("box_xywhn" in item for item in candidates):
            candidate_boxes = torch.tensor([item["box_xywhn"] for item in candidates], dtype=torch.float32)
            overlaps = _iou_xywhn(teacher_box_xywhn.detach().cpu().float(), candidate_boxes)
            best = int(torch.argmax(overlaps))
            if float(overlaps[best]) >= 0.999:
                return candidates[best]
        if len(candidates) == 1:
            return candidates[0]
        raise RuntimeError(
            f"KD hard-example object mismatch for {image_path}: index={teacher_index}, "
            f"class={fine_class}, candidates={len(candidates)}"
        )


class VehicleNegativeStore:
    """Portable TRAIN-only vehicle-background boxes mined from a fixed detector."""

    def __init__(self, manifest_path: str | Path | None):
        self.path = Path(manifest_path).resolve() if manifest_path else None
        self.images: dict[str, dict] = {}
        if self.path is None:
            return
        if not self.path.exists():
            raise FileNotFoundError(f"Vehicle-negative manifest missing: {self.path}")
        try:
            manifest = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Invalid vehicle-negative manifest {self.path}: {error}") from error
        if (
            manifest.get("format") != 1
            or manifest.get("kind") != "vehicle_background"
            or manifest.get("split") != "train"
        ):
            raise RuntimeError(
                "Vehicle-negative manifest must be format=1, kind='vehicle_background', split='train'."
            )
        self.images = manifest.get("images", {})
        if not isinstance(self.images, dict) or not self.images:
            raise RuntimeError(f"Vehicle-negative manifest {self.path} contains no images.")
        for relative, entry in self.images.items():
            relative_path = Path(relative)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise RuntimeError(f"Vehicle-negative manifest contains an unsafe TRAIN path: {relative!r}")
            boxes = entry.get("boxes", []) if isinstance(entry, dict) else []
            if not boxes:
                raise RuntimeError(f"Vehicle-negative manifest image {relative!r} contains no boxes.")
            for item in boxes:
                box = item.get("box_xywhn", [])
                score = float(item.get("score", -1.0))
                if len(box) != 4 or not all(math.isfinite(float(value)) for value in box):
                    raise RuntimeError(f"Vehicle-negative manifest image {relative!r} has an invalid box.")
                if not 0.0 <= score <= 1.0:
                    raise RuntimeError(f"Vehicle-negative manifest image {relative!r} has an invalid score.")

    @property
    def enabled(self) -> bool:
        return self.path is not None

    def boxes(self, image_path: str | Path) -> list[dict]:
        if not self.enabled:
            return []
        return list(self.images.get(_train_relative_image(image_path), {}).get("boxes", []))


class StudentDistillAddons(nn.Module):
    """Training-only modules. Attached to YOLO before optimizer construction."""
    def __init__(
        self,
        student_channels: list[int],
        teacher_channels: int,
        num_classes: int,
        enable_vehicle_bg: bool = False,
    ):
        super().__init__()
        self.projectors = nn.ModuleList([
            nn.Sequential(nn.Conv2d(channels, teacher_channels, 1, bias=False), nn.GroupNorm(16, teacher_channels), nn.SiLU(), nn.Conv2d(teacher_channels, teacher_channels, 1))
            for channels in student_channels
        ])
        self.student_roi_head = nn.Sequential(
            nn.Flatten(), nn.Linear(student_channels[0] * 7 * 7, 512), nn.SiLU(), nn.Dropout(0.1), nn.Linear(512, num_classes)
        )
        self.vehicle_bg_head = (
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(student_channels[0], 128),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 1),
            )
            if enable_vehicle_bg else None
        )
        # Persistent training-only calibration state. Keeping it in the attached
        # module makes an Ultralytics last.pt resume reproduce the same KD
        # coefficients instead of silently recalibrating from a later epoch.
        self.register_buffer("feature_kd_weight", torch.tensor(1.0))
        self.register_buffer("cls_kd_weight", torch.tensor(1.0))
        self.register_buffer("feature_kd_log_sum", torch.tensor(0.0))
        self.register_buffer("cls_kd_log_sum", torch.tensor(0.0))
        self.register_buffer("feature_kd_calibration_count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("cls_kd_calibration_count", torch.tensor(0, dtype=torch.long))
        if enable_vehicle_bg:
            self.register_buffer("vehicle_bg_kd_weight", torch.tensor(1.0))
            self.register_buffer("vehicle_bg_kd_log_sum", torch.tensor(0.0))
            self.register_buffer("vehicle_bg_kd_calibration_count", torch.tensor(0, dtype=torch.long))


class DistillationLoss:
    def __init__(self, original_loss, addons: StudentDistillAddons, signal_store: TeacherSignalStore, cfg: dict):
        self.aot_donated_buffers_disabled = allow_retained_backward_through_compiled_ops()
        self.original_loss = original_loss
        self.addons = addons
        self.signal_store = signal_store
        self.cfg = cfg
        dcfg = cfg["distillation"]
        manifest_path = dcfg.get("hard_example_manifest")
        if manifest_path and not Path(manifest_path).is_absolute():
            manifest_path = Path(cfg["paths"]["project_root"]) / manifest_path
        self.hard_examples = HardExampleStore(manifest_path)
        negative_manifest = dcfg.get("vehicle_negative_manifest")
        if negative_manifest and not Path(negative_manifest).is_absolute():
            negative_manifest = Path(cfg["paths"]["project_root"]) / negative_manifest
        self.vehicle_negatives = VehicleNegativeStore(negative_manifest)
        self.vehicle_bg_enabled = bool(dcfg.get("vehicle_bg_enabled", False))
        if self.vehicle_bg_enabled and not self.vehicle_negatives.enabled:
            raise RuntimeError("vehicle_bg_enabled requires a TRAIN vehicle_negative_manifest.")
        if self.vehicle_bg_enabled and self.addons.vehicle_bg_head is None:
            raise RuntimeError("vehicle_bg_enabled but StudentDistillAddons has no vehicle_bg_head.")
        if self.vehicle_bg_enabled and dcfg.get("weighting_mode") != "gradient_calibrated":
            raise RuntimeError("vehicle background auxiliary loss requires weighting_mode='gradient_calibrated'.")
        self.fine_to_group = {
            int(fine_class): group
            for group, fine_classes in cfg["dataset"].get("class_groups", {}).items()
            for fine_class in fine_classes
        }
        hierarchical = dcfg.get("hierarchical_kl", {})
        if bool(hierarchical.get("enabled", False)):
            expected = set(range(int(cfg["dataset"]["nc"])))
            actual = set(self.fine_to_group)
            if actual != expected:
                raise RuntimeError(
                    "hierarchical_kl requires class_groups to cover every fine class exactly once; "
                    f"missing={sorted(expected - actual)} extra={sorted(actual - expected)}"
                )
        self.features: list[Tensor] = []
        self.epoch = 0
        self.batch_calls = 0
        self.zero_streak = {"feature": 0, "cls": 0}
        self.grad_events = {"feature_projector": 0, "roi_head": 0, "vehicle_bg_head": 0}
        self.grad_norm_sum = {"feature_projector": 0.0, "roi_head": 0.0, "vehicle_bg_head": 0.0}
        self.epoch_totals = {
            "batches": 0, "feature_raw_sum": 0.0, "cls_raw_sum": 0.0, "vehicle_bg_raw_sum": 0.0, "kd_sum": 0.0,
            "feature_nonzero_batches": 0, "cls_nonzero_batches": 0, "vehicle_bg_nonzero_batches": 0, "valid_rois": 0,
            "vehicle_bg_positive_rois": 0, "vehicle_bg_negative_rois": 0,
            "teacher_candidates": 0, "teacher_kept": 0, "teacher_confidence_sum": 0.0,
        }
        self.gradient_probe = {
            pair: {"count": 0, "negative": 0, "strong_negative": 0, "cosine_sum": 0.0}
            for pair in (
                "det_feature", "det_cls", "feature_cls", "det_vehicle_bg",
                "feature_vehicle_bg", "cls_vehicle_bg",
            )
        }
        self.gradient_norm_sum = {name: 0.0 for name in ("det", "feature", "cls", "vehicle_bg")}
        self.gradient_norm_count = {name: 0 for name in ("det", "feature", "cls", "vehicle_bg")}
        self.last = {
            "feature_raw": 0.0, "cls_raw": 0.0, "vehicle_bg_raw": 0.0, "kd": 0.0, "valid_rois": 0,
            "feature_weight": 0.0, "cls_weight": 0.0, "vehicle_bg_weight": 0.0,
        }

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    @staticmethod
    def prediction_features(preds: Any) -> list[Tensor]:
        """Ultralytics 8.4.90 exposes the three Detect inputs in preds['feats']."""
        if isinstance(preds, dict) and isinstance(preds.get("feats"), (list, tuple)):
            return [item for item in preds["feats"] if isinstance(item, Tensor) and item.ndim == 4][-3:]
        return []

    def _schedule(self, branch: str) -> float:
        """Epoch-level warmup/hold/cosine-decay schedule for one KD branch.

        Old configurations retain their original monotonic warmup. Direction-1
        configs can stop KD early so it cannot keep pulling an already converged
        detector away from its localization optimum.
        """
        # A health check never updates weights, so force the selected branch
        # on and verify the exact final-backward path even when its production
        # schedule intentionally starts in a later epoch.
        experiment = self.cfg["runtime"]["experiment"]
        health_check = self.cfg["runtime"].get("health_batches") is not None
        branch_enabled = (
            (branch == "feature" and experiment in {"f", "fk", "e1", "e3"})
            or (branch == "cls" and experiment in {"k", "fk", "e2", "e3"})
            or (branch == "vehicle_bg" and self.vehicle_bg_enabled)
        )
        if not branch_enabled:
            return 0.0
        if health_check and branch_enabled:
            return 1.0
        schedule = self.cfg["distillation"].get(f"{branch}_schedule")
        if not schedule:
            return min(1.0, (self.epoch + 1) / max(1, self.cfg["student"]["kd_warmup_epochs"]))
        start = int(schedule.get("start_epoch", 0))
        warmup = max(0, int(schedule.get("warmup_epochs", 0)))
        hold = max(0, int(schedule.get("hold_epochs", 0)))
        decay = max(0, int(schedule.get("decay_epochs", 0)))
        local_epoch = self.epoch - start
        if local_epoch < 0:
            return 0.0
        if warmup and local_epoch < warmup:
            return float(local_epoch + 1) / float(warmup)
        local_epoch -= warmup
        if local_epoch < hold:
            return 1.0
        local_epoch -= hold
        if decay and local_epoch < decay:
            progress = float(local_epoch + 1) / float(decay)
            return 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)).item())
        return 0.0

    def _feature_mask(
        self,
        boxes: Tensor,
        batch_indices: Tensor,
        object_weights: Tensor,
        h: int,
        w: int,
        batch: int,
        device: torch.device,
    ) -> Tensor:
        dcfg = self.cfg["distillation"]
        mask = torch.full((batch, 1, h, w), float(dcfg["background_weight"]), device=device)
        for box, b, object_weight in zip(boxes, batch_indices, object_weights):
            x, y, bw, bh = box
            x1 = int(torch.floor((x - bw / 2) * w).clamp(0, w - 1)); x2 = int(torch.ceil((x + bw / 2) * w).clamp(1, w))
            y1 = int(torch.floor((y - bh / 2) * h).clamp(0, h - 1)); y2 = int(torch.ceil((y + bh / 2) * h).clamp(1, h))
            foreground = float(dcfg["foreground_weight"]) * object_weight
            boundary = float(dcfg["boundary_weight"]) * object_weight
            mask[int(b), :, y1:y2, x1:x2] = torch.maximum(mask[int(b), :, y1:y2, x1:x2], foreground)
            # One-cell context ring gets an intermediate value for tiny-object boundaries.
            mask[int(b), :, max(0, y1 - 1):min(h, y2 + 1), max(0, x1 - 1):min(w, x2 + 1)] = torch.maximum(
                mask[int(b), :, max(0, y1 - 1):min(h, y2 + 1), max(0, x1 - 1):min(w, x2 + 1)],
                boundary,
            )
            mask[int(b), :, y1:y2, x1:x2] = torch.maximum(mask[int(b), :, y1:y2, x1:x2], foreground)
        return mask

    def _match_signal_object(self, box: Tensor, fine_class: int, batch_index: int, signals: list[dict]) -> int | None:
        signal = signals[batch_index] if batch_index < len(signals) else None
        if signal is None or not len(signal["classes"]):
            return None
        target_classes = signal["classes"].to(dtype=torch.long)
        candidates = torch.where(target_classes == fine_class)[0]
        if not len(candidates):
            return None
        overlaps = _iou_xywhn(box.detach().cpu(), signal["boxes_xywhn"].float()[candidates])
        best_position = torch.argmax(overlaps)
        if float(overlaps[best_position]) < float(self.cfg["distillation"].get("roi_match_iou_floor", 0.5)):
            return None
        return int(candidates[best_position])

    def _object_weight(
        self,
        image_path: str,
        teacher_index: int,
        fine_class: int,
        branch: str,
        teacher_box_xywhn: Tensor | None = None,
    ) -> float:
        dcfg = self.cfg["distillation"]
        group = self.fine_to_group.get(fine_class)
        group_weights = dcfg.get("group_distill_weights", {})
        weight = float(group_weights.get(group, 1.0))
        hard_object = self.hard_examples.object(image_path, teacher_index, fine_class, teacher_box_xywhn)
        if hard_object is not None:
            error_weights = dcfg.get(f"{branch}_error_weights", {})
            weight *= float(error_weights.get(hard_object.get("error_type"), 1.0))
            if bool(hard_object.get("crowded")):
                weight *= float(dcfg.get(f"{branch}_crowded_multiplier", 1.0))
            if bool(hard_object.get("edge")):
                weight *= float(dcfg.get(f"{branch}_edge_multiplier", 1.0))
            if hard_object.get("size") == "small":
                weight *= float(dcfg.get(f"{branch}_small_multiplier", 1.0))
        lower, upper = dcfg.get("object_weight_bounds", [0.25, 4.0])
        return min(max(weight, float(lower)), float(upper))

    def _batch_object_weights(self, batch: dict, signals: list[dict], branch: str) -> Tensor:
        device = self.features[0].device
        boxes = batch["bboxes"].to(device)
        dcfg = self.cfg["distillation"]
        if not dcfg.get("group_distill_weights") and not self.hard_examples.enabled:
            return torch.ones(len(boxes), device=device, dtype=torch.float32)
        classes = batch["cls"].view(-1).long().to(device)
        batch_indices = batch["batch_idx"].long().to(device)
        image_files = list(batch.get("im_file", [""] * int(batch["img"].shape[0])))
        values = []
        for box, fine_class, batch_index in zip(boxes, classes, batch_indices):
            index = self._match_signal_object(box, int(fine_class), int(batch_index), signals)
            if index is None:
                raise RuntimeError(
                    f"KD object weighting could not match class={int(fine_class)} in {image_files[int(batch_index)]}"
                )
            teacher_box = signals[int(batch_index)]["boxes_xywhn"][index]
            values.append(self._object_weight(image_files[int(batch_index)], index, int(fine_class), branch, teacher_box))
        weights = torch.tensor(values, device=device, dtype=torch.float32)
        # Only redistribute a branch's fixed gradient budget. A normalized mean
        # of one prevents class/hardness weighting from silently strengthening KD.
        return weights / weights.mean().clamp_min(1e-8) if len(weights) else weights

    @staticmethod
    def _teacher_feature_batch(signals: list[dict], level: str, device: torch.device) -> Tensor:
        return torch.stack([signal[level] for signal in signals]).to(device=device, dtype=torch.float32, non_blocking=True)

    def feature_loss(self, batch: dict, signals: list[dict]) -> Tensor:
        if len(self.features) < 3:
            raise RuntimeError("KD health failure: Ultralytics predictions did not expose three Detect feature maps.")
        names = ("p3", "p4", "p5"); weights = self.cfg["distillation"]["feature_scale_weights"]
        boxes = batch["bboxes"].to(self.features[0].device); batch_indices = batch["batch_idx"].long().to(self.features[0].device)
        object_weights = self._batch_object_weights(batch, signals, "feature")
        # The projection branch must stay FP32 even inside Ultralytics AMP.
        # Its normalized MSE needs a power-of-two pre-calibration scale to keep
        # gradients representable at the FP16 student-feature boundary; running
        # the projector itself in FP16 would then overflow its GradScaler-scaled
        # internal backward before the shared gradient reaches that boundary.
        total = torch.zeros((), device=self.features[0].device, dtype=torch.float32)
        for student, projector, name, weight in zip(self.features, self.addons.projectors, names, weights):
            with torch.autocast(device_type=student.device.type, enabled=False):
                teacher = self._teacher_feature_batch(signals, name, student.device)
                projected = F.normalize(projector(student.float()), dim=1)
                if teacher.shape != projected.shape:
                    raise RuntimeError(f"KD health failure: {name} teacher shape {tuple(teacher.shape)} does not match projected student {tuple(projected.shape)}")
                teacher = F.normalize(teacher, dim=1)
                mask = self._feature_mask(boxes, batch_indices, object_weights, student.shape[-2], student.shape[-1], student.shape[0], student.device)
                total = total + float(weight) * (((projected - teacher) ** 2) * mask).sum() / (mask.sum() * projected.shape[1]).clamp_min(1.0)
        # This is a numerical re-parameterization, not a stronger objective.
        # The normalized multi-scale mean produces ~1e-7 gradients at the
        # FP16 Detect inputs, which can underflow before gradient calibration.
        # Calibration inversely adjusts the final weight, preserving the
        # requested shared-gradient ratio while keeping the raw gradient finite.
        return total * float(self.cfg["distillation"].get("feature_numerical_scale", 1.0))

    def _hierarchical_kl(
        self,
        student_logits: Tensor,
        teacher_logits: Tensor,
        gt_classes: Tensor,
        temperature: float,
    ) -> Tensor:
        """Competition-aware 25-fine/3-coarse KL for each RoI.

        The coarse term protects ship/aircraft/vehicle separation. The
        conditional term transfers only relations within the GT coarse group,
        so saturated aircraft fine classes cannot dominate ship/vehicle errors.
        """
        settings = self.cfg["distillation"].get("hierarchical_kl", {})
        student_prob = (student_logits / temperature).softmax(dim=1)
        teacher_prob = (teacher_logits / temperature).softmax(dim=1)
        groups = list(self.cfg["dataset"]["class_groups"].items())
        student_coarse = torch.stack(
            [student_prob[:, [int(value) for value in fine_ids]].sum(dim=1) for _, fine_ids in groups], dim=1
        )
        teacher_coarse = torch.stack(
            [teacher_prob[:, [int(value) for value in fine_ids]].sum(dim=1) for _, fine_ids in groups], dim=1
        )
        coarse = (
            teacher_coarse
            * (teacher_coarse.clamp_min(1e-9).log() - student_coarse.clamp_min(1e-9).log())
        ).sum(dim=1)
        conditional = torch.zeros_like(coarse)
        for _, fine_ids in groups:
            indices = torch.tensor([int(value) for value in fine_ids], device=student_logits.device, dtype=torch.long)
            item_mask = torch.isin(gt_classes, indices)
            if not bool(item_mask.any()):
                continue
            teacher_within = teacher_prob[item_mask][:, indices]
            student_within = student_prob[item_mask][:, indices]
            teacher_within = teacher_within / teacher_within.sum(dim=1, keepdim=True).clamp_min(1e-9)
            student_within = student_within / student_within.sum(dim=1, keepdim=True).clamp_min(1e-9)
            conditional[item_mask] = (
                teacher_within
                * (teacher_within.clamp_min(1e-9).log() - student_within.clamp_min(1e-9).log())
            ).sum(dim=1)
        coarse_weight = float(settings.get("coarse_weight", 0.6))
        within_weight = float(settings.get("within_group_weight", 0.4))
        if coarse_weight < 0 or within_weight < 0 or coarse_weight + within_weight <= 0:
            raise ValueError("hierarchical_kl weights must be non-negative and not both zero.")
        scale = temperature ** 2 / (coarse_weight + within_weight)
        return (coarse_weight * coarse + within_weight * conditional) * scale

    def cls_loss(self, batch: dict, signals: list[dict]) -> tuple[Tensor, int, dict[str, float | int]]:
        if not self.features:
            raise RuntimeError("KD health failure: P3 feature is unavailable for classification KL.")
        p3 = self.features[0]
        boxes = batch["bboxes"].to(p3.device); classes = batch["cls"].view(-1).long().to(p3.device); batch_indices = batch["batch_idx"].long().to(p3.device)
        rois, targets, target_gt_classes, distill_weights = [], [], [], []
        image_files = list(batch.get("im_file", [""] * int(p3.shape[0])))
        for box, cls, batch_index in zip(boxes, classes, batch_indices):
            signal = signals[int(batch_index)] if int(batch_index) < len(signals) else None
            best = self._match_signal_object(box, int(cls), int(batch_index), signals)
            if signal is None or best is None:
                continue
            teacher_logit = signal["roi_logits"][best].to(p3.device, dtype=torch.float32)
            x1, y1, x2, y2 = _xywhn_to_xyxy(box.unsqueeze(0), p3.shape[-1], p3.shape[-2])[0]
            rois.append(torch.stack([batch_index.float(), x1, y1, x2, y2]))
            targets.append(teacher_logit)
            target_gt_classes.append(cls)
            distill_weights.append(
                self._object_weight(
                    image_files[int(batch_index)], best, int(cls), "cls", signal["boxes_xywhn"][best]
                )
            )
        if not rois:
            return p3.new_zeros(()), 0, {"candidates": 0, "kept": 0, "confidence_sum": 0.0}
        student_logits = self.addons.student_roi_head(roi_align(p3.float(), torch.stack(rois), output_size=7, spatial_scale=1.0, sampling_ratio=2, aligned=True))
        teacher_logits = torch.stack(targets)
        temperature = float(self.cfg["distillation"]["temperature"])
        # Reliability must be measured on the teacher's native distribution.
        # Measuring it after temperature softening made the old 0.05 threshold
        # nearly equivalent to accepting a uniform 25-class prediction.
        reliability_prob = teacher_logits.softmax(dim=1)
        confidence = reliability_prob.max(dim=1).values
        normalized_entropy = -(reliability_prob.clamp_min(1e-9).log() * reliability_prob).sum(dim=1) / torch.log(
            reliability_prob.new_tensor(float(reliability_prob.shape[1]))
        )
        teacher_prob = (teacher_logits / temperature).softmax(dim=1)
        keep = confidence >= float(self.cfg["distillation"]["teacher_confidence_floor"])
        entropy_ceiling = self.cfg["distillation"].get("teacher_entropy_ceiling")
        if entropy_ceiling is not None:
            keep &= normalized_entropy <= float(entropy_ceiling)
        gt_classes = torch.stack(target_gt_classes).long().to(teacher_logits.device)
        if bool(self.cfg["distillation"].get("require_teacher_correct", False)):
            correctness = self.cfg["distillation"].get("teacher_correctness", "fine")
            predictions = reliability_prob.argmax(dim=1)
            if correctness == "fine":
                keep &= predictions == gt_classes
            elif correctness == "coarse":
                predicted_groups = torch.tensor(
                    [self.fine_to_group[int(value)] == self.fine_to_group[int(truth)] for value, truth in zip(predictions, gt_classes)],
                    device=teacher_logits.device,
                    dtype=torch.bool,
                )
                keep &= predicted_groups
            else:
                raise ValueError("teacher_correctness must be 'fine' or 'coarse'.")
        stats = {
            "candidates": int(len(rois)),
            "kept": int(keep.sum()),
            "confidence_sum": float(confidence[keep].sum()) if bool(keep.any()) else 0.0,
        }
        if not bool(keep.any()):
            return p3.new_zeros(()), 0, stats
        if bool(self.cfg["distillation"].get("hierarchical_kl", {}).get("enabled", False)):
            per_item = self._hierarchical_kl(
                student_logits[keep], teacher_logits[keep], gt_classes[keep], temperature
            )
        else:
            per_item = F.kl_div(
                (student_logits[keep] / temperature).log_softmax(dim=1),
                teacher_prob[keep], reduction="none",
            ).sum(dim=1) * (temperature ** 2)
        kept_weights = torch.tensor(distill_weights, device=p3.device, dtype=torch.float32)[keep]
        kept_weights = kept_weights / kept_weights.mean().clamp_min(1e-8)
        return (per_item * confidence[keep].square() * kept_weights).mean(), int(keep.sum()), stats

    def vehicle_background_loss(self, batch: dict) -> tuple[Tensor, int, int]:
        """Separate real GT foreground from mined vehicle-like background RoIs.

        This head and loss exist only during training. Positives are all GT
        objects in a batch, while negatives are strictly TRAIN-only vehicle
        predictions classified as background (never localization/duplicates).
        The two sides are averaged independently so abundant aircraft positives
        cannot dilute the sparse hard negatives.
        """
        if not self.vehicle_bg_enabled:
            return self.features[0].new_zeros(()), 0, 0
        p3 = self.features[0]
        assert self.addons.vehicle_bg_head is not None
        boxes = batch["bboxes"].to(p3.device)
        batch_indices = batch["batch_idx"].long().to(p3.device)
        image_files = list(batch["im_file"])
        max_positives = max(1, int(self.cfg["distillation"].get("vehicle_bg_max_positive_rois", 64)))
        max_negatives = max(1, int(self.cfg["distillation"].get("vehicle_bg_max_negatives_per_image", 4)))

        positive_rois = []
        for box, batch_index in list(zip(boxes, batch_indices))[:max_positives]:
            x1, y1, x2, y2 = _xywhn_to_xyxy(box.unsqueeze(0), p3.shape[-1], p3.shape[-2])[0]
            positive_rois.append(torch.stack([batch_index.float(), x1, y1, x2, y2]))

        negative_rois, negative_scores = [], []
        for batch_index, image_file in enumerate(image_files):
            for item in self.vehicle_negatives.boxes(image_file)[:max_negatives]:
                box = torch.tensor(item["box_xywhn"], device=p3.device, dtype=torch.float32)
                x1, y1, x2, y2 = _xywhn_to_xyxy(box.unsqueeze(0), p3.shape[-1], p3.shape[-2])[0]
                negative_rois.append(torch.stack([box.new_tensor(float(batch_index)), x1, y1, x2, y2]))
                negative_scores.append(float(item["score"]))

        # Do not turn ordinary batches into a generic objectness objective. The
        # auxiliary branch is activated only when a mined vehicle background is
        # present; replay sampling guarantees enough such batches per epoch.
        if not positive_rois or not negative_rois:
            return p3.new_zeros(()), len(positive_rois), len(negative_rois)
        rois = torch.stack([*positive_rois, *negative_rois])
        pooled = roi_align(
            p3.float(), rois, output_size=7, spatial_scale=1.0,
            sampling_ratio=2, aligned=True,
        )
        logits = self.addons.vehicle_bg_head(pooled).flatten()
        positive_count = len(positive_rois)
        positive_loss = F.binary_cross_entropy_with_logits(
            logits[:positive_count], torch.ones_like(logits[:positive_count]), reduction="mean"
        )
        negative_logits = logits[positive_count:]
        negative_per_item = F.binary_cross_entropy_with_logits(
            negative_logits, torch.zeros_like(negative_logits), reduction="none"
        )
        score_weights = torch.tensor(negative_scores, device=p3.device, dtype=torch.float32)
        score_power = float(self.cfg["distillation"].get("vehicle_bg_score_power", 1.0))
        score_weights = score_weights.clamp_min(1e-6).pow(score_power)
        score_weights = score_weights / score_weights.mean().clamp_min(1e-8)
        negative_loss = (negative_per_item * score_weights).mean()
        return 0.5 * (positive_loss + negative_loss), positive_count, len(negative_rois)

    @staticmethod
    def _budgeted(raw: Tensor, detection_loss: Tensor, ratio: float, warmup: float) -> Tensor:
        if not torch.isfinite(raw) or raw.detach().item() <= 0:
            return raw.new_zeros(())
        # This detached scale gives the requested relative loss contribution without changing the gradient direction.
        scale = ratio * detection_loss.detach().abs() / raw.detach().clamp_min(1e-8)
        return raw * scale * warmup

    @staticmethod
    def _gradients(loss: Tensor, features: list[Tensor]) -> tuple[Tensor | None, ...]:
        if not loss.requires_grad or not bool(torch.isfinite(loss)) or float(loss.detach()) <= 0:
            return tuple(None for _ in features)
        return torch.autograd.grad(loss, features, retain_graph=True, allow_unused=True)

    @staticmethod
    def _gradient_norm(gradients: tuple[Tensor | None, ...]) -> Tensor:
        values = [gradient.detach().float().square().sum() for gradient in gradients if gradient is not None]
        if not values:
            return torch.tensor(0.0)
        return torch.stack(values).sum().sqrt()

    @staticmethod
    def _gradient_cosine(left: tuple[Tensor | None, ...], right: tuple[Tensor | None, ...]) -> float | None:
        products, left_norms, right_norms = [], [], []
        for a, b in zip(left, right):
            if a is None or b is None:
                continue
            a = a.detach().float(); b = b.detach().float()
            products.append((a * b).sum())
            left_norms.append(a.square().sum()); right_norms.append(b.square().sum())
        if not products:
            return None
        denominator = torch.stack(left_norms).sum().sqrt() * torch.stack(right_norms).sum().sqrt()
        if float(denominator) <= 1e-12:
            return None
        return float(torch.stack(products).sum() / denominator)

    def _record_gradient_probe(
        self,
        det_gradients: tuple[Tensor | None, ...],
        feature_gradients: tuple[Tensor | None, ...],
        cls_gradients: tuple[Tensor | None, ...],
        vehicle_bg_gradients: tuple[Tensor | None, ...],
    ) -> None:
        gradient_sets = {
            "det": det_gradients,
            "feature": feature_gradients,
            "cls": cls_gradients,
            "vehicle_bg": vehicle_bg_gradients,
        }
        for name, gradients in gradient_sets.items():
            norm = float(self._gradient_norm(gradients))
            if norm > 0:
                self.gradient_norm_sum[name] += norm
                self.gradient_norm_count[name] += 1
        for pair, left_name, right_name in (
            ("det_feature", "det", "feature"),
            ("det_cls", "det", "cls"),
            ("feature_cls", "feature", "cls"),
            ("det_vehicle_bg", "det", "vehicle_bg"),
            ("feature_vehicle_bg", "feature", "vehicle_bg"),
            ("cls_vehicle_bg", "cls", "vehicle_bg"),
        ):
            cosine = self._gradient_cosine(gradient_sets[left_name], gradient_sets[right_name])
            if cosine is None:
                continue
            stats = self.gradient_probe[pair]
            stats["count"] += 1
            stats["negative"] += int(cosine < 0)
            threshold = abs(float(self.cfg["distillation"].get("gradient_conflict_threshold", 0.05)))
            stats["strong_negative"] += int(cosine < -threshold)
            stats["cosine_sum"] += cosine

    def _calibrated_weight(
        self,
        branch: str,
        detection_gradients: tuple[Tensor | None, ...],
        auxiliary_gradients: tuple[Tensor | None, ...],
    ) -> float:
        dcfg = self.cfg["distillation"]
        count_buffer = getattr(self.addons, f"{branch}_kd_calibration_count")
        weight_buffer = getattr(self.addons, f"{branch}_kd_weight")
        log_sum_buffer = getattr(self.addons, f"{branch}_kd_log_sum")
        count = int(count_buffer)
        weight = float(weight_buffer)
        log_sum = float(log_sum_buffer)
        if count < 0 or not math.isfinite(weight) or weight <= 0 or not math.isfinite(log_sum):
            raise RuntimeError(
                f"KD health failure: invalid {branch} calibration state "
                f"(count={count}, weight={weight}, log_sum={log_sum})."
            )
        if count == 0 and abs(log_sum) > 1e-7:
            raise RuntimeError(
                f"KD health failure: inconsistent {branch} calibration state "
                f"(count=0 but log_sum={log_sum}); checkpoint EMA state was not synchronized."
            )
        calibration_batches = max(1, int(dcfg.get("calibration_batches", 128)))
        if count >= calibration_batches:
            return weight
        det_norm = float(self._gradient_norm(detection_gradients))
        auxiliary_norm = float(self._gradient_norm(auxiliary_gradients))
        if not math.isfinite(det_norm) or not math.isfinite(auxiliary_norm):
            raise RuntimeError(f"KD health failure: non-finite {branch} gradient norm during calibration.")
        if det_norm <= 0 or auxiliary_norm <= 0:
            return float(weight_buffer)
        target = float(dcfg.get(f"{branch}_gradient_ratio", 0.03 if branch == "feature" else 0.01))
        lower, upper = dcfg.get(f"{branch}_weight_bounds", [1e-6, 1e4])
        observed = min(max(target * det_norm / auxiliary_norm, float(lower)), float(upper))
        with torch.no_grad():
            log_sum_buffer.add_(torch.log(log_sum_buffer.new_tensor(observed)))
            count_buffer.add_(1)
            weight_buffer.copy_(torch.exp(log_sum_buffer / count_buffer.to(log_sum_buffer.dtype)))
        return float(weight_buffer)

    def _weighted_branch(
        self,
        branch: str,
        raw: Tensor,
        detection_loss: Tensor,
        detection_gradients: tuple[Tensor | None, ...],
        auxiliary_gradients: tuple[Tensor | None, ...],
        legacy_ratio: float | None = None,
    ) -> tuple[Tensor, float]:
        if not bool(torch.isfinite(raw)):
            raise RuntimeError(f"KD health failure: raw {branch} loss is NaN/Inf.")
        if float(raw.detach()) <= 0:
            return raw.new_zeros(()), 0.0
        dcfg = self.cfg["distillation"]
        mode = dcfg.get("weighting_mode", "legacy_relative")
        schedule = self._schedule(branch)
        if mode == "legacy_relative":
            ratio = float(dcfg[f"{branch}_budget_ratio"] if legacy_ratio is None else legacy_ratio)
            return self._budgeted(raw, detection_loss, ratio, schedule), ratio
        if mode == "fixed":
            weight = float(dcfg.get(f"{branch}_loss_weight", 1.0))
        elif mode == "gradient_calibrated":
            weight = self._calibrated_weight(branch, detection_gradients, auxiliary_gradients)
        else:
            raise ValueError(f"Unknown distillation weighting_mode={mode!r}")
        weighted = raw * weight * schedule
        if not math.isfinite(weight) or not math.isfinite(schedule) or not bool(torch.isfinite(weighted)):
            raise RuntimeError(
                f"KD health failure: weighted {branch} loss is NaN/Inf "
                f"(raw={float(raw.detach())}, weight={weight}, schedule={schedule})."
            )
        return weighted, weight * schedule

    def record_gradient(self, group: str, gradient: Tensor) -> Tensor:
        norm = float(gradient.detach().float().norm())
        if not math.isfinite(norm):
            raise RuntimeError(f"KD health failure: {group} received a NaN/Inf parameter gradient.")
        self.grad_events[group] += 1
        self.grad_norm_sum[group] += norm
        return gradient

    def assert_health(self, patience_batches: int) -> None:
        experiment = self.cfg["runtime"]["experiment"]
        if self.batch_calls < patience_batches:
            return
        feature_active = self._schedule("feature") > 0.0
        cls_active = self._schedule("cls") > 0.0
        vehicle_bg_active = self._schedule("vehicle_bg") > 0.0
        if experiment in {"f", "fk"} and self.zero_streak["feature"] >= patience_batches:
            raise RuntimeError(f"KD health failure: feature loss stayed zero for {patience_batches} consecutive batches.")
        if experiment in {"k", "fk"} and self.zero_streak["cls"] >= patience_batches:
            raise RuntimeError(f"KD health failure: classification KL stayed zero for {patience_batches} consecutive batches.")
        if feature_active and experiment in {"f", "fk"} and (self.grad_events["feature_projector"] == 0 or self.grad_norm_sum["feature_projector"] <= 0):
            raise RuntimeError("KD health failure: feature projectors received no gradient.")
        if cls_active and experiment in {"k", "fk"} and (self.grad_events["roi_head"] == 0 or self.grad_norm_sum["roi_head"] <= 0):
            raise RuntimeError("KD health failure: ROI classification head received no gradient.")
        vehicle_bg_patience = int(self.cfg["distillation"].get("vehicle_bg_health_patience_batches", 128))
        # ``batch_calls`` is cumulative across epochs, while sparse replay
        # health must wait for enough batches in the *current* active epoch.
        # Otherwise an intentionally disabled first epoch exhausts the waiting
        # period and epoch 2 can fail before its first replay image is sampled.
        current_epoch_batches = int(self.epoch_totals["batches"])
        if self.vehicle_bg_enabled and vehicle_bg_active and current_epoch_batches >= vehicle_bg_patience:
            if self.epoch_totals["vehicle_bg_nonzero_batches"] <= 0:
                raise RuntimeError("KD health failure: no replay batch contained both GT foreground and vehicle-background RoIs.")
            if self.grad_events["vehicle_bg_head"] == 0 or self.grad_norm_sum["vehicle_bg_head"] <= 0:
                raise RuntimeError("KD health failure: vehicle background head received no gradient.")
        if self.cfg["runtime"].get("health_batches") is not None and self.cfg["distillation"].get("weighting_mode") == "gradient_calibrated":
            expected = min(self.batch_calls, int(self.cfg["distillation"].get("calibration_batches", 128)))
            if experiment in {"f", "fk"} and int(self.addons.feature_kd_calibration_count) < expected:
                raise RuntimeError(
                    "KD health failure: feature gradient calibration did not receive a finite non-zero shared gradient "
                    f"for every health batch ({int(self.addons.feature_kd_calibration_count)}/{expected})."
                )
            if experiment in {"k", "fk"} and int(self.addons.cls_kd_calibration_count) < expected:
                raise RuntimeError(
                    "KD health failure: classification gradient calibration did not receive a finite non-zero shared gradient "
                    f"for every health batch ({int(self.addons.cls_kd_calibration_count)}/{expected})."
                )
            if self.vehicle_bg_enabled and current_epoch_batches >= vehicle_bg_patience:
                bg_expected = min(
                    int(self.epoch_totals["vehicle_bg_nonzero_batches"]),
                    int(self.cfg["distillation"].get("calibration_batches", 128)),
                )
                if int(self.addons.vehicle_bg_kd_calibration_count) < bg_expected:
                    raise RuntimeError(
                        "KD health failure: vehicle-background calibration missed an active replay batch "
                        f"({int(self.addons.vehicle_bg_kd_calibration_count)}/{bg_expected})."
                    )

    def epoch_summary(self, reset: bool = False) -> dict[str, float | int]:
        batches = max(int(self.epoch_totals["batches"]), 1)
        summary = {
            "epoch": self.epoch + 1,
            "feature_schedule_multiplier": self._schedule("feature"),
            "cls_schedule_multiplier": self._schedule("cls"),
            "vehicle_bg_schedule_multiplier": self._schedule("vehicle_bg"),
            "batches": int(self.epoch_totals["batches"]),
            "feature_raw_mean": self.epoch_totals["feature_raw_sum"] / batches,
            "cls_raw_mean": self.epoch_totals["cls_raw_sum"] / batches,
            "vehicle_bg_raw_mean": self.epoch_totals["vehicle_bg_raw_sum"] / batches,
            "kd_mean": self.epoch_totals["kd_sum"] / batches,
            "feature_nonzero_batches": int(self.epoch_totals["feature_nonzero_batches"]),
            "cls_nonzero_batches": int(self.epoch_totals["cls_nonzero_batches"]),
            "vehicle_bg_nonzero_batches": int(self.epoch_totals["vehicle_bg_nonzero_batches"]),
            "vehicle_bg_positive_rois": int(self.epoch_totals["vehicle_bg_positive_rois"]),
            "vehicle_bg_negative_rois": int(self.epoch_totals["vehicle_bg_negative_rois"]),
            "valid_rois": int(self.epoch_totals["valid_rois"]),
            "cache_requests": self.signal_store.requests,
            "cache_memory_hits": self.signal_store.memory_hits,
            "cache_disk_hits": self.signal_store.disk_hits,
            "cache_misses": self.signal_store.misses,
            "feature_grad_events": self.grad_events["feature_projector"],
            "feature_grad_norm_sum": self.grad_norm_sum["feature_projector"],
            "roi_grad_events": self.grad_events["roi_head"],
            "roi_grad_norm_sum": self.grad_norm_sum["roi_head"],
            "vehicle_bg_grad_events": self.grad_events["vehicle_bg_head"],
            "vehicle_bg_grad_norm_sum": self.grad_norm_sum["vehicle_bg_head"],
            "teacher_candidates": int(self.epoch_totals["teacher_candidates"]),
            "teacher_kept": int(self.epoch_totals["teacher_kept"]),
            "teacher_kept_confidence_mean": self.epoch_totals["teacher_confidence_sum"] / max(int(self.epoch_totals["teacher_kept"]), 1),
            "feature_kd_weight": float(self.addons.feature_kd_weight),
            "cls_kd_weight": float(self.addons.cls_kd_weight),
            "feature_kd_calibration_count": int(self.addons.feature_kd_calibration_count),
            "cls_kd_calibration_count": int(self.addons.cls_kd_calibration_count),
            "aot_donated_buffers_disabled": self.aot_donated_buffers_disabled,
        }
        if self.vehicle_bg_enabled:
            summary.update(
                vehicle_bg_kd_weight=float(self.addons.vehicle_bg_kd_weight),
                vehicle_bg_kd_calibration_count=int(self.addons.vehicle_bg_kd_calibration_count),
            )
        else:
            summary.update(vehicle_bg_kd_weight=0.0, vehicle_bg_kd_calibration_count=0)
        for name in ("det", "feature", "cls", "vehicle_bg"):
            summary[f"{name}_shared_grad_norm_mean"] = self.gradient_norm_sum[name] / max(self.gradient_norm_count[name], 1)
        det_norm_mean = summary["det_shared_grad_norm_mean"]
        summary["feature_calibrated_shared_grad_ratio"] = (
            float(self.addons.feature_kd_weight) * summary["feature_shared_grad_norm_mean"] / det_norm_mean
            if det_norm_mean > 0 else 0.0
        )
        summary["cls_calibrated_shared_grad_ratio"] = (
            float(self.addons.cls_kd_weight) * summary["cls_shared_grad_norm_mean"] / det_norm_mean
            if det_norm_mean > 0 else 0.0
        )
        summary["vehicle_bg_calibrated_shared_grad_ratio"] = (
            float(self.addons.vehicle_bg_kd_weight) * summary["vehicle_bg_shared_grad_norm_mean"] / det_norm_mean
            if self.vehicle_bg_enabled and det_norm_mean > 0 else 0.0
        )
        for pair, stats in self.gradient_probe.items():
            summary[f"{pair}_grad_cosine_mean"] = stats["cosine_sum"] / max(stats["count"], 1)
            summary[f"{pair}_grad_negative_rate"] = stats["negative"] / max(stats["count"], 1)
            summary[f"{pair}_grad_strong_negative_rate"] = stats["strong_negative"] / max(stats["count"], 1)
            summary[f"{pair}_grad_samples"] = stats["count"]
        if reset:
            for key in self.epoch_totals:
                self.epoch_totals[key] = 0
            for key in self.grad_events:
                self.grad_events[key] = 0; self.grad_norm_sum[key] = 0.0
            # Gradient probing is intentionally cumulative. It runs only on the
            # first N batches, and later epoch summaries must retain that result
            # instead of misleadingly reporting zero conflict after probing ends.
            self.signal_store.requests = self.signal_store.memory_hits = self.signal_store.disk_hits = self.signal_store.misses = 0
        return summary

    def __call__(self, preds, batch):
        self.features = self.prediction_features(preds)
        detection_components, items = self.original_loss(preds, batch)
        # Ultralytics reuses the EMA model criterion to report validation loss.
        # Teacher signals are intentionally cached only for the training split;
        # validation must measure the plain detector and must never query/train on
        # teacher signals. model.eval() recursively sets the attached addons to
        # evaluation mode, giving us an unambiguous lifecycle boundary here.
        if not self.addons.training:
            self.features = []
            return detection_components, items
        # Ultralytics detect criterion returns [box, cls, dfl]. The trainer sums
        # these components after model(batch). KD is a fourth scalar objective,
        # so budget and add it exactly once instead of broadcasting it 3 times.
        detection_loss = detection_components.sum()
        experiment = self.cfg["runtime"]["experiment"]
        image_files = list(batch.get("im_file", []))
        if not image_files or len(image_files) != int(batch["img"].shape[0]):
            raise RuntimeError(f"KD health failure: im_file count {len(image_files)} does not match batch size {int(batch['img'].shape[0])}.")
        signals = []
        for image_file in image_files:
            signal = self.signal_store.get(image_file)
            if signal is None:
                raise RuntimeError(f"KD health failure: teacher cache miss for {image_file}")
            signals.append(signal)
        feature_raw = self.feature_loss(batch, signals) if experiment in {"f", "fk", "e1", "e3"} else detection_loss.new_zeros(())
        if experiment in {"k", "fk", "e2", "e3"}:
            cls_raw, valid_rois, teacher_stats = self.cls_loss(batch, signals)
        else:
            cls_raw, valid_rois = detection_loss.new_zeros(()), 0
            teacher_stats = {"candidates": 0, "kept": 0, "confidence_sum": 0.0}
        if self.vehicle_bg_enabled:
            vehicle_bg_raw, vehicle_bg_positive_rois, vehicle_bg_negative_rois = self.vehicle_background_loss(batch)
        else:
            vehicle_bg_raw = detection_loss.new_zeros(())
            vehicle_bg_positive_rois = vehicle_bg_negative_rois = 0

        dcfg = self.cfg["distillation"]
        weighting_mode = dcfg.get("weighting_mode", "legacy_relative")
        calibration_batches = int(dcfg.get("calibration_batches", 128))
        probe_batches = int(dcfg.get("gradient_probe_batches", 0))
        needs_feature_calibration = weighting_mode == "gradient_calibrated" and experiment in {"f", "fk", "e1", "e3"} and int(self.addons.feature_kd_calibration_count) < calibration_batches
        needs_cls_calibration = weighting_mode == "gradient_calibrated" and experiment in {"k", "fk", "e2", "e3"} and int(self.addons.cls_kd_calibration_count) < calibration_batches
        needs_vehicle_bg_calibration = (
            weighting_mode == "gradient_calibrated"
            and self.vehicle_bg_enabled
            and int(self.addons.vehicle_bg_kd_calibration_count) < calibration_batches
        )
        needs_probe = self.batch_calls < probe_batches
        if needs_feature_calibration or needs_cls_calibration or needs_vehicle_bg_calibration or needs_probe:
            detection_gradients = self._gradients(detection_loss, self.features)
            feature_gradients = self._gradients(feature_raw, self.features) if feature_raw.requires_grad else tuple(None for _ in self.features)
            cls_gradients = self._gradients(cls_raw, self.features) if cls_raw.requires_grad else tuple(None for _ in self.features)
            vehicle_bg_gradients = self._gradients(vehicle_bg_raw, self.features) if vehicle_bg_raw.requires_grad else tuple(None for _ in self.features)
        else:
            empty_gradients = tuple(None for _ in self.features)
            detection_gradients = feature_gradients = cls_gradients = vehicle_bg_gradients = empty_gradients
        if needs_probe:
            self._record_gradient_probe(detection_gradients, feature_gradients, cls_gradients, vehicle_bg_gradients)

        # Backward-compatible old FK used one total budget split evenly. New
        # modes always keep feature and class controls independent.
        if weighting_mode == "legacy_relative" and experiment in {"fk", "e3"} and bool(dcfg.get("legacy_joint_split", True)):
            split_ratio = float(dcfg["feature_budget_ratio"]) / 2.0
            feature_kd, feature_weight = self._weighted_branch("feature", feature_raw, detection_loss, detection_gradients, feature_gradients, split_ratio)
            cls_kd, cls_weight = self._weighted_branch("cls", cls_raw, detection_loss, detection_gradients, cls_gradients, split_ratio)
        else:
            feature_kd, feature_weight = self._weighted_branch("feature", feature_raw, detection_loss, detection_gradients, feature_gradients)
            cls_kd, cls_weight = self._weighted_branch("cls", cls_raw, detection_loss, detection_gradients, cls_gradients)
        if self.vehicle_bg_enabled:
            vehicle_bg_kd, vehicle_bg_weight = self._weighted_branch(
                "vehicle_bg", vehicle_bg_raw, detection_loss, detection_gradients, vehicle_bg_gradients
            )
        else:
            vehicle_bg_kd, vehicle_bg_weight = detection_loss.new_zeros(()), 0.0
        kd = feature_kd + cls_kd + vehicle_bg_kd
        if not bool(torch.isfinite(kd)):
            raise RuntimeError("KD health failure: weighted distillation loss is NaN/Inf.")
        total_loss = detection_loss + kd
        feature_value = float(feature_raw.detach())
        cls_value = float(cls_raw.detach())
        vehicle_bg_value = float(vehicle_bg_raw.detach())
        kd_value = float(kd.detach())
        self.batch_calls += 1
        self.last = {
            "feature_raw": feature_value, "cls_raw": cls_value, "vehicle_bg_raw": vehicle_bg_value,
            "kd": kd_value, "valid_rois": valid_rois,
            "feature_weight": feature_weight, "cls_weight": cls_weight, "vehicle_bg_weight": vehicle_bg_weight,
        }
        self.zero_streak["feature"] = 0 if feature_value > 0 else self.zero_streak["feature"] + 1
        self.zero_streak["cls"] = 0 if cls_value > 0 else self.zero_streak["cls"] + 1
        self.epoch_totals["batches"] += 1
        self.epoch_totals["feature_raw_sum"] += feature_value
        self.epoch_totals["cls_raw_sum"] += cls_value
        self.epoch_totals["vehicle_bg_raw_sum"] += vehicle_bg_value
        self.epoch_totals["kd_sum"] += kd_value
        self.epoch_totals["feature_nonzero_batches"] += int(feature_value > 0)
        self.epoch_totals["cls_nonzero_batches"] += int(cls_value > 0)
        self.epoch_totals["vehicle_bg_nonzero_batches"] += int(vehicle_bg_value > 0)
        self.epoch_totals["vehicle_bg_positive_rois"] += vehicle_bg_positive_rois
        self.epoch_totals["vehicle_bg_negative_rois"] += vehicle_bg_negative_rois
        self.epoch_totals["valid_rois"] += valid_rois
        self.epoch_totals["teacher_candidates"] += int(teacher_stats["candidates"])
        self.epoch_totals["teacher_kept"] += int(teacher_stats["kept"])
        self.epoch_totals["teacher_confidence_sum"] += float(teacher_stats["confidence_sum"])
        self.features = []
        return total_loss, items
