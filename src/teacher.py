from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torchvision.ops import roi_align


DINO_SAT_MEAN = (0.430, 0.411, 0.296)
DINO_SAT_STD = (0.213, 0.156, 0.143)


class DINOFeatureTeacher(nn.Module):
    """Frozen DINOv3-SAT backbone with a trainable FPN and object classifier.

    This is intentionally a training-only teacher. The exported student stays a plain YOLO11m.
    """

    def __init__(self, dino_repo: str | Path, weights: str | Path, feature_channels: int = 256, roi_size: int = 7, num_classes: int = 25):
        super().__init__()
        dino_repo = Path(dino_repo)
        if not dino_repo.exists():
            raise FileNotFoundError(f"DINOv3 source not found: {dino_repo}")
        self.backbone = torch.hub.load(str(dino_repo), "dinov3_vitl16", source="local", weights=str(weights))
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.backbone.eval()
        self.register_buffer("mean", torch.tensor(DINO_SAT_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(DINO_SAT_STD).view(1, 3, 1, 1), persistent=False)
        self.fuse = nn.Sequential(
            nn.Conv2d(1024 * 4, feature_channels, kernel_size=1, bias=False),
            nn.GroupNorm(16, feature_channels),
            nn.SiLU(),
        )
        self.p3 = nn.Sequential(nn.Conv2d(feature_channels, feature_channels, 3, padding=1, bias=False), nn.GroupNorm(16, feature_channels), nn.SiLU())
        self.p4 = nn.Sequential(nn.Conv2d(feature_channels, feature_channels, 3, padding=1, bias=False), nn.GroupNorm(16, feature_channels), nn.SiLU())
        self.p5 = nn.Sequential(nn.Conv2d(feature_channels, feature_channels, 3, padding=1, bias=False), nn.GroupNorm(16, feature_channels), nn.SiLU())
        self.roi_size = roi_size
        self.roi_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_channels * roi_size * roi_size, 512),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes),
        )

    def normalize(self, x: Tensor) -> Tensor:
        return (x - self.mean.to(dtype=x.dtype)) / self.std.to(dtype=x.dtype)

    @torch.no_grad()
    def _dino_features(self, x: Tensor) -> Sequence[Tensor]:
        self.backbone.eval()
        return self.backbone.get_intermediate_layers(self.normalize(x), n=(5, 11, 17, 23), reshape=True, norm=True)

    def pyramid(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        layers = self._dino_features(x)
        fused = self.fuse(torch.cat(layers, dim=1))
        p4 = self.p4(fused)
        p3 = self.p3(F.interpolate(p4, scale_factor=2.0, mode="bilinear", align_corners=False))
        p5 = self.p5(F.avg_pool2d(p4, kernel_size=2, stride=2))
        return p3, p4, p5

    def classify_rois(self, p3: Tensor, rois_xyxy: Tensor) -> Tensor:
        if rois_xyxy.numel() == 0:
            return p3.new_zeros((0, self.roi_classifier[-1].out_features))
        aligned = roi_align(p3, rois_xyxy, output_size=self.roi_size, spatial_scale=1.0 / 8.0, sampling_ratio=2, aligned=True)
        return self.roi_classifier(aligned)

    def forward(self, x: Tensor, rois_xyxy: Tensor | None = None) -> dict[str, Tensor | tuple[Tensor, Tensor, Tensor]]:
        p3, p4, p5 = self.pyramid(x)
        result: dict[str, Tensor | tuple[Tensor, Tensor, Tensor]] = {"features": (p3, p4, p5)}
        if rois_xyxy is not None:
            result["roi_logits"] = self.classify_rois(p3, rois_xyxy)
        return result
