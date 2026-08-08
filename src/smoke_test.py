from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .common import load_config
from .teacher import DINOFeatureTeacher


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the DINOv3 teacher produces the expected 640->P3/P4/P5 tensors.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    args = parser.parse_args(); cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DINOFeatureTeacher(cfg["paths"]["dino_repo"], cfg["paths"]["dino_weights"], cfg["teacher"]["feature_channels"], cfg["teacher"]["roi_size"], cfg["dataset"]["nc"]).to(device).eval()
    x = torch.zeros(1, 3, cfg["dataset"]["image_size"], cfg["dataset"]["image_size"], device=device)
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
        p3, p4, p5 = model(x)["features"]
    expected = [(1, cfg["teacher"]["feature_channels"], 80, 80), (1, cfg["teacher"]["feature_channels"], 40, 40), (1, cfg["teacher"]["feature_channels"], 20, 20)]
    actual = [tuple(t.shape) for t in (p3, p4, p5)]
    assert actual == expected, f"Expected {expected}, got {actual}"
    print("PASS", actual)


if __name__ == "__main__":
    main()
