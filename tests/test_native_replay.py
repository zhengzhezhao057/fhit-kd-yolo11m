from __future__ import annotations

import pytest

from src.build_native_replay_pools import build_native_replay_pools


def hard_manifest() -> dict:
    return {
        "format": 1, "split": "train",
        "images": {
            "ship1.jpg": {"objects": [
                {"coarse_group": "ship", "error_type": "low_confidence"},
                {"coarse_group": "ship", "error_type": "crowded"},
            ]},
            "car1.jpg": {"objects": [{"coarse_group": "vehicle", "error_type": "no_candidate"}]},
            "plain.jpg": {"objects": [{"coarse_group": "ship", "error_type": "detected"}]},
        },
    }


def background_manifest() -> dict:
    return {"format": 1, "split": "train", "images": {"bg1.jpg": {"boxes": [{"x": 0.5, "y": 0.5, "w": 0.1, "h": 0.1}]}}}


def test_replay_pools_select_only_hard_errors() -> None:
    manifest = build_native_replay_pools(hard_manifest(), background_manifest(), dataset_fingerprint="fp")
    assert set(manifest["images"]) == {"ship1.jpg", "car1.jpg", "bg1.jpg"}
    assert manifest["images"]["car1.jpg"]["pool"] == "vehicle_hard_positive"
    assert manifest["images"]["car1.jpg"]["repeat_count"] == 2
    assert manifest["images"]["ship1.jpg"]["reasons"] == ["low_confidence"]
    assert manifest["images"]["bg1.jpg"]["pool"] == "vehicle_background"
    assert manifest["images"]["bg1.jpg"]["reasons"] == ["background_fp"]
    assert manifest["format"] == 2
    assert manifest["max_repeat_per_image"] == 3


def test_replay_pools_reject_non_train_sources() -> None:
    with pytest.raises(RuntimeError, match="TRAIN"):
        build_native_replay_pools({"format": 1, "split": "val", "images": {}}, background_manifest(), dataset_fingerprint="fp")


def test_replay_pools_reject_empty_selection() -> None:
    with pytest.raises(RuntimeError, match="survived"):
        build_native_replay_pools({"format": 1, "split": "train", "images": {}}, {"format": 1, "split": "train", "images": {}}, dataset_fingerprint="fp")
