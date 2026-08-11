from __future__ import annotations

from pathlib import Path

import pytest
import torch
from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import DetectionModel

from src.distillation import DistillationLoss, StudentDistillAddons, TeacherSignalStore
from src.distill_trainer import export_plain_yolo_checkpoint, infer_student_channels
from src.prototype_bank import PrototypeBank
from src.train_ablation import require_fhit_cache_manifest


def _config(experiment: str) -> dict:
    return {
        "paths": {"project_root": "."},
        "dataset": {
            "image_size": 640,
            "nc": 25,
            "class_groups": {
                "ship": [0, 1, 2, 3],
                "aircraft": list(range(4, 24)),
                "vehicle": [24],
            },
        },
        "teacher": {"feature_channels": 32},
        "student": {"kd_warmup_epochs": 1},
        "runtime": {"experiment": experiment},
        "distillation": {
            "temperature": 3.0,
            "feature_budget_ratio": 0.01,
            "cls_budget_ratio": 0.01,
            "global_budget_ratio": 0.01,
            "prototype_budget_ratio": 0.01,
            "feature_scale_weights": [0.6, 0.3, 0.1],
            "foreground_weight": 1.0,
            "boundary_weight": 0.3,
            "background_weight": 0.05,
            "teacher_confidence_floor": 0.0,
            "roi_match_iou_floor": 0.999,
            "global_feature_levels": ["p4", "p5"],
            "global_scale_weights": [0.65, 0.35],
            "prototype_teacher_confidence_floor": 0.7,
            "prototype_teacher_margin_floor": 0.1,
            "prototype_require_coarse_correct": True,
            "prototype_embedding_dim": 512,
            "prototype_relation_temperature": 0.2,
        },
    }


def _unit(index: int, dim: int = 512) -> torch.Tensor:
    value = torch.zeros(dim)
    value[index] = 1.0
    return value


def _prototype_bank(*, min_count: int = 2) -> PrototypeBank:
    e0, e1, e2 = _unit(0), _unit(1), _unit(2)
    global_stats = {
        "fine:4": {"sum": 3 * e0 + e2, "count": 4},
        "fine:5": {"sum": 3 * e1 + e2, "count": 4},
        "coarse:aircraft": {"sum": 3 * e0 + e1 + 2 * e2, "count": 6},
        "size:medium": {"sum": 3 * e0 + e1 + 2 * e2, "count": 6},
    }
    scene_stats = {
        "scene-a": {
            "fine:4": {"sum": e2, "count": 1},
            "fine:5": {"sum": e2, "count": 1},
            "coarse:aircraft": {"sum": 2 * e2, "count": 2},
            "size:medium": {"sum": 2 * e2, "count": 2},
        }
    }
    return PrototypeBank(
        {
            "embedding_dim": 512,
            "min_count": min_count,
            "global_stats": global_stats,
            "scene_stats": scene_stats,
            "image_to_scene": {"image.jpg": "scene-a"},
        }
    )


def test_leave_one_scene_out_lookup_excludes_current_scene() -> None:
    bank = _prototype_bank()
    value, count = bank.lookup("scene-a", "fine:4", torch.device("cpu"))
    assert count == 3
    assert value is not None
    assert torch.allclose(value, _unit(0))


def test_leave_one_scene_out_low_count_is_skipped() -> None:
    bank = _prototype_bank(min_count=4)
    value, count = bank.lookup("scene-a", "fine:4", torch.device("cpu"))
    assert count == 3
    assert value is None


def test_g_uses_only_p4_p5_and_ignores_interpolated_p3() -> None:
    torch.manual_seed(3)
    cfg = _config("g")
    addons = StudentDistillAddons([16, 32, 64], 32, 25, enable_global=True)
    wrapper = DistillationLoss(lambda _preds, _batch: None, addons, TeacherSignalStore(Path(".")), cfg)
    wrapper.features = [
        torch.randn(1, 16, 8, 8, requires_grad=True),
        torch.randn(1, 32, 4, 4, requires_grad=True),
        torch.randn(1, 64, 2, 2, requires_grad=True),
    ]
    batch = {
        "img": torch.zeros(1, 3, 640, 640),
        "bboxes": torch.tensor([[0.5, 0.5, 0.1, 0.1]]),
        "batch_idx": torch.tensor([0]),
    }
    base = {
        "p3": torch.randn(32, 8, 8),
        "p4": torch.randn(32, 4, 4),
        "p5": torch.randn(32, 2, 2),
    }
    route = torch.tensor([True])
    first = wrapper.global_feature_loss(batch, [base], route)
    changed = {**base, "p3": torch.randn_like(base["p3"]) * 1000}
    second = wrapper.global_feature_loss(batch, [changed], route)
    assert float(first.detach()) > 0
    assert torch.allclose(first, second)
    assert float(wrapper.global_feature_loss(batch, [base], torch.tensor([False]))) == 0.0


def test_g_rejects_p3_as_a_fake_dino_scale() -> None:
    cfg = _config("g")
    cfg["distillation"]["global_feature_levels"] = ["p3", "p4"]
    wrapper = DistillationLoss(
        lambda _preds, _batch: None,
        StudentDistillAddons([16, 32, 64], 32, 25, enable_global=True),
        TeacherSignalStore(Path(".")),
        cfg,
    )
    wrapper.features = [
        torch.randn(1, 16, 8, 8),
        torch.randn(1, 32, 4, 4),
        torch.randn(1, 64, 2, 2),
    ]
    batch = {
        "img": torch.zeros(1, 3, 640, 640),
        "bboxes": torch.tensor([[0.5, 0.5, 0.1, 0.1]]),
        "batch_idx": torch.tensor([0]),
    }
    with pytest.raises(RuntimeError, match="not a genuine DINO fine scale"):
        wrapper.global_feature_loss(
            batch,
            [{"p3": torch.randn(32, 8, 8), "p4": torch.randn(32, 4, 4)}],
            torch.tensor([True]),
        )


def test_gp_target_router_is_exclusive_and_failure_size_group_aware() -> None:
    cfg = _config("gp")
    addons = StudentDistillAddons(
        [16, 32, 64], 32, 25, enable_global=True, enable_prototype=True
    )
    wrapper = DistillationLoss(
        lambda _preds, _batch: None,
        addons,
        TeacherSignalStore(Path(".")),
        cfg,
        prototype_bank=_prototype_bank(),
    )
    wrapper.features = [torch.randn(1, 16, 8, 8)]
    boxes = torch.tensor(
        [
            [0.3, 0.3, 0.10, 0.10],  # medium aircraft -> G
            [0.5, 0.5, 0.02, 0.02],  # small aircraft -> P
            [0.7, 0.7, 0.10, 0.10],  # ship group -> P
        ]
    )
    classes = torch.tensor([4, 4, 0])
    batch = {
        "img": torch.zeros(1, 3, 640, 640),
        "bboxes": boxes,
        "cls": classes[:, None].float(),
        "batch_idx": torch.zeros(3, dtype=torch.long),
        "im_file": [str(Path("dataset") / "images" / "train" / "image.jpg")],
    }
    signals = [{"boxes_xywhn": boxes, "classes": classes}]
    global_route, prototype_route, stats = wrapper._batch_routes(batch, signals)
    assert global_route.tolist() == [True, False, False]
    assert prototype_route.tolist() == [False, True, True]
    assert not bool((global_route & prototype_route).any())
    assert stats == {"global": 1, "prototype": 2, "overlap": 0}


def test_p_loss_uses_student_p3_p4_and_backpropagates() -> None:
    torch.manual_seed(5)
    cfg = _config("p")
    addons = StudentDistillAddons([16, 32, 64], 32, 25, enable_prototype=True)
    wrapper = DistillationLoss(
        lambda _preds, _batch: None,
        addons,
        TeacherSignalStore(Path(".")),
        cfg,
        prototype_bank=_prototype_bank(),
    )
    wrapper.features = [
        torch.randn(1, 16, 8, 8, requires_grad=True),
        torch.randn(1, 32, 4, 4, requires_grad=True),
        torch.randn(1, 64, 2, 2, requires_grad=True),
    ]
    box = torch.tensor([[0.5, 0.5, 0.10, 0.10]])
    logits = torch.full((1, 25), -8.0)
    logits[0, 4] = 8.0
    signals = [
        {
            "boxes_xywhn": box,
            "classes": torch.tensor([4]),
            "roi_logits": logits,
            "roi_embeddings": _unit(2).unsqueeze(0),
        }
    ]
    batch = {
        "img": torch.zeros(1, 3, 640, 640),
        "bboxes": box,
        "cls": torch.tensor([[4.0]]),
        "batch_idx": torch.tensor([0]),
        "im_file": [str(Path("dataset") / "images" / "train" / "image.jpg")],
    }
    loss, valid, stats = wrapper.prototype_loss(batch, signals, torch.tensor([True]))
    loss.backward()
    assert float(loss.detach()) > 0
    assert valid == 1
    assert stats["kept"] == 1
    assert next(addons.prototype_projectors.parameters()).grad is not None
    assert next(addons.prototype_fuse.parameters()).grad is not None


def test_new_experiments_reject_old_cache_manifest() -> None:
    for experiment in ("g", "p", "gp"):
        with pytest.raises(RuntimeError, match="format=3"):
            require_fhit_cache_manifest({"format": 2}, experiment)
    with pytest.raises(RuntimeError, match="roi_embeddings"):
        require_fhit_cache_manifest({"format": 3}, "p")
    # Legacy F/K/FK compatibility is unchanged.
    require_fhit_cache_manifest({"format": 2}, "fk")


@pytest.mark.parametrize(
    ("experiment", "expected"),
    [("g", "G P4/P5"), ("p", "P prototype"), ("gp", "P prototype")],
)
def test_g_p_gp_ten_batch_silent_zero_is_fatal(
    experiment: str, expected: str
) -> None:
    cfg = _config(experiment)
    addons = StudentDistillAddons(
        [16, 32, 64],
        32,
        25,
        enable_global=experiment in {"g", "gp"},
        enable_prototype=experiment in {"p", "gp"},
    )
    wrapper = DistillationLoss(
        lambda _preds, _batch: None,
        addons,
        TeacherSignalStore(Path(".")),
        cfg,
        prototype_bank=_prototype_bank() if experiment in {"p", "gp"} else None,
    )
    wrapper.batch_calls = 10
    if experiment == "gp":
        # Prove P's independent gate instead of being masked by G's earlier gate.
        wrapper.zero_streak["global"] = 0
        wrapper.grad_events["global_projector"] = 1
        wrapper.grad_norm_sum["global_projector"] = 1.0
    wrapper.zero_streak["global" if experiment == "g" else "prototype"] = 10
    with pytest.raises(RuntimeError, match=expected):
        wrapper.assert_health(10)


def test_gp_deploy_checkpoint_is_plain_yolo(monkeypatch: pytest.MonkeyPatch) -> None:
    model = DetectionModel("yolo11n.yaml", ch=3, nc=25, verbose=False)
    model.args = get_cfg()
    channels = infer_student_channels(model, 64)
    model.add_module(
        "distill_addons",
        StudentDistillAddons(
            channels, 32, 25, enable_global=True, enable_prototype=True
        ),
    )
    detector_before = {
        name: value.clone()
        for name, value in model.state_dict().items()
        if not name.startswith("distill_addons.")
    }
    source, destination = Path("best.pt"), Path("best_deploy.pt")
    checkpoint = {"model": model, "optimizer": None, "epoch": -1}
    saved: dict[str, object] = {}
    monkeypatch.setattr(torch, "load", lambda *_args, **_kwargs: checkpoint)
    monkeypatch.setattr(torch, "save", lambda value, _path: saved.update(checkpoint=value))
    assert export_plain_yolo_checkpoint(source, destination) > 0
    deployed = saved["checkpoint"]["model"]  # type: ignore[index]
    assert not hasattr(deployed, "distill_addons")
    assert detector_before.keys() == deployed.state_dict().keys()
    assert all(
        torch.equal(detector_before[name], deployed.state_dict()[name])
        for name in detector_before
    )
