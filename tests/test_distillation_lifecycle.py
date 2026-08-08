from __future__ import annotations

from pathlib import Path

import torch
from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import DetectionModel

from src.common import stable_image_key
from src.distillation import DistillationLoss, StudentDistillAddons, TeacherSignalStore, allow_retained_backward_through_compiled_ops
from src.distill_trainer import DistillationDetectionTrainer, export_plain_yolo_checkpoint, infer_student_channels


def kd_config(experiment: str = "fk") -> dict:
    return {
        "dataset": {"image_size": 64, "nc": 25},
        "teacher": {"feature_channels": 32},
        "student": {"kd_warmup_epochs": 1},
        "runtime": {"experiment": experiment},
        "distillation": {
            "temperature": 3.0,
            "feature_budget_ratio": 0.15,
            "cls_budget_ratio": 0.15,
            "feature_scale_weights": [0.6, 0.3, 0.1],
            "foreground_weight": 1.0,
            "boundary_weight": 0.3,
            "background_weight": 0.05,
            "teacher_confidence_floor": 0.0,
            "roi_match_iou_floor": 0.5,
        },
    }


def make_model() -> DetectionModel:
    model = DetectionModel("yolo11n.yaml", ch=3, nc=25, verbose=False)
    model.args = get_cfg()
    return model


def test_retained_backward_disables_aot_buffer_donation() -> None:
    configured = allow_retained_backward_through_compiled_ops()
    try:
        from torch._functorch import config as functorch_config
    except (ImportError, AttributeError):
        assert not configured
    else:
        if hasattr(functorch_config, "donated_buffer"):
            assert configured
            assert functorch_config.donated_buffer is False


def test_custom_trainer_rebuild_preserves_addon_weights(tmp_path: Path) -> None:
    cfg = kd_config("f")
    source = make_model()
    channels = infer_student_channels(source, 64)
    source.add_module("distill_addons", StudentDistillAddons(channels, 32, 25))
    with torch.no_grad():
        next(source.distill_addons.parameters()).fill_(0.123)
        source.distill_addons.feature_kd_weight.fill_(7.5)
        source.distill_addons.feature_kd_calibration_count.fill_(17)

    DistillationDetectionTrainer.configure(cfg, tmp_path)
    trainer = object.__new__(DistillationDetectionTrainer)
    trainer.data = {"nc": 25, "channels": 3}
    rebuilt = trainer.get_model(cfg=source.yaml, weights=source, verbose=False)

    assert rebuilt is not source
    assert isinstance(rebuilt.distill_addons, StudentDistillAddons)
    assert torch.allclose(next(rebuilt.distill_addons.parameters()), torch.full_like(next(rebuilt.distill_addons.parameters()), 0.123))
    assert float(rebuilt.distill_addons.feature_kd_weight) == 7.5
    assert int(rebuilt.distill_addons.feature_kd_calibration_count) == 17


def test_joint_kd_loss_is_nonzero_and_backpropagates(tmp_path: Path) -> None:
    torch.manual_seed(0)
    cfg = kd_config("fk")
    image_path = tmp_path / "image.jpg"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    model = make_model().train()
    with torch.no_grad():
        predictions = model(torch.zeros(1, 3, 64, 64))
    features = predictions["feats"]
    channels = [int(feature.shape[1]) for feature in features]
    addons = StudentDistillAddons(channels, 32, 25)
    model.add_module("distill_addons", addons)

    box = torch.tensor([[0.5, 0.5, 0.25, 0.25]], dtype=torch.float32)
    torch.save(
        {
            "path": str(image_path.resolve()),
            "boxes_xywhn": box,
            "classes": torch.tensor([4]),
            "roi_logits": torch.randn(1, 25),
            "p3": torch.randn(32, *features[0].shape[-2:]),
            "p4": torch.randn(32, *features[1].shape[-2:]),
            "p5": torch.randn(32, *features[2].shape[-2:]),
        },
        cache_dir / f"{stable_image_key(image_path)}.pt",
    )

    wrapper = DistillationLoss(model.init_criterion(), addons, TeacherSignalStore(cache_dir), cfg)
    model.criterion = wrapper
    batch = {
        "img": torch.rand(1, 3, 64, 64),
        "batch_idx": torch.tensor([0]),
        "cls": torch.tensor([[4.0]]),
        "bboxes": box,
        "im_file": [str(image_path)],
    }
    loss, _items = model(batch)
    loss.backward()

    assert wrapper.last["feature_raw"] > 0
    assert wrapper.last["cls_raw"] > 0
    assert wrapper.last["kd"] > 0
    assert wrapper.last["valid_rois"] == 1
    assert next(addons.projectors.parameters()).grad is not None
    assert next(addons.student_roi_head.parameters()).grad is not None


def test_cache_miss_fails_instead_of_returning_zero(tmp_path: Path) -> None:
    cfg = kd_config("f")
    model = make_model().train()
    channels = infer_student_channels(model, 64)
    addons = StudentDistillAddons(channels, 32, 25)
    model.add_module("distill_addons", addons)
    model.criterion = DistillationLoss(model.init_criterion(), addons, TeacherSignalStore(tmp_path), cfg)
    batch = {
        "img": torch.rand(1, 3, 64, 64),
        "batch_idx": torch.tensor([0]),
        "cls": torch.tensor([[4.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]]),
        "im_file": [str(tmp_path / "missing.jpg")],
    }
    try:
        model(batch)
    except RuntimeError as error:
        assert "teacher cache miss" in str(error)
    else:
        raise AssertionError("A missing teacher cache entry must terminate KD training.")


def test_validation_uses_plain_detection_loss_without_teacher_cache(tmp_path: Path) -> None:
    cfg = kd_config("f")
    model = make_model()
    channels = infer_student_channels(model, 64)
    addons = StudentDistillAddons(channels, 32, 25)
    model.add_module("distill_addons", addons)
    wrapper = DistillationLoss(model.init_criterion(), addons, TeacherSignalStore(tmp_path), cfg)
    model.criterion = wrapper
    model.eval()
    batch = {
        "img": torch.rand(1, 3, 64, 64),
        "batch_idx": torch.tensor([0]),
        "cls": torch.tensor([[4.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]]),
        "im_file": [str(tmp_path / "uncached_val_image.jpg")],
    }

    loss_components, _items = model(batch)

    assert loss_components.numel() == 3
    assert wrapper.batch_calls == 0
    assert wrapper.signal_store.requests == 0


def test_deploy_export_removes_only_training_addons(tmp_path: Path) -> None:
    model = make_model()
    channels = infer_student_channels(model, 64)
    model.add_module("distill_addons", StudentDistillAddons(channels, 32, 25))
    detector_before = {name: value.clone() for name, value in model.state_dict().items() if not name.startswith("distill_addons.")}
    source, destination = tmp_path / "best.pt", tmp_path / "best_deploy.pt"
    torch.save({"model": model, "optimizer": None, "epoch": -1}, source)

    removed = export_plain_yolo_checkpoint(source, destination)
    deployed = torch.load(destination, map_location="cpu", weights_only=False)["model"]

    assert removed > 0
    assert not hasattr(deployed, "distill_addons")
    assert detector_before.keys() == deployed.state_dict().keys()
    assert all(torch.equal(detector_before[name], deployed.state_dict()[name]) for name in detector_before)


def test_teacher_reliability_uses_native_probability_and_can_require_correct_label(tmp_path: Path) -> None:
    cfg = kd_config("k")
    cfg["distillation"].update(teacher_confidence_floor=0.7, require_teacher_correct=True)
    addons = StudentDistillAddons([16, 32, 64], 32, 25)
    wrapper = DistillationLoss(lambda _preds, _batch: None, addons, TeacherSignalStore(tmp_path), cfg)
    wrapper.features = [torch.randn(1, 16, 8, 8, requires_grad=True)]
    box = torch.tensor([[0.5, 0.5, 0.25, 0.25]])
    wrong_logits = torch.full((1, 25), -10.0); wrong_logits[0, 5] = 10.0
    batch = {"bboxes": box, "cls": torch.tensor([[4.0]]), "batch_idx": torch.tensor([0])}
    signals = [{"boxes_xywhn": box, "classes": torch.tensor([4]), "roi_logits": wrong_logits}]

    loss, valid, stats = wrapper.cls_loss(batch, signals)

    assert float(loss) == 0.0
    assert valid == 0
    assert stats["candidates"] == 1
    assert stats["kept"] == 0


def test_gradient_calibration_persists_weight_in_addon_buffers(tmp_path: Path) -> None:
    cfg = kd_config("f")
    cfg["distillation"].update(
        weighting_mode="gradient_calibrated",
        calibration_batches=1,
        gradient_probe_batches=1,
        feature_gradient_ratio=0.03,
        feature_weight_bounds=[1e-6, 1e6],
    )
    image_path = tmp_path / "image.jpg"
    cache_dir = tmp_path / "cache"; cache_dir.mkdir()
    model = make_model().train()
    with torch.no_grad():
        predictions = model(torch.zeros(1, 3, 64, 64))
    features = predictions["feats"]
    addons = StudentDistillAddons([int(feature.shape[1]) for feature in features], 32, 25)
    model.add_module("distill_addons", addons)
    box = torch.tensor([[0.5, 0.5, 0.25, 0.25]])
    torch.save({
        "path": str(image_path.resolve()), "boxes_xywhn": box, "classes": torch.tensor([4]),
        "roi_logits": torch.randn(1, 25),
        "p3": torch.randn(32, *features[0].shape[-2:]), "p4": torch.randn(32, *features[1].shape[-2:]), "p5": torch.randn(32, *features[2].shape[-2:]),
    }, cache_dir / f"{stable_image_key(image_path)}.pt")
    wrapper = DistillationLoss(model.init_criterion(), addons, TeacherSignalStore(cache_dir), cfg)
    model.criterion = wrapper
    batch = {"img": torch.rand(1, 3, 64, 64), "batch_idx": torch.tensor([0]), "cls": torch.tensor([[4.0]]), "bboxes": box, "im_file": [str(image_path)]}

    loss, _items = model(batch)
    loss.backward()

    assert int(addons.feature_kd_calibration_count) == 1
    assert float(addons.feature_kd_weight) > 0
    summary = wrapper.epoch_summary()
    assert summary["det_feature_grad_samples"] == 1
    assert -1.0 <= summary["det_feature_grad_cosine_mean"] <= 1.0


def test_branch_schedule_warms_holds_decays_and_stops(tmp_path: Path) -> None:
    cfg = kd_config("f")
    cfg["distillation"]["feature_schedule"] = {"start_epoch": 1, "warmup_epochs": 2, "hold_epochs": 1, "decay_epochs": 2}
    wrapper = DistillationLoss(lambda _preds, _batch: None, StudentDistillAddons([16, 32, 64], 32, 25), TeacherSignalStore(tmp_path), cfg)
    values = []
    for epoch in range(7):
        wrapper.set_epoch(epoch); values.append(wrapper._schedule("feature"))
    assert values[0] == 0.0
    assert values[1] == 0.5
    assert values[2] == 1.0
    assert values[3] == 1.0
    assert 0.0 < values[4] < 1.0
    assert values[5] == 0.0
    assert values[6] == 0.0


def test_health_check_forces_scheduled_branch_into_final_backward(tmp_path: Path) -> None:
    cfg = kd_config("k")
    cfg["runtime"]["health_batches"] = 10
    cfg["distillation"]["cls_schedule"] = {"start_epoch": 2, "warmup_epochs": 1, "hold_epochs": 1, "decay_epochs": 1}
    wrapper = DistillationLoss(
        lambda _preds, _batch: None,
        StudentDistillAddons([16, 32, 64], 32, 25),
        TeacherSignalStore(tmp_path),
        cfg,
    )
    wrapper.set_epoch(0)
    assert wrapper._schedule("cls") == 1.0
    assert wrapper._schedule("feature") == 0.0


def test_intentionally_stopped_branch_does_not_fail_gradient_health_gate(tmp_path: Path) -> None:
    cfg = kd_config("f")
    cfg["distillation"]["feature_schedule"] = {"start_epoch": 0, "warmup_epochs": 0, "hold_epochs": 0, "decay_epochs": 0}
    wrapper = DistillationLoss(
        lambda _preds, _batch: None,
        StudentDistillAddons([16, 32, 64], 32, 25),
        TeacherSignalStore(tmp_path),
        cfg,
    )
    wrapper.batch_calls = 10
    wrapper.zero_streak["feature"] = 0  # Raw teacher/student signal remains valid; only its scheduled weight is zero.
    assert wrapper._schedule("feature") == 0.0
    wrapper.assert_health(10)


def test_feature_numerical_scale_is_an_exact_loss_rescaling(tmp_path: Path) -> None:
    cfg = kd_config("f")
    addons = StudentDistillAddons([16, 32, 64], 32, 25)
    wrapper = DistillationLoss(lambda _preds, _batch: None, addons, TeacherSignalStore(tmp_path), cfg)
    wrapper.features = [
        torch.randn(1, 16, 8, 8, requires_grad=True),
        torch.randn(1, 32, 4, 4, requires_grad=True),
        torch.randn(1, 64, 2, 2, requires_grad=True),
    ]
    signals = [{
        "p3": torch.randn(32, 8, 8),
        "p4": torch.randn(32, 4, 4),
        "p5": torch.randn(32, 2, 2),
    }]
    batch = {"bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]]), "batch_idx": torch.tensor([0])}
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        base = wrapper.feature_loss(batch, signals)
    assert base.dtype == torch.float32
    cfg["distillation"]["feature_numerical_scale"] = 1024.0
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        scaled = wrapper.feature_loss(batch, signals)
    assert scaled.dtype == torch.float32
    assert torch.allclose(scaled, base * 1024.0)


def test_nonfinite_parameter_gradient_is_rejected(tmp_path: Path) -> None:
    cfg = kd_config("f")
    wrapper = DistillationLoss(
        lambda _preds, _batch: None,
        StudentDistillAddons([16, 32, 64], 32, 25),
        TeacherSignalStore(tmp_path),
        cfg,
    )
    try:
        wrapper.record_gradient("feature_projector", torch.tensor([float("inf")]))
    except RuntimeError as error:
        assert "NaN/Inf" in str(error)
    else:
        raise AssertionError("A non-finite KD parameter gradient must terminate training.")
