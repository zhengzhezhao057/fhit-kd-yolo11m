from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.torch_utils import unwrap_model

from .distillation import (
    DistillationLoss,
    StudentDistillAddons,
    TeacherSignalStore,
    VehicleNegativeStore,
    _train_relative_image,
)
from .prototype_bank import PrototypeBank


KD_CALIBRATION_BRANCHES = ("feature", "cls", "global", "prototype", "vehicle_bg")


def sync_kd_calibration_buffers(source: torch.nn.Module, destination: torch.nn.Module) -> int:
    """Copy non-EMA KD calibration state exactly into a checkpoint model.

    Ultralytics checkpoints derive their model from EMA. Floating buffers are
    EMA-averaged while integer counters are ignored, which makes a geometric
    calibration accumulator internally inconsistent after resume. These are
    optimizer-control state, not learned detector parameters, and must be
    copied exactly like an optimizer step counter.
    """
    copied = 0
    for branch in KD_CALIBRATION_BRANCHES:
        for suffix in ("kd_weight", "kd_log_sum", "kd_calibration_count"):
            name = f"{branch}_{suffix}"
            if not hasattr(source, name):
                continue
            if not hasattr(destination, name):
                raise RuntimeError(f"KD checkpoint lifecycle failure: EMA addons lack {name}.")
            getattr(destination, name).copy_(getattr(source, name).detach())
            copied += 1
    return copied


def reset_kd_calibration_buffers(addons: torch.nn.Module) -> int:
    """Reset gradient calibration only when starting a genuinely new objective."""
    reset = 0
    for branch in KD_CALIBRATION_BRANCHES:
        count = getattr(addons, f"{branch}_kd_calibration_count", None)
        weight = getattr(addons, f"{branch}_kd_weight", None)
        log_sum = getattr(addons, f"{branch}_kd_log_sum", None)
        if count is None:
            continue
        count.zero_(); weight.fill_(1.0); log_sum.zero_(); reset += 3
    return reset


def restore_legacy_kd_calibration(
    model: torch.nn.Module,
    health_file: str | Path,
    checkpoint_epoch: int,
) -> int:
    """Reconstruct calibration buffers in checkpoints saved before exact sync.

    Each epoch health record stores the live geometric-mean weight and count;
    therefore ``log_sum = count * log(weight)`` exactly reconstructs the state
    required by the next calibration batch.
    """
    if bool(getattr(model, "kd_calibration_buffers_exact", False)):
        return 0
    addons = getattr(model, "distill_addons", None)
    if not isinstance(addons, StudentDistillAddons):
        raise RuntimeError("KD resume recovery failed: checkpoint has no distill_addons.")
    path = Path(health_file)
    if not path.exists():
        raise RuntimeError(
            f"KD resume recovery requires the matching epoch health log, but {path} is missing."
        )
    target_epoch = int(checkpoint_epoch) + 1
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    matches = [record for record in records if int(record.get("epoch", -1)) == target_epoch]
    if not matches:
        raise RuntimeError(
            f"KD resume recovery found no health record for checkpoint epoch {target_epoch} in {path}."
        )
    record = matches[-1]
    restored = 0
    for branch in KD_CALIBRATION_BRANCHES:
        count_name = f"{branch}_kd_calibration_count"
        if not hasattr(addons, count_name):
            continue
        count = int(record.get(count_name, 0))
        weight = float(record.get(f"{branch}_kd_weight", 1.0))
        if count < 0 or not torch.isfinite(torch.tensor(weight)) or weight <= 0:
            raise RuntimeError(f"KD resume recovery found invalid {branch} state: count={count}, weight={weight}.")
        getattr(addons, count_name).fill_(count)
        getattr(addons, f"{branch}_kd_weight").fill_(weight)
        getattr(addons, f"{branch}_kd_log_sum").fill_(count * torch.log(torch.tensor(weight)).item())
        restored += 3
    model.kd_calibration_buffers_exact = True
    return restored


def kd_objective_fingerprint(cfg: dict[str, Any]) -> str:
    """Fingerprint every setting that changes a KD continuation trajectory."""
    dataset_payload = {
        key: cfg["dataset"].get(key)
        for key in ("nc", "image_size", "class_groups")
    }
    if bool(cfg.get("runtime", {}).get("dataset_identity", {}).get("strict")):
        dataset_payload.update(
            {
                "id": cfg["dataset"].get("id"),
                "dataset_fingerprint": cfg["dataset"].get("dataset_fingerprint"),
                "split_fingerprint": cfg["dataset"].get("split_fingerprint"),
            }
        )
    payload = {
        "experiment": cfg["runtime"]["experiment"],
        "dataset": dataset_payload,
        "student": cfg["student"],
        "distillation": copy.deepcopy(cfg["distillation"]),
    }
    for manifest_key in (
        "hard_example_manifest",
        "vehicle_negative_manifest",
        "hard_image_replay_manifest",
        "prototype_bank",
    ):
        manifest = payload["distillation"].get(manifest_key)
        if not manifest:
            continue
        manifest_path = Path(manifest)
        if not manifest_path.is_absolute():
            manifest_path = Path(cfg["paths"]["project_root"]) / manifest_path
        if not manifest_path.exists():
            raise FileNotFoundError(f"KD hard-example manifest missing: {manifest_path}")
        payload["distillation"][manifest_key] = {
            "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def find_detect_module(model: torch.nn.Module) -> torch.nn.Module:
    for module in model.modules():
        if hasattr(module, "stride") and hasattr(module, "nc") and (hasattr(module, "cv2") or hasattr(module, "cv3")):
            return module
    raise RuntimeError("Could not find the Ultralytics Detect head. Check the pinned ultralytics version.")


def infer_student_channels(model: torch.nn.Module, image_size: int) -> list[int]:
    """Read Detect input channels without leaving hooks or changing BN statistics."""
    captured: dict[str, list[int]] = {}

    def capture(_module, inputs) -> None:
        value = inputs[0] if len(inputs) == 1 else inputs
        features = value if isinstance(value, (list, tuple)) else [value]
        captured["channels"] = [int(item.shape[1]) for item in features if isinstance(item, torch.Tensor) and item.ndim == 4][-3:]

    was_training = model.training
    model.eval()
    handle = find_detect_module(model).register_forward_pre_hook(capture)
    try:
        with torch.no_grad():
            model(torch.zeros(1, 3, image_size, image_size, device=next(model.parameters()).device))
    finally:
        handle.remove()
        model.train(was_training)
    channels = captured.get("channels", [])
    if len(channels) != 3:
        raise RuntimeError(f"Expected three Detect inputs, got {channels}. Check model/head compatibility.")
    return channels


def export_plain_yolo_checkpoint(source: str | Path, destination: str | Path) -> int:
    """Remove training-only KD modules while preserving the trained YOLO detector."""
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    deployment_model = checkpoint.get("model")
    if deployment_model is None or not hasattr(deployment_model, "distill_addons"):
        raise RuntimeError(f"KD lifecycle failure: {source} has no trained distill_addons.")
    addon_parameters = sum(parameter.numel() for parameter in deployment_model.distill_addons.parameters())
    delattr(deployment_model, "distill_addons")
    deployment_model.criterion = None
    torch.save(checkpoint, destination)
    return addon_parameters


class HardImageReplayDataset:
    """Deterministically append TRAIN images containing OOF hard positives or negatives."""

    def __init__(self, base, relative_images: set[str], repeats: int):
        self.base = base
        if repeats <= 0:
            raise ValueError("Hard-image replay repeats must be positive.")
        matched = [
            index for index, image_path in enumerate(base.im_files)
            if _train_relative_image(image_path) in relative_images
        ]
        if not matched:
            raise RuntimeError("Hard-image replay matched zero training images.")
        self._indices = list(range(len(base))) + matched * repeats
        self.replay_images = len(matched)
        self.replay_samples = len(matched) * repeats

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index):
        return self.base[self._indices[int(index)]]

    def __getattr__(self, name):
        return getattr(self.base, name)


def load_hard_replay_images(cfg: dict[str, Any]) -> set[str]:
    manifest = cfg["distillation"].get("hard_image_replay_manifest")
    if not manifest:
        return set()
    path = Path(manifest)
    if not path.is_absolute():
        path = Path(cfg["paths"]["project_root"]) / path
    if not path.exists():
        raise FileNotFoundError(f"Hard-image replay manifest missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Invalid hard-image replay manifest {path}: {error}") from error
    if payload.get("format") != 1 or payload.get("kind") != "hard_image_replay" or payload.get("split") != "train":
        raise RuntimeError("Hard-image replay manifest must be format=1, kind='hard_image_replay', split='train'.")
    images = payload.get("images", {})
    if not isinstance(images, dict) or not images:
        raise RuntimeError(f"Hard-image replay manifest {path} contains no images.")
    for relative in images:
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise RuntimeError(f"Hard-image replay manifest contains unsafe path {relative!r}.")
    return set(images)


def wrap_hard_replay_dataset(dataset, cfg: dict[str, Any], mode: str):
    if mode != "train":
        return dataset
    images = load_hard_replay_images(cfg)
    repeats = int(cfg["distillation"].get("hard_image_replay_repeats", 0))
    if not images or repeats <= 0:
        return dataset
    replay = HardImageReplayDataset(dataset, images, repeats)
    LOGGER.info(
        f"OOF HARD REPLAY: images={replay.replay_images}, repeats={repeats}, "
        f"extra_samples={replay.replay_samples}, epoch_samples={len(replay)}"
    )
    return replay


class ReplayDetectionTrainer(DetectionTrainer):
    """Plain detector control with exactly the same OOF image replay as KD trials."""

    experiment_config: dict[str, Any] | None = None

    @classmethod
    def configure(cls, cfg: dict[str, Any]) -> None:
        cls.experiment_config = copy.deepcopy(cfg)

    def __init__(self, *args, **kwargs):
        if self.experiment_config is None:
            raise RuntimeError("ReplayDetectionTrainer.configure() must be called before YOLO.train().")
        super().__init__(*args, **kwargs)

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        dataset = super().build_dataset(img_path, mode, batch)
        assert self.experiment_config is not None
        return wrap_hard_replay_dataset(dataset, self.experiment_config, mode)


class DistillationDetectionTrainer(DetectionTrainer):
    """Attach KD to the actual model that Ultralytics trains, validates and checkpoints.

    Ultralytics reconstructs a new DetectionModel inside Model.train(). Attaching KD to
    the caller's YOLO.model therefore silently loses it. This trainer adds the modules
    in get_model(), before weight loading and optimizer construction, and installs the
    criterion after the normal model attributes are set.
    """

    experiment_config: dict[str, Any] | None = None
    teacher_cache_dir: Path | None = None

    @classmethod
    def configure(cls, cfg: dict[str, Any], cache_dir: str | Path) -> None:
        cls.experiment_config = copy.deepcopy(cfg)
        cls.teacher_cache_dir = Path(cache_dir).resolve()

    def __init__(self, *args, **kwargs):
        if self.experiment_config is None or self.teacher_cache_dir is None:
            raise RuntimeError("DistillationDetectionTrainer.configure() must be called before YOLO.train().")
        super().__init__(*args, **kwargs)
        if self.ddp:
            raise RuntimeError("The cache-based custom KD trainer currently supports one GPU per process. Use device=0.")
        self.kd_loss: DistillationLoss | None = None
        self._kd_gradient_handles: list[Any] = []
        self._kd_addon_parameters = 0
        self._health_optimizer_steps_skipped = 0
        self.add_callback("on_train_epoch_start", self._on_kd_epoch_start)
        self.add_callback("on_train_batch_end", self._on_kd_batch_end)
        self.add_callback("on_train_epoch_end", self._on_kd_epoch_end)

    @property
    def kd_cfg(self) -> dict[str, Any]:
        assert self.experiment_config is not None
        return self.experiment_config

    def get_model(self, cfg: str | dict | None = None, weights=None, verbose: bool = True):
        model = DetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        channels = infer_student_channels(model, int(self.kd_cfg["dataset"]["image_size"]))
        experiment = self.kd_cfg["runtime"]["experiment"]
        model.add_module(
            "distill_addons",
            StudentDistillAddons(
                channels,
                int(self.kd_cfg["teacher"]["feature_channels"]),
                int(self.kd_cfg["dataset"]["nc"]),
                enable_vehicle_bg=bool(self.kd_cfg["distillation"].get("vehicle_bg_enabled", False)),
                enable_global=experiment in {"g", "gp"},
                enable_prototype=experiment in {"p", "gp"},
                prototype_dim=int(self.kd_cfg["distillation"].get("prototype_embedding_dim", 512)),
            ),
        )
        if weights:
            model.load(weights)
        if bool(self.kd_cfg["runtime"].get("reset_kd_calibration", False)):
            reset = reset_kd_calibration_buffers(model.distill_addons)
            LOGGER.info(f"KD NEW OBJECTIVE: reset {reset} inherited calibration buffers before training.")
        recovery = self.kd_cfg["runtime"].get("legacy_calibration_recovery")
        if recovery:
            restored = restore_legacy_kd_calibration(
                model,
                recovery["health_file"],
                int(recovery["checkpoint_epoch"]),
            )
            LOGGER.info(
                f"Recovered {restored} pre-fix KD calibration buffers inside the real trainer model "
                f"from {recovery['health_file']}."
            )
        return model

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        dataset = super().build_dataset(img_path, mode, batch)
        replay_manifest = self.kd_cfg["distillation"].get("hard_image_replay_manifest")
        if replay_manifest:
            return wrap_hard_replay_dataset(dataset, self.kd_cfg, mode)
        manifest = self.kd_cfg["distillation"].get("vehicle_negative_manifest")
        repeats = int(self.kd_cfg["distillation"].get("vehicle_negative_replay_repeats", 0))
        if mode != "train" or not manifest or repeats <= 0:
            return dataset
        manifest_path = Path(manifest)
        if not manifest_path.is_absolute():
            manifest_path = Path(self.kd_cfg["paths"]["project_root"]) / manifest_path
        store = VehicleNegativeStore(manifest_path)
        replay = HardImageReplayDataset(dataset, set(store.images), repeats)
        LOGGER.info(
            f"VEHICLE NEGATIVE REPLAY: images={replay.replay_images}, repeats={repeats}, "
            f"extra_samples={replay.replay_samples}, epoch_samples={len(replay)}"
        )
        return replay

    def set_model_attributes(self) -> None:
        super().set_model_attributes()
        model = unwrap_model(self.model)
        addons = getattr(model, "distill_addons", None)
        if not isinstance(addons, StudentDistillAddons):
            raise RuntimeError("KD lifecycle failure: distill_addons are absent from the real trainer.model.")
        model.kd_experiment = self.kd_cfg["runtime"]["experiment"]
        model.kd_cache_manifest = copy.deepcopy(self.kd_cfg["runtime"]["cache_manifest"])
        from .provenance import portable_dataset_identity

        model.kd_dataset_identity = portable_dataset_identity(
            self.kd_cfg["runtime"]["dataset_identity"]
        )
        model.kd_artifact_namespace = self.kd_cfg["runtime"].get(
            "artifact_namespace"
        )
        model.kd_prototype_bank = copy.deepcopy(
            self.kd_cfg["runtime"].get("prototype_bank")
        )
        model.kd_objective_fingerprint = kd_objective_fingerprint(self.kd_cfg)
        original_criterion = model.init_criterion()
        experiment = self.kd_cfg["runtime"]["experiment"]
        prototype_bank = None
        required_fields: set[str] = set()
        if experiment in {"p", "gp"}:
            bank_path = self.kd_cfg["distillation"].get("prototype_bank")
            if not bank_path:
                raise RuntimeError("P/GP requires distillation.prototype_bank.")
            prototype_bank = PrototypeBank.load(
                bank_path,
                self.kd_cfg,
                self.kd_cfg["runtime"]["cache_manifest"],
            )
            required_fields.add("roi_embeddings")
        self.kd_loss = DistillationLoss(
            original_criterion,
            addons,
            TeacherSignalStore(
                self.teacher_cache_dir,
                manifest_fingerprint=self.kd_cfg["runtime"]["cache_manifest"].get(
                    "compatibility_fingerprint"
                ),
                required_fields=required_fields,
            ),
            self.kd_cfg,
            prototype_bank=prototype_bank,
        )
        model.criterion = self.kd_loss

    def _setup_train(self) -> None:
        super()._setup_train()
        model = unwrap_model(self.model)
        addons = getattr(model, "distill_addons", None)
        if not isinstance(addons, StudentDistillAddons) or self.kd_loss is None:
            raise RuntimeError("KD lifecycle failure: trainer setup discarded the KD modules or criterion.")

        optimizer_ids = {id(parameter) for group in self.optimizer.param_groups for parameter in group["params"]}
        missing = [name for name, parameter in addons.named_parameters() if parameter.requires_grad and id(parameter) not in optimizer_ids]
        if missing:
            raise RuntimeError(f"KD lifecycle failure: optimizer is missing addon parameters: {missing[:5]}")

        experiment = self.kd_cfg["runtime"]["experiment"]
        if experiment in {"f", "fk"}:
            parameter = next(parameter for parameter in addons.projectors.parameters() if parameter.requires_grad)
            self._kd_gradient_handles.append(parameter.register_hook(lambda grad: self.kd_loss.record_gradient("feature_projector", grad)))
        if experiment in {"k", "fk"}:
            parameter = next(parameter for parameter in addons.student_roi_head.parameters() if parameter.requires_grad)
            self._kd_gradient_handles.append(parameter.register_hook(lambda grad: self.kd_loss.record_gradient("roi_head", grad)))
        if experiment in {"g", "gp"}:
            # Hook P4 explicitly: G never relies on the interpolated teacher P3.
            parameter = next(
                parameter
                for parameter in addons.projectors[1].parameters()
                if parameter.requires_grad
            )
            self._kd_gradient_handles.append(
                parameter.register_hook(
                    lambda grad: self.kd_loss.record_gradient("global_projector", grad)
                )
            )
        if experiment in {"p", "gp"}:
            if addons.prototype_fuse is None:
                raise RuntimeError("KD lifecycle failure: prototype_fuse is missing.")
            parameter = next(
                parameter
                for parameter in addons.prototype_fuse.parameters()
                if parameter.requires_grad
            )
            self._kd_gradient_handles.append(
                parameter.register_hook(
                    lambda grad: self.kd_loss.record_gradient("prototype_head", grad)
                )
            )
        if bool(self.kd_cfg["distillation"].get("vehicle_bg_enabled", False)):
            if addons.vehicle_bg_head is None:
                raise RuntimeError("KD lifecycle failure: vehicle_bg_head is missing.")
            parameter = next(parameter for parameter in addons.vehicle_bg_head.parameters() if parameter.requires_grad)
            self._kd_gradient_handles.append(
                parameter.register_hook(lambda grad: self.kd_loss.record_gradient("vehicle_bg_head", grad))
            )

        addon_parameters = sum(parameter.numel() for parameter in addons.parameters())
        self._kd_addon_parameters = addon_parameters
        LOGGER.info(
            f"KD ACTIVE: experiment={experiment}, trainer_model_id={id(model)}, "
            f"addon_parameters={addon_parameters:,}, optimizer_verified=True, cache={self.teacher_cache_dir}"
        )

    def _on_kd_epoch_start(self, _trainer) -> None:
        assert self.kd_loss is not None
        self.kd_loss.set_epoch(self.epoch)

    def _on_kd_batch_end(self, _trainer) -> None:
        assert self.kd_loss is not None
        patience = int(self.kd_cfg["distillation"].get("health_patience_batches", 10))
        self.kd_loss.assert_health(patience)
        health_batches = self.kd_cfg["runtime"].get("health_batches")
        if health_batches is not None and self.kd_loss.batch_calls >= int(health_batches):
            LOGGER.info(f"KD health-check reached {health_batches} batches; ending this smoke-test run cleanly.")
            self.stop = True

    def _on_kd_epoch_end(self, _trainer) -> None:
        assert self.kd_loss is not None
        patience = int(self.kd_cfg["distillation"].get("health_patience_batches", 10))
        self.kd_loss.assert_health(patience)
        summary = self.kd_loss.epoch_summary(reset=True)
        summary.update(
            experiment=self.kd_cfg["runtime"]["experiment"],
            addon_parameters=self._kd_addon_parameters,
            optimizer_verified=True,
            health_check_no_updates=self.kd_cfg["runtime"].get("health_batches") is not None,
            health_optimizer_steps_skipped=self._health_optimizer_steps_skipped,
        )
        health_file = Path(self.save_dir) / "kd_health.jsonl"
        with health_file.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(summary, ensure_ascii=False) + "\n")
        LOGGER.info(
            "KD HEALTH: "
            f"epoch={summary['epoch']} feature={summary['feature_raw_mean']:.6g} "
            f"cls={summary['cls_raw_mean']:.6g} "
            f"global={summary['global_raw_mean']:.6g} "
            f"prototype={summary['prototype_raw_mean']:.6g} kd={summary['kd_mean']:.6g} "
            f"vehicle_bg={summary['vehicle_bg_raw_mean']:.6g} "
            f"rois={summary['valid_rois']} prototype_rois={summary['prototype_valid_rois']} "
            f"routes=G{summary['global_routed_objects']}/P{summary['prototype_routed_objects']} "
            f"cache_misses={summary['cache_misses']} "
            f"teacher_keep={summary['teacher_kept']}/{summary['teacher_candidates']} "
            f"weights=({summary['feature_kd_weight']:.4g},{summary['cls_kd_weight']:.4g}) "
            f"conflict(det/F)={summary['det_feature_grad_negative_rate']:.3f} "
            f"conflict(det/K)={summary['det_cls_grad_negative_rate']:.3f} "
            f"feature_grad_events={summary['feature_grad_events']} roi_grad_events={summary['roi_grad_events']}"
            f" global_grad_events={summary['global_grad_events']}"
            f" prototype_grad_events={summary['prototype_grad_events']}"
            f" vehicle_bg_grad_events={summary['vehicle_bg_grad_events']}"
        )

    def optimizer_step(self) -> None:
        """A wiring check measures gradients but must never mutate the baseline.

        Full runs use Ultralytics' normal scaler/optimizer/EMA lifecycle.  Only
        ``--health-batches`` runs discard the already-observed gradients, which
        makes their final validation a checkpoint-equivalence guard instead of
        a misleading ten-batch training result.
        """
        if self.kd_cfg["runtime"].get("health_batches") is not None:
            self.optimizer.zero_grad(set_to_none=True)
            self._health_optimizer_steps_skipped += 1
            return
        super().optimizer_step()

    def save_model(self):
        """Keep KD calibration control state out of EMA averaging."""
        live = unwrap_model(self.model)
        ema = unwrap_model(self.ema.ema)
        live_addons = getattr(live, "distill_addons", None)
        ema_addons = getattr(ema, "distill_addons", None)
        if not isinstance(live_addons, StudentDistillAddons) or not isinstance(ema_addons, StudentDistillAddons):
            raise RuntimeError("KD checkpoint lifecycle failure: live/EMA addons are missing before save.")
        copied = sync_kd_calibration_buffers(live_addons, ema_addons)
        ema.kd_calibration_buffers_exact = True
        LOGGER.info(f"KD CHECKPOINT: copied {copied} calibration buffers exactly into EMA checkpoint state.")
        return super().save_model()

    def _model_train(self) -> None:
        super()._model_train()
        if self.kd_cfg["runtime"].get("health_batches") is not None:
            # Optimizer-free wiring checks must not alter BatchNorm running
            # statistics either. Dropout in the auxiliary RoI head remains in
            # train mode so the checked graph matches a real KD batch.
            for module in unwrap_model(self.model).modules():
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    module.eval()

    def final_eval(self) -> None:
        super().final_eval()
        if RANK not in {-1, 0} or not self.best.exists():
            return
        exports = [(self.best, self.wdir / "best_deploy.pt")]
        if self.last.exists():
            exports.append((self.last, self.wdir / "last_deploy.pt"))
        for source, deploy_path in exports:
            addon_parameters = export_plain_yolo_checkpoint(source, deploy_path)
            LOGGER.info(
                f"Exported plain-YOLO deployment checkpoint to {deploy_path} "
                f"(removed {addon_parameters:,} training-only KD parameters)."
            )
