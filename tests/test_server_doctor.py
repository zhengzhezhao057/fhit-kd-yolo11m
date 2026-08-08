from pathlib import Path

from src.server_doctor import DINO_REQUIRED_MODULES, full_checks, vram_capacity


def test_marketed_24gb_4090_is_not_rejected_as_23_point_5_gib():
    decimal_gb, binary_gib, accepted = vram_capacity(25_245_122_560)
    assert decimal_gb >= 24
    assert 23 <= binary_gib < 24
    assert accepted


def test_actual_sub_24gb_card_is_rejected():
    _decimal_gb, _binary_gib, accepted = vram_capacity(23_000_000_000)
    assert not accepted


def test_explicit_disk_override_is_reflected_in_report(monkeypatch, tmp_path: Path):
    class Usage:
        free = 50 * 1024**3

    monkeypatch.setattr("src.server_doctor.shutil.disk_usage", lambda _path: Usage())
    checks = full_checks(tmp_path / "missing.yaml", require_config=False, min_free_gb=45)
    disk = next(check for check in checks if check.name == "Free disk")
    assert disk.ok
    assert "requires >= 45 GiB" in disk.detail


def test_dinov3_runtime_import_dependencies_are_preflighted():
    assert DINO_REQUIRED_MODULES["torchmetrics"] == "torchmetrics"
    assert DINO_REQUIRED_MODULES["omegaconf"] == "omegaconf"
