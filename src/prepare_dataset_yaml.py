from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .common import json_dump, load_config
from .artifact_paths import config_dir, dataset_root, server_dataset_root

GENERATOR_VERSION = 1
REQUIRED_KEYS = ("train", "val", "test", "nc", "names")


def build_dataset_yaml(source_yaml: Path, path: str, target: Path, *, dataset_id: str = "scene811_v1") -> dict:
    """Write a machine-specific dataset.yaml from the neutral package template."""
    source = Path(source_yaml)
    if not source.is_file():
        raise FileNotFoundError(source)
    with source.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    missing = [key for key in REQUIRED_KEYS if key not in data]
    if missing:
        raise ValueError(f"Source dataset.yaml is missing keys {missing}: {source}")
    names = {int(key): value for key, value in data["names"].items()}
    if sorted(names) != list(range(len(names))):
        raise ValueError(f"Class ids must be contiguous 0..n-1, got {sorted(names)}")
    for key in ("train", "val", "test"):
        if not isinstance(data[key], str):
            raise ValueError(f"Split {key!r} must be a directory path relative to the dataset root.")
    output = {
        "# dataset_id": dataset_id,
        "# generator": f"src.prepare_dataset_yaml v{GENERATOR_VERSION}",
        "path": str(path).replace("\\", "/"),
        "train": data["train"],
        "val": data["val"],
        "test": data["test"],
        "nc": int(data["nc"]),
        "names": names,
    }
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(output, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return output


def prepare_configs(*, source_yaml: Path, local_out: Path, server_out: Path, local_path: Path, server_path: Path) -> dict:
    build_dataset_yaml(source_yaml, str(local_path), local_out)
    build_dataset_yaml(source_yaml, str(server_path), server_out)
    return {
        "source": str(Path(source_yaml).resolve()),
        "local_yaml": str(Path(local_out).resolve()),
        "server_yaml": str(Path(server_out).resolve()),
        "local_path": str(local_path),
        "server_path": str(server_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate machine-specific Scene811 dataset YAML files.")
    parser.add_argument("--source-yaml", default=None, help="Dataset package dataset.yaml; defaults to dataset_root/dataset.yaml")
    parser.add_argument("--local-out", default=None)
    parser.add_argument("--server-out", default=None)
    parser.add_argument("--local-path", default=None, help="Absolute path embedded in the local YAML")
    parser.add_argument("--server-path", default=None, help="Absolute path embedded in the server YAML")
    args = parser.parse_args()
    source = Path(args.source_yaml) if args.source_yaml else dataset_root() / "dataset.yaml"
    local_out = Path(args.local_out) if args.local_out else config_dir() / "scene811.local.yaml"
    server_out = Path(args.server_out) if args.server_out else config_dir() / "scene811.server.yaml"
    local_path = Path(args.local_path) if args.local_path else dataset_root()
    server_path = Path(args.server_path) if args.server_path else server_dataset_root()
    report = prepare_configs(
        source_yaml=source, local_out=local_out, server_out=server_out,
        local_path=local_path, server_path=server_path,
    )
    json_dump(report, config_dir() / "scene811_yaml_manifest.json")
    for key in ("local_yaml", "server_yaml"):
        print(f"{key}: {report[key]}")


if __name__ == "__main__":
    main()
