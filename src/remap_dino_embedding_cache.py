"""Safely reuse content-bound DINO embeddings after a split-only rebuild."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .audit_dino_scene_neighbors import (
    _index_payload,
    build_cache_contract,
    canonical_sha256,
    read_csv,
    select_official_rows,
    write_csv,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source-cache", required=True, type=Path)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    source_meta = json.loads((args.source_cache / "metadata.json").read_text("utf-8"))
    source_index = read_csv(args.source_cache / "index.csv")
    source_embeddings = np.load(args.source_cache / "embeddings.npy", mmap_mode="r")
    source_valid = np.load(args.source_cache / "valid.npy", mmap_mode="r")
    if not bool(source_valid.all()):
        raise RuntimeError("Source embedding cache is incomplete")
    old = {row["image_sha256"]: int(row["row_index"]) for row in source_index}
    rows = select_official_rows(args.root / "split_manifest.csv")
    missing = [row["image"] for row in rows if row["image_sha256"] not in old]
    if missing or len(old) != len(rows):
        raise RuntimeError(f"Cache inventory differs: missing={missing[:3]} old={len(old)} new={len(rows)}")
    info = json.loads((args.root / "dataset_fingerprint.json").read_text("utf-8"))
    contract = build_cache_contract(
        dataset_fingerprint=info["dataset_fingerprint"],
        weights_sha256=source_meta["weights_sha256"],
        dino_repo_git=source_meta["dino_repo_git"],
        index_fingerprint=canonical_sha256(_index_payload(rows)),
        image_size=source_meta["preprocess"]["image_size"],
        layers=tuple(source_meta["embedding"]["layers"]),
        model_name=source_meta["model_name"],
        code_sha256=source_meta["audit_code_sha256"],
    )
    args.out.mkdir(parents=True, exist_ok=True)
    embeddings = np.lib.format.open_memmap(
        args.out / "embeddings.npy", mode="w+", dtype=np.float32,
        shape=(len(rows), source_embeddings.shape[1]),
    )
    for index, row in enumerate(rows):
        embeddings[index] = source_embeddings[old[row["image_sha256"]]]
    embeddings.flush()
    np.save(args.out / "valid.npy", np.ones(len(rows), dtype=np.bool_))
    index_rows = [{"row_index": index, **payload} for index, payload in enumerate(_index_payload(rows))]
    write_csv(args.out / "index.csv", list(index_rows[0]), index_rows)
    metadata = {**contract, "embedding_dim": int(source_embeddings.shape[1]), "rows": len(rows),
                "embeddings_file": "embeddings.npy", "valid_file": "valid.npy", "index_file": "index.csv",
                "reused_from_cache_fingerprint": source_meta["cache_fingerprint"]}
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "cache_fingerprint": metadata["cache_fingerprint"],
                      "reused_from": metadata["reused_from_cache_fingerprint"]}, indent=2))


if __name__ == "__main__":
    main()
