from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.audit_dino_scene_neighbors import (
    build_cache_contract,
    cross_split_topk,
    initialize_or_validate_cache,
    summarize_neighbors,
)


def make_contract(*, image_size: int = 448, weights: str = "a" * 64) -> dict:
    return build_cache_contract(
        dataset_fingerprint="d" * 64,
        weights_sha256=weights,
        dino_repo_git="1" * 40,
        index_fingerprint="i" * 64,
        image_size=image_size,
        layers=(11, 17, 23),
        model_name="dinov3_vitl16",
        code_sha256="c" * 64,
    )


def cache_rows() -> list[dict[str, str]]:
    return [
        {
            "split": split,
            "source": "official",
            "source_family": "official_mar20",
            "scene_id": f"mar20:MAR20_{index}",
            "cluster_id": f"official:mar20:MAR20_{index}",
            "image": f"MAR20_{index}.jpg",
            "image_sha256": str(index) * 64,
        }
        for index, split in enumerate(("train", "val", "test"), start=1)
    ]


def test_cache_fingerprint_binds_weights_preprocess_and_rejects_mismatch(
    tmp_path: Path,
) -> None:
    base = make_contract()
    assert base["cache_fingerprint"] != make_contract(image_size=384)["cache_fingerprint"]
    assert base["cache_fingerprint"] != make_contract(weights="b" * 64)["cache_fingerprint"]

    embeddings, valid = initialize_or_validate_cache(
        tmp_path / "cache",
        rows=cache_rows(),
        contract=base,
        embedding_dim=6,
    )
    embeddings[:] = 1
    valid[:] = True
    embeddings.flush()
    valid.flush()
    resumed_embeddings, resumed_valid = initialize_or_validate_cache(
        tmp_path / "cache",
        rows=cache_rows(),
        contract=base,
        embedding_dim=6,
    )
    assert resumed_embeddings.shape == (3, 6)
    assert bool(resumed_valid.all())
    metadata = json.loads((tmp_path / "cache/metadata.json").read_text("utf-8"))
    assert metadata["dataset_fingerprint"] == "d" * 64
    assert metadata["weights_sha256"] == "a" * 64
    with pytest.raises(RuntimeError, match="fingerprint mismatch"):
        initialize_or_validate_cache(
            tmp_path / "cache",
            rows=cache_rows(),
            contract=make_contract(image_size=384),
            embedding_dim=6,
        )


def test_neighbors_are_cross_split_and_sorted_by_cosine() -> None:
    rows = [
        {
            "split": "train",
            "image": "MAR20_1.jpg",
            "scene_id": "mar20:MAR20_1",
        },
        {
            "split": "train",
            "image": "MAR20_2.jpg",
            "scene_id": "mar20:MAR20_2",
        },
        {
            "split": "val",
            "image": "MAR20_3.jpg",
            "scene_id": "mar20:MAR20_3",
        },
        {
            "split": "test",
            "image": "MAR20_100.jpg",
            "scene_id": "mar20:MAR20_100",
        },
    ]
    embeddings = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.999, 0.001, 0.0],
            [0.98, 0.20, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    neighbors = cross_split_topk(embeddings, rows, top_k=2, block_size=2)
    assert len(neighbors) == len(rows) * 2
    assert all(row["query_split"] != row["neighbor_split"] for row in neighbors)
    for query_index in range(len(rows)):
        query_rows = [row for row in neighbors if row["query_index"] == query_index]
        assert [row["neighbor_rank"] for row in query_rows] == [1, 2]
        assert query_rows[0]["cosine_similarity"] >= query_rows[1]["cosine_similarity"]
    # Same-split vector 1 is closest to vector 0, but must never be returned.
    first = [row for row in neighbors if row["query_index"] == 0]
    assert all(row["neighbor_index"] != 1 for row in first)
    assert first[0]["neighbor_index"] == 2

    candidates, summary = summarize_neighbors(
        neighbors, candidate_threshold=0.95, very_high_threshold=0.995
    )
    assert candidates == sorted(
        candidates,
        key=lambda row: (
            -float(row["cosine_similarity"]),
            row["query_image"],
            row["neighbor_rank"],
        ),
    )
    assert summary["automatic_dataset_mutation"] is False
    assert summary["requires_manual_review"] is True
