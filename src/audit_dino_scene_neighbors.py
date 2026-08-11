from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import io
import json
import math
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from .teacher import DINO_SAT_MEAN, DINO_SAT_STD


SPLITS = ("train", "val", "test")
DEFAULT_LAYERS = (11, 17, 23)


def canonical_sha256(payload: object) -> str:
    blob = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision(repo: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(Path(repo)), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, fields: list[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def select_official_rows(
    manifest: Path, *, max_images: int | None = None
) -> list[dict[str, str]]:
    rows = [row for row in read_csv(manifest) if row.get("source") == "official"]
    rows.sort(key=lambda row: (row["split"], row["image"].casefold()))
    if not rows:
        raise ValueError(f"No official rows in {manifest}")
    invalid = [row for row in rows if row["split"] not in SPLITS]
    if invalid:
        raise ValueError(f"Invalid official split values: {invalid[:3]}")
    if max_images is None or max_images >= len(rows):
        return rows
    if max_images <= 0:
        raise ValueError("--max-images must be positive")
    # A smoke subset must still compare across splits. Round-robin by split,
    # rather than taking the first lexicographic rows from one split.
    buckets = {
        split: [row for row in rows if row["split"] == split] for split in SPLITS
    }
    selected: list[dict[str, str]] = []
    positions = Counter()
    while len(selected) < max_images:
        progressed = False
        for split in SPLITS:
            if len(selected) >= max_images:
                break
            position = positions[split]
            if position < len(buckets[split]):
                selected.append(buckets[split][position])
                positions[split] += 1
                progressed = True
        if not progressed:
            break
    selected.sort(key=lambda row: (row["split"], row["image"].casefold()))
    return selected


def build_cache_contract(
    *,
    dataset_fingerprint: str,
    weights_sha256: str,
    dino_repo_git: str,
    index_fingerprint: str,
    image_size: int,
    layers: tuple[int, ...],
    model_name: str,
    code_sha256: str,
) -> dict:
    contract = {
        "format": 1,
        "kind": "dino_scene_embedding_cache",
        "dataset_fingerprint": dataset_fingerprint,
        "weights_sha256": weights_sha256,
        "dino_repo_git": dino_repo_git,
        "index_fingerprint": index_fingerprint,
        "model_name": model_name,
        "preprocess": {
            "name": "whole_image_square_letterbox",
            "image_size": int(image_size),
            "resize": "PIL.Image.Resampling.BILINEAR",
            "pad_rgb": [round(value * 255) for value in DINO_SAT_MEAN],
            "input_scale": "uint8_to_float32_div_255",
            "mean_rgb": list(DINO_SAT_MEAN),
            "std_rgb": list(DINO_SAT_STD),
        },
        "embedding": {
            "layers": list(layers),
            "layer_pool": "l2_normalize(cls)+l2_normalize(mean_patch), then l2",
            "fusion": "concatenate layer vectors then final l2 normalization",
            "storage_dtype": "float32",
        },
        "audit_code_sha256": code_sha256,
    }
    contract["cache_fingerprint"] = canonical_sha256(contract)
    return contract


def preprocess_image(data: bytes, image_size: int) -> np.ndarray:
    if image_size <= 0 or image_size % 16:
        raise ValueError("DINO image size must be a positive multiple of 16")
    with Image.open(io.BytesIO(data)) as source:
        image = source.convert("RGB")
        width, height = image.size
        ratio = min(image_size / max(width, 1), image_size / max(height, 1))
        resized = image.resize(
            (max(1, round(width * ratio)), max(1, round(height * ratio))),
            Image.Resampling.BILINEAR,
        )
    canvas = Image.new(
        "RGB",
        (image_size, image_size),
        tuple(round(value * 255) for value in DINO_SAT_MEAN),
    )
    left = (image_size - resized.width) // 2
    top = (image_size - resized.height) // 2
    canvas.paste(resized, (left, top))
    array = np.asarray(canvas, dtype=np.float32) / 255.0
    array = (array - np.asarray(DINO_SAT_MEAN, dtype=np.float32)) / np.asarray(
        DINO_SAT_STD, dtype=np.float32
    )
    return np.transpose(array, (2, 0, 1)).copy()


def fused_global_embedding(outputs) -> "object":
    import torch
    import torch.nn.functional as functional

    layers = []
    for patch_tokens, class_token in outputs:
        cls = functional.normalize(class_token.float(), dim=-1)
        patch = functional.normalize(patch_tokens.float().mean(dim=1), dim=-1)
        layers.append(functional.normalize(cls + patch, dim=-1))
    return functional.normalize(torch.cat(layers, dim=-1), dim=-1)


def load_local_backbone(dino_repo: Path, model_name: str, weights: Path):
    """Load only the backbone module, avoiding hubconf's unrelated eval dependencies."""
    import torch

    repository = str(Path(dino_repo).resolve())
    sys.path.insert(0, repository)
    try:
        backbones = importlib.import_module("dinov3.hub.backbones")
        factory = getattr(backbones, model_name, None)
        if factory is None or not callable(factory):
            raise ValueError(f"Unknown DINOv3 backbone factory: {model_name}")
        # The SAT ViT-L checkpoint uses an untied local CLS norm; passing the
        # filename lets the official factory select that architecture without
        # asking torch.hub to copy the 1.2 GB local file into its download cache.
        model = factory(pretrained=False, weights=str(Path(weights).resolve()))
        state = torch.load(Path(weights), map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        model.load_state_dict(state, strict=True)
        return model
    finally:
        try:
            sys.path.remove(repository)
        except ValueError:
            pass


def _index_payload(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "split": row["split"],
            "source": row["source"],
            "source_family": row.get("source_family", ""),
            "scene_id": row["scene_id"],
            "cluster_id": row.get("cluster_id", row["scene_id"]),
            "image": row["image"],
            "image_sha256": row["image_sha256"],
        }
        for row in rows
    ]


def initialize_or_validate_cache(
    cache_dir: Path,
    *,
    rows: list[dict[str, str]],
    contract: dict,
    embedding_dim: int,
) -> tuple[np.memmap, np.memmap]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = cache_dir / "metadata.json"
    embeddings_path = cache_dir / "embeddings.npy"
    valid_path = cache_dir / "valid.npy"
    index_path = cache_dir / "index.csv"
    index_rows = [
        {"row_index": index, **payload}
        for index, payload in enumerate(_index_payload(rows))
    ]
    index_fields = [
        "row_index",
        "split",
        "source",
        "source_family",
        "scene_id",
        "cluster_id",
        "image",
        "image_sha256",
    ]
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text("utf-8"))
        if metadata.get("cache_fingerprint") != contract["cache_fingerprint"]:
            raise RuntimeError(
                "Embedding cache fingerprint mismatch. Preserve the old cache and use a new --out directory."
            )
        if int(metadata.get("embedding_dim", -1)) != embedding_dim:
            raise RuntimeError("Embedding cache dimension mismatch")
        if read_csv(index_path) != [
            {field: str(row[field]) for field in index_fields} for row in index_rows
        ]:
            raise RuntimeError("Embedding cache index differs from the current manifest")
        embeddings = np.load(embeddings_path, mmap_mode="r+")
        valid = np.load(valid_path, mmap_mode="r+")
        if embeddings.shape != (len(rows), embedding_dim) or valid.shape != (len(rows),):
            raise RuntimeError("Embedding cache array shape mismatch")
        return embeddings, valid

    if any(path.exists() for path in (embeddings_path, valid_path, index_path)):
        raise RuntimeError("Incomplete embedding cache exists without metadata.json")
    embeddings = np.lib.format.open_memmap(
        embeddings_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(rows), embedding_dim),
    )
    valid = np.lib.format.open_memmap(
        valid_path, mode="w+", dtype=np.bool_, shape=(len(rows),)
    )
    valid[:] = False
    embeddings.flush()
    valid.flush()
    write_csv(index_path, index_fields, index_rows)
    metadata = {
        **contract,
        "embedding_dim": embedding_dim,
        "rows": len(rows),
        "embeddings_file": "embeddings.npy",
        "valid_file": "valid.npy",
        "index_file": "index.csv",
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return embeddings, valid


def extract_embeddings(
    *,
    root: Path,
    rows: list[dict[str, str]],
    dino_repo: Path,
    weights: Path,
    cache_dir: Path,
    contract: dict,
    image_size: int,
    layers: tuple[int, ...],
    model_name: str,
    device: str,
    batch_size: int,
) -> dict:
    import torch

    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    model = load_local_backbone(dino_repo, model_name, weights).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    embedding_dim = int(model.embed_dim) * len(layers)
    embeddings, valid = initialize_or_validate_cache(
        cache_dir,
        rows=rows,
        contract=contract,
        embedding_dim=embedding_dim,
    )
    pending = [index for index in range(len(rows)) if not bool(valid[index])]
    started = time.time()
    processed = 0
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    mean = torch.tensor(DINO_SAT_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(DINO_SAT_STD, device=device).view(1, 3, 1, 1)
    del mean, std  # preprocessing is deterministic on CPU; kept in contract above.
    try:
        for offset in range(0, len(pending), batch_size):
            indices = pending[offset : offset + batch_size]
            arrays = []
            for index in indices:
                row = rows[index]
                path = root / "images" / row["split"] / row["image"]
                data = path.read_bytes()
                digest = hashlib.sha256(data).hexdigest()
                if digest != row["image_sha256"]:
                    raise RuntimeError(
                        f"Image hash mismatch before DINO extraction: {row['split']}/{row['image']}"
                    )
                arrays.append(preprocess_image(data, image_size))
            tensor = torch.from_numpy(np.stack(arrays)).to(device=device, dtype=torch.float32)
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if device.startswith("cuda")
                else torch.autocast(device_type="cpu", enabled=False)
            )
            with torch.inference_mode(), autocast:
                outputs = model.get_intermediate_layers(
                    tensor,
                    n=layers,
                    reshape=False,
                    return_class_token=True,
                    norm=True,
                )
                fused = fused_global_embedding(outputs).cpu().numpy().astype(np.float32)
            embeddings[indices] = fused
            valid[indices] = True
            processed += len(indices)
            if processed % max(batch_size * 10, 10) == 0 or offset + batch_size >= len(pending):
                embeddings.flush()
                valid.flush()
                print(
                    f"DINO EMBEDDINGS: {int(valid.sum())}/{len(rows)} valid; "
                    f"this_run={processed}; elapsed={time.time() - started:.1f}s",
                    flush=True,
                )
    except torch.OutOfMemoryError as error:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise RuntimeError(
            "DINOv3 ViT-L extraction ran out of VRAM. Retry with --batch 1 "
            "and, if necessary, a new output directory with --image-size 384."
        ) from error
    peak_vram = (
        int(torch.cuda.max_memory_allocated(device)) if device.startswith("cuda") else 0
    )
    return {
        "rows": len(rows),
        "valid": int(valid.sum()),
        "processed_this_run": processed,
        "embedding_dim": embedding_dim,
        "elapsed_seconds": time.time() - started,
        "peak_vram_bytes": peak_vram,
        "peak_vram_gib": peak_vram / 1024**3,
        "complete": bool(valid.all()),
    }


def source_metadata(image: str, scene_id: str) -> dict[str, object]:
    import re

    stem = Path(image).stem
    pan = re.match(
        r"^(\d{2})-PAN-(\d{8})-(\d+)-(\d+)-(L\d+)-CCD", stem, re.IGNORECASE
    )
    if pan:
        return {
            "family": "pan_l_product",
            "product": f"{pan.group(1)}:{pan.group(5)}",
            "date": pan.group(2),
            "x": int(pan.group(3)),
            "y": int(pan.group(4)),
        }
    coordinate = re.match(
        r"^([EW])(\d+(?:\.\d+)?)_([NS])(\d+(?:\.\d+)?)_(\d{8})_(L1A\d+)-PAN",
        stem,
        re.IGNORECASE,
    )
    if coordinate:
        lon = float(coordinate.group(2)) * (-1 if coordinate.group(1).upper() == "W" else 1)
        lat = float(coordinate.group(4)) * (-1 if coordinate.group(3).upper() == "S" else 1)
        return {
            "family": "coordinate_l1a",
            "product": coordinate.group(6).upper(),
            "date": coordinate.group(5),
            "lon": lon,
            "lat": lat,
        }
    fsc = re.search(r"-([NS])(\d+(?:\.\d+)?)-([EW])(\d+(?:\.\d+)?)", stem, re.IGNORECASE)
    if stem.casefold().startswith("fsc_") and fsc:
        lat = float(fsc.group(2)) * (-1 if fsc.group(1).upper() == "S" else 1)
        lon = float(fsc.group(4)) * (-1 if fsc.group(3).upper() == "W" else 1)
        return {
            "family": "fsc_location",
            "product": scene_id,
            "lat": lat,
            "lon": lon,
        }
    mar = re.match(r"^MAR20_(\d+)$", stem, re.IGNORECASE)
    if mar:
        return {"family": "mar20", "product": scene_id, "sequence": int(mar.group(1))}
    return {"family": "other", "product": scene_id}


def metadata_relation(left: dict, right: dict) -> tuple[bool, bool, str]:
    same_product = left.get("product") == right.get("product")
    if left.get("family") != right.get("family"):
        return same_product, False, "different_family"
    family = str(left.get("family"))
    if family == "pan_l_product":
        nearby = left.get("date") == right.get("date") and abs(int(left["x"]) - int(right["x"])) <= 1 and abs(
            int(left["y"]) - int(right["y"])
        ) <= 1
        return same_product, nearby, "same_date_adjacent_grid" if nearby else ""
    if family in {"coordinate_l1a", "fsc_location"}:
        distance = math.hypot(float(left["lat"]) - float(right["lat"]), float(left["lon"]) - float(right["lon"]))
        nearby = distance <= (0.05 if family == "fsc_location" else 0.25)
        return same_product, nearby, f"coordinate_distance_deg={distance:.4f}" if nearby else ""
    if family == "mar20":
        difference = abs(int(left["sequence"]) - int(right["sequence"]))
        return same_product, difference <= 2, f"sequence_delta={difference}" if difference <= 2 else ""
    return same_product, False, ""


def cross_split_topk(
    embeddings: np.ndarray,
    rows: list[dict[str, str]],
    *,
    top_k: int,
    block_size: int = 256,
) -> list[dict]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    vectors = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(~np.isfinite(vectors)) or np.any(norms <= 0):
        raise ValueError("Embeddings contain NaN/Inf or zero vectors")
    vectors = vectors / norms
    splits = np.asarray([row["split"] for row in rows])
    minimum_cross_count = min(int(np.sum(splits != split)) for split in set(splits))
    if minimum_cross_count <= 0:
        raise ValueError("At least two non-empty splits are required for cross-split neighbors")
    effective_k = min(top_k, minimum_cross_count)
    metadata = [source_metadata(row["image"], row["scene_id"]) for row in rows]
    output: list[dict] = []
    for start in range(0, len(rows), block_size):
        stop = min(start + block_size, len(rows))
        scores = vectors[start:stop] @ vectors.T
        scores[splits[start:stop, None] == splits[None, :]] = -np.inf
        candidate_indices = np.argpartition(scores, -effective_k, axis=1)[:, -effective_k:]
        for local_index, neighbors in enumerate(candidate_indices):
            query_index = start + local_index
            ordered = sorted(
                (int(index) for index in neighbors),
                key=lambda index: (-float(scores[local_index, index]), index),
            )
            for rank, neighbor_index in enumerate(ordered, start=1):
                left, right = rows[query_index], rows[neighbor_index]
                same_product, nearby, reason = metadata_relation(
                    metadata[query_index], metadata[neighbor_index]
                )
                output.append(
                    {
                        "query_index": query_index,
                        "query_split": left["split"],
                        "query_image": left["image"],
                        "query_scene": left["scene_id"],
                        "neighbor_rank": rank,
                        "neighbor_index": neighbor_index,
                        "neighbor_split": right["split"],
                        "neighbor_image": right["image"],
                        "neighbor_scene": right["scene_id"],
                        "cosine_similarity": float(scores[local_index, neighbor_index]),
                        "same_source_product": same_product,
                        "nearby_source_metadata": nearby,
                        "metadata_reason": reason,
                    }
                )
    return output


def summarize_neighbors(
    neighbor_rows: list[dict], *, candidate_threshold: float, very_high_threshold: float
) -> tuple[list[dict], dict]:
    scores = np.asarray(
        [float(row["cosine_similarity"]) for row in neighbor_rows], dtype=np.float64
    )
    candidates = [
        row
        for row in neighbor_rows
        if float(row["cosine_similarity"]) >= candidate_threshold
        or bool(row["same_source_product"])
        or bool(row["nearby_source_metadata"])
    ]
    candidates.sort(
        key=lambda row: (-float(row["cosine_similarity"]), row["query_image"], row["neighbor_rank"])
    )
    quantiles = {}
    if len(scores):
        quantiles = {
            key: float(np.quantile(scores, value))
            for key, value in (("p50", 0.50), ("p90", 0.90), ("p95", 0.95), ("p99", 0.99), ("max", 1.0))
        }
    unique_pairs = {
        tuple(sorted((row["query_image"], row["neighbor_image"]))) for row in candidates
    }
    summary = {
        "candidate_threshold": candidate_threshold,
        "very_high_threshold": very_high_threshold,
        "neighbor_rows": len(neighbor_rows),
        "candidate_rows": len(candidates),
        "candidate_unique_pairs": len(unique_pairs),
        "very_high_rows": sum(
            float(row["cosine_similarity"]) >= very_high_threshold for row in neighbor_rows
        ),
        "same_source_product_rows": sum(bool(row["same_source_product"]) for row in neighbor_rows),
        "nearby_source_metadata_rows": sum(bool(row["nearby_source_metadata"]) for row in neighbor_rows),
        "similarity_quantiles": quantiles,
        "requires_manual_review": bool(candidates),
        "automatic_dataset_mutation": False,
    }
    return candidates, summary


def run_neighbors(
    *,
    cache_dir: Path,
    out_dir: Path,
    top_k: int,
    block_size: int,
    candidate_threshold: float,
    very_high_threshold: float,
) -> dict:
    metadata = json.loads((cache_dir / "metadata.json").read_text("utf-8"))
    index = read_csv(cache_dir / "index.csv")
    valid = np.load(cache_dir / "valid.npy", mmap_mode="r")
    if not bool(valid.all()):
        raise RuntimeError(
            f"Embedding cache is incomplete: {int(valid.sum())}/{len(valid)} valid"
        )
    embeddings = np.load(cache_dir / "embeddings.npy", mmap_mode="r")
    rows = cross_split_topk(embeddings, index, top_k=top_k, block_size=block_size)
    candidates, summary = summarize_neighbors(
        rows,
        candidate_threshold=candidate_threshold,
        very_high_threshold=very_high_threshold,
    )
    fields = [
        "query_index",
        "query_split",
        "query_image",
        "query_scene",
        "neighbor_rank",
        "neighbor_index",
        "neighbor_split",
        "neighbor_image",
        "neighbor_scene",
        "cosine_similarity",
        "same_source_product",
        "nearby_source_metadata",
        "metadata_reason",
    ]
    write_csv(out_dir / "cross_split_topk.csv", fields, rows)
    write_csv(out_dir / "manual_review_candidates.csv", fields, candidates)
    report = {
        "format": 1,
        "kind": "dino_scene_neighbor_audit",
        "cache_fingerprint": metadata["cache_fingerprint"],
        "dataset_fingerprint": metadata["dataset_fingerprint"],
        "weights_sha256": metadata["weights_sha256"],
        "dino_repo_git": metadata["dino_repo_git"],
        "index_fingerprint": metadata["index_fingerprint"],
        "model_name": metadata["model_name"],
        "audit_code_sha256": metadata["audit_code_sha256"],
        "preprocess": metadata["preprocess"],
        "embedding": metadata["embedding"],
        "top_k": top_k,
        "block_size": block_size,
        **summary,
        "candidate_file": "manual_review_candidates.csv",
        "all_neighbors_file": "cross_split_topk.csv",
        "interpretation": "Candidates require human review. This audit never deletes images or changes split assignments.",
    }
    (out_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DINOv3-SAT global-embedding cross-split scene-neighbor audit."
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--dino-repo", default=None)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model-name", default="dinov3_vitl16")
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--layers", type=int, nargs="+", default=list(DEFAULT_LAYERS))
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--candidate-threshold", type=float, default=0.985)
    parser.add_argument("--very-high-threshold", type=float, default=0.995)
    parser.add_argument("--max-images", type=int, default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--extract-only", action="store_true")
    mode.add_argument("--neighbors-only", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    manifest = Path(args.manifest) if args.manifest else root / "split_manifest.csv"
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "embedding_cache"
    rows = select_official_rows(manifest, max_images=args.max_images)
    dataset_info = json.loads((root / "dataset_fingerprint.json").read_text("utf-8"))
    index_fingerprint = canonical_sha256(_index_payload(rows))
    layers = tuple(args.layers)
    if not layers or sorted(set(layers)) != list(layers):
        raise ValueError("--layers must be unique and strictly increasing")

    if args.neighbors_only:
        if not (cache_dir / "metadata.json").is_file():
            raise FileNotFoundError(f"Missing embedding cache: {cache_dir}")
        cache_metadata = json.loads((cache_dir / "metadata.json").read_text("utf-8"))
        if cache_metadata.get("dataset_fingerprint") != dataset_info["dataset_fingerprint"]:
            raise RuntimeError("Embedding cache belongs to a different dataset fingerprint")
        if cache_metadata.get("index_fingerprint") != index_fingerprint:
            raise RuntimeError("Embedding cache index differs from the requested manifest/subset")
        if cache_metadata.get("model_name") != args.model_name:
            raise RuntimeError("Embedding cache model differs from --model-name")
        if cache_metadata.get("preprocess", {}).get("image_size") != args.image_size:
            raise RuntimeError("Embedding cache preprocessing differs from --image-size")
        if cache_metadata.get("embedding", {}).get("layers") != list(layers):
            raise RuntimeError("Embedding cache layers differ from --layers")
    else:
        if not args.dino_repo or not args.weights:
            raise ValueError("--dino-repo and --weights are required for extraction")
        dino_repo, weights = Path(args.dino_repo), Path(args.weights)
        if not (dino_repo / "hubconf.py").is_file():
            raise FileNotFoundError(f"Invalid DINOv3 repository: {dino_repo}")
        if not weights.is_file():
            raise FileNotFoundError(f"Missing DINOv3 weights: {weights}")
        import torch

        device = args.device
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        contract = build_cache_contract(
            dataset_fingerprint=dataset_info["dataset_fingerprint"],
            weights_sha256=file_sha256(weights),
            dino_repo_git=git_revision(dino_repo),
            index_fingerprint=index_fingerprint,
            image_size=args.image_size,
            layers=layers,
            model_name=args.model_name,
            code_sha256=file_sha256(Path(__file__)),
        )
        extraction = extract_embeddings(
            root=root,
            rows=rows,
            dino_repo=dino_repo,
            weights=weights,
            cache_dir=cache_dir,
            contract=contract,
            image_size=args.image_size,
            layers=layers,
            model_name=args.model_name,
            device=device,
            batch_size=args.batch,
        )
        print(json.dumps({"extraction": extraction}, ensure_ascii=False, indent=2))
        if not extraction["complete"]:
            raise RuntimeError("Embedding extraction did not complete")

    if not args.extract_only:
        report = run_neighbors(
            cache_dir=cache_dir,
            out_dir=out_dir,
            top_k=args.top_k,
            block_size=args.block_size,
            candidate_threshold=args.candidate_threshold,
            very_high_threshold=args.very_high_threshold,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
