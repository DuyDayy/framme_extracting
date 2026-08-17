#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import modal

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from framme_extracting.config import PipelineConfig

APP_NAME = "aic-framme-extracting"
DATA_VOLUME = "aic-data-vol"
WORK_VOLUME = "aic-framme-vol"
CACHE_VOLUME = "hf-cache"
VIDEO_ROOT = Path("/data/video")
BOUNDARY_ROOT = Path("/data/shot_boundaries")
RUN_ROOT = Path("/work/runs")
EXPECTED_VIDEOS = 216
GPU_WORKERS = 10
GPU_TYPE = "A10"
CANDIDATE_CPU_WORKERS = 100
SELECTION_CPU_WORKERS = 100

app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name(DATA_VOLUME, create_if_missing=False)
work_volume = modal.Volume.from_name(WORK_VOLUME, create_if_missing=True)
cache_volume = modal.Volume.from_name(CACHE_VOLUME, create_if_missing=True)

package_dir = str(SRC_ROOT)
base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install("av==14.0.1", "numpy==1.26.4", "pillow==11.1.0")
)
cpu_image = base_image.add_local_dir(package_dir, remote_path="/root/src")
gpu_image = (
    base_image.pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "transformers==4.48.0",
        "einops==0.8.0",
        "timm==1.0.13",
        "peft==0.14.0",
    )
    .env({"HF_HOME": "/cache", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(package_dir, remote_path="/root/src")
)


def _config(value: dict[str, Any]) -> PipelineConfig:
    return PipelineConfig(**value)


def _volume_run_path(run_id: str, *parts: str) -> str:
    return str(Path("/runs") / run_id / Path(*parts))


def _read_manifest(run: Path, config: PipelineConfig) -> dict[str, Any]:
    path = run / "RUN_MANIFEST.json"
    if not path.exists():
        raise RuntimeError(f"missing production manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("config_fingerprint") != config.fingerprint:
        raise RuntimeError("run manifest belongs to a different pipeline config")
    if manifest.get("encoder_fingerprint") != config.encoder_fingerprint:
        raise RuntimeError("run manifest belongs to a different encoder")
    return manifest


@app.function(
    image=cpu_image,
    volumes={"/data": data_volume, "/work": work_volume},
    timeout=60 * 10,
)
def prepare_run_remote(
    run_id: str, config_value: dict[str, Any], video_ids: list[str]
) -> dict[str, Any]:
    from datetime import datetime, timezone

    from framme_extracting.production import make_budget_plan
    from framme_extracting.storage import atomic_write_json, sha256_file

    config = _config(config_value)
    if len(video_ids) != EXPECTED_VIDEOS or len(set(video_ids)) != EXPECTED_VIDEOS:
        raise RuntimeError(f"production requires exactly {EXPECTED_VIDEOS} unique videos")
    paths = [BOUNDARY_ROOT / f"{video_id}.json" for video_id in video_ids]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise RuntimeError(f"missing {len(missing)} boundary files; first={missing[:3]}")

    plan = make_budget_plan(paths, config)
    if plan["status"] != "pass" or plan["assigned_rows"] != config.target_embedding_rows:
        raise RuntimeError(f"invalid production budget: {plan['failures']}")

    run = RUN_ROOT / run_id
    code_hashes = {
        path.name: sha256_file(path)
        for path in sorted(Path("/root/src/framme_extracting").glob("*.py"))
    }
    manifest = {
        "schema_version": "framme-production/v2",
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "video_ids": video_ids,
        "expected_videos": EXPECTED_VIDEOS,
        "gpu_workers": GPU_WORKERS,
        "gpu_type": GPU_TYPE,
        "config": config.as_dict(),
        "config_fingerprint": config.fingerprint,
        "encoder_fingerprint": config.encoder_fingerprint,
        "code_hashes": code_hashes,
    }
    manifest_path = run / "RUN_MANIFEST.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        immutable = (
            "video_ids",
            "config_fingerprint",
            "encoder_fingerprint",
            "code_hashes",
        )
        if any(existing.get(key) != manifest.get(key) for key in immutable):
            raise RuntimeError("run_id already belongs to different code/config; use a new run_id")
        previous_gpu = existing.get("gpu_type")
        if previous_gpu not in (None, GPU_TYPE):
            raise RuntimeError(f"run_id already used GPU type {previous_gpu}")
        existing["gpu_type"] = GPU_TYPE
        existing["gpu_workers"] = GPU_WORKERS
        manifest = existing
    atomic_write_json(manifest_path, manifest)
    atomic_write_json(run / "BUDGET_PLAN.json", plan)
    work_volume.commit()
    return plan


@app.function(
    image=cpu_image,
    cpu=4.0,
    memory=8192,
    volumes={"/data": data_volume, "/work": work_volume},
    timeout=60 * 60 * 4,
    retries=2,
    max_containers=CANDIDATE_CPU_WORKERS,
)
def build_candidates_remote(
    video_id: str, run_id: str, config_value: dict[str, Any]
) -> dict[str, Any]:
    from framme_extracting.pipeline import build_video_candidates
    from framme_extracting.storage import sha256_file

    config = _config(config_value)
    run = RUN_ROOT / run_id
    manifest = _read_manifest(run, config)
    if video_id not in manifest["video_ids"]:
        raise RuntimeError(f"{video_id} is not part of this production run")
    budget = json.loads((run / "BUDGET_PLAN.json").read_text(encoding="utf-8"))
    if budget.get("config_fingerprint") != config.fingerprint:
        raise RuntimeError("budget/config fingerprint mismatch")
    targets = {item["video_id"]: int(item["target_rows"]) for item in budget["videos"]}

    output = run / "candidates" / video_id
    checkpoint = output / "candidate.done.json"
    candidate_path = output / "candidates.jsonl"
    if checkpoint.exists() and candidate_path.exists():
        current = json.loads(checkpoint.read_text(encoding="utf-8"))
        if (
            current.get("config_fingerprint") == config.fingerprint
            and current.get("target_rows") == targets[video_id]
            and current.get("decode_missing") == 0
            and current.get("candidate_sha256") == sha256_file(candidate_path)
        ):
            return {**current, "skipped": True}

    result = build_video_candidates(
        VIDEO_ROOT / f"{video_id}.mp4",
        BOUNDARY_ROOT / f"{video_id}.json",
        output,
        config,
        target_rows=targets[video_id],
    )
    if result.get("decode_missing") != 0:
        raise RuntimeError(f"{video_id}: candidate decode was incomplete")
    work_volume.commit()
    return result


@app.cls(
    image=gpu_image,
    gpu=GPU_TYPE,
    cpu=4.0,
    memory=16384,
    volumes={"/data": data_volume, "/work": work_volume, "/cache": cache_volume},
    timeout=60 * 60 * 4,
    retries=2,
    buffer_containers=GPU_WORKERS,
    max_containers=GPU_WORKERS,
    scaledown_window=60,
)
class CandidateEncoder:
    @modal.enter()
    def load_model(self) -> None:
        import torch
        from transformers import AutoModel

        config = PipelineConfig()
        self.model = AutoModel.from_pretrained(
            config.encoder_model,
            revision=config.encoder_revision,
            code_revision=config.encoder_code_revision,
            trust_remote_code=True,
        ).to("cuda").eval()
        self.torch = torch
        self.encoder_fingerprint = config.encoder_fingerprint

    @modal.method()
    def encode_video(
        self,
        video_id: str,
        run_id: str,
        config_value: dict[str, Any],
        image_batch_size: int = 32,
    ) -> dict[str, Any]:
        import numpy as np
        from PIL import Image

        from framme_extracting.core import Candidate, pixel_sha256
        from framme_extracting.production import validate_vector_store
        from framme_extracting.storage import (
            atomic_save_npy,
            atomic_write_json,
            atomic_write_jsonl,
            read_jsonl,
            sha256_file,
        )
        from framme_extracting.video_io import iter_selected_frames

        if image_batch_size < 1:
            raise ValueError("image_batch_size must be positive")
        config = _config(config_value)
        if config.encoder_fingerprint != self.encoder_fingerprint:
            raise RuntimeError("runtime encoder differs from the model loaded in this container")
        run = RUN_ROOT / run_id
        manifest = _read_manifest(run, config)
        if video_id not in manifest["video_ids"]:
            raise RuntimeError(f"{video_id} is not part of this production run")

        candidate_path = run / "candidates" / video_id / "candidates.jsonl"
        candidate_done_path = run / "candidates" / video_id / "candidate.done.json"
        if not candidate_path.exists() or not candidate_done_path.exists():
            raise RuntimeError(f"{video_id}: candidate stage is incomplete")
        candidate_done = json.loads(candidate_done_path.read_text(encoding="utf-8"))
        if (
            candidate_done.get("config_fingerprint") != config.fingerprint
            or candidate_done.get("candidate_sha256") != sha256_file(candidate_path)
            or candidate_done.get("decode_missing") != 0
        ):
            raise RuntimeError(f"{video_id}: candidate checkpoint is invalid")

        rows = [Candidate.from_dict(row) for row in read_jsonl(candidate_path)]
        output = run / "vectors" / video_id
        done_path = output / "vector.done.json"
        vector_path = output / "candidate_vectors.npy"
        metadata_path = output / "vector_rows.jsonl"
        if done_path.exists() and vector_path.exists() and metadata_path.exists():
            done = json.loads(done_path.read_text(encoding="utf-8"))
            if (
                done.get("encoder_fingerprint") == config.encoder_fingerprint
                and done.get("config_fingerprint") == config.fingerprint
                and done.get("gpu_type") == GPU_TYPE
                and done.get("candidate_sha256") == sha256_file(candidate_path)
                and done.get("vector_sha256") == sha256_file(vector_path)
                and done.get("metadata_sha256") == sha256_file(metadata_path)
            ):
                return {**done, "skipped": True}

        expected_hash = {
            row.frame_idx: row.metrics.pixel_sha256
            for row in rows
            if row.metrics is not None
        }
        if len(expected_hash) != len(rows):
            raise RuntimeError(f"{video_id}: candidate without pixel hash")

        vectors: list[np.ndarray] = []
        metadata: list[dict[str, Any]] = []
        images: list[Image.Image] = []
        encode_seconds = 0.0

        def flush() -> None:
            nonlocal encode_seconds
            if not images:
                return
            started = time.perf_counter()
            with self.torch.inference_mode():
                encoded = self.model.encode_image(images, batch_size=len(images))
            self.torch.cuda.synchronize()
            encode_seconds += time.perf_counter() - started
            values = np.asarray(encoded, dtype=np.float32)
            if values.ndim != 2 or values.shape[1] != config.embedding_dim:
                raise RuntimeError(f"unexpected Jina output shape {values.shape}")
            norms = np.linalg.norm(values, axis=1, keepdims=True)
            if np.any(norms <= 1e-12):
                raise RuntimeError("Jina emitted a zero vector")
            vectors.append((values / norms).astype(np.float16))
            images.clear()

        seen: set[int] = set()
        by_frame = {row.frame_idx: row for row in rows}
        for frame_idx, rgb in iter_selected_frames(VIDEO_ROOT / f"{video_id}.mp4", by_frame):
            row = by_frame[frame_idx]
            actual_hash = pixel_sha256(rgb)
            if actual_hash != expected_hash[frame_idx]:
                raise RuntimeError(f"pixel hash mismatch for {video_id}:{frame_idx}")
            seen.add(frame_idx)
            images.append(Image.fromarray(rgb, mode="RGB"))
            metadata.append(
                {
                    "video_id": video_id,
                    "frame_idx": frame_idx,
                    "shot_id": row.shot_id,
                    "pixel_sha256": actual_hash,
                    "encoder_fingerprint": config.encoder_fingerprint,
                }
            )
            if len(images) >= image_batch_size:
                flush()
        flush()
        if seen != set(by_frame):
            raise RuntimeError(f"{video_id}: failed to decode {len(set(by_frame) - seen)} candidates")

        matrix = (
            np.concatenate(vectors, axis=0)
            if vectors
            else np.zeros((0, config.embedding_dim), dtype=np.float16)
        )
        validation = validate_vector_store(rows, matrix, config)
        if validation["status"] != "pass":
            raise RuntimeError(f"vector integrity check failed: {validation['failures']}")
        atomic_save_npy(vector_path, matrix)
        atomic_write_jsonl(metadata_path, metadata)
        checkpoint = {
            "schema_version": config.schema_version,
            "video_id": video_id,
            "rows": len(rows),
            "image_batch_size": image_batch_size,
            "gpu_type": GPU_TYPE,
            "encode_seconds": encode_seconds,
            "encode_fps": len(rows) / encode_seconds if encode_seconds else 0.0,
            "candidate_sha256": sha256_file(candidate_path),
            "vector_sha256": sha256_file(vector_path),
            "metadata_sha256": sha256_file(metadata_path),
            "encoder_fingerprint": config.encoder_fingerprint,
            "config_fingerprint": config.fingerprint,
            "integrity": validation,
        }
        atomic_write_json(done_path, checkpoint)
        work_volume.commit()
        return checkpoint


@app.function(
    image=cpu_image,
    cpu=4.0,
    memory=8192,
    volumes={"/data": data_volume, "/work": work_volume},
    timeout=60 * 60 * 2,
    retries=2,
    max_containers=SELECTION_CPU_WORKERS,
)
def select_video_remote(
    video_id: str, run_id: str, config_value: dict[str, Any]
) -> dict[str, Any]:
    import numpy as np
    from PIL import Image

    from framme_extracting.core import Candidate, load_segmentation, pixel_sha256
    from framme_extracting.production import validate_vector_store
    from framme_extracting.selection import promote_vectors, select_views
    from framme_extracting.storage import (
        atomic_save_npy,
        atomic_write_csv,
        atomic_write_json,
        atomic_write_jsonl,
        read_jsonl,
        sha256_file,
    )
    from framme_extracting.video_io import iter_selected_frames

    config = _config(config_value)
    run = RUN_ROOT / run_id
    work_volume.reload()
    manifest = _read_manifest(run, config)
    if video_id not in manifest["video_ids"]:
        raise RuntimeError(f"{video_id} is not part of this production run")

    candidate_path = run / "candidates" / video_id / "candidates.jsonl"
    vector_path = run / "vectors" / video_id / "candidate_vectors.npy"
    vector_done_path = run / "vectors" / video_id / "vector.done.json"
    # The GPU commits before returning, but a freshly scheduled CPU container can
    # briefly see the previous Volume snapshot. Reload until both atomic outputs
    # become visible instead of burning retries on an expected consistency delay.
    for _attempt in range(30):
        work_volume.reload()
        if vector_path.exists() and vector_done_path.exists():
            break
        time.sleep(2)
    if not vector_path.exists() or not vector_done_path.exists():
        raise RuntimeError(f"{video_id}: vector stage is incomplete")
    vector_done = json.loads(vector_done_path.read_text(encoding="utf-8"))
    if (
        vector_done.get("config_fingerprint") != config.fingerprint
        or vector_done.get("encoder_fingerprint") != config.encoder_fingerprint
        or vector_done.get("vector_sha256") != sha256_file(vector_path)
    ):
        raise RuntimeError(f"{video_id}: vector checkpoint is invalid")

    output = run / "dataset" / video_id
    done_path = output / "selection.done.json"
    if done_path.exists():
        done = json.loads(done_path.read_text(encoding="utf-8"))
        images_dir = output / "images"
        actual_images = sum(1 for _ in images_dir.glob("*.webp"))
        persisted = {
            "discovery_vector_sha256": output / "discovery_vectors.npy",
            "locator_vector_sha256": output / "locator_vectors.npy",
            "metadata_sha256": output / "metadata.jsonl",
            "metadata_csv_sha256": output / "metadata.csv",
        }
        if (
            done.get("config_fingerprint") == config.fingerprint
            and done.get("candidate_vector_sha256") == sha256_file(vector_path)
            and done.get("images") == actual_images
            and all(
                path.exists() and done.get(key) == sha256_file(path)
                for key, path in persisted.items()
            )
        ):
            return {**done, "skipped": True}

    rows = [Candidate.from_dict(row) for row in read_jsonl(candidate_path)]
    vectors = np.load(vector_path, allow_pickle=False)
    vector_report = validate_vector_store(rows, vectors, config)
    if vector_report["status"] != "pass":
        raise RuntimeError(f"{video_id}: invalid vector store")
    segmentation = load_segmentation(BOUNDARY_ROOT / f"{video_id}.json")
    shots = {shot.shot_id: shot for shot in segmentation.shots}
    views = select_views(segmentation, rows, vectors, config)
    discovery_rows, discovery_vectors = promote_vectors(rows, vectors, views.discovery_indices)
    locator_rows, locator_vectors = promote_vectors(rows, vectors, views.localization_indices)

    images_dir = output / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    by_frame = {row.frame_idx: (n, row) for n, row in enumerate(discovery_rows, start=1)}
    seen: set[int] = set()
    metadata: list[dict[str, Any]] = []
    for frame_idx, rgb in iter_selected_frames(VIDEO_ROOT / f"{video_id}.mp4", by_frame):
        n, row = by_frame[frame_idx]
        actual_hash = pixel_sha256(rgb)
        if row.metrics is None or actual_hash != row.metrics.pixel_sha256:
            raise RuntimeError(f"final pixel mismatch for {video_id}:{frame_idx}")
        target = images_dir / f"{n:06d}.webp"
        temporary = images_dir / f".{n:06d}.webp.tmp"
        Image.fromarray(rgb, mode="RGB").save(
            temporary, format="WEBP", lossless=True, method=4
        )
        temporary.replace(target)
        with Image.open(target) as image:
            persisted = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if pixel_sha256(persisted) != actual_hash:
            raise RuntimeError(f"lossless WebP verification failed for {video_id}:{frame_idx}")
        metadata.append(
            {
                "video_id": video_id,
                "n": n,
                "frame_idx": frame_idx,
                "shot_id": row.shot_id,
                "shot_start_frame": shots[row.shot_id].start_frame,
                "shot_end_frame": shots[row.shot_id].end_frame,
                "fps": segmentation.video.fps,
                "pts_time": frame_idx / segmentation.video.fps,
                "pixel_sha256": actual_hash,
                "encoder_fingerprint": config.encoder_fingerprint,
                "reasons": sorted(row.reasons),
            }
        )
        seen.add(frame_idx)
    if seen != set(by_frame):
        raise RuntimeError(f"{video_id}: final materialization missing frames")
    expected_images = {f"{n:06d}.webp" for n in range(1, len(discovery_rows) + 1)}
    actual_images = {path.name for path in images_dir.glob("*.webp")}
    if actual_images != expected_images:
        raise RuntimeError(
            f"{video_id}: stale/missing images; use a new run_id "
            f"(expected {len(expected_images)}, got {len(actual_images)})"
        )

    atomic_write_jsonl(output / "metadata.jsonl", metadata)
    atomic_write_csv(
        output / "metadata.csv",
        metadata,
        [
            "video_id",
            "n",
            "frame_idx",
            "pts_time",
            "fps",
            "shot_id",
            "shot_start_frame",
            "shot_end_frame",
            "pixel_sha256",
            "encoder_fingerprint",
            "reasons",
        ],
    )
    atomic_save_npy(output / "discovery_vectors.npy", discovery_vectors)
    atomic_write_jsonl(output / "locator_rows.jsonl", (row.to_dict() for row in locator_rows))
    atomic_save_npy(output / "locator_vectors.npy", locator_vectors)
    for out_index, source_index in enumerate(views.discovery_indices):
        if discovery_vectors[out_index].tobytes() != vectors[source_index].tobytes():
            raise RuntimeError("discovery vector reuse proof failed")

    checkpoint = {
        "schema_version": config.schema_version,
        "video_id": video_id,
        "images": len(discovery_rows),
        "frames_path": _volume_run_path(run_id, "dataset", video_id, "images"),
        "discovery_rows": len(discovery_rows),
        "locator_rows": len(locator_rows),
        "semantic_rejected": len(views.rejected_semantic_indices),
        "config_fingerprint": config.fingerprint,
        "encoder_fingerprint": config.encoder_fingerprint,
        "gpu_type": run_manifest.get("gpu_type"),
        "candidate_vector_sha256": sha256_file(vector_path),
        "discovery_vector_sha256": sha256_file(output / "discovery_vectors.npy"),
        "locator_vector_sha256": sha256_file(output / "locator_vectors.npy"),
        "metadata_sha256": sha256_file(output / "metadata.jsonl"),
        "metadata_csv_sha256": sha256_file(output / "metadata.csv"),
        "vector_integrity": vector_report,
    }
    atomic_write_json(done_path, checkpoint)
    work_volume.commit()
    return checkpoint


@app.function(
    image=cpu_image,
    cpu=4.0,
    memory=8192,
    volumes={"/work": work_volume},
    timeout=60 * 60 * 2,
)
def freeze_remote(run_id: str, config_value: dict[str, Any]) -> dict[str, Any]:
    import os

    import numpy as np

    from framme_extracting.storage import (
        atomic_save_npy,
        atomic_write_json,
        atomic_write_jsonl,
        read_jsonl,
        sha256_file,
    )

    config = _config(config_value)
    run = RUN_ROOT / run_id
    run_manifest = _read_manifest(run, config)
    expected_ids = set(run_manifest["video_ids"])
    done_paths = sorted((run / "dataset").glob("*/selection.done.json"))
    failures: list[str] = []
    videos: list[dict[str, Any]] = []
    for path in done_paths:
        item = json.loads(path.read_text(encoding="utf-8"))
        video_id = item.get("video_id")
        output = path.parent
        if video_id not in expected_ids:
            failures.append(f"unexpected selected video: {video_id}")
        if item.get("config_fingerprint") != config.fingerprint:
            failures.append(f"{video_id}: config mismatch")
        checks = {
            "discovery_vector_sha256": output / "discovery_vectors.npy",
            "locator_vector_sha256": output / "locator_vectors.npy",
            "metadata_sha256": output / "metadata.jsonl",
            "metadata_csv_sha256": output / "metadata.csv",
        }
        for key, target in checks.items():
            if not target.exists() or item.get(key) != sha256_file(target):
                failures.append(f"{video_id}: final hash mismatch for {target.name}")
        image_count = sum(1 for _ in (output / "images").glob("*.webp"))
        if image_count != item.get("images"):
            failures.append(f"{video_id}: expected {item.get('images')} images, found {image_count}")
        videos.append({**item, "checkpoint_sha256": sha256_file(path)})

    selected_ids = {str(item.get("video_id")) for item in videos}
    missing_ids = sorted(expected_ids - selected_ids)
    if missing_ids:
        failures.append(f"missing {len(missing_ids)} selected videos; first={missing_ids[:5]}")
    if len(videos) != EXPECTED_VIDEOS:
        failures.append(f"expected {EXPECTED_VIDEOS} selected videos, found {len(videos)}")
    if failures:
        manifest = {
            "schema_version": "framme-frozen/v2",
            "status": "fail",
            "run_id": run_id,
            "config_fingerprint": config.fingerprint,
            "videos": videos,
            "failures": failures,
        }
        atomic_write_json(run / "FROZEN_MANIFEST.json", manifest)
        work_volume.commit()
        return manifest

    total_rows = sum(int(item["discovery_rows"]) for item in videos)
    index_dir = run / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    temporary_emb = index_dir / ".emb.npy.tmp"
    matrix = np.lib.format.open_memmap(
        temporary_emb,
        mode="w+",
        dtype=np.float16,
        shape=(total_rows, config.embedding_dim),
    )
    ids = np.empty((total_rows, 2), dtype="<U24")
    frame_indices = np.empty(total_rows, dtype=np.int32)
    ranges: dict[str, list[int]] = {}
    global_metadata: list[dict[str, Any]] = []
    cursor = 0
    for item in sorted(videos, key=lambda value: value["video_id"]):
        video_id = item["video_id"]
        output = run / "dataset" / video_id
        values = np.load(output / "discovery_vectors.npy", mmap_mode="r", allow_pickle=False)
        metadata = read_jsonl(output / "metadata.jsonl")
        if values.shape != (len(metadata), config.embedding_dim):
            raise RuntimeError(f"{video_id}: vector/metadata mismatch")
        end = cursor + len(metadata)
        matrix[cursor:end] = values
        for offset, row in enumerate(metadata):
            n = offset + 1
            if row["video_id"] != video_id or int(row["n"]) != n:
                raise RuntimeError(f"{video_id}: non-canonical metadata order")
            ids[cursor + offset] = (video_id, str(n))
            frame_indices[cursor + offset] = int(row["frame_idx"])
            global_metadata.append(row)
        ranges[video_id] = [cursor, end]
        cursor = end
    matrix.flush()
    del matrix
    os.replace(temporary_emb, index_dir / "emb.npy")
    atomic_save_npy(index_dir / "ids.npy", ids)
    atomic_save_npy(index_dir / "frame_idx.npy", frame_indices)
    atomic_write_json(index_dir / "ranges.json", ranges)
    atomic_write_jsonl(index_dir / "metadata.jsonl", global_metadata)
    index_hashes = {
        name: sha256_file(index_dir / name)
        for name in ("emb.npy", "ids.npy", "frame_idx.npy", "ranges.json", "metadata.jsonl")
    }
    manifest = {
        "schema_version": "framme-frozen/v2",
        "status": "pass",
        "run_id": run_id,
        "config": config.as_dict(),
        "config_fingerprint": config.fingerprint,
        "encoder_fingerprint": config.encoder_fingerprint,
        "gpu_type": run_manifest.get("gpu_type"),
        "frames_root": _volume_run_path(run_id, "dataset"),
        "videos": videos,
        "index": {
            "rows": total_rows,
            "dimension": config.embedding_dim,
            "dtype": config.embedding_dtype,
            "hashes": index_hashes,
        },
        "provenance": {
            name: sha256_file(run / name)
            for name in ("RUN_MANIFEST.json", "BUDGET_PLAN.json")
        },
        "failures": [],
    }
    manifest_path = run / "FROZEN_MANIFEST.json"
    atomic_write_json(manifest_path, manifest)
    atomic_write_json(
        Path("/work/CURRENT.json"),
        {
            "run_id": run_id,
            "frames_root": _volume_run_path(run_id, "dataset"),
            "index_root": _volume_run_path(run_id, "index"),
            "manifest": _volume_run_path(run_id, "FROZEN_MANIFEST.json"),
            "manifest_sha256": sha256_file(manifest_path),
        },
    )
    work_volume.commit()
    return manifest


@app.function(
    image=cpu_image,
    volumes={"/work": work_volume},
    timeout=60 * 5,
)
def status_remote(run_id: str) -> dict[str, Any]:
    run = RUN_ROOT / run_id
    selected = sorted((run / "dataset").glob("*/selection.done.json"))
    frame_count = 0
    for path in selected:
        frame_count += int(json.loads(path.read_text(encoding="utf-8")).get("images", 0))
    frozen_path = run / "FROZEN_MANIFEST.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8")) if frozen_path.exists() else None
    return {
        "run_id": run_id,
        "candidate_videos": len(list((run / "candidates").glob("*/candidate.done.json"))),
        "encoded_videos": len(list((run / "vectors").glob("*/vector.done.json"))),
        "selected_videos": len(selected),
        "materialized_webp_frames": frame_count,
        "frames_root": _volume_run_path(run_id, "dataset"),
        "frozen_status": frozen.get("status") if frozen else None,
    }


def _local_boundaries(boundary_dir: Path, groups: str) -> list[Path]:
    prefixes = tuple(f"{item.strip()}_" for item in groups.split(",") if item.strip())
    return sorted(path for path in boundary_dir.glob("*.json") if path.stem.startswith(prefixes))


@app.local_entrypoint()
def main(
    stage: str = "run",
    run_id: str = "l21-l25-prod-v1",
    groups: str = "L21,L22,L23,L24,L25",
    image_batch_size: int = 32,
    boundaries: str = "",
) -> None:
    if stage == "status":
        print(json.dumps(status_remote.remote(run_id), indent=2, ensure_ascii=False))
        return
    if stage not in {"run", "sync"}:
        raise ValueError("stage must be one of: run, sync, status")
    if not boundaries:
        raise ValueError("--boundaries is required for run/sync")

    local_paths = _local_boundaries(Path(boundaries), groups)
    if len(local_paths) != EXPECTED_VIDEOS:
        raise RuntimeError(
            f"expected {EXPECTED_VIDEOS} L21-L25 boundary files, found {len(local_paths)}"
        )
    video_ids = [path.stem for path in local_paths]
    with data_volume.batch_upload(force=True) as batch:
        for path in local_paths:
            batch.put_file(path, f"/shot_boundaries/{path.name}")
    if stage == "sync":
        print(json.dumps({"uploaded_boundaries": len(local_paths)}, indent=2))
        return

    config = PipelineConfig()
    config_value = config.as_dict()
    plan = prepare_run_remote.remote(run_id, config_value, video_ids)
    print(f"[1/4] candidate extraction: {len(video_ids)} videos")
    candidate_results = list(
        build_candidates_remote.starmap(
            ((video_id, run_id, config_value) for video_id in video_ids),
            order_outputs=False,
        )
    )
    candidate_rows = sum(int(item["candidate_rows"]) for item in candidate_results)
    if candidate_rows != plan["assigned_rows"]:
        raise RuntimeError(f"candidate rows {candidate_rows} != budget {plan['assigned_rows']}")

    print(
        f"[2/4] Jina encode + streaming WebP: {candidate_rows} images, "
        f"batch={image_batch_size}, workers={GPU_WORKERS}x{GPU_TYPE}"
    )
    encoder = CandidateEncoder()
    target_rows = {
        item["video_id"]: int(item["target_rows"])
        for item in plan["videos"]
    }
    # Longest-processing-time first keeps all ten GPUs occupied and reduces the
    # final straggler compared with lexicographic video order.
    encode_ids = sorted(video_ids, key=lambda value: (-target_rows[value], value))
    encoded_rows = 0
    encoded_videos = 0
    selection_calls = []
    for result in encoder.encode_video.starmap(
        (
            (video_id, run_id, config_value, image_batch_size)
            for video_id in encode_ids
        ),
        order_outputs=False,
    ):
        encoded_rows += int(result["rows"])
        encoded_videos += 1
        selection_calls.append(
            select_video_remote.spawn(result["video_id"], run_id, config_value)
        )
        if encoded_videos % 10 == 0 or encoded_videos == len(video_ids):
            print(f"encoded {encoded_videos}/{len(video_ids)} videos")
    if encoded_rows != candidate_rows:
        raise RuntimeError("encoded row count differs from candidate row count")

    print("[3/4] wait for canonical selection and lossless WebP materialization")
    selection_results = [call.get() for call in selection_calls]
    if len(selection_results) != len(video_ids):
        raise RuntimeError("selection did not return every production video")
    image_rows = sum(int(item["images"]) for item in selection_results)
    print(
        f"materialized {image_rows} WebP frames under "
        f"/runs/{run_id}/dataset/<video_id>/images"
    )

    print("[4/4] freeze flat index and CURRENT.json")
    frozen = freeze_remote.remote(run_id, config_value)
    if frozen["status"] != "pass":
        raise RuntimeError(f"freeze failed: {frozen['failures'][:10]}")
    print(
        json.dumps(
            {
                "status": "pass",
                "run_id": run_id,
                "videos": len(frozen["videos"]),
                "webp_frames": image_rows,
                "index_rows": frozen["index"]["rows"],
                "frames_root": frozen["frames_root"],
                "gpu_workers": GPU_WORKERS,
                "gpu_type": GPU_TYPE,
                "image_batch_size": image_batch_size,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
