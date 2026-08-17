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
PILOT_BOUNDARY_ROOT = Path("/data/pilot_boundaries")
RUN_ROOT = Path("/work/runs")
EXPECTED_FULL_VIDEOS = 216

app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name(DATA_VOLUME, create_if_missing=False)
work_volume = modal.Volume.from_name(WORK_VOLUME, create_if_missing=True)
cache_volume = modal.Volume.from_name(CACHE_VOLUME, create_if_missing=True)

package_dir = str(SRC_ROOT)
base_cpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install("av==14.0.1", "numpy==1.26.4", "pillow==11.1.0")
)
cpu_image = base_cpu_image.add_local_dir(package_dir, remote_path="/root/src")
gpu_image = (
    base_cpu_image.pip_install(
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


def _video_ids(groups: str = "L21,L22,L23,L24,L25") -> list[str]:
    prefixes = tuple(f"{item.strip()}_" for item in groups.split(",") if item.strip())
    return sorted(path.stem for path in BOUNDARY_ROOT.glob("*.json") if path.stem.startswith(prefixes))


@app.function(
    image=cpu_image,
    cpu=4.0,
    memory=8192,
    volumes={"/data": data_volume, "/work": work_volume},
    timeout=60 * 60 * 4,
    retries=2,
    max_containers=12,
)
def build_candidates_remote(
    video_id: str,
    run_id: str,
    config_value: dict[str, Any],
    enforce_cpu_pilot_gate: bool = True,
) -> dict[str, Any]:
    from framme_extracting.pipeline import build_video_candidates

    run = RUN_ROOT / run_id
    if enforce_cpu_pilot_gate:
        cpu_gate_path = run / "reports" / "cpu_pilot_gate.json"
        if not cpu_gate_path.exists():
            raise RuntimeError("missing CPU pilot gate; run --stage cpu-pilot first")
        cpu_gate = json.loads(cpu_gate_path.read_text(encoding="utf-8"))
        config = _config(config_value)
        if cpu_gate.get("status") != "pass" or cpu_gate.get("config_fingerprint") != config.fingerprint:
            raise RuntimeError("CPU pilot gate failed or is stale")
    output = run / "candidates" / video_id
    checkpoint = output / "candidate.done.json"
    if checkpoint.exists():
        current = json.loads(checkpoint.read_text(encoding="utf-8"))
        if current.get("config_fingerprint") == _config(config_value).fingerprint:
            return {**current, "skipped": True}
    budget_path = RUN_ROOT / run_id / "BUDGET_PLAN.json"
    if not budget_path.exists():
        raise RuntimeError("missing BUDGET_PLAN.json; candidates stage must prepare the run first")
    budget = json.loads(budget_path.read_text(encoding="utf-8"))
    if budget.get("config_fingerprint") != _config(config_value).fingerprint:
        raise RuntimeError("budget/config fingerprint mismatch")
    targets = {item["video_id"]: item["target_rows"] for item in budget["videos"]}
    result = build_video_candidates(
        VIDEO_ROOT / f"{video_id}.mp4",
        BOUNDARY_ROOT / f"{video_id}.json",
        output,
        _config(config_value),
        target_rows=int(targets[video_id]),
    )
    work_volume.commit()
    return result


@app.function(
    image=cpu_image,
    cpu=4.0,
    memory=8192,
    volumes={"/data": data_volume, "/work": work_volume},
    timeout=60 * 60 * 4,
    retries=1,
    max_containers=3,
)
def build_sample_remote(
    video_id: str,
    run_id: str,
    config_value: dict[str, Any],
    target_rows: int,
) -> dict[str, Any]:
    from framme_extracting.pipeline import build_video_candidates

    config = _config(config_value)
    output = RUN_ROOT / run_id / "sample_candidates" / video_id
    result = build_video_candidates(
        VIDEO_ROOT / f"{video_id}.mp4",
        PILOT_BOUNDARY_ROOT / run_id / f"{video_id}.json",
        output,
        config,
        target_rows=target_rows,
    )
    work_volume.commit()
    return result


@app.function(
    image=cpu_image,
    volumes={"/data": data_volume, "/work": work_volume},
    timeout=60 * 10,
)
def evaluate_sample_remote(
    video_ids: list[str], run_id: str, config_value: dict[str, Any]
) -> dict[str, Any]:
    from collections import Counter

    from framme_extracting.core import Candidate, load_segmentation
    from framme_extracting.evaluation import WINDOW_LENGTHS, temporal_window_coverage
    from framme_extracting.storage import atomic_write_json, read_jsonl

    config = _config(config_value)
    failures: list[str] = []
    reason_counts: Counter[str] = Counter()
    rejection_counts: Counter[str] = Counter()
    videos = []
    for video_id in video_ids:
        boundary = PILOT_BOUNDARY_ROOT / run_id / f"{video_id}.json"
        output = RUN_ROOT / run_id / "sample_candidates" / video_id
        done = json.loads((output / "candidate.done.json").read_text(encoding="utf-8"))
        rows = [Candidate.from_dict(row) for row in read_jsonl(output / "candidates.jsonl")]
        segmentation = load_segmentation(boundary)
        coverage = temporal_window_coverage(segmentation, rows, WINDOW_LENGTHS)
        if done.get("decode_missing") != 0:
            failures.append(f"{video_id}: decode_missing={done.get('decode_missing')}")
        if coverage[str(config.answer_window_frames)] < 1.0 - 1e-12:
            failures.append(f"{video_id}: answer-window coverage below 1")
        if any(row.metrics is None for row in rows):
            failures.append(f"{video_id}: candidate without metrics/pixel hash")
        reason_counts.update(reason for row in rows for reason in row.reasons)
        rejection_counts.update(row.rejected_reason for row in rows if row.rejected_reason)
        videos.append(
            {
                "video_id": video_id,
                "shots": len(segmentation.shots),
                "raw_frames": segmentation.video.frame_count,
                "candidate_rows": len(rows),
                "locator_rows": sum(row.locator_keep for row in rows),
                "scan_seconds": done["scan_seconds"],
                "scan_target_fps": done["decode_targets"] / done["scan_seconds"],
                "window_coverage": coverage,
            }
        )
    report = {
        "schema_version": "framme-sample-eval/v1",
        "status": "pass" if not failures else "fail",
        "run_id": run_id,
        "config_fingerprint": config.fingerprint,
        "videos": videos,
        "totals": {
            "videos": len(videos),
            "raw_frames": sum(item["raw_frames"] for item in videos),
            "candidate_rows": sum(item["candidate_rows"] for item in videos),
            "locator_rows": sum(item["locator_rows"] for item in videos),
            "scan_seconds": sum(item["scan_seconds"] for item in videos),
        },
        "reason_counts": dict(sorted(reason_counts.items())),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "failures": failures,
    }
    atomic_write_json(RUN_ROOT / run_id / "reports" / "sample_eval.json", report)
    work_volume.commit()
    return report


@app.function(
    image=cpu_image,
    volumes={"/data": data_volume, "/work": work_volume},
    timeout=60 * 10,
)
def prepare_run_remote(run_id: str, config_value: dict[str, Any]) -> dict[str, Any]:
    from datetime import datetime, timezone

    from framme_extracting.evaluation import make_budget_plan
    from framme_extracting.storage import atomic_write_json, sha256_file

    config = _config(config_value)
    plan = make_budget_plan(sorted(BOUNDARY_ROOT.glob("*.json")), config)
    if plan["status"] != "pass" or len(plan["videos"]) != EXPECTED_FULL_VIDEOS:
        raise RuntimeError(f"invalid budget plan: {plan['failures']}")
    run = RUN_ROOT / run_id
    code_hashes = {
        path.name: sha256_file(path)
        for path in sorted(Path("/root/src/framme_extracting").glob("*.py"))
    }
    run_manifest = {
        "schema_version": "framme-run/v1",
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": config.as_dict(),
        "config_fingerprint": config.fingerprint,
        "encoder_fingerprint": config.encoder_fingerprint,
        "code_hashes": code_hashes,
    }
    existing_path = run / "RUN_MANIFEST.json"
    if existing_path.exists():
        existing = json.loads(existing_path.read_text(encoding="utf-8"))
        immutable_fields = ("config_fingerprint", "encoder_fingerprint", "code_hashes")
        if any(existing.get(key) != run_manifest.get(key) for key in immutable_fields):
            raise RuntimeError("run_id already belongs to different config/code; choose a new run_id")
        run_manifest = existing
    atomic_write_json(run / "RUN_MANIFEST.json", run_manifest)
    atomic_write_json(run / "BUDGET_PLAN.json", plan)
    work_volume.commit()
    return plan


@app.function(
    image=cpu_image,
    volumes={"/data": data_volume, "/work": work_volume},
    timeout=60 * 10,
)
def finalize_cpu_pilot_remote(
    run_id: str,
    config_value: dict[str, Any],
    stats: list[dict[str, Any]],
    max_projected_wall_hours: float,
) -> dict[str, Any]:
    from framme_extracting.evaluation import evaluate_sampling_plan
    from framme_extracting.storage import atomic_write_json

    config = _config(config_value)
    valid = [item for item in stats if item.get("decode_targets", 0) and item.get("scan_seconds", 0)]
    targets = sum(int(item["decode_targets"]) for item in valid)
    seconds = sum(float(item["scan_seconds"]) for item in valid)
    fps = targets / seconds if seconds else 0.0
    sampling = evaluate_sampling_plan(sorted(BOUNDARY_ROOT.glob("*.json")), config)
    projected_cpu_seconds = sampling["decode_target_rows"] / fps if fps else float("inf")
    projected_wall_hours = projected_cpu_seconds / 12 / 3600
    failures = []
    if len(valid) < 2:
        failures.append("CPU pilot requires at least two videos")
    if sampling["status"] != "pass":
        failures.append("analytical sampling eval failed")
    if fps <= 0:
        failures.append("invalid measured CPU throughput")
    if projected_wall_hours > max_projected_wall_hours:
        failures.append(
            f"projected CPU wall time {projected_wall_hours:.2f}h exceeds cap {max_projected_wall_hours:.2f}h"
        )
    report = {
        "schema_version": "framme-cpu-pilot/v1",
        "status": "pass" if not failures else "fail",
        "config_fingerprint": config.fingerprint,
        "pilot_videos": [item["video_id"] for item in valid],
        "measured_decode_targets": targets,
        "measured_scan_seconds": seconds,
        "measured_target_fps": fps,
        "projected_decode_targets": sampling["decode_target_rows"],
        "projected_cpu_seconds": projected_cpu_seconds,
        "assumed_parallel_containers": 12,
        "projected_wall_hours": projected_wall_hours,
        "max_projected_wall_hours": max_projected_wall_hours,
        "failures": failures,
    }
    atomic_write_json(RUN_ROOT / run_id / "reports" / "cpu_pilot_gate.json", report)
    work_volume.commit()
    return report


@app.function(
    image=cpu_image,
    cpu=2.0,
    memory=4096,
    volumes={"/data": data_volume, "/work": work_volume},
    timeout=60 * 30,
)
def evaluate_remote(run_id: str, config_value: dict[str, Any], expected_videos: int) -> dict[str, Any]:
    from framme_extracting.evaluation import evaluate_candidate_run, write_report

    config = _config(config_value)
    paths = sorted(BOUNDARY_ROOT.glob("*.json"))
    report = evaluate_candidate_run(
        paths,
        RUN_ROOT / run_id / "candidates",
        config,
        expected_videos=expected_videos,
    )
    report_path = RUN_ROOT / run_id / "reports" / "eval_gate.json"
    write_report(report_path, report)
    work_volume.commit()
    return report


@app.cls(
    image=gpu_image,
    gpu="T4",
    cpu=4.0,
    memory=16384,
    volumes={"/data": data_volume, "/work": work_volume, "/cache": cache_volume},
    timeout=60 * 60 * 4,
    retries=2,
    max_containers=8,
    scaledown_window=300,
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
        batch_size: int = 32,
        enforce_full_gate: bool = True,
    ) -> dict[str, Any]:
        import numpy as np
        from PIL import Image

        from framme_extracting.core import Candidate, pixel_sha256
        from framme_extracting.evaluation import require_passing_gate, validate_vector_store
        from framme_extracting.storage import (
            atomic_save_npy,
            atomic_write_json,
            atomic_write_jsonl,
            read_jsonl,
            sha256_file,
        )
        from framme_extracting.video_io import iter_selected_frames

        config = _config(config_value)
        if config.encoder_fingerprint != self.encoder_fingerprint:
            raise RuntimeError("runtime encoder config differs from the model loaded in this container")
        run = RUN_ROOT / run_id
        if enforce_full_gate:
            require_passing_gate(run / "reports" / "eval_gate.json", config, EXPECTED_FULL_VIDEOS)
            pilot_path = run / "reports" / "pilot_gate.json"
            if not pilot_path.exists():
                raise RuntimeError("missing pilot_gate.json; run --stage pilot first")
            pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
            if pilot.get("status") != "pass" or pilot.get("encoder_fingerprint") != config.encoder_fingerprint:
                raise RuntimeError("pilot gate rejected full encoding")
            if pilot.get("config_fingerprint") != config.fingerprint:
                raise RuntimeError("pilot gate was produced by a different pipeline config")

        candidate_path = run / "candidates" / video_id / "candidates.jsonl"
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
                and done.get("candidate_sha256") == sha256_file(candidate_path)
                and done.get("vector_sha256") == sha256_file(vector_path)
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
        batch_rows: list[Candidate] = []
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
            values = values / norms
            vectors.append(values.astype(np.float16))
            images.clear()
            batch_rows.clear()

        seen: set[int] = set()
        by_frame = {row.frame_idx: row for row in rows}
        for frame_idx, rgb in iter_selected_frames(VIDEO_ROOT / f"{video_id}.mp4", by_frame):
            row = by_frame[frame_idx]
            actual_hash = pixel_sha256(rgb)
            if actual_hash != expected_hash[frame_idx]:
                raise RuntimeError(f"pixel hash mismatch for {video_id}:{frame_idx}")
            seen.add(frame_idx)
            images.append(Image.fromarray(rgb, mode="RGB"))
            batch_rows.append(row)
            metadata.append(
                {
                    "video_id": video_id,
                    "frame_idx": frame_idx,
                    "shot_id": row.shot_id,
                    "pixel_sha256": actual_hash,
                    "encoder_fingerprint": config.encoder_fingerprint,
                }
            )
            if len(images) >= batch_size:
                flush()
        flush()
        if seen != set(by_frame):
            raise RuntimeError(f"{video_id}: failed to decode {len(set(by_frame) - seen)} candidates")
        matrix = np.concatenate(vectors, axis=0) if vectors else np.zeros((0, config.embedding_dim), np.float16)
        report = validate_vector_store(rows, matrix, config)
        if report["status"] != "pass":
            raise RuntimeError(f"vector validation failed: {report['failures']}")
        atomic_save_npy(vector_path, matrix)
        atomic_write_jsonl(metadata_path, metadata)
        checkpoint = {
            "schema_version": config.schema_version,
            "video_id": video_id,
            "rows": len(rows),
            "encode_seconds": encode_seconds,
            "encode_fps": len(rows) / encode_seconds if encode_seconds else 0.0,
            "candidate_sha256": sha256_file(candidate_path),
            "vector_sha256": sha256_file(vector_path),
            "metadata_sha256": sha256_file(metadata_path),
            "encoder_fingerprint": config.encoder_fingerprint,
            "config_fingerprint": config.fingerprint,
            "validation": report,
        }
        atomic_write_json(done_path, checkpoint)
        work_volume.commit()
        return checkpoint


@app.function(
    image=cpu_image,
    volumes={"/work": work_volume},
    timeout=60 * 10,
)
def finalize_pilot_remote(
    run_id: str,
    config_value: dict[str, Any],
    stats: list[dict[str, Any]],
    gpu_usd_per_hour: float,
    max_usd: float,
) -> dict[str, Any]:
    from framme_extracting.evaluation import require_passing_gate
    from framme_extracting.storage import atomic_write_json

    config = _config(config_value)
    valid = [item for item in stats if item.get("rows", 0) and item.get("encode_seconds", 0)]
    rows = sum(int(item["rows"]) for item in valid)
    seconds = sum(float(item["encode_seconds"]) for item in valid)
    fps = rows / seconds if seconds else 0.0
    eval_report = require_passing_gate(
        RUN_ROOT / run_id / "reports" / "eval_gate.json", config, EXPECTED_FULL_VIDEOS
    )
    total_rows = int(eval_report["candidate_metrics"]["rows"])
    projected_hours = total_rows / fps / 3600 if fps else float("inf")
    projected_usd = projected_hours * gpu_usd_per_hour
    failures = []
    if len(valid) < 2:
        failures.append("pilot requires at least two successfully encoded videos")
    if fps <= 0:
        failures.append("invalid measured throughput")
    if projected_usd > max_usd:
        failures.append(f"projected ${projected_usd:.2f} exceeds cap ${max_usd:.2f}")
    report = {
        "schema_version": "framme-pilot/v1",
        "status": "pass" if not failures else "fail",
        "config_fingerprint": config.fingerprint,
        "encoder_fingerprint": config.encoder_fingerprint,
        "pilot_videos": [item["video_id"] for item in valid],
        "measured_rows": rows,
        "measured_encode_seconds": seconds,
        "measured_encode_fps": fps,
        "projected_total_rows": total_rows,
        "gpu_usd_per_hour": gpu_usd_per_hour,
        "projected_gpu_hours": projected_hours,
        "projected_usd": projected_usd,
        "max_usd": max_usd,
        "failures": failures,
    }
    atomic_write_json(RUN_ROOT / run_id / "reports" / "pilot_gate.json", report)
    work_volume.commit()
    return report


@app.function(
    image=cpu_image,
    cpu=4.0,
    memory=8192,
    volumes={"/data": data_volume, "/work": work_volume},
    timeout=60 * 60 * 2,
    retries=2,
    max_containers=12,
)
def select_video_remote(video_id: str, run_id: str, config_value: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    from PIL import Image

    from framme_extracting.core import Candidate, load_segmentation, pixel_sha256
    from framme_extracting.evaluation import require_passing_gate, validate_vector_store
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
    require_passing_gate(run / "reports" / "eval_gate.json", config, EXPECTED_FULL_VIDEOS)
    required_provenance = [
        run / "RUN_MANIFEST.json",
        run / "BUDGET_PLAN.json",
        run / "reports" / "pilot_gate.json",
    ]
    missing_provenance = [str(path) for path in required_provenance if not path.exists()]
    if missing_provenance:
        raise RuntimeError(f"cannot select without provenance: {missing_provenance}")
    run_manifest = json.loads((run / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    pilot_gate = json.loads((run / "reports" / "pilot_gate.json").read_text(encoding="utf-8"))
    if run_manifest.get("config_fingerprint") != config.fingerprint:
        raise RuntimeError("RUN_MANIFEST config mismatch")
    if (
        pilot_gate.get("status") != "pass"
        or pilot_gate.get("config_fingerprint") != config.fingerprint
        or pilot_gate.get("encoder_fingerprint") != config.encoder_fingerprint
    ):
        raise RuntimeError("pilot provenance is stale or failed")
    candidate_path = run / "candidates" / video_id / "candidates.jsonl"
    vector_path = run / "vectors" / video_id / "candidate_vectors.npy"
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
    output = run / "dataset" / video_id
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
        Image.fromarray(rgb, mode="RGB").save(temporary, format="WEBP", lossless=True, method=4)
        temporary.replace(target)
        persisted = np.asarray(Image.open(target).convert("RGB"), dtype=np.uint8)
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
            f"{video_id}: stale/missing images; use a new run_id (expected {len(expected_images)}, got {len(actual_images)})"
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
    # The only accepted promotion is a byte-identical gather from the master store.
    for out_index, source_index in enumerate(views.discovery_indices):
        if discovery_vectors[out_index].tobytes() != vectors[source_index].tobytes():
            raise RuntimeError("discovery vector reuse proof failed")
    checkpoint = {
        "schema_version": config.schema_version,
        "video_id": video_id,
        "discovery_rows": len(discovery_rows),
        "locator_rows": len(locator_rows),
        "semantic_rejected": len(views.rejected_semantic_indices),
        "config_fingerprint": config.fingerprint,
        "encoder_fingerprint": config.encoder_fingerprint,
        "candidate_vector_sha256": sha256_file(vector_path),
        "discovery_vector_sha256": sha256_file(output / "discovery_vectors.npy"),
        "locator_vector_sha256": sha256_file(output / "locator_vectors.npy"),
        "metadata_sha256": sha256_file(output / "metadata.jsonl"),
        "metadata_csv_sha256": sha256_file(output / "metadata.csv"),
        "vector_validation": vector_report,
    }
    atomic_write_json(output / "selection.done.json", checkpoint)
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

    from framme_extracting.evaluation import require_passing_gate
    from framme_extracting.storage import (
        atomic_save_npy,
        atomic_write_json,
        atomic_write_jsonl,
        read_jsonl,
        sha256_file,
    )

    config = _config(config_value)
    run = RUN_ROOT / run_id
    require_passing_gate(run / "reports" / "eval_gate.json", config, EXPECTED_FULL_VIDEOS)
    required_provenance = [
        run / "RUN_MANIFEST.json",
        run / "BUDGET_PLAN.json",
        run / "reports" / "pilot_gate.json",
    ]
    missing_provenance = [str(path) for path in required_provenance if not path.exists()]
    if missing_provenance:
        raise RuntimeError(f"cannot freeze without provenance: {missing_provenance}")
    run_manifest = json.loads((run / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    pilot_gate = json.loads((run / "reports" / "pilot_gate.json").read_text(encoding="utf-8"))
    if run_manifest.get("config_fingerprint") != config.fingerprint:
        raise RuntimeError("RUN_MANIFEST config mismatch")
    if (
        pilot_gate.get("status") != "pass"
        or pilot_gate.get("config_fingerprint") != config.fingerprint
        or pilot_gate.get("encoder_fingerprint") != config.encoder_fingerprint
    ):
        raise RuntimeError("pilot provenance is stale or failed")
    done_paths = sorted((run / "dataset").glob("*/selection.done.json"))
    failures = []
    videos = []
    for path in done_paths:
        item = json.loads(path.read_text(encoding="utf-8"))
        if item.get("config_fingerprint") != config.fingerprint:
            failures.append(f"{item.get('video_id')}: config mismatch")
        output = path.parent
        checks = {
            "discovery_vector_sha256": output / "discovery_vectors.npy",
            "locator_vector_sha256": output / "locator_vectors.npy",
            "metadata_sha256": output / "metadata.jsonl",
            "metadata_csv_sha256": output / "metadata.csv",
        }
        for key, target in checks.items():
            if not target.exists() or item.get(key) != sha256_file(target):
                failures.append(f"{item.get('video_id')}: final hash mismatch for {target.name}")
        videos.append({**item, "checkpoint_sha256": sha256_file(path)})
    if len(videos) != EXPECTED_FULL_VIDEOS:
        failures.append(f"expected {EXPECTED_FULL_VIDEOS} selected videos, found {len(videos)}")
    if failures:
        manifest = {
            "schema_version": "framme-frozen/v1",
            "status": "fail",
            "run_id": run_id,
            "config": config.as_dict(),
            "config_fingerprint": config.fingerprint,
            "encoder_fingerprint": config.encoder_fingerprint,
            "videos": videos,
            "failures": failures,
        }
        atomic_write_json(run / "FROZEN_MANIFEST.json", manifest)
        work_volume.commit()
        return manifest

    # Build the exact flat-index format consumed by src.ingestion.vector_index.
    # Copying to the memmap is a byte-preserving gather; no model is invoked here.
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
            raise RuntimeError(f"{video_id}: vector/metadata mismatch during consolidation")
        end = cursor + len(metadata)
        matrix[cursor:end] = values
        for offset, row in enumerate(metadata):
            expected_n = offset + 1
            if row["video_id"] != video_id or int(row["n"]) != expected_n:
                raise RuntimeError(f"{video_id}: non-canonical metadata order")
            ids[cursor + offset] = (video_id, str(expected_n))
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
        "schema_version": "framme-frozen/v1",
        "status": "pass" if not failures else "fail",
        "run_id": run_id,
        "config": config.as_dict(),
        "config_fingerprint": config.fingerprint,
        "encoder_fingerprint": config.encoder_fingerprint,
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
        }
        | {
            "eval_gate.json": sha256_file(run / "reports" / "eval_gate.json"),
            "pilot_gate.json": sha256_file(run / "reports" / "pilot_gate.json"),
        },
        "failures": failures,
    }
    manifest_path = run / "FROZEN_MANIFEST.json"
    atomic_write_json(manifest_path, manifest)
    pointer = {
        "run_id": run_id,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
    }
    atomic_write_json(Path("/work/CURRENT.json"), pointer)
    work_volume.commit()
    return manifest


@app.local_entrypoint()
def main(
    stage: str = "preflight",
    run_id: str = "l21-l25-v1",
    groups: str = "L21,L22,L23,L24,L25",
    limit: int = 0,
    batch_size: int = 32,
    gpu_usd_per_hour: float = 0.59,
    max_usd: float = 12.0,
    max_cpu_wall_hours: float = 4.0,
    boundaries: str = "data/Framme/L21-L25/shots",
) -> None:
    from framme_extracting.evaluation import (
        evaluate_sampling_plan,
        evaluate_segmentation_files,
        make_budget_plan,
    )

    config = PipelineConfig()
    boundary_dir = Path(boundaries)
    local_paths = sorted(boundary_dir.glob("*.json"))
    if stage == "preflight":
        print(
            json.dumps(
                {
                    "boundaries": evaluate_segmentation_files(local_paths),
                    "sampling_plan": evaluate_sampling_plan(local_paths, config),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return
    if stage == "sample-eval":
        prefixes = tuple(f"{item.strip()}_" for item in groups.split(",") if item.strip())
        selected = [path for path in local_paths if path.stem.startswith(prefixes)]
        selected = selected[: (limit or 3)]
        if not 2 <= len(selected) <= 5:
            raise RuntimeError("sample-eval requires 2-5 videos")
        structural = evaluate_segmentation_files(selected)
        if structural["status"] != "pass":
            raise RuntimeError(f"sample boundary eval failed: {structural['failures']}")
        plan = make_budget_plan(local_paths, config)
        if plan["status"] != "pass":
            raise RuntimeError(f"global budget plan failed: {plan['failures']}")
        targets = {item["video_id"]: item["target_rows"] for item in plan["videos"]}
        with data_volume.batch_upload() as batch:
            for path in selected:
                batch.put_file(path, f"/pilot_boundaries/{run_id}/{path.name}")
        config_value = config.as_dict()
        results = list(
            build_sample_remote.starmap(
                (
                    (path.stem, run_id, config_value, int(targets[path.stem]))
                    for path in selected
                ),
                order_outputs=False,
            )
        )
        report = evaluate_sample_remote.remote(
            [path.stem for path in selected], run_id, config_value
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return
    if stage == "sync":
        report = evaluate_segmentation_files(local_paths)
        if report["status"] != "pass" or len(local_paths) != EXPECTED_FULL_VIDEOS:
            raise RuntimeError("local boundary preflight failed or did not contain exactly 216 videos")
        with data_volume.batch_upload(force=True) as batch:
            for path in local_paths:
                batch.put_file(path, f"/shot_boundaries/{path.name}")
        print(f"uploaded {len(local_paths)} boundary files")
        return

    prefixes = tuple(f"{item.strip()}_" for item in groups.split(",") if item.strip())
    ids = sorted(path.stem for path in local_paths if path.stem.startswith(prefixes))
    if limit:
        ids = ids[:limit]
    config_value = config.as_dict()
    if stage == "cpu-pilot":
        prepare_run_remote.remote(run_id, config_value)
        pilot_ids = ids[: min(3, len(ids))]
        if len(pilot_ids) < 2:
            raise RuntimeError("CPU pilot needs at least two videos")
        results = list(
            build_candidates_remote.starmap(
                (
                    (video_id, run_id, config_value, False)
                    for video_id in pilot_ids
                ),
                order_outputs=False,
            )
        )
        report = finalize_cpu_pilot_remote.remote(
            run_id, config_value, results, max_cpu_wall_hours
        )
        print(json.dumps(report, indent=2))
        return
    if stage == "candidates":
        plan = prepare_run_remote.remote(run_id, config_value)
        if plan["assigned_rows"] != config.target_embedding_rows:
            raise RuntimeError("prepared budget does not match configured target")
        results = list(
            build_candidates_remote.starmap(
                ((video_id, run_id, config_value, True) for video_id in ids),
                order_outputs=False,
            )
        )
        print(json.dumps({"videos": len(results), "rows": sum(item["candidate_rows"] for item in results)}, indent=2))
        return
    if stage == "evaluate":
        report = evaluate_remote.remote(run_id, config_value, EXPECTED_FULL_VIDEOS)
        print(json.dumps({key: report[key] for key in ("status", "scope", "evaluated_videos", "candidate_metrics", "failures")}, indent=2))
        return
    encoder = CandidateEncoder()
    if stage == "pilot":
        gate = evaluate_remote.remote(run_id, config_value, EXPECTED_FULL_VIDEOS)
        if gate["status"] != "pass":
            raise RuntimeError(f"candidate eval failed before pilot: {gate['failures'][:10]}")
        pilot_ids = ids[: min(3, len(ids))]
        if len(pilot_ids) < 2:
            raise RuntimeError("pilot needs at least two videos")
        results = list(
            encoder.encode_video.starmap(
                ((video_id, run_id, config_value, batch_size, False) for video_id in pilot_ids),
                order_outputs=False,
            )
        )
        report = finalize_pilot_remote.remote(run_id, config_value, results, gpu_usd_per_hour, max_usd)
        print(json.dumps(report, indent=2))
        return
    if stage == "encode":
        results = list(
            encoder.encode_video.starmap(
                ((video_id, run_id, config_value, batch_size, True) for video_id in ids),
                order_outputs=False,
            )
        )
        print(json.dumps({"videos": len(results), "rows": sum(item["rows"] for item in results)}, indent=2))
        return
    if stage == "select":
        results = list(
            select_video_remote.starmap(
                ((video_id, run_id, config_value) for video_id in ids), order_outputs=False
            )
        )
        print(json.dumps({"videos": len(results), "discovery_rows": sum(item["discovery_rows"] for item in results)}, indent=2))
        return
    if stage == "freeze":
        report = freeze_remote.remote(run_id, config_value)
        print(json.dumps({"status": report["status"], "videos": len(report["videos"]), "failures": report["failures"]}, indent=2))
        return
    if stage == "full":
        raise RuntimeError(
            "The one-shot full stage is intentionally disabled. Run candidates -> evaluate, "
            "inspect eval_gate.json, then pilot -> encode -> select -> freeze."
        )
    raise ValueError(f"unknown stage: {stage}")
