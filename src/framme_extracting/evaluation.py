from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .config import PipelineConfig
from .core import (
    Candidate,
    Segmentation,
    expanded_decode_targets,
    load_segmentation,
    seed_candidates,
)
from .storage import atomic_write_json, read_jsonl, sha256_file

WINDOW_LENGTHS = (5, 7, 9, 10, 11, 21)


def _percentile(values: Iterable[float], quantile: float) -> float:
    data = np.asarray(list(values), dtype=np.float64)
    return float(np.percentile(data, quantile)) if data.size else 0.0


def _covered_window_starts(shot_start: int, shot_end: int, frames: list[int], length: int) -> int:
    possible = shot_end - shot_start - length + 2
    if possible <= 0:
        return 0
    last_start = shot_end - length + 1
    intervals: list[tuple[int, int]] = []
    for frame_idx in frames:
        left = max(shot_start, frame_idx - length + 1)
        right = min(last_start, frame_idx)
        if left <= right:
            intervals.append((left, right))
    if not intervals:
        return 0
    intervals.sort()
    covered = 0
    left, right = intervals[0]
    for next_left, next_right in intervals[1:]:
        if next_left <= right + 1:
            right = max(right, next_right)
        else:
            covered += right - left + 1
            left, right = next_left, next_right
    return covered + right - left + 1


def temporal_window_coverage(
    segmentation: Segmentation, candidates: list[Candidate], lengths: Iterable[int]
) -> dict[str, float]:
    locator_by_shot: dict[int, list[int]] = {}
    for row in candidates:
        if row.locator_keep and row.rejected_reason != "decode_missing":
            locator_by_shot.setdefault(row.shot_id, []).append(row.frame_idx)
    output: dict[str, float] = {}
    for length in lengths:
        covered = total = 0
        for shot in segmentation.shots:
            possible = shot.length - length + 1
            if possible <= 0:
                continue
            total += possible
            covered += _covered_window_starts(
                shot.start_frame,
                shot.end_frame,
                sorted(locator_by_shot.get(shot.shot_id, [])),
                length,
            )
        output[str(length)] = covered / total if total else 1.0
    return output


def _max_gap(shot_start: int, shot_end: int, frames: list[int]) -> int:
    if not frames:
        return shot_end - shot_start + 1
    anchors = [shot_start, *sorted(set(frames)), shot_end]
    return max(right - left for left, right in zip(anchors, anchors[1:]))


def evaluate_segmentation_files(paths: Iterable[str | Path]) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []
    segmentations: list[Segmentation] = []
    input_hashes: dict[str, str] = {}
    for path in sorted((Path(value) for value in paths), key=lambda value: value.name):
        try:
            segmentation = load_segmentation(path)
            segmentations.append(segmentation)
            input_hashes[segmentation.video.video_id] = sha256_file(path)
        except Exception as error:  # report every malformed boundary file in one run
            failures.append(f"{path.name}: {error}")
    video_ids = [item.video.video_id for item in segmentations]
    duplicates = sorted(video_id for video_id, count in Counter(video_ids).items() if count > 1)
    if duplicates:
        failures.append(f"duplicate video ids: {duplicates}")
    shot_lengths = [shot.length for item in segmentations for shot in item.shots]
    transition_lengths = [
        zone.end_frame - zone.start_frame + 1
        for item in segmentations
        for zone in item.transition_zones
    ]
    if transition_lengths:
        warnings.append(
            f"preserved {len(transition_lengths)} unassigned transition zones; they are never widened into shots"
        )
    return {
        "status": "pass" if not failures else "fail",
        "scope": {
            "videos": len(segmentations),
            "shots": sum(len(item.shots) for item in segmentations),
            "raw_frames": sum(item.video.frame_count for item in segmentations),
        },
        "shot_length_frames": {
            "min": min(shot_lengths, default=0),
            "p10": _percentile(shot_lengths, 10),
            "p50": _percentile(shot_lengths, 50),
            "p90": _percentile(shot_lengths, 90),
            "p99": _percentile(shot_lengths, 99),
            "max": max(shot_lengths, default=0),
        },
        "transition_zones": {
            "count": len(transition_lengths),
            "frames": sum(transition_lengths),
            "max_length": max(transition_lengths, default=0),
        },
        "input_hashes": input_hashes,
        "failures": failures,
        "warnings": warnings,
    }


def evaluate_sampling_plan(
    paths: Iterable[str | Path], config: PipelineConfig
) -> dict[str, Any]:
    """Analytical pre-run evaluation; no video decode or GPU is involved."""

    paths = tuple(Path(value) for value in paths)
    total_seed = total_locator = total_decode = total_rescue_upper = 0
    sampling_failures: list[str] = []
    reason_counts: Counter[str] = Counter()
    coverage_numerator = {str(length): 0.0 for length in WINDOW_LENGTHS}
    coverage_denominator = {str(length): 0 for length in WINDOW_LENGTHS}
    for path in paths:
        segmentation = load_segmentation(path)
        rows = seed_candidates(segmentation, config)
        total_seed += len(rows)
        total_locator += sum(row.locator_keep for row in rows)
        for shot in segmentation.shots:
            shot_locators = sum(row.locator_keep and row.shot_id == shot.shot_id for row in rows)
            total_rescue_upper += int(math.ceil(shot_locators * config.max_rescue_fraction))
        total_decode += len(expanded_decode_targets(segmentation, rows, config.neighbor_radius))
        reason_counts.update(reason for row in rows for reason in row.reasons)
        coverage = temporal_window_coverage(segmentation, rows, WINDOW_LENGTHS)
        answer_coverage = coverage[str(config.answer_window_frames)]
        if answer_coverage < 1.0 - 1e-12:
            sampling_failures.append(
                f"{segmentation.video.video_id}: answer-window coverage {answer_coverage:.9f} < 1"
            )
        for length, value in coverage.items():
            weight = sum(max(0, shot.length - int(length) + 1) for shot in segmentation.shots)
            coverage_numerator[length] += value * weight
            coverage_denominator[length] += weight
    rescue_upper = total_rescue_upper
    row_upper = total_seed + rescue_upper
    budget = make_budget_plan(paths, config)
    bytes_per_row = config.embedding_dim * np.dtype(config.embedding_dtype).itemsize
    return {
        "config_fingerprint": config.fingerprint,
        "seed_rows": total_seed,
        "locator_rows": total_locator,
        "decode_target_rows": total_decode,
        "rescue_row_upper_bound": rescue_upper,
        "candidate_row_upper_bound": row_upper,
        "embedding_budget": budget,
        "candidate_cap": config.max_candidate_rows,
        "raw_seed_union_vector_gib_if_fully_encoded": total_seed * bytes_per_row / (1024**3),
        "raw_union_upper_vector_gib_if_fully_encoded": row_upper * bytes_per_row / (1024**3),
        "budgeted_vector_gib": config.target_embedding_rows * bytes_per_row / (1024**3),
        "vector_gib_cap": config.max_vector_gib,
        "time_weighted_window_coverage": {
            length: coverage_numerator[length] / coverage_denominator[length]
            if coverage_denominator[length]
            else 1.0
            for length in coverage_numerator
        },
        "reason_counts": dict(sorted(reason_counts.items())),
        "failures": sampling_failures,
        "status": (
            "pass"
            if not sampling_failures
            and budget["status"] == "pass"
            and config.target_embedding_rows <= config.max_candidate_rows
            and config.target_embedding_rows * bytes_per_row / (1024**3) <= config.max_vector_gib
            else "fail"
        ),
    }


def make_budget_plan(
    paths: Iterable[str | Path], config: PipelineConfig
) -> dict[str, Any]:
    """Allocate exactly the global embedding budget without dropping locators."""

    videos: list[dict[str, Any]] = []
    for path in sorted((Path(value) for value in paths), key=lambda value: value.name):
        segmentation = load_segmentation(path)
        rows = seed_candidates(segmentation, config)
        locators = sum(row.locator_keep for row in rows)
        capacity = len(rows) - locators
        videos.append(
            {
                "video_id": segmentation.video.video_id,
                "locator_rows": locators,
                "extra_capacity": capacity,
                "target_rows": locators,
            }
        )
    locator_total = sum(item["locator_rows"] for item in videos)
    remaining = config.target_embedding_rows - locator_total
    failures: list[str] = []
    if remaining < 0:
        failures.append(
            f"locator skeleton {locator_total} exceeds target {config.target_embedding_rows}"
        )
        remaining = 0
    capacity_total = sum(item["extra_capacity"] for item in videos)
    if remaining > capacity_total:
        failures.append(f"only {capacity_total} source extras available for {remaining} slots")
        remaining = capacity_total
    if remaining and capacity_total:
        fractions: list[tuple[float, str, int]] = []
        allocated = 0
        for index, item in enumerate(videos):
            exact = remaining * item["extra_capacity"] / capacity_total
            base = min(item["extra_capacity"], int(math.floor(exact)))
            item["target_rows"] += base
            allocated += base
            fractions.append((exact - base, item["video_id"], index))
        for _fraction, _video_id, index in sorted(fractions, reverse=True)[: remaining - allocated]:
            videos[index]["target_rows"] += 1
    assigned = sum(item["target_rows"] for item in videos)
    if assigned != config.target_embedding_rows:
        failures.append(f"assigned {assigned} rows, expected {config.target_embedding_rows}")
    return {
        "schema_version": "framme-budget/v1",
        "status": "pass" if not failures else "fail",
        "config_fingerprint": config.fingerprint,
        "target_embedding_rows": config.target_embedding_rows,
        "locator_rows": locator_total,
        "source_extra_rows": assigned - locator_total,
        "assigned_rows": assigned,
        "videos": videos,
        "failures": failures,
    }


def evaluate_candidate_run(
    boundary_paths: Iterable[str | Path],
    candidate_dir: str | Path,
    config: PipelineConfig,
    expected_videos: int | None = None,
) -> dict[str, Any]:
    paths = sorted((Path(value) for value in boundary_paths), key=lambda value: value.name)
    preflight = evaluate_segmentation_files(paths)
    failures = list(preflight["failures"])
    warnings = list(preflight["warnings"])
    if expected_videos is not None and len(paths) != expected_videos:
        failures.append(f"expected {expected_videos} boundary files, found {len(paths)}")

    total_rows = 0
    locator_rows = 0
    discovery_eligible = 0
    reason_counts: Counter[str] = Counter()
    rejection_counts: Counter[str] = Counter()
    max_locator_gap = 0
    coverage_weighted: dict[str, list[tuple[float, int]]] = {
        str(length): [] for length in WINDOW_LENGTHS
    }
    per_video: list[dict[str, Any]] = []
    root = Path(candidate_dir)
    budget_path = root.parent / "BUDGET_PLAN.json"
    cpu_gate_path = root.parent / "reports" / "cpu_pilot_gate.json"
    budget_targets: dict[str, int] = {}
    if not budget_path.exists():
        failures.append("missing BUDGET_PLAN.json")
    else:
        import json

        budget = json.loads(budget_path.read_text(encoding="utf-8"))
        if budget.get("config_fingerprint") != config.fingerprint:
            failures.append("budget/config fingerprint mismatch")
        budget_targets = {
            item["video_id"]: int(item["target_rows"]) for item in budget.get("videos", [])
        }
    if not cpu_gate_path.exists():
        failures.append("missing cpu_pilot_gate.json")
    else:
        import json

        cpu_gate = json.loads(cpu_gate_path.read_text(encoding="utf-8"))
        if cpu_gate.get("status") != "pass" or cpu_gate.get("config_fingerprint") != config.fingerprint:
            failures.append("CPU pilot gate failed or is stale")
    for boundary_path in paths:
        segmentation = load_segmentation(boundary_path)
        candidate_path = root / segmentation.video.video_id / "candidates.jsonl"
        done_path = root / segmentation.video.video_id / "candidate.done.json"
        if not candidate_path.exists() or not done_path.exists():
            failures.append(f"{segmentation.video.video_id}: missing candidate output/checkpoint")
            continue
        import json

        done = json.loads(done_path.read_text(encoding="utf-8"))
        if done.get("config_fingerprint") != config.fingerprint:
            failures.append(f"{segmentation.video.video_id}: stale candidate config")
        if done.get("candidate_sha256") != sha256_file(candidate_path):
            failures.append(f"{segmentation.video.video_id}: candidate checkpoint hash mismatch")
        if done.get("boundary_sha256") != sha256_file(boundary_path):
            failures.append(f"{segmentation.video.video_id}: boundary hash changed after candidate build")
        if int(done.get("decode_missing", -1)) != 0:
            failures.append(
                f"{segmentation.video.video_id}: {done.get('decode_missing')} requested frames failed to decode"
            )
        rows = [Candidate.from_dict(row) for row in read_jsonl(candidate_path)]
        target_rows = budget_targets.get(segmentation.video.video_id)
        if target_rows is None or len(rows) != target_rows or done.get("target_rows") != target_rows:
            failures.append(
                f"{segmentation.video.video_id}: rows/checkpoint/budget disagree "
                f"({len(rows)}, {done.get('target_rows')}, {target_rows})"
            )
        frame_ids = [row.frame_idx for row in rows]
        if len(frame_ids) != len(set(frame_ids)):
            failures.append(f"{segmentation.video.video_id}: duplicate frame_idx")
        transition_frames = {
            frame_idx
            for zone in segmentation.transition_zones
            for frame_idx in range(zone.start_frame, zone.end_frame + 1)
        }
        shot_by_id = {shot.shot_id: shot for shot in segmentation.shots}
        for row in rows:
            shot = shot_by_id.get(row.shot_id)
            if shot is None or not shot.contains(row.frame_idx):
                failures.append(
                    f"{segmentation.video.video_id}:{row.frame_idx} outside declared shot"
                )
            if row.frame_idx in transition_frames:
                failures.append(
                    f"{segmentation.video.video_id}:{row.frame_idx} inside transition zone"
                )
            reason_counts.update(row.reasons)
            if row.rejected_reason:
                rejection_counts[row.rejected_reason] += 1
        video_max_gap = 0
        for shot in segmentation.shots:
            frames = [
                row.frame_idx
                for row in rows
                if row.shot_id == shot.shot_id
                and row.locator_keep
                and row.rejected_reason != "decode_missing"
            ]
            if not frames:
                failures.append(
                    f"{segmentation.video.video_id}: shot {shot.shot_id} has no locator"
                )
            if not any(
                row.shot_id == shot.shot_id and row.rejected_reason is None for row in rows
            ):
                failures.append(
                    f"{segmentation.video.video_id}: shot {shot.shot_id} has no discovery-eligible frame"
                )
            video_max_gap = max(video_max_gap, _max_gap(shot.start_frame, shot.end_frame, frames))
        if video_max_gap > config.sample_stride:
            failures.append(
                f"{segmentation.video.video_id}: locator max gap {video_max_gap} > stride {config.sample_stride}"
            )
        max_locator_gap = max(max_locator_gap, video_max_gap)
        coverage = temporal_window_coverage(segmentation, rows, WINDOW_LENGTHS)
        answer_coverage = coverage[str(config.answer_window_frames)]
        if answer_coverage < 1.0 - 1e-12:
            failures.append(
                f"{segmentation.video.video_id}: answer-window coverage {answer_coverage:.9f} < 1"
            )
        for length, value in coverage.items():
            weight = sum(max(0, shot.length - int(length) + 1) for shot in segmentation.shots)
            coverage_weighted[length].append((value, weight))
        total_rows += len(rows)
        locator_rows += sum(row.locator_keep for row in rows)
        discovery_eligible += sum(row.rejected_reason is None for row in rows)
        per_video.append(
            {
                "video_id": segmentation.video.video_id,
                "rows": len(rows),
                "locator_rows": sum(row.locator_keep for row in rows),
                "max_locator_gap": video_max_gap,
                "window_coverage": coverage,
                "candidate_sha256": sha256_file(candidate_path),
            }
        )

    vector_bytes = total_rows * config.embedding_dim * np.dtype(config.embedding_dtype).itemsize
    vector_gib = vector_bytes / (1024**3)
    if total_rows > config.max_candidate_rows:
        failures.append(f"candidate rows {total_rows} exceed cap {config.max_candidate_rows}")
    if total_rows != config.target_embedding_rows:
        failures.append(
            f"candidate rows {total_rows} != target {config.target_embedding_rows}"
        )
    if vector_gib > config.max_vector_gib:
        failures.append(f"projected vector size {vector_gib:.3f} GiB exceeds cap {config.max_vector_gib}")
    weighted_coverage = {}
    for length, values in coverage_weighted.items():
        denominator = sum(weight for _, weight in values)
        weighted_coverage[length] = (
            sum(value * weight for value, weight in values) / denominator if denominator else 1.0
        )
    scope_name = "full" if expected_videos is not None and len(per_video) == expected_videos else "partial"
    return {
        "schema_version": "framme-eval/v1",
        "status": "pass" if not failures else "fail",
        "scope": scope_name,
        "expected_videos": expected_videos,
        "evaluated_videos": len(per_video),
        "config_fingerprint": config.fingerprint,
        "encoder_fingerprint": config.encoder_fingerprint,
        "preflight": preflight,
        "candidate_metrics": {
            "rows": total_rows,
            "locator_rows": locator_rows,
            "discovery_eligible_rows": discovery_eligible,
            "max_locator_gap": max_locator_gap,
            "time_weighted_window_coverage": weighted_coverage,
            "reason_counts": dict(sorted(reason_counts.items())),
            "rejection_counts": dict(sorted(rejection_counts.items())),
            "projected_vector_gib_fp16_1024d": vector_gib,
        },
        "per_video": per_video,
        "failures": failures,
        "warnings": warnings,
    }


def validate_vector_store(
    candidates: list[Candidate], vectors: np.ndarray, config: PipelineConfig
) -> dict[str, Any]:
    failures: list[str] = []
    expected_shape = (len(candidates), config.embedding_dim)
    if vectors.shape != expected_shape:
        failures.append(f"shape {vectors.shape} != {expected_shape}")
    if vectors.dtype != np.dtype(config.embedding_dtype):
        failures.append(f"dtype {vectors.dtype} != {config.embedding_dtype}")
    if vectors.ndim == 2 and vectors.shape[0]:
        norms = np.linalg.norm(vectors.astype(np.float32), axis=1)
        if not np.all(np.isfinite(norms)):
            failures.append("non-finite vector norm")
        if config.normalize_vectors and float(np.max(np.abs(norms - 1.0))) > 2e-3:
            failures.append("vectors are not L2 normalized within fp16 tolerance")
        norm_stats = {
            "min": float(norms.min()),
            "max": float(norms.max()),
            "mean": float(norms.mean()),
        }
    else:
        norm_stats = {"min": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "status": "pass" if not failures else "fail",
        "shape": list(vectors.shape),
        "dtype": str(vectors.dtype),
        "norm": norm_stats,
        "failures": failures,
    }


def require_passing_gate(
    path: str | Path,
    config: PipelineConfig,
    expected_videos: int,
) -> dict[str, Any]:
    import json

    gate_path = Path(path)
    if not gate_path.exists():
        raise RuntimeError(f"missing eval gate: {gate_path}")
    report = json.loads(gate_path.read_text(encoding="utf-8"))
    requirements = {
        "status": "pass",
        "scope": "full",
        "expected_videos": expected_videos,
        "evaluated_videos": expected_videos,
        "config_fingerprint": config.fingerprint,
        "encoder_fingerprint": config.encoder_fingerprint,
    }
    mismatches = [
        f"{key}: expected {expected!r}, got {report.get(key)!r}"
        for key, expected in requirements.items()
        if report.get(key) != expected
    ]
    if mismatches:
        raise RuntimeError("eval gate rejected run: " + "; ".join(mismatches))
    actual_rows = report.get("candidate_metrics", {}).get("rows")
    if actual_rows != config.target_embedding_rows:
        raise RuntimeError(
            f"eval gate candidate rows: expected {config.target_embedding_rows}, got {actual_rows}"
        )
    return report


def write_report(path: str | Path, report: dict[str, Any]) -> None:
    atomic_write_json(path, report)
