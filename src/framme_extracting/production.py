from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .config import PipelineConfig
from .core import Candidate, load_segmentation, seed_candidates


def make_budget_plan(
    paths: Iterable[str | Path], config: PipelineConfig
) -> dict[str, Any]:
    """Allocate the fixed global budget without dropping periodic locators."""

    videos: list[dict[str, Any]] = []
    for path in sorted((Path(value) for value in paths), key=lambda value: value.name):
        segmentation = load_segmentation(path)
        rows = seed_candidates(segmentation, config)
        locators = sum(row.locator_keep for row in rows)
        videos.append(
            {
                "video_id": segmentation.video.video_id,
                "locator_rows": locators,
                "extra_capacity": len(rows) - locators,
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
        for _fraction, _video_id, index in sorted(fractions, reverse=True)[
            : remaining - allocated
        ]:
            videos[index]["target_rows"] += 1

    assigned = sum(item["target_rows"] for item in videos)
    if assigned != config.target_embedding_rows:
        failures.append(f"assigned {assigned} rows, expected {config.target_embedding_rows}")
    return {
        "schema_version": "framme-budget/v2",
        "status": "pass" if not failures else "fail",
        "config_fingerprint": config.fingerprint,
        "target_embedding_rows": config.target_embedding_rows,
        "locator_rows": locator_total,
        "source_extra_rows": assigned - locator_total,
        "assigned_rows": assigned,
        "videos": videos,
        "failures": failures,
    }


def reconcile_budget_plan(
    plan: dict[str, Any], candidate_results: Iterable[dict[str, Any]]
) -> dict[str, Any]:
    """Freeze a static budget ceiling to the rows actually admitted after decode."""

    planned = {str(item["video_id"]): dict(item) for item in plan["videos"]}
    results: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    for value in candidate_results:
        item = dict(value)
        video_id = str(item.get("video_id", ""))
        if not video_id:
            failures.append("candidate result without video_id")
        elif video_id in results:
            failures.append(f"duplicate candidate result: {video_id}")
        else:
            results[video_id] = item

    unexpected = sorted(set(results) - set(planned))
    missing = sorted(set(planned) - set(results))
    if unexpected:
        failures.append(f"unexpected candidate videos: {unexpected[:5]}")
    if missing:
        failures.append(f"missing {len(missing)} candidate videos; first={missing[:5]}")

    videos: list[dict[str, Any]] = []
    for video_id in sorted(planned):
        item = planned[video_id]
        result = results.get(video_id)
        requested = int(item.get("requested_rows", item["target_rows"]))
        actual = int(result["candidate_rows"]) if result is not None else 0
        locator_rows = int(item["locator_rows"])
        if result is not None and int(result.get("decode_missing", 0)) != 0:
            failures.append(f"{video_id}: candidate decode is incomplete")
        if actual < locator_rows:
            failures.append(f"{video_id}: admitted {actual} below {locator_rows} locators")
        if actual > requested:
            failures.append(f"{video_id}: admitted {actual} above ceiling {requested}")
        videos.append(
            {
                **item,
                "requested_rows": requested,
                "target_rows": actual,
                "underfilled_rows": max(0, requested - actual),
            }
        )

    locator_total = sum(int(item["locator_rows"]) for item in videos)
    requested_total = sum(int(item["requested_rows"]) for item in videos)
    assigned_total = sum(int(item["target_rows"]) for item in videos)
    return {
        **plan,
        "schema_version": "framme-budget/v3",
        "status": "pass" if not failures else "fail",
        "realized": True,
        "requested_assigned_rows": requested_total,
        "locator_rows": locator_total,
        "source_extra_rows": assigned_total - locator_total,
        "assigned_rows": assigned_total,
        "underfilled_rows": requested_total - assigned_total,
        "videos": videos,
        "failures": failures,
    }


def validate_vector_store(
    candidates: list[Candidate], vectors: np.ndarray, config: PipelineConfig
) -> dict[str, Any]:
    """Cheap production integrity check; this does not score retrieval quality."""

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
