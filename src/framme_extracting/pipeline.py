from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .config import PipelineConfig
from .core import enrich_candidates, expanded_decode_targets, load_segmentation, seed_candidates
from .storage import atomic_write_json, atomic_write_jsonl, sha256_file
from .video_io import decode_selected_metrics, ffprobe_video, validate_probe


def build_video_candidates(
    video_path: str | Path,
    boundary_path: str | Path,
    output_dir: str | Path,
    config: PipelineConfig,
    target_rows: int | None = None,
) -> dict[str, Any]:
    video_path = Path(video_path)
    boundary_path = Path(boundary_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    segmentation = load_segmentation(boundary_path)
    probe_started = time.perf_counter()
    observed = ffprobe_video(video_path)
    probe_errors = validate_probe(segmentation.video, observed)
    if probe_errors:
        raise RuntimeError("probe/boundary mismatch: " + "; ".join(probe_errors))
    probe_seconds = time.perf_counter() - probe_started

    seeds = seed_candidates(segmentation, config)
    targets = expanded_decode_targets(segmentation, seeds, config.neighbor_radius)
    scan_started = time.perf_counter()
    decoded = decode_selected_metrics(video_path, targets, config)
    rows = enrich_candidates(segmentation, seeds, decoded, config)
    scan_seconds = time.perf_counter() - scan_started
    if target_rows is not None:
        mandatory = [row for row in rows if row.locator_keep]
        if len(mandatory) > target_rows:
            raise RuntimeError(
                f"locator rows {len(mandatory)} exceed assigned budget {target_rows}"
            )
        extras = [
            row
            for row in rows
            if not row.locator_keep and row.rejected_reason is None
        ]

        def admission_score(row):
            source_priority = sum(
                {
                    "quality_rescue": 5.0,
                    "text_change": 4.0,
                    "appearance_change": 3.0,
                    "motion_change": 2.0,
                    "boundary_start": 1.0,
                    "boundary_end": 1.0,
                }.get(reason, 0.25)
                for reason in row.reasons
            )
            quality = row.metrics.quality_score if row.metrics else 0.0
            return (row.hard_keep, source_priority, quality, -row.frame_idx)

        extras.sort(key=admission_score, reverse=True)
        rows = sorted(mandatory + extras[: target_rows - len(mandatory)], key=lambda row: row.frame_idx)
        if len(rows) != target_rows:
            raise RuntimeError(f"could only admit {len(rows)} of assigned {target_rows} rows")
    candidate_path = output / "candidates.jsonl"
    atomic_write_jsonl(candidate_path, (row.to_dict() for row in rows))
    checkpoint = {
        "schema_version": config.schema_version,
        "video_id": segmentation.video.video_id,
        "config_fingerprint": config.fingerprint,
        "video_sha256": sha256_file(video_path),
        "boundary_sha256": sha256_file(boundary_path),
        "candidate_sha256": sha256_file(candidate_path),
        "seed_rows": len(seeds),
        "decode_targets": len(targets),
        "decoded_rows": len(decoded),
        "candidate_rows": len(rows),
        "target_rows": target_rows,
        "decode_missing": len(targets) - len(decoded),
        "probe_seconds": probe_seconds,
        "scan_seconds": scan_seconds,
        "total_seconds": time.perf_counter() - started,
    }
    atomic_write_json(output / "candidate.done.json", checkpoint)
    # Read-after-write is intentional: the done marker is trusted by later stages.
    persisted = json.loads((output / "candidate.done.json").read_text(encoding="utf-8"))
    if persisted != checkpoint or sha256_file(candidate_path) != checkpoint["candidate_sha256"]:
        raise RuntimeError("candidate checkpoint readback failed")
    return checkpoint
