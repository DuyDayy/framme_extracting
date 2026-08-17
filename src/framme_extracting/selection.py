from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .config import PipelineConfig
from .core import Candidate, Segmentation, Shot


@dataclass(frozen=True)
class SelectionViews:
    discovery_indices: tuple[int, ...]
    localization_indices: tuple[int, ...]
    rejected_semantic_indices: tuple[int, ...]


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("zero embedding row")
    return values / norms


def search_view(master_vectors: np.ndarray, search_dim: int) -> np.ndarray:
    if master_vectors.ndim != 2 or search_dim > master_vectors.shape[1]:
        raise ValueError("invalid master vector shape/search_dim")
    return normalize_rows(master_vectors[:, :search_dim])


def _semantic_rejects(
    candidates: list[Candidate], vectors: np.ndarray, config: PipelineConfig
) -> set[int]:
    normalized = normalize_rows(vectors)
    rejected: set[int] = set()
    last_kept_by_shot: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(candidates):
        if row.rejected_reason is not None or row.locator_keep or row.hard_keep:
            if row.rejected_reason is None:
                last_kept_by_shot[row.shot_id].append(index)
            continue
        local = last_kept_by_shot[row.shot_id]
        duplicate = False
        for previous_index in reversed(local):
            previous = candidates[previous_index]
            if row.frame_idx - previous.frame_idx > config.semantic_local_window:
                break
            similarity = float(normalized[index] @ normalized[previous_index])
            if similarity >= config.semantic_duplicate_cosine:
                duplicate = True
                break
        if duplicate:
            rejected.add(index)
        else:
            local.append(index)
    return rejected


def _utility(row: Candidate) -> tuple[float, int]:
    quality = row.metrics.quality_score if row.metrics else 0.0
    reason_bonus = min(0.30, 0.05 * len(row.reasons))
    hard_bonus = 1.0 if row.hard_keep else 0.0
    return hard_bonus + quality + reason_bonus, -row.frame_idx


def _repair_shot_gap(
    shot: Shot,
    selected: set[int],
    eligible: list[int],
    candidates: list[Candidate],
    max_gap: int,
) -> None:
    """Greedily add the best candidate inside every uncovered temporal gap."""

    while True:
        frames = sorted(candidates[index].frame_idx for index in selected if candidates[index].shot_id == shot.shot_id)
        anchors = [shot.start_frame, *frames, shot.end_frame]
        widest = max(zip(anchors, anchors[1:]), key=lambda pair: pair[1] - pair[0])
        if widest[1] - widest[0] <= max_gap:
            return
        choices = [
            index
            for index in eligible
            if index not in selected and widest[0] < candidates[index].frame_idx < widest[1]
        ]
        if not choices:
            return
        midpoint = (widest[0] + widest[1]) / 2.0
        chosen = max(
            choices,
            key=lambda index: (
                -abs(candidates[index].frame_idx - midpoint),
                *_utility(candidates[index]),
            ),
        )
        selected.add(chosen)


def select_views(
    segmentation: Segmentation,
    candidates: list[Candidate],
    master_vectors: np.ndarray,
    config: PipelineConfig,
) -> SelectionViews:
    if master_vectors.shape != (len(candidates), config.embedding_dim):
        raise ValueError(
            f"vector shape {master_vectors.shape} != ({len(candidates)}, {config.embedding_dim})"
        )
    semantic_rejected = _semantic_rejects(candidates, master_vectors, config)
    selected: set[int] = set()
    by_shot: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(candidates):
        if row.locator_keep and row.rejected_reason != "decode_missing":
            selected.add(index)
        if row.rejected_reason is None and index not in semantic_rejected:
            by_shot[row.shot_id].append(index)
            if row.hard_keep:
                selected.add(index)

    for shot in segmentation.shots:
        eligible = by_shot[shot.shot_id]
        if not eligible:
            continue
        # Every shot is represented. Prefer the highest-quality source-rich row.
        selected.add(max(eligible, key=lambda index: _utility(candidates[index])))
        _repair_shot_gap(
            shot, selected, eligible, candidates, config.discovery_max_gap
        )

    # Source-aware extras: after mandatory shot/gap coverage, reserve the remaining
    # budget round-robin across active sources, then refill globally after dedup.
    eligible_all = [
        index
        for indices in by_shot.values()
        for index in indices
        if index not in selected
    ]
    base_target = int(
        np.ceil(segmentation.video.frame_count / config.discovery_max_gap)
        * config.discovery_budget_multiplier
    )
    target = min(len(selected) + len(eligible_all), max(len(selected), base_target))
    source_queues: dict[str, list[int]] = defaultdict(list)
    for index in eligible_all:
        for reason in candidates[index].reasons:
            source_queues[reason].append(index)
    for queue in source_queues.values():
        queue.sort(key=lambda index: _utility(candidates[index]), reverse=True)
    sources = sorted(source_queues)
    cursors = {source: 0 for source in sources}
    progressed = True
    while len(selected) < target and progressed:
        progressed = False
        for source in sources:
            queue = source_queues[source]
            cursor = cursors[source]
            while cursor < len(queue) and queue[cursor] in selected:
                cursor += 1
            cursors[source] = cursor
            if cursor < len(queue) and len(selected) < target:
                selected.add(queue[cursor])
                cursors[source] += 1
                progressed = True
    if len(selected) < target:
        refill = sorted(
            (index for index in eligible_all if index not in selected),
            key=lambda index: _utility(candidates[index]),
            reverse=True,
        )
        selected.update(refill[: target - len(selected)])

    discovery = tuple(sorted(selected, key=lambda index: candidates[index].frame_idx))
    localization = tuple(
        index
        for index, row in enumerate(candidates)
        if row.locator_keep and row.rejected_reason != "decode_missing"
    )
    for index in discovery:
        candidates[index].selected = True
    return SelectionViews(
        discovery_indices=discovery,
        localization_indices=localization,
        rejected_semantic_indices=tuple(sorted(semantic_rejected)),
    )


def promote_vectors(
    candidates: list[Candidate], master_vectors: np.ndarray, indices: Iterable[int]
) -> tuple[list[Candidate], np.ndarray]:
    ordered_indices = sorted(
        set(int(index) for index in indices),
        key=lambda index: (candidates[index].video_id, candidates[index].frame_idx),
    )
    promoted = np.take(master_vectors, ordered_indices, axis=0)
    if promoted.dtype != master_vectors.dtype:
        raise AssertionError("np.take changed vector dtype")
    for output_index, source_index in enumerate(ordered_indices):
        if promoted[output_index].tobytes() != master_vectors[source_index].tobytes():
            raise AssertionError("promoted vector bytes differ from candidate store")
    return [candidates[index] for index in ordered_indices], promoted
