from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from .config import PipelineConfig


@dataclass(frozen=True)
class VideoSpec:
    video_id: str
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float


@dataclass(frozen=True)
class Shot:
    shot_id: int
    start_frame: int
    end_frame: int
    boundary_score: float | None = None

    @property
    def length(self) -> int:
        return self.end_frame - self.start_frame + 1

    def contains(self, frame_idx: int) -> bool:
        return self.start_frame <= frame_idx <= self.end_frame


@dataclass(frozen=True)
class TransitionZone:
    start_frame: int
    end_frame: int
    kind: str


@dataclass(frozen=True)
class Segmentation:
    video: VideoSpec
    shots: tuple[Shot, ...]
    transition_zones: tuple[TransitionZone, ...]

    def shot_for_frame(self, frame_idx: int) -> Shot | None:
        # Shot counts are small enough for generation; callers doing dense work use
        # the per-shot loops rather than invoking this for every raw frame.
        for shot in self.shots:
            if shot.contains(frame_idx):
                return shot
        return None


@dataclass(frozen=True)
class FrameMetrics:
    pixel_sha256: str
    dhash: int
    luma_mean: float
    luma_std: float
    black_ratio: float
    sharpness: float
    entropy: float
    text_proxy: float
    quality_score: float
    hard_invalid: bool


@dataclass
class Candidate:
    video_id: str
    frame_idx: int
    shot_id: int
    reasons: set[str] = field(default_factory=set)
    locator_keep: bool = False
    hard_keep: bool = False
    cluster_id: int | None = None
    metrics: FrameMetrics | None = None
    selected: bool = False
    rejected_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["reasons"] = sorted(self.reasons)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Candidate":
        raw = dict(value)
        raw["reasons"] = set(raw.get("reasons", []))
        if raw.get("metrics") is not None:
            raw["metrics"] = FrameMetrics(**raw["metrics"])
        return cls(**raw)


def load_segmentation(path: str | Path) -> Segmentation:
    import json

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    video_raw = raw["video"]
    video = VideoSpec(
        video_id=str(video_raw["video_id"]),
        path=str(video_raw.get("path", "")),
        width=int(video_raw["width"]),
        height=int(video_raw["height"]),
        fps=float(video_raw["fps"]),
        frame_count=int(video_raw["frame_count"]),
        duration_seconds=float(video_raw.get("duration_seconds", 0.0)),
    )
    if video.frame_count <= 0 or video.fps <= 0:
        raise ValueError(f"invalid video metadata for {video.video_id}")

    shots = tuple(
        Shot(
            shot_id=int(item["shot_id"]),
            start_frame=int(item["start_frame"]),
            end_frame=int(item["end_frame"]),
            boundary_score=(
                None if item.get("boundary_score") is None else float(item["boundary_score"])
            ),
        )
        for item in raw["shots"]
    )
    if not shots:
        raise ValueError(f"no shots for {video.video_id}")

    transitions: list[TransitionZone] = []
    previous_end = -1
    seen_ids: set[int] = set()
    for shot in shots:
        if shot.shot_id in seen_ids:
            raise ValueError(f"duplicate shot_id={shot.shot_id} in {video.video_id}")
        seen_ids.add(shot.shot_id)
        if shot.start_frame < 0 or shot.end_frame >= video.frame_count or shot.length <= 0:
            raise ValueError(f"invalid shot bounds {shot} in {video.video_id}")
        if shot.start_frame <= previous_end:
            raise ValueError(f"overlapping or unsorted shots in {video.video_id}")
        if shot.start_frame > previous_end + 1:
            transitions.append(
                TransitionZone(previous_end + 1, shot.start_frame - 1, "unassigned_gap")
            )
        previous_end = shot.end_frame
    if previous_end < video.frame_count - 1:
        transitions.append(
            TransitionZone(previous_end + 1, video.frame_count - 1, "unassigned_tail")
        )
    return Segmentation(video=video, shots=shots, transition_zones=tuple(transitions))


def _upsert_candidate(
    rows: dict[int, Candidate], shot: Shot, frame_idx: int, reason: str, *, locator: bool = False
) -> None:
    frame_idx = max(shot.start_frame, min(frame_idx, shot.end_frame))
    row = rows.setdefault(
        frame_idx,
        Candidate(video_id="", frame_idx=frame_idx, shot_id=shot.shot_id),
    )
    row.reasons.add(reason)
    row.locator_keep |= locator


def seed_candidates(segmentation: Segmentation, config: PipelineConfig) -> list[Candidate]:
    rows: dict[int, Candidate] = {}
    for shot in segmentation.shots:
        # Minimum hitting set for every stride-sized window: select the right edge
        # of each full cell. A shot shorter than the answer window still gets one
        # locator so it remains retrievable.
        if shot.length < config.sample_stride:
            periodic = [(shot.start_frame + shot.end_frame) // 2]
        else:
            periodic = list(
                range(
                    shot.start_frame + config.sample_stride - 1,
                    shot.end_frame + 1,
                    config.sample_stride,
                )
            )
        for frame_idx in periodic:
            _upsert_candidate(rows, shot, frame_idx, "periodic", locator=True)

        _upsert_candidate(rows, shot, shot.start_frame, "boundary_start")
        _upsert_candidate(rows, shot, shot.end_frame, "boundary_end")
        if shot.length <= config.sample_stride:
            _upsert_candidate(
                rows, shot, (shot.start_frame + shot.end_frame) // 2, "short_shot"
            )
        else:
            for fraction, label in ((0.25, "stable_q1"), (0.5, "stable_mid"), (0.75, "stable_q3")):
                offset = int(round((shot.length - 1) * fraction))
                _upsert_candidate(rows, shot, shot.start_frame + offset, label)

    for row in rows.values():
        row.video_id = segmentation.video.video_id
    return sorted(rows.values(), key=lambda item: item.frame_idx)


def expanded_decode_targets(
    segmentation: Segmentation, seeds: Iterable[Candidate], neighbor_radius: int
) -> list[int]:
    targets: set[int] = set()
    by_id = {shot.shot_id: shot for shot in segmentation.shots}
    for row in seeds:
        shot = by_id[row.shot_id]
        for delta in range(-neighbor_radius, neighbor_radius + 1):
            frame_idx = row.frame_idx + delta
            if shot.contains(frame_idx):
                targets.add(frame_idx)
    return sorted(targets)


def _resize_gray(rgb: np.ndarray, size: int) -> np.ndarray:
    image = Image.fromarray(rgb, mode="RGB").convert("L")
    return np.asarray(image.resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32)


def _dhash(gray: np.ndarray) -> int:
    image = Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8), mode="L")
    pixels = np.asarray(image.resize((9, 8), Image.Resampling.BILINEAR))
    bits = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in bits.ravel():
        value = (value << 1) | int(bit)
    return value


def hamming64(a: int, b: int) -> int:
    return (int(a) ^ int(b)).bit_count()


def compute_frame_metrics(rgb: np.ndarray, config: PipelineConfig) -> FrameMetrics:
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("frame must be uint8 RGB")
    gray = _resize_gray(rgb, config.thumbnail_size)
    luma_mean = float(gray.mean())
    luma_std = float(gray.std())
    black_ratio = float(np.mean(gray <= config.black_luma_threshold))
    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    sharpness_raw = float((gx.var() + gy.var()) / (255.0 * 255.0))
    hist, _ = np.histogram(gray, bins=32, range=(0, 256))
    probabilities = hist.astype(np.float64) / max(1, hist.sum())
    probabilities = probabilities[probabilities > 0]
    entropy = float(-(probabilities * np.log2(probabilities)).sum() / 5.0)
    lower = gray[gray.shape[0] // 2 :, :]
    text_proxy = float(
        (np.mean(np.abs(np.diff(lower, axis=1))) + np.mean(np.abs(np.diff(lower, axis=0))))
        / 510.0
    )
    hard_invalid = black_ratio >= config.black_ratio_threshold
    exposure = max(0.0, 1.0 - abs(luma_mean - 127.5) / 127.5)
    quality = (
        0.40 * min(1.0, sharpness_raw / 0.02)
        + 0.25 * min(1.0, entropy)
        + 0.20 * exposure
        + 0.15 * min(1.0, luma_std / 64.0)
    )
    if hard_invalid:
        quality = 0.0
    return FrameMetrics(
        pixel_sha256=pixel_sha256(rgb),
        dhash=_dhash(gray),
        luma_mean=luma_mean,
        luma_std=luma_std,
        black_ratio=black_ratio,
        sharpness=sharpness_raw,
        entropy=entropy,
        text_proxy=text_proxy,
        quality_score=float(quality),
        hard_invalid=hard_invalid,
    )


def pixel_sha256(rgb: np.ndarray) -> str:
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("frame must be uint8 RGB")
    return hashlib.sha256(rgb.tobytes(order="C")).hexdigest()


def _robust_threshold(values: np.ndarray, scale: float) -> float:
    if values.size == 0:
        return math.inf
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return median + scale * max(mad, 1e-9)


def enrich_candidates(
    segmentation: Segmentation,
    seeds: list[Candidate],
    decoded_rgb: dict[int, np.ndarray | FrameMetrics],
    config: PipelineConfig,
) -> list[Candidate]:
    """Attach metrics, change-source provenance, rescue neighbors and cheap dedup.

    All periodic rows remain present as the localization view even when a better
    neighbor is added for discovery. This is crucial for temporal coverage.
    """

    by_shot: dict[int, list[Candidate]] = {}
    for row in seeds:
        if row.frame_idx not in decoded_rgb:
            row.rejected_reason = "decode_missing"
            continue
        value = decoded_rgb[row.frame_idx]
        row.metrics = value if isinstance(value, FrameMetrics) else compute_frame_metrics(value, config)
        if row.metrics.hard_invalid:
            row.rejected_reason = "hard_invalid"
        by_shot.setdefault(row.shot_id, []).append(row)

    result = {row.frame_idx: row for row in seeds}
    shots = {shot.shot_id: shot for shot in segmentation.shots}
    for shot_id, shot_rows in by_shot.items():
        shot_rows.sort(key=lambda row: row.frame_idx)
        valid = [row for row in shot_rows if row.metrics is not None]
        appearance: list[float] = []
        motion: list[float] = []
        text_change: list[float] = []
        for previous, current in zip(valid, valid[1:]):
            assert previous.metrics and current.metrics
            appearance.append(hamming64(previous.metrics.dhash, current.metrics.dhash) / 64.0)
            motion.append(abs(current.metrics.luma_std - previous.metrics.luma_std) / 255.0)
            text_change.append(abs(current.metrics.text_proxy - previous.metrics.text_proxy))
        a_thr = _robust_threshold(np.asarray(appearance), config.appearance_mad_scale)
        m_thr = _robust_threshold(np.asarray(motion), config.motion_mad_scale)
        t_thr = _robust_threshold(np.asarray(text_change), config.text_mad_scale)
        for index in range(1, len(valid)):
            row = valid[index]
            if appearance[index - 1] >= a_thr:
                row.reasons.add("appearance_change")
            if motion[index - 1] >= m_thr:
                row.reasons.add("motion_change")
            if text_change[index - 1] >= t_thr:
                row.reasons.add("text_change")

        shot = shots[shot_id]
        rescue_proposals: dict[int, tuple[float, FrameMetrics]] = {}
        locator_count = 0
        for original in list(valid):
            if not original.locator_keep or original.metrics is None:
                continue
            locator_count += 1
            best_idx = original.frame_idx
            best_metrics = original.metrics
            for delta in range(-config.neighbor_radius, config.neighbor_radius + 1):
                frame_idx = original.frame_idx + delta
                if not shot.contains(frame_idx) or frame_idx not in decoded_rgb:
                    continue
                value = decoded_rgb[frame_idx]
                metrics = value if isinstance(value, FrameMetrics) else compute_frame_metrics(value, config)
                if not metrics.hard_invalid and metrics.quality_score > best_metrics.quality_score:
                    best_idx, best_metrics = frame_idx, metrics
            if (
                best_idx != original.frame_idx
                and best_metrics.quality_score - original.metrics.quality_score >= config.rescue_min_gain
            ):
                gain = best_metrics.quality_score - original.metrics.quality_score
                previous = rescue_proposals.get(best_idx)
                if previous is None or gain > previous[0]:
                    rescue_proposals[best_idx] = (gain, best_metrics)
        rescue_limit = int(math.ceil(locator_count * config.max_rescue_fraction))
        for best_idx, (_gain, best_metrics) in sorted(
            rescue_proposals.items(), key=lambda item: (-item[1][0], item[0])
        )[:rescue_limit]:
            rescue = result.get(best_idx)
            if rescue is None:
                rescue = Candidate(
                    video_id=segmentation.video.video_id,
                    frame_idx=best_idx,
                    shot_id=shot_id,
                    metrics=best_metrics,
                )
                result[best_idx] = rescue
            rescue.reasons.add("quality_rescue")
            rescue.hard_keep = True

    ordered = sorted(result.values(), key=lambda row: row.frame_idx)
    previous_by_shot: dict[int, Candidate] = {}
    for row in ordered:
        row.cluster_id = row.frame_idx // config.cluster_radius
        previous = previous_by_shot.get(row.shot_id)
        if (
            previous is not None
            and not row.locator_keep
            and not row.hard_keep
            and previous.metrics is not None
            and row.metrics is not None
            and row.frame_idx - previous.frame_idx <= config.cluster_radius
            and hamming64(previous.metrics.dhash, row.metrics.dhash)
            <= config.dhash_duplicate_distance
        ):
            row.rejected_reason = "cheap_duplicate"
        elif row.rejected_reason is None:
            previous_by_shot[row.shot_id] = row
    return ordered


def candidates_from_rows(rows: Iterable[dict[str, Any]]) -> list[Candidate]:
    return [Candidate.from_dict(row) for row in rows]
