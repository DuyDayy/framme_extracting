from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from .config import PipelineConfig
from .core import FrameMetrics, VideoSpec, compute_frame_metrics


def ffprobe_video(path: str | Path) -> VideoSpec:
    source = Path(path)
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=width,height,avg_frame_rate,nb_read_frames,nb_frames:format=duration",
        "-of",
        "json",
        str(source),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(completed.stdout)
    stream = payload["streams"][0]
    numerator, denominator = (int(value) for value in stream["avg_frame_rate"].split("/"))
    fps = numerator / denominator
    frame_count = int(stream.get("nb_read_frames") or stream.get("nb_frames") or 0)
    return VideoSpec(
        video_id=source.stem,
        path=str(source),
        width=int(stream["width"]),
        height=int(stream["height"]),
        fps=fps,
        frame_count=frame_count,
        duration_seconds=float(payload.get("format", {}).get("duration", 0.0)),
    )


def validate_probe(expected: VideoSpec, observed: VideoSpec) -> list[str]:
    errors: list[str] = []
    if expected.video_id != observed.video_id:
        errors.append(f"video_id: expected {expected.video_id}, got {observed.video_id}")
    for field in ("width", "height", "frame_count"):
        left, right = getattr(expected, field), getattr(observed, field)
        if left != right:
            errors.append(f"{field}: expected {left}, got {right}")
    if abs(expected.fps - observed.fps) > 1e-3:
        errors.append(f"fps: expected {expected.fps}, got {observed.fps}")
    return errors


def decode_selected_frames(path: str | Path, frame_indices: Iterable[int]) -> dict[int, np.ndarray]:
    """Decode selected frames in one sequential pass with PyAV.

    Frame indices here are decode-order indices and match the TransNet boundaries.
    The function deliberately avoids seeking: keyframe seeking can return a different
    decoded pixel for the same requested index on codecs with reordering.
    """

    import av

    targets = sorted(set(int(value) for value in frame_indices))
    if not targets:
        return {}
    if targets[0] < 0:
        raise ValueError("negative frame index")
    wanted = set(targets)
    last = targets[-1]
    decoded: dict[int, np.ndarray] = {}
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame_idx, frame in enumerate(container.decode(stream)):
            if frame_idx in wanted:
                decoded[frame_idx] = frame.to_ndarray(format="rgb24")
            if frame_idx >= last:
                break
    return decoded


def iter_selected_frames(
    path: str | Path, frame_indices: Iterable[int]
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield requested RGB frames and immediately release non-requested pixels."""

    import av

    targets = sorted(set(int(value) for value in frame_indices))
    if not targets:
        return
    if targets[0] < 0:
        raise ValueError("negative frame index")
    cursor = 0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame_idx, frame in enumerate(container.decode(stream)):
            while cursor < len(targets) and targets[cursor] < frame_idx:
                cursor += 1
            if cursor >= len(targets):
                break
            if targets[cursor] == frame_idx:
                yield frame_idx, frame.to_ndarray(format="rgb24")
                cursor += 1


def decode_selected_metrics(
    path: str | Path, frame_indices: Iterable[int], config: PipelineConfig
) -> dict[int, FrameMetrics]:
    return {
        frame_idx: compute_frame_metrics(rgb, config)
        for frame_idx, rgb in iter_selected_frames(path, frame_indices)
    }
