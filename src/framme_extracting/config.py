from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PipelineConfig:
    """All values that can change the produced candidate or vector dataset."""

    schema_version: str = "framme-extracting/v1"
    sample_stride: int = 10
    answer_window_frames: int = 10
    neighbor_radius: int = 4
    cluster_radius: int = 8
    discovery_max_gap: int = 10
    discovery_budget_multiplier: float = 1.15
    raw_refine_radius: int = 9
    thumbnail_size: int = 64
    black_luma_threshold: float = 8.0
    black_ratio_threshold: float = 0.985
    hard_blur_threshold: float = 0.15
    rescue_min_gain: float = 0.08
    max_rescue_fraction: float = 0.15
    appearance_mad_scale: float = 3.0
    motion_mad_scale: float = 3.0
    text_mad_scale: float = 3.0
    dhash_duplicate_distance: int = 2
    semantic_duplicate_cosine: float = 0.985
    semantic_local_window: int = 24
    min_frames_per_shot: int = 1
    target_embedding_rows: int = 627_000
    max_candidate_rows: int = 627_000
    max_vector_gib: float = 1.25
    embedding_dim: int = 1024
    search_dim: int = 512
    embedding_dtype: str = "float16"
    encoder_model: str = "jinaai/jina-clip-v2"
    encoder_revision: str = "e10d47f5691d0454a0fb5d13f46f2199b74cb436"
    encoder_code_revision: str = "39e6a55ae971b59bea6e44675d237c99762e7ee2"
    image_size: int = 512
    image_resize: str = "shortest"
    image_interpolation: str = "bicubic"
    normalize_vectors: bool = True

    def __post_init__(self) -> None:
        if self.sample_stride < 1:
            raise ValueError("sample_stride must be positive")
        if self.sample_stride > self.answer_window_frames:
            raise ValueError("sample_stride cannot exceed the guaranteed answer window")
        if not 0 <= self.neighbor_radius < self.sample_stride:
            raise ValueError("neighbor_radius must be in [0, sample_stride)")
        if self.discovery_max_gap < self.sample_stride:
            raise ValueError("discovery_max_gap cannot be below sample_stride")
        if self.discovery_budget_multiplier < 1.0:
            raise ValueError("discovery_budget_multiplier must be at least 1")
        if not 0 <= self.max_rescue_fraction <= 1:
            raise ValueError("max_rescue_fraction must be in [0, 1]")
        if not 0 < self.search_dim <= self.embedding_dim:
            raise ValueError("search_dim must be in (0, embedding_dim]")
        if self.target_embedding_rows > self.max_candidate_rows:
            raise ValueError("target_embedding_rows cannot exceed max_candidate_rows")
        if self.raw_refine_radius < self.sample_stride - 1:
            raise ValueError("raw_refine_radius is too small for the periodic skeleton")
        if not 0 < self.semantic_duplicate_cosine <= 1:
            raise ValueError("semantic_duplicate_cosine must be in (0, 1]")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    def canonical_json(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @property
    def encoder_fingerprint(self) -> str:
        payload = {
            "model": self.encoder_model,
            "revision": self.encoder_revision,
            "code_revision": self.encoder_code_revision,
            "dim": self.embedding_dim,
            "dtype": self.embedding_dtype,
            "image_size": self.image_size,
            "resize": self.image_resize,
            "interpolation": self.image_interpolation,
            "normalized": self.normalize_vectors,
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
