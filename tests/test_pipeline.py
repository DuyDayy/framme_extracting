from __future__ import annotations

import json
from dataclasses import replace

import numpy as np

from framme_extracting.config import PipelineConfig
from framme_extracting.core import (
    Candidate,
    FrameMetrics,
    Segmentation,
    Shot,
    TransitionZone,
    VideoSpec,
    load_segmentation,
    seed_candidates,
)
from framme_extracting.evaluation import (
    temporal_window_coverage,
)
from framme_extracting.production import (
    make_budget_plan,
    reconcile_budget_plan,
    validate_vector_store,
)
from framme_extracting.selection import promote_vectors, search_view, select_views
from framme_extracting.storage import atomic_save_npy, atomic_write_json


def _segmentation() -> Segmentation:
    return Segmentation(
        video=VideoSpec("L21_V001", "x.mp4", 320, 180, 25.0, 42, 1.68),
        shots=(Shot(1, 0, 19), Shot(2, 22, 41)),
        transition_zones=(TransitionZone(20, 21, "unassigned_gap"),),
    )


def _metrics(frame_idx: int, quality: float = 0.5) -> FrameMetrics:
    return FrameMetrics(
        pixel_sha256=f"{frame_idx:064x}",
        dhash=frame_idx,
        luma_mean=100.0,
        luma_std=20.0,
        black_ratio=0.0,
        sharpness=0.02,
        entropy=0.8,
        text_proxy=0.1,
        quality_score=quality,
        hard_invalid=False,
    )


def test_config_fingerprint_is_deterministic_and_sensitive() -> None:
    config = PipelineConfig()
    assert config.fingerprint == PipelineConfig().fingerprint
    assert config.fingerprint != replace(config, sample_stride=7).fingerprint
    assert config.embedding_dim == 1024


def test_boundaries_preserve_transition_gap(tmp_path) -> None:
    path = tmp_path / "L21_V001.json"
    path.write_text(
        json.dumps(
            {
                "video": {
                    "video_id": "L21_V001",
                    "path": "x.mp4",
                    "width": 320,
                    "height": 180,
                    "fps": 25,
                    "frame_count": 42,
                    "duration_seconds": 1.68,
                },
                "shots": [
                    {"shot_id": 1, "start_frame": 0, "end_frame": 19},
                    {"shot_id": 2, "start_frame": 22, "end_frame": 41},
                ],
            }
        ),
        encoding="utf-8",
    )
    segmentation = load_segmentation(path)
    assert segmentation.transition_zones == (TransitionZone(20, 21, "unassigned_gap"),)
    assert segmentation.shot_for_frame(20) is None


def test_periodic_candidates_cover_each_shot_without_entering_gap() -> None:
    config = PipelineConfig(sample_stride=8)
    rows = seed_candidates(_segmentation(), config)
    periodic = [row for row in rows if row.locator_keep]
    assert [row.frame_idx for row in periodic] == [7, 15, 29, 37]
    assert all(row.frame_idx not in {20, 21} for row in rows)
    assert temporal_window_coverage(_segmentation(), rows, [1, 5, 8])["8"] == 1.0


def test_search_view_truncates_then_renormalizes() -> None:
    master = np.asarray([[3.0, 4.0, 12.0], [0.0, 2.0, 0.0]], dtype=np.float16)
    view = search_view(master, 2)
    np.testing.assert_allclose(np.linalg.norm(view, axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(view[0], [0.6, 0.8], atol=1e-6)


def test_budget_keeps_every_locator_and_assigns_exact_target(tmp_path) -> None:
    path = tmp_path / "L21_V001.json"
    path.write_text(
        json.dumps(
            {
                "video": {
                    "video_id": "L21_V001",
                    "path": "x.mp4",
                    "width": 320,
                    "height": 180,
                    "fps": 25,
                    "frame_count": 42,
                    "duration_seconds": 1.68,
                },
                "shots": [
                    {"shot_id": 1, "start_frame": 0, "end_frame": 19},
                    {"shot_id": 2, "start_frame": 22, "end_frame": 41},
                ],
            }
        ),
        encoding="utf-8",
    )
    config = replace(
        PipelineConfig(), target_embedding_rows=7, max_candidate_rows=7, max_vector_gib=1
    )
    plan = make_budget_plan([path], config)
    assert plan["status"] == "pass"
    assert plan["locator_rows"] == 4
    assert plan["source_extra_rows"] == 3
    assert plan["assigned_rows"] == 7


def test_reconciled_budget_accepts_quality_underfill_but_keeps_locators(tmp_path) -> None:
    path = tmp_path / "L21_V001.json"
    path.write_text(
        json.dumps(
            {
                "video": {
                    "video_id": "L21_V001",
                    "path": "x.mp4",
                    "width": 320,
                    "height": 180,
                    "fps": 25,
                    "frame_count": 42,
                    "duration_seconds": 1.68,
                },
                "shots": [
                    {"shot_id": 1, "start_frame": 0, "end_frame": 19},
                    {"shot_id": 2, "start_frame": 22, "end_frame": 41},
                ],
            }
        ),
        encoding="utf-8",
    )
    config = replace(
        PipelineConfig(), target_embedding_rows=7, max_candidate_rows=7, max_vector_gib=1
    )
    static = make_budget_plan([path], config)
    realized = reconcile_budget_plan(
        static,
        [{"video_id": "L21_V001", "candidate_rows": 6, "decode_missing": 0}],
    )
    assert realized["status"] == "pass"
    assert realized["requested_assigned_rows"] == 7
    assert realized["assigned_rows"] == 6
    assert realized["underfilled_rows"] == 1
    assert realized["videos"][0]["requested_rows"] == 7
    assert realized["videos"][0]["target_rows"] == 6


def test_selection_gathers_byte_identical_vectors_and_keeps_locator() -> None:
    config = replace(
        PipelineConfig(),
        sample_stride=8,
        raw_refine_radius=7,
        embedding_dim=4,
        search_dim=2,
        discovery_max_gap=8,
    )
    rows = seed_candidates(_segmentation(), config)
    for index, row in enumerate(rows):
        row.metrics = _metrics(row.frame_idx, quality=0.5 + index / 100)
    rng = np.random.default_rng(7)
    vectors = rng.normal(size=(len(rows), 4)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors.astype(np.float16)
    views = select_views(_segmentation(), rows, vectors, config)
    assert set(views.localization_indices) == {i for i, row in enumerate(rows) if row.locator_keep}
    promoted_rows, promoted = promote_vectors(rows, vectors, views.discovery_indices)
    assert [row.frame_idx for row in promoted_rows] == sorted(row.frame_idx for row in promoted_rows)
    for output_index, source_index in enumerate(views.discovery_indices):
        assert promoted[output_index].tobytes() == vectors[source_index].tobytes()


def test_vector_integrity_validation() -> None:
    config = replace(PipelineConfig(), embedding_dim=4, search_dim=2)
    rows = [Candidate("L21_V001", 0, 1, metrics=_metrics(0))]
    vectors = np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float16)
    assert validate_vector_store(rows, vectors, config)["status"] == "pass"


def test_frozen_index_files_have_the_flat_index_contract(tmp_path) -> None:
    vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float16)
    ids = np.asarray([["L21_V001", "1"], ["L21_V001", "2"]], dtype="<U24")
    atomic_save_npy(tmp_path / "emb.npy", vectors)
    atomic_save_npy(tmp_path / "ids.npy", ids)
    atomic_save_npy(tmp_path / "frame_idx.npy", np.asarray([9, 19], dtype=np.int32))
    atomic_write_json(tmp_path / "ranges.json", {"L21_V001": [0, 2]})
    loaded_vectors = np.load(tmp_path / "emb.npy").astype(np.float32)
    loaded_ids = [(str(video), int(n)) for video, n in np.load(tmp_path / "ids.npy")]
    loaded_frames = np.load(tmp_path / "frame_idx.npy").astype(np.int32)
    ranges = json.loads((tmp_path / "ranges.json").read_text())
    assert loaded_vectors.shape == (2, 2)
    assert loaded_ids == [("L21_V001", 1), ("L21_V001", 2)]
    assert int(loaded_frames[1]) == 19
    assert ranges == {"L21_V001": [0, 2]}
