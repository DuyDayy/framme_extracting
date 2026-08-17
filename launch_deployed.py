#!/usr/bin/env python3
from __future__ import annotations

import argparse

import modal

from framme_extracting.config import PipelineConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the deployed production coordinator")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--video-start", type=int, required=True)
    parser.add_argument("--video-end", type=int, required=True)
    parser.add_argument("--target-embedding-rows", type=int, required=True)
    parser.add_argument("--max-vector-gib", type=float, required=True)
    parser.add_argument("--image-batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.video_start < 1 or args.video_end < args.video_start:
        raise ValueError("invalid inclusive video range")
    video_ids = [
        f"{args.group}_V{ordinal:03d}"
        for ordinal in range(args.video_start, args.video_end + 1)
    ]
    config = PipelineConfig(
        target_embedding_rows=args.target_embedding_rows,
        max_candidate_rows=args.target_embedding_rows,
        max_vector_gib=args.max_vector_gib,
    )
    function = modal.Function.from_name(
        "aic-framme-extracting", "run_production_remote"
    )
    call = function.spawn(
        args.run_id,
        config.as_dict(),
        video_ids,
        args.image_batch_size,
    )
    print(f"launched run_id={args.run_id} function_call_id={call.object_id}")


if __name__ == "__main__":
    main()
