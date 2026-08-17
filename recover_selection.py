#!/usr/bin/env python3
from __future__ import annotations

import argparse

import modal


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run selection only for existing vector checkpoints"
    )
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    function = modal.Function.from_name(
        "aic-framme-extracting", "recover_selection_remote"
    )
    call = function.spawn(args.run_id)
    print(
        f"launched selection-only run_id={args.run_id} "
        f"function_call_id={call.object_id}"
    )


if __name__ == "__main__":
    main()
