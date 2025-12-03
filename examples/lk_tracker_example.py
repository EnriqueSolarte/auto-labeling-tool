#!/usr/bin/env python3
"""
Interactive example script for LK keypoint tracking utilities.

Usage:
    python -m exmaples.lk_tracker_example [--sequence-dir /path/to/frames]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from utils.lk_tracker import DEFAULT_SEQ_DIR, run_interactive_lk_tracker


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive LK keypoint tracking demo.")
    parser.add_argument(
        "--sequence-dir",
        type=str,
        default=str(DEFAULT_SEQ_DIR),
        help=f"Directory containing a sequence of images (default: {DEFAULT_SEQ_DIR}).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    sequence_dir = Path(args.sequence_dir).expanduser().resolve()
    trajectories, statuses = run_interactive_lk_tracker(sequence_dir)

    if trajectories is None or statuses is None:
        print("No UV points were selected; nothing to track.")
        return 0

    print(f"Tracked {trajectories.shape[1]} points over {trajectories.shape[0]} frames.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

