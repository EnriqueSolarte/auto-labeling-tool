#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import shutil

import cv2
import numpy as np


def _list_episode_dirs(root: Path) -> List[Path]:
    """Return all immediate subdirectories of `root` (episodes)."""
    if not root.is_dir():
        raise FileNotFoundError(f"Root directory not found: {root}")
    return [p for p in root.iterdir() if p.is_dir()]


def _list_version_dirs(episode_dir: Path) -> List[Path]:
    """Return all mask version directories under an episode, excluding `average_mask`."""
    return [
        p
        for p in episode_dir.iterdir()
        if p.is_dir() and p.name != "average_mask"
    ]


def _build_filename_index(version_dirs: Iterable[Path]) -> Dict[str, List[Path]]:
    """
    Map each filename to the list of corresponding mask files across all versions.

    This allows fast aggregation per frame by grouping paths with the same name.
    """
    index: Dict[str, List[Path]] = defaultdict(list)
    for vdir in version_dirs:
        for path in vdir.iterdir():
            if path.is_file():
                index[path.name].append(path)
    return index


def _aggregate_masks_for_episode(
    episode_dir: Path,
    debug: bool = False,
    debug_rgb_root: Optional[Path] = None,
    thr: float = 0.5,
) -> None:
    """
    Aggregate all mask versions for a single episode using mean > `thr`.

    For each frame filename, all corresponding mask images across version
    directories are loaded, averaged, thresholded at 0.5, and written as a
    binary mask into `average_mask` under the episode directory.
    """
    version_dirs = _list_version_dirs(episode_dir)
    if not version_dirs:
        return

    out_dir = episode_dir / "average_mask"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    filename_index = _build_filename_index(version_dirs)

    for filename, paths in filename_index.items():
        # Each mask file may be:
        #   - a stacked array [n, H, W] stored as .npy, or
        #   - a single 2D mask [H, W] image, which we treat as [1, H, W].
        mask_arrays: List[np.ndarray] = []
        base_shape: Optional[tuple[int, int, int]] = None

        for path in paths:
            if path.suffix.lower() == ".npy":
                try:
                    arr = np.load(str(path))
                except Exception:
                    continue
                if arr.ndim == 2:
                    arr = arr[None, ...]  # [1, H, W]
                elif arr.ndim != 3:
                    # Unsupported shape; skip this file.
                    continue
                masks = arr > 0
            else:
                img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                masks = (img > 0)[None, ...]  # [1, H, W]

            if base_shape is None:
                base_shape = masks.shape  # (n, H, W)
            else:
                if masks.shape != base_shape:
                    # Mismatched shapes across versions; skip this file.
                    continue
            mask_arrays.append(masks.astype(np.float32))

        if not mask_arrays or base_shape is None:
            continue

        # stack: [V, n, H, W] where V is number of versions.
        stack = np.stack(mask_arrays, axis=0)
        mean_stack = stack.mean(axis=0)  # [n, H, W]
        binary_stack = mean_stack > thr  # bool [n, H, W]

        name = Path(filename)
        stem = name.stem

        # Save stacked per-object mask as .npy with shape [n, H, W].
        out_npy = out_dir / f"{stem}.npy"
        np.save(str(out_npy), binary_stack)

        # Also save a union mask over all objects as PNG for visualization.
        union_mask = binary_stack.any(axis=0).astype(np.uint8) * 255  # [H, W]
        out_png = out_dir / f"{stem}.png"
        cv2.imwrite(
            str(out_png),
            union_mask,
            [cv2.IMWRITE_PNG_COMPRESSION, 9],
        )

        if debug:
            try:
                # Precompute a simple color mask for per-object visualization.
                palette = np.array(
                    [
                        [0, 255, 0],
                        [255, 0, 0],
                        [0, 200, 255],
                        [255, 165, 0],
                        [255, 0, 255],
                        [0, 255, 255],
                    ],
                    dtype=np.uint8,
                )
                color_mask = np.zeros((*union_mask.shape, 3), dtype=np.uint8)
                for idx, obj_mask in enumerate(binary_stack):
                    if obj_mask.any():
                        color_mask[obj_mask] = palette[idx % len(palette)]

                if debug_rgb_root is not None:
                    # Try to find the corresponding RGB image either under
                    # RGB_ROOT/<episode>/<stem>.png or RGB_ROOT/<filename>.
                    rgb_paths = [
                        debug_rgb_root / episode_dir.name / f"{stem}.png",
                        debug_rgb_root / episode_dir.name / f"{stem}.jpg",
                        debug_rgb_root / episode_dir.name / filename,
                        debug_rgb_root / f"{stem}.png",
                        debug_rgb_root / f"{stem}.jpg",
                        debug_rgb_root / filename,
                    ]
                    rgb_img = None
                    for rgb_path in rgb_paths:
                        if rgb_path.is_file():
                            rgb_img = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
                        if rgb_img is not None:
                            break

                    if rgb_img is not None:
                        if rgb_img.shape[:2] != union_mask.shape[:2]:
                            rgb_img = cv2.resize(
                                rgb_img,
                                (union_mask.shape[1], union_mask.shape[0]),
                                interpolation=cv2.INTER_LINEAR,
                            )

                        overlay = rgb_img.copy()
                        mask_bool = union_mask > 0
                        overlay[mask_bool] = (
                            0.4 * overlay[mask_bool]
                            + 0.6 * color_mask[mask_bool].astype(np.float32)
                        ).astype(np.uint8)

                        cv2.imshow("Aggregated mask overlay", overlay)
                    else:
                        # Proceed without RGB; show per-object colors only.
                        cv2.imshow("Aggregated mask", color_mask)
                else:
                    cv2.imshow("Aggregated mask", color_mask if color_mask.any() else union_mask)

                # Short delay to allow the window to refresh; close with standard UI.
                cv2.waitKey(1)
            except Exception:
                # If visualization is not supported (e.g., headless), ignore errors.
                debug = False


def aggregate_all_episodes(
    masks_dir: Path,
    debug: bool = False,
    debug_rgb_root: Optional[Path] = None,
    thr: float = 0.5,
) -> None:
    """
    Aggregate masks for all episodes under `masks_dir`.

    `masks_dir` is expected to contain episode directories such as:

        /data/masks/
        └── rosbag2_2025_11_28-11_43_17
            ├── 20251205_115246    # masks v0
            ├── 20251205_115711    # masks v1
            └── average_mask       # output (created by this script)
    """
    # If the user passes a single episode directory instead of the root that
    # contains episodes, handle it transparently.
    if any(p.is_dir() for p in masks_dir.iterdir()):
        # If masks_dir itself looks like an episode (has subdirs that look like
        # versions), treat it as a single episode.
        if _list_version_dirs(masks_dir):
            _aggregate_masks_for_episode(
                masks_dir,
                debug=debug,
                debug_rgb_root=debug_rgb_root,
                thr=thr,
            )
            return

    for episode_dir in _list_episode_dirs(masks_dir):
        _aggregate_masks_for_episode(
            episode_dir,
            debug=debug,
            debug_rgb_root=debug_rgb_root,
            thr=thr,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate multiple mask versions per frame using mean > THR.\n"
            "Expects a directory structure like /data/masks/<episode>/<version>/frame.png "
            "and writes aggregated masks into /data/masks/<episode>/average_mask/.\n"
            "Example:\n"
            "  python aggregate_masks.py --masks-dir /data/masks"
        )
    )
    parser.add_argument(
        "--masks-dir",
        type=Path,
        default=Path("/data/masks"),
        help=(
            "Directory containing episode subdirectories with masks, e.g. "
            "/data/masks. You may also pass a single episode directory."
        ),
    )
    parser.add_argument(
        "--debug",
        nargs="?",
        const="",
        metavar="RGB_DIR",
        help=(
            "Visualize aggregated masks as they are generated. Optionally pass "
            "RGB_DIR to look up corresponding RGB frames and show a mask overlay. "
            "RGB_DIR is expected to contain either <episode>/<frame>.png or "
            "directly <frame>.png."
        ),
    )
    parser.add_argument(
        "--thr",
        type=float,
        default=0.5,
        help="Threshold in [0, 1] on the mean mask used to binarize; default 0.5.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    debug = args.debug is not None
    debug_rgb_root: Optional[Path] = None
    if isinstance(args.debug, str) and args.debug != "":
        debug_rgb_root = Path(args.debug)

    aggregate_all_episodes(
        args.masks_dir,
        debug=debug,
        debug_rgb_root=debug_rgb_root,
        thr=args.thr,
    )


if __name__ == "__main__":
    main()
