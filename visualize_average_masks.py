#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def _get_image_files(root: Path) -> list[Path]:
    return [
        p
        for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]


def _build_rgb_index(rgb_root: Path) -> dict[str, Path]:
    """
    Recursively index RGB images under `rgb_root` by filename.

    If duplicate filenames exist, the first one found is kept.
    """
    index: dict[str, Path] = {}
    for path in rgb_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        name = path.name
        if name not in index:
            index[name] = path
    return index


def _find_rgb_image(mask_path: Path, rgb_index: dict[str, Path]) -> Optional[Path]:
    # First try exact filename match.
    if mask_path.name in rgb_index:
        return rgb_index[mask_path.name]

    # Then try any supported extension with the same stem.
    for ext in IMAGE_EXTS:
        candidate_name = f"{mask_path.stem}{ext}"
        if candidate_name in rgb_index:
            return rgb_index[candidate_name]
    return None


def _annotate_controls(
    img,
    frame_idx: Optional[int] = None,
    total_frames: Optional[int] = None,
) -> None:
    cv2.putText(
        img,
        "SPACE: next  Q: quit",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    if frame_idx is not None and total_frames is not None:
        h = img.shape[0]
        cv2.putText(
            img,
            f"Frame {frame_idx}/{total_frames}",
            (10, max(h - 10, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )


def _make_overlay(
    mask: np.ndarray,
    rgb: np.ndarray,
    frame_idx: Optional[int] = None,
    total_frames: Optional[int] = None,
) -> np.ndarray:
    if rgb.shape[:2] != mask.shape[:2]:
        rgb = cv2.resize(
            rgb,
            (mask.shape[1], mask.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    overlay = rgb.copy()
    mask_bool = mask > 0
    if mask_bool.any():
        overlay[mask_bool] = (
            0.4 * overlay[mask_bool]
            + 0.6 * np.array([0, 0, 255], dtype=np.float32)
        ).astype(np.uint8)
    _annotate_controls(
        overlay,
        frame_idx=frame_idx,
        total_frames=total_frames,
    )
    return overlay


def _colorize_mask(
    mask: np.ndarray,
    frame_idx: Optional[int] = None,
    total_frames: Optional[int] = None,
) -> np.ndarray:
    mask_u8 = cv2.normalize(
        mask,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_8U,
    )
    color = cv2.applyColorMap(mask_u8, cv2.COLORMAP_JET)
    _annotate_controls(
        color,
        frame_idx=frame_idx,
        total_frames=total_frames,
    )
    return color


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize a directory of average mask images, optionally overlaid on RGB frames.\n"
            "Typical usage:\n"
            "  python visualize_average_masks.py --masks-dir /data/masks/episode/average_mask\n"
            "  python visualize_average_masks.py --masks-dir /data/masks/episode/average_mask "
            "--rgb-dir /data/rgb_frames/episode"
        )
    )
    parser.add_argument(
        "--masks-dir",
        required=True,
        type=Path,
        help=(
            "Directory containing mask images (e.g. an 'average_mask' directory "
            "produced by aggregate_masks.py)."
        ),
    )
    parser.add_argument(
        "--rgb-dir",
        type=Path,
        default='/data/raw_data/with_bed/',
        help=(
            "Optional directory containing RGB images with filenames matching the masks. "
            "If provided, masks are shown as an overlay on the corresponding RGB frame."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    masks_dir: Path = args.masks_dir.expanduser().resolve()
    rgb_dir: Optional[Path] = (
        args.rgb_dir.expanduser().resolve() if args.rgb_dir is not None else None
    )
    rgb_index: Optional[dict[str, Path]] = None

    if not masks_dir.is_dir():
        raise FileNotFoundError(f"masks_dir is not a directory: {masks_dir}")

    mask_paths = sorted(_get_image_files(masks_dir))
    if not mask_paths:
        # If no masks are found directly under the given directory,
        # fall back to an "average_mask" subdirectory if it exists.
        fallback_dir = masks_dir / "average_mask"
        if fallback_dir.is_dir():
            fallback_masks = sorted(_get_image_files(fallback_dir))
            if fallback_masks:
                print(
                    f"No mask images found in {masks_dir}, "
                    f"falling back to {fallback_dir}"
                )
                masks_dir = fallback_dir
                mask_paths = fallback_masks

    if not mask_paths:
        print(f"No mask images found in {masks_dir}")
        return

    # Derive the episode directory name from the masks directory.
    # If masks_dir is an 'average_mask' subdir, the episode is its parent.
    if masks_dir.name == "average_mask":
        episode_dir = masks_dir.parent
    else:
        episode_dir = masks_dir

    rgb_root_for_index: Optional[Path] = None
    if rgb_dir is not None and not rgb_dir.is_dir():
        print(f"Warning: rgb_dir is not a directory, ignoring: {rgb_dir}")
        rgb_dir = None
    if rgb_dir is not None:
        # The user may pass a root directory that contains multiple episodes.
        # Prefer the subdirectory that matches the masks episode name.
        candidate = rgb_dir / episode_dir.name
        rgb_root_for_index = candidate if candidate.is_dir() else rgb_dir

        rgb_index = _build_rgb_index(rgb_root_for_index)
        if not rgb_index:
            print(
                f"Warning: no RGB images found under {rgb_root_for_index}; "
                "showing masks only."
            )

    rgb_frame_positions: Optional[dict[str, int]] = None
    total_frames: int
    if rgb_index:
        # When RGB images are provided, base the frame index on
        # the ordering of RGB frames, not the masks.
        sorted_rgb_names = sorted(rgb_index.keys())
        rgb_frame_positions = {
            name: i for i, name in enumerate(sorted_rgb_names, start=1)
        }
        total_frames = len(sorted_rgb_names)
    else:
        total_frames = len(mask_paths)

    window_name = "average masks"
    print(f"Found {len(mask_paths)} mask images in {masks_dir}")
    if rgb_root_for_index is not None:
        print(f"Using RGB images from: {rgb_root_for_index}")

    for idx, mask_path in enumerate(mask_paths, start=1):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        vis_img = None
        display_idx: Optional[int] = None
        rgb_path: Optional[Path] = None
        if rgb_index is not None:
            rgb_path = _find_rgb_image(mask_path, rgb_index)
            if rgb_path is not None:
                rgb_img = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
                if rgb_img is not None:
                    if rgb_frame_positions is not None:
                        display_idx = rgb_frame_positions.get(rgb_path.name)
                    vis_img = _make_overlay(
                        mask,
                        rgb_img,
                        frame_idx=display_idx,
                        total_frames=total_frames if display_idx is not None else None,
                    )

        if vis_img is None:
            # If there is no RGB context, use the mask sequence index.
            if rgb_index is None:
                display_idx = idx
            vis_img = _colorize_mask(
                mask,
                frame_idx=display_idx,
                total_frames=total_frames if display_idx is not None else None,
            )

        cv2.imshow(window_name, vis_img)
        key = cv2.waitKey(0) & 0xFF
        if key in (27, ord("q"), ord("Q")):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
