"""Read-only visualization of aggregated mask stacks."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif")
PALETTE = np.array(
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


def find_aggregated_mask_paths(masks_root: Path) -> list[Path]:
    """Return sorted ``.npy`` masks stored in ``average_mask`` directories."""
    masks_root = masks_root.expanduser().resolve()
    if not masks_root.is_dir():
        raise FileNotFoundError(f"Masks directory not found: {masks_root}")

    if masks_root.name == "average_mask":
        average_dirs = [masks_root]
    else:
        # Aggregation supports both a direct layout
        # ``masks_root/average_mask`` and an episode layout
        # ``masks_root/<episode>/average_mask``.
        average_dirs = []
        direct_average_dir = masks_root / "average_mask"
        if direct_average_dir.is_dir():
            average_dirs.append(direct_average_dir)
        average_dirs.extend(
            path / "average_mask"
            for path in masks_root.iterdir()
            if path.is_dir() and path.name != "average_mask"
            and (path / "average_mask").is_dir()
        )
    return sorted(
        path
        for average_dir in average_dirs
        for path in average_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".npy"
    )


def load_mask_stack(mask_path: Path) -> np.ndarray:
    """Load an aggregated mask as a boolean stack with shape ``[n, H, W]``."""
    try:
        stack = np.load(str(mask_path))
    except Exception as exc:
        raise ValueError(f"Could not load mask {mask_path}: {exc}") from exc
    if stack.ndim == 2:
        stack = stack[None, ...]
    if stack.ndim != 3:
        raise ValueError(
            f"Invalid mask shape in {mask_path}: expected [n, H, W], got {stack.shape}"
        )
    return stack.astype(bool)


def render_mask_stack(mask_stack: np.ndarray) -> np.ndarray:
    """Render each mask slice in a stable, distinct BGR color."""
    if mask_stack.ndim != 3:
        raise ValueError("mask_stack must have shape [n, H, W]")
    color_mask = np.zeros((*mask_stack.shape[1:], 3), dtype=np.uint8)
    for index, mask in enumerate(mask_stack):
        color_mask[mask] = PALETTE[index % len(PALETTE)]
    return color_mask


def _find_rgb_image(images_dir: Path, episode_name: str, stem: str) -> np.ndarray | None:
    for root in (images_dir / episode_name, images_dir):
        for extension in IMAGE_EXTS:
            image_path = root / f"{stem}{extension}"
            if not image_path.is_file():
                continue
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is not None:
                return image
    return None


def _display_image(mask_path: Path, images_dir: Path) -> np.ndarray:
    stack = load_mask_stack(mask_path)
    color_mask = render_mask_stack(stack)
    episode_name = mask_path.parent.parent.name
    rgb_image = _find_rgb_image(images_dir, episode_name, mask_path.stem)
    if rgb_image is None:
        return color_mask
    if rgb_image.shape[:2] != color_mask.shape[:2]:
        rgb_image = cv2.resize(rgb_image, (color_mask.shape[1], color_mask.shape[0]))
    result = rgb_image.copy()
    covered = stack.any(axis=0)
    result[covered] = (
        0.4 * result[covered] + 0.6 * color_mask[covered]
    ).astype(np.uint8)
    return result


def visualize_aggregated_masks(masks_root: Path, images_dir: Path) -> None:
    """Interactively display existing aggregate masks without modifying them."""
    paths = find_aggregated_mask_paths(masks_root)
    if not paths:
        raise RuntimeError(f"No aggregated .npy masks found under: {masks_root}")

    index = 0
    window_name = "Aggregated masks"
    try:
        while True:
            mask_path = paths[index]
            try:
                image = _display_image(mask_path, images_dir)
            except ValueError as exc:
                print(f"[auto-label-tool] Skipping {mask_path}: {exc}")
                index = (index + 1) % len(paths)
                continue

            label = f"{index + 1}/{len(paths)}  {mask_path.parent.parent.name}/{mask_path.stem}"
            cv2.putText(image, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(window_name, image)
            key = cv2.waitKey(0) & 0xFF
            if key in (27, ord("q")):
                break
            if key in (ord("p"), 81):  # p or left arrow
                index = (index - 1) % len(paths)
            else:  # n, space, right arrow, or any other key advances.
                index = (index + 1) % len(paths)
    except cv2.error as exc:
        raise RuntimeError(
            "Could not open an OpenCV display window. Run visualize-mask from a graphical session."
        ) from exc
    finally:
        cv2.destroyAllWindows()
