#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import re
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from utils.segmentation import (
    get_segmentation,
    load_sam_model,
    overlay_mask,
)

SAM2_DEFAULT_DIR = Path("/models/sam2")
DEFAULT_SAM2_CONFIG = SAM2_DEFAULT_DIR / "configs/sam2.1/sam2.1_hiera_l.yaml"
DEFAULT_SAM2_CHECKPOINT = SAM2_DEFAULT_DIR / "sam2.1_hiera_large.pt"

BBox = Tuple[float, float, float, float]


def aggregate_mask(masks: List[np.ndarray], bbox: BBox) -> np.ndarray:
    if not masks:
        raise ValueError("masks must contain at least one mask.")
    height, width = masks[0].shape
    x_min, y_min, x_max, y_max = bbox

    x0 = max(0, int(np.floor(x_min)))
    y0 = max(0, int(np.floor(y_min)))
    x1 = min(width, int(np.ceil(x_max)))
    y1 = min(height, int(np.ceil(y_max)))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Invalid bounding box for mask aggregation.")
    # Build a stack restricted to the ROI for efficiency.
    roi_stack = np.stack([m[y0:y1, x0:x1] for m in masks])  # (N, roi_h, roi_w)

    # Compute per-mask ROI size (area) to use as weights.
    areas = roi_stack.reshape(roi_stack.shape[0], -1).sum(axis=1).astype(np.float32)
    total_area = float(areas.sum())

    if total_area > 0.0:
        weights = areas / total_area  # (N,)
        weighted_roi = (roi_stack.astype(np.float32) * weights[:, None, None]).sum(
            axis=0
        )
        roi_mean = weighted_roi > 0.5
    else:
        # Fallback to unweighted majority if all masks are empty.
        roi_mean = roi_stack.mean(axis=0) > 0.5

    mean_mask = np.zeros((height, width), dtype=bool)
    mean_mask[y0:y1, x0:x1] = roi_mean
    return mean_mask


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a SAM2 mask from a randomly selected self-label bounding box."
        )
    )
    parser.add_argument(
        "--bbx-dir",
        type=Path,
        # required=True,
        default='/data/self_labeling/with_bed/silver-sweep-21__2025.12.5/self_labels/rosbag2_2025_11_28-14_53_38',
        help="Directory containing bounding box files (one file per image).",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        # required=True,
        default='/data/with_bed/rosbag2_2025_11_28-14_53_38',
        help="Directory containing the corresponding RGB images.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_SAM2_CHECKPOINT,
        help="Path to the SAM2 checkpoint (.pt).",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device for SAM2 (e.g. cuda:0 or cpu).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=2,
        help="Number of random points to sample inside the selected bounding box.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducibility.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of times to run SAM for averaging masks.",
    )
    return parser.parse_args()


def _list_bbx_files(bbx_dir: Path) -> List[Path]:
    if not bbx_dir.is_dir():
        raise FileNotFoundError(f"Bounding box directory not found: {bbx_dir}")
    files = [p for p in bbx_dir.iterdir() if p.is_file()]
    if not files:
        raise ValueError(f"No bounding box files found in {bbx_dir}")
    return files


def _pick_random_bbx_file(files: List[Path], seed: int | None = None) -> Path:
    if seed is not None:
        random.seed(seed)
    return random.choice(files)


def _parse_bbox_file(path: Path, image_size: Tuple[int, int] | None = None) -> BBox:
    """
    Parse a bounding box file.

    Supported formats (first non-empty line):

    1) Explicit corners:
           x_min y_min x_max y_max

    2) YOLO-like row:
           class_id cx cy w h [score]
       where (cx, cy, w, h) are either:
         - normalized to [0, 1] and `image_size` must be provided, or
         - absolute pixel center/width/height.

    Values can be separated by spaces or commas.
    """
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = re.split(r"[,\s]+", line)
            # Keep only numeric tokens
            nums: List[float] = []
            for p in parts:
                try:
                    nums.append(float(p))
                except ValueError:
                    continue
            if len(nums) < 4:
                continue

            # Case 1: assume corners (x_min, y_min, x_max, y_max)
            x_min, y_min, x_max, y_max = nums[0], nums[1], nums[2], nums[3]
            if x_max > x_min and y_max > y_min:
                return x_min, y_min, x_max, y_max

            # Case 2: YOLO-style (class_id, cx, cy, w, h, [score])
            if len(nums) >= 5:
                _, cx, cy, w, h = nums[:5]
                if image_size is None:
                    raise ValueError(
                        f"Bounding box in {path} looks like center/width/height format "
                        "but image_size was not provided."
                    )
                img_w, img_h = image_size

                # Detect whether coords are normalized or absolute.
                if 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0:
                    scale_x, scale_y = img_w, img_h
                else:
                    scale_x = scale_y = 1.0

                x_min = (cx - w / 2.0) * scale_x
                y_min = (cy - h / 2.0) * scale_y
                x_max = (cx + w / 2.0) * scale_x
                y_max = (cy + h / 2.0) * scale_y

                # Clip to image bounds.
                x_min = max(0.0, min(x_min, img_w - 1.0))
                x_max = max(0.0, min(x_max, img_w - 1.0))
                y_min = max(0.0, min(y_min, img_h - 1.0))
                y_max = max(0.0, min(y_max, img_h - 1.0))

                if x_max > x_min and y_max > y_min:
                    return x_min, y_min, x_max, y_max

    raise ValueError(f"Could not parse a valid bounding box from {path}")


def _find_image_for_bbox(bbx_file: Path, image_dir: Path) -> Path:
    stem = bbx_file.stem
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    for ext in exts:
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No image found for bounding box file {bbx_file.name} in {image_dir}"
    )


def _sample_points_in_bbox(bbox: BBox, num_samples: int, rng: np.random.Generator) -> List[Tuple[float, float]]:
    x_min, y_min, x_max, y_max = bbox
    if num_samples <= 0:
        return []
    xs = rng.uniform(x_min, x_max, size=num_samples)
    ys = rng.uniform(y_min, y_max, size=num_samples)
    return list(zip(xs.tolist(), ys.tolist()))


def main() -> None:
    args = _parse_args()
    bbx_dir: Path = args.bbx_dir.expanduser().resolve()
    image_dir: Path = args.image_dir.expanduser().resolve()

    bbx_files = _list_bbx_files(bbx_dir)
    rng = np.random.default_rng(args.seed)
    predictor = load_sam_model(
        checkpoint_path=args.checkpoint,
        model_config=None,
        device=args.device,
    )

    while True:
        bbx_file = _pick_random_bbx_file(bbx_files, seed=args.seed)
        image_path = _find_image_for_bbox(bbx_file, image_dir)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image for bounding box: {image_path}")
        height, width = image.shape[:2]
        bbox = _parse_bbox_file(bbx_file, image_size=(width, height))

        list_masks = []
        for _ in range(args.iterations):
            points = _sample_points_in_bbox(bbox, args.num_samples, rng)
            if not points:
                raise RuntimeError("No points were sampled inside the bounding box.")

            mask, _ = get_segmentation(
                image=image_path,
                keypoints_uv=points,
                predictor=predictor,
                multimask_output=False,
            )
            list_masks.append(mask)
        mask = aggregate_mask(list_masks, bbox)

        overlay_rgb = overlay_mask(image_path, mask)
        overlay = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)

        x_min, y_min, x_max, y_max = bbox
        pt1 = (int(round(x_min)), int(round(y_min)))
        pt2 = (int(round(x_max)), int(round(y_max)))
        cv2.rectangle(overlay, pt1, pt2, (0, 0, 255), 2)
        for u, v in points:
            cv2.circle(
                overlay,
                (int(round(u)), int(round(v))),
                3,
                (0, 255, 0),
                -1,
            )
        cv2.imshow("SAM2 mask from self-label", overlay)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()

        print(key)
        if key in (ord("t"), ord("q")):
            break
        if key == 32:  # Spacebar
            continue


if __name__ == "__main__":
    main()
