#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import re
from pathlib import Path
from typing import List, Tuple
from datetime import datetime

import cv2
import numpy as np

from utils.segmentation import (
    get_segmentation,
    load_sam_model,
    overlay_mask,
)
from utils.sampling_mask import uniform_mask_sampling as mask_sampling
from propagate_sam2 import (
    list_frame_paths,
    load_video_predictor,
    read_frame,
    tensor_mask_to_numpy,
)

SAM2_DEFAULT_DIR = Path("/models/sam2")
DEFAULT_SAM2_CONFIG = SAM2_DEFAULT_DIR / "configs/sam2.1/sam2.1_hiera_l.yaml"
DEFAULT_SAM2_CHECKPOINT = SAM2_DEFAULT_DIR / "sam2.1_hiera_large.pt"

BBox = Tuple[float, float, float, float]


def _save_mask(mask: np.ndarray, out_path: Path) -> bool:
    """
    Save a binary mask image.

    Returns:
        True if an image file was written, False otherwise.
    """
    if mask is None:
        return False
    mask_arr = np.asarray(mask)
    if mask_arr.size == 0 or not mask_arr.any():
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mask_u8 = (mask_arr.astype(np.uint8) * 255)
    ok = cv2.imwrite(
        str(out_path),
        mask_u8,
        [cv2.IMWRITE_PNG_COMPRESSION, 9],
    )
    return bool(ok)


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


def _propagate_mask_from_image(
    sequence_dir: Path,
    annotation_image: Path,
    initial_mask: np.ndarray,
    device: str,
    checkpoint: Path,
) -> None:
    """
    Use the SAM2 video predictor to propagate `initial_mask` from `annotation_image`
    across the full image sequence in `sequence_dir`, without manual prompts.
    """
    try:
        frame_paths = list_frame_paths(sequence_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"[propagate_from_self_labels] Failed to list frames: {exc}")
        return

    try:
        annotation_idx = next(
            i for i, p in enumerate(frame_paths) if p.name == annotation_image.name
        )
    except StopIteration:
        print(
            f"[propagate_from_self_labels] Annotation image {annotation_image.name} "
            f"not found in sequence {sequence_dir}"
        )
        return

    out_root = Path("/data/masks")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_root / sequence_dir.name / timestamp

    print(f"[propagate_from_self_labels] Sequence dir: {sequence_dir}")
    print(f"[propagate_from_self_labels] Annotation frame index: {annotation_idx}")
    print(f"[propagate_from_self_labels] Masks will be saved under: {out_dir}")

    points = mask_sampling(initial_mask, num_samples=10)
    if not points:
        print("[propagate_from_self_labels] No points sampled from mask; aborting.")
        return

    predictor = load_video_predictor(
        checkpoint=checkpoint,
        model_config=str(DEFAULT_SAM2_CONFIG),
        device=device,
        vos_optimized=False,
    )
    inference_state = predictor.init_state(
        video_path=str(sequence_dir),
        offload_video_to_cpu=True,
        offload_state_to_cpu=False,
        async_loading_frames=True,
    )

    points_arr = np.asarray(points, dtype=np.float32)
    labels = np.ones(points_arr.shape[0], dtype=np.int32)
    object_id = 1

    frame_idx, obj_ids, masks = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=annotation_idx,
        obj_id=object_id,
        points=points_arr,
        labels=labels,
    )

    if object_id in obj_ids:
        pos = obj_ids.index(object_id)
        mask_np = tensor_mask_to_numpy(masks[pos])
        mask_path = out_dir / frame_paths[frame_idx].name
        if _save_mask(mask_np, mask_path):
            print(
                f"[propagate_from_self_labels] Saved mask for frame {frame_idx}: "
                f"{mask_path}"
            )
        else:
            print(
                f"[propagate_from_self_labels] Mask NOT SAVED for frame {frame_idx}: "
                f"{mask_path}"
            )

    print("[propagate_from_self_labels] Starting propagation...")
    for out_idx, obj_ids, mask_scores in predictor.propagate_in_video(
        inference_state,
        start_frame_idx=annotation_idx,
        max_frame_num_to_track=None,
    ):
        if object_id not in obj_ids:
            continue
        pos = obj_ids.index(object_id)
        mask_np = tensor_mask_to_numpy(mask_scores[pos])
        mask_path = out_dir / frame_paths[out_idx].name
        if _save_mask(mask_np, mask_path):
            print(
                f"[propagate_from_self_labels] Saved mask for frame {out_idx}: "
                f"{mask_path}"
            )
        else:
            print(
                f"[propagate_from_self_labels] Mask NOT SAVED for frame {out_idx}: "
                f"{mask_path}"
            )

    print("[propagate_from_self_labels] Finished.")


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

    bbx_file = _pick_random_bbx_file(bbx_files, seed=args.seed)
    image_path = _find_image_for_bbox(bbx_file, image_dir)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image for bounding box: {image_path}")
    height, width = image.shape[:2]
    bbox = _parse_bbox_file(bbx_file, image_size=(width, height))

    list_masks: List[np.ndarray] = []
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

    accepted_mask = aggregate_mask(list_masks, bbox)
    _propagate_mask_from_image(
        sequence_dir=image_dir,
        annotation_image=image_path,
        initial_mask=accepted_mask,
        device=args.device,
        checkpoint=args.checkpoint,
    )


if __name__ == "__main__":
    main()
