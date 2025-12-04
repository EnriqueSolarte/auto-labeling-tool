#!/usr/bin/env python3
"""Interactive SAM 2 video propagation demo.

This script mirrors propagate_seg.py but uses SAM 2's video predictor to
propagate a segmentation mask across an ordered sequence of RGB frames.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict

from utils import overlay_mask
from utils.segmentation import _apply_postprocessing_overrides
from utils.sampling_mask import uniform_mask_sampling as mask_sampling

SAM2_DEFAULT_DIR = Path("/models/sam2")
DEFAULT_SAM2_CONFIG = SAM2_DEFAULT_DIR / "configs/sam2.1/sam2.1_l.yaml"
DEFAULT_SAM2_CHECKPOINT = SAM2_DEFAULT_DIR / "sam2.1_hiera_large.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track a SAM 2 mask across frames")
    parser.add_argument(
        "--sequence-dir",
        type=Path,
        default=Path("/workspace/assets/seq_images"),
        help="Directory containing an ordered list of RGB frames.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_SAM2_CHECKPOINT,
        help="Path to the SAM 2 checkpoint (.pt).",
    )
    parser.add_argument(
        "--model-config",
        default=DEFAULT_SAM2_CONFIG,
        help="Hydra config name or YAML path for the SAM 2 model.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device for model execution (e.g., cuda:0 or cpu).",
    )
    parser.add_argument(
        "--object-id",
        type=int,
        default=1,
        help="Identifier for the tracked object (useful when extending to multi-object).",
    )
    parser.add_argument(
        "--annotation-frame",
        type=int,
        default=0,
        help="Frame index that will receive the initial clicks.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on the number of frames to propagate.",
    )
    parser.add_argument(
        "--vos-optimized",
        action="store_true",
        help="Enable SAM 2's VOS-optimized predictor variant.",
    )
    parser.add_argument(
        "--async-loading",
        action="store_true",
        help="Stream frames asynchronously while initializing the predictor.",
    )
    parser.add_argument(
        "--offload-video-to-cpu",
        action="store_true",
        help="Keep frames on CPU in the inference state to save GPU memory.",
    )
    parser.add_argument(
        "--offload-state-to-cpu",
        action="store_true",
        help="Keep cached state on CPU to reduce GPU footprint (slower).",
    )
    parser.add_argument(
        "--samples-per-mask",
        type=int,
        default=10,
        help="How many prompt points to resample from the predicted mask. "
        "Set to 0 to keep only the manually clicked points.",
    )
    parser.add_argument(
        "--sampling-noise-std",
        type=float,
        default=1.5,
        help="Std. dev. for jitter when sampling new points from the mask.",
    )
    return parser.parse_args()


def list_frame_paths(sequence_dir: Path) -> List[Path]:
    if not sequence_dir.is_dir():
        raise FileNotFoundError(f"Sequence directory not found: {sequence_dir}")
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    paths = sorted(p for p in sequence_dir.iterdir() if p.suffix.lower() in exts)
    if not paths:
        raise ValueError(f"No image files found in {sequence_dir}")
    return paths


def read_frame(path: Path) -> np.ndarray:
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Failed to read frame: {path}")
    return frame


def get_initial_xy_points(image: np.ndarray) -> List[tuple[float, float]]:
    window = "Select initial points"
    img_display = image.copy()
    selected: List[tuple[float, float]] = []

    def on_mouse(event, x, y, flags, param):  # type: ignore[unused-argument]
        if event == cv2.EVENT_LBUTTONDOWN:
            selected.append((float(x), float(y)))
            cv2.circle(img_display, (x, y), 3, (0, 0, 255), -1)
            cv2.imshow(window, img_display)

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)
    cv2.imshow(window, img_display)

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 27, 32):  # ENTER, ESC, SPACE
            break

    cv2.destroyWindow(window)
    return selected


def _build_video_predictor_from_local_config(
    config_path: Path,
    checkpoint: Path,
    device: str | None,
    vos_optimized: bool,
):
    cfg = OmegaConf.load(str(config_path))
    _apply_postprocessing_overrides(cfg)
    with open_dict(cfg):
        model_cfg = cfg.model
        if model_cfg is None:
            raise ValueError("Invalid SAM 2 config: missing 'model' section.")
        with open_dict(model_cfg):
            target = (
                "sam2.sam2_video_predictor.SAM2VideoPredictorVOS"
                if vos_optimized
                else "sam2.sam2_video_predictor.SAM2VideoPredictor"
            )
            model_cfg._target_ = target
            model_cfg.binarize_mask_from_pts_for_mem_enc = True
            model_cfg.fill_hole_area = model_cfg.get("fill_hole_area", 8)
            if vos_optimized:
                model_cfg.compile_image_encoder = True
    model = instantiate(cfg.model, _recursive_=True)
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)["model"]
    missing, unexpected = model.load_state_dict(state)
    if missing:
        raise RuntimeError(f"Missing checkpoint keys: {missing}")
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected}")
    if device is not None:
        model = model.to(device)
    model.eval()
    return model


def load_video_predictor(
    checkpoint: Path,
    model_config: str,
    device: str,
    vos_optimized: bool,
):
    from sam2.build_sam import build_sam2_video_predictor

    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"SAM 2 checkpoint not found: {checkpoint}")

    config_path = Path(model_config)
    if config_path.exists():
        return _build_video_predictor_from_local_config(
            config_path=config_path,
            checkpoint=checkpoint,
            device=device,
            vos_optimized=vos_optimized,
        )

    predictor = build_sam2_video_predictor(
        config_file=model_config,
        ckpt_path=str(checkpoint),
        device=device,
        vos_optimized=vos_optimized,
    )
    return predictor


def tensor_mask_to_numpy(mask_tensor: torch.Tensor) -> np.ndarray:
    mask_np = mask_tensor.squeeze(0).detach().cpu().numpy()
    return mask_np > 0.0


def sample_points_from_mask(
    mask_tensor: torch.Tensor,
    num_samples: int,
) -> List[tuple[float, float]]:
    if num_samples <= 0:
        return []
    mask = tensor_mask_to_numpy(mask_tensor)
    return mask_sampling(mask, num_samples)


def render_overlay(frame: np.ndarray, mask_tensor: torch.Tensor) -> np.ndarray:
    mask = tensor_mask_to_numpy(mask_tensor)
    return overlay_mask(frame, mask)


def main() -> None:
    args = parse_args()
    frame_paths = list_frame_paths(args.sequence_dir)
    annotation_idx = int(np.clip(args.annotation_frame, 0, len(frame_paths) - 1))
    print(f"Loaded {len(frame_paths)} frames from {args.sequence_dir}")
    print(f"Annotating frame {annotation_idx}")

    anno_frame = read_frame(frame_paths[annotation_idx])
    initial_points = get_initial_xy_points(anno_frame)
    if not initial_points:
        print("No points selected; exiting.")
        return

    predictor = load_video_predictor(
        checkpoint=args.checkpoint,
        model_config=args.model_config,
        device=args.device,
        vos_optimized=args.vos_optimized,
    )
    inference_state = predictor.init_state(
        video_path=str(args.sequence_dir),
        offload_video_to_cpu=args.offload_video_to_cpu,
        offload_state_to_cpu=args.offload_state_to_cpu,
        async_loading_frames=args.async_loading,
    )

    points = np.asarray(initial_points, dtype=np.float32)
    labels = np.ones(points.shape[0], dtype=np.int32)

    frame_idx, obj_ids, masks = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=annotation_idx,
        obj_id=args.object_id,
        points=points,
        labels=labels,
    )

    display_window = "SAM2 Video Tracking"
    print("Initial mask computed; starting propagation...")
    try:
        idx = obj_ids.index(args.object_id)
        overlay = render_overlay(read_frame(frame_paths[frame_idx]), masks[idx])
        cv2.imshow(display_window, overlay)
        cv2.waitKey(1)
        if args.samples_per_mask > 0:
            sampled_points = sample_points_from_mask(
                masks[idx],
                num_samples=args.samples_per_mask,
            )
            if sampled_points:
                print(
                    f"Resampled {len(sampled_points)} prompt points from frame {frame_idx}."
                )
                points = np.asarray(sampled_points, dtype=np.float32)
                labels = np.ones(points.shape[0], dtype=np.int32)
                frame_idx, obj_ids, masks = predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=annotation_idx,
                    obj_id=args.object_id,
                    points=points,
                    labels=labels,
                    clear_old_points=True,
                )
                initial_points = sampled_points
                idx = obj_ids.index(args.object_id)
                overlay = render_overlay(read_frame(frame_paths[frame_idx]), masks[idx])
                cv2.imshow(display_window, overlay)
                cv2.waitKey(1)
            else:
                print("Mask sampling returned no points; using original clicks.")
    except ValueError:
        idx = None

    for out_idx, obj_ids, mask_scores in predictor.propagate_in_video(
        inference_state,
        start_frame_idx=annotation_idx,
        max_frame_num_to_track=args.max_frames,
    ):
        if args.object_id not in obj_ids:
            continue
        obj_pos = obj_ids.index(args.object_id)
        frame = read_frame(frame_paths[out_idx])
        overlay = render_overlay(frame, mask_scores[obj_pos])
        cv2.putText(
            overlay,
            f"Frame {out_idx+1}/{len(frame_paths)}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(display_window, overlay)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
