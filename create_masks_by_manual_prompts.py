#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from propagate_sam2 import (
    DEFAULT_SAM2_CHECKPOINT as VID_DEFAULT_CKPT,
    DEFAULT_SAM2_CONFIG as VID_DEFAULT_CFG,
    get_initial_xy_points as get_points_video,
    list_frame_paths,
    load_video_predictor,
    read_frame,
    tensor_mask_to_numpy,
)
from propagate_sam2_recursive import sample_points_from_mask as sample_points_recursive
from propagate_sam2_lkt import (
    DEFAULT_SAM2_CHECKPOINT as LKT_DEFAULT_CKPT,
    DEFAULT_SAM2_CONFIG as LKT_DEFAULT_CFG,
    compute_seg_mask,
    get_initial_xy_points as get_points_lkt,
    get_sam_model,
    track_xy,
)
from utils import overlay_mask
from utils.lk_tracker import load_image_sequence


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
        # Skip saving empty masks
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mask_u8 = (mask_arr.astype(np.uint8) * 255)
    ok = cv2.imwrite(
        str(out_path),
        mask_u8,
        [cv2.IMWRITE_PNG_COMPRESSION, 9],
    )
    return bool(ok)


def _select_annotation_frame(frame_paths: list[Path], window: str = "Select frame") -> int | None:
    """
    Let the user choose the initial frame.

    - SPACE: skip current frame and show the next one.
    - ENTER: select the current frame.
    - ESC/q: cancel selection and return None.
    """
    if not frame_paths:
        return None

    idx = 0
    total = len(frame_paths)
    while True:
        frame = read_frame(frame_paths[idx])
        disp = frame.copy()
        text_top = f"Frame {idx} / {total-1}"
        text_bottom = "SPACE: next  ENTER: select  ESC/q: cancel"
        cv2.putText(
            disp,
            text_top,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            disp,
            text_bottom,
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.imshow(window, disp)
        key = cv2.waitKey(0) & 0xFF
        if key == 32:  # SPACE -> next frame
            idx = (idx + 1) % len(frame_paths)
            continue
        if key in (13,):  # ENTER -> select
            cv2.destroyWindow(window)
            return idx
        if key in (27, ord("q")):  # ESC/q -> cancel
            cv2.destroyWindow(window)
            return None


def _show_overlay(
    frame: np.ndarray,
    mask: np.ndarray,
    window: str = "Masks",
    idx: int | None = None,
    total: int | None = None,
) -> bool:
    """Show an overlay of the mask on top of the frame.

    Returns True if the user requested to quit (q/ESC), False otherwise.
    """
    overlay = overlay_mask(frame, mask)
    disp = overlay.copy()
    if idx is not None:
        label = f"Frame {idx}" if total is None else f"Frame {idx} / {total-1}"
        cv2.putText(
            disp,
            label,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    info = "q/ESC: quit"
    cv2.putText(
        disp,
        info,
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.imshow(window, disp)
    key = cv2.waitKey(1) & 0xFF
    return key in (27, ord("q"))


def run_sam2_prop(sequence_dir: Path, out_dir: Path) -> None:
    frame_paths = list_frame_paths(sequence_dir)
    if not frame_paths:
        print(f"[sam2_prop] No frames found in {sequence_dir}")
        return

    print(f"[sam2_prop] Found {len(frame_paths)} frames in {sequence_dir}")
    print(f"[sam2_prop] Output directory: {out_dir}")

    annotation_idx_opt = _select_annotation_frame(frame_paths, "Select initial frame (sam2_prop)")
    if annotation_idx_opt is None:
        print("[sam2_prop] No frame selected; aborting.")
        return
    annotation_idx = int(annotation_idx_opt)
    print(f"[sam2_prop] Selected frame index: {annotation_idx}")

    anno_frame = read_frame(frame_paths[annotation_idx])
    points = get_points_video(anno_frame)
    if not points:
        print("[sam2_prop] No points selected; aborting.")
        return

    print(f"[sam2_prop] Collected {len(points)} initial points on frame {annotation_idx}")

    predictor = load_video_predictor(
        checkpoint=VID_DEFAULT_CKPT,
        model_config=str(VID_DEFAULT_CFG),
        device="cuda:0",
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
            print(f"[sam2_prop] Image saved for frame {frame_idx}: {mask_path}")
        else:
            print(f"[sam2_prop] Image NOT SAVED for frame {frame_idx}: {mask_path}")
        if _show_overlay(anno_frame, mask_np, "SAM2 Masks", frame_idx, len(frame_paths)):
            print("[sam2_prop] User requested exit during initial overlay.")
            cv2.destroyAllWindows()
            return

    print("[sam2_prop] Starting propagation...")
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
            print(f"[sam2_prop] Image saved for frame {out_idx}: {mask_path}")
        else:
            print(f"[sam2_prop] Image NOT SAVED for frame {out_idx}: {mask_path}")

        frame = read_frame(frame_paths[out_idx])
        if _show_overlay(frame, mask_np, "SAM2 Masks", out_idx, len(frame_paths)):
            print(f"[sam2_prop] User requested exit at frame {out_idx}.")
            break

    cv2.destroyAllWindows()
    print("[sam2_prop] Finished.")


def run_sam2_prop_recursive(sequence_dir: Path, out_dir: Path) -> None:
    frame_paths = list_frame_paths(sequence_dir)
    if not frame_paths:
        print(f"[sam2_prop_recur] No frames found in {sequence_dir}")
        return

    print(f"[sam2_prop_recur] Found {len(frame_paths)} frames in {sequence_dir}")
    print(f"[sam2_prop_recur] Output directory: {out_dir}")

    annotation_idx_opt = _select_annotation_frame(frame_paths, "Select initial frame (sam2_prop_recur)")
    if annotation_idx_opt is None:
        print("[sam2_prop_recur] No frame selected; aborting.")
        return
    annotation_idx = int(annotation_idx_opt)
    print(f"[sam2_prop_recur] Selected frame index: {annotation_idx}")

    anno_frame = read_frame(frame_paths[annotation_idx])
    current_points = get_points_video(anno_frame)
    if not current_points:
        print("[sam2_prop_recur] No points selected; aborting.")
        return

    print(f"[sam2_prop_recur] Collected {len(current_points)} initial points on frame {annotation_idx}")

    predictor = load_video_predictor(
        checkpoint=VID_DEFAULT_CKPT,
        model_config=str(VID_DEFAULT_CFG),
        device="cuda:0",
        vos_optimized=False,
    )
    inference_state = predictor.init_state(
        video_path=str(sequence_dir),
        offload_video_to_cpu=True,
        offload_state_to_cpu=False,
        async_loading_frames=True,
    )

    object_id = 1
    samples_per_mask = 10

    points_arr = np.asarray(current_points, dtype=np.float32)
    labels = np.ones(points_arr.shape[0], dtype=np.int32)

    _, obj_ids, masks = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=annotation_idx,
        obj_id=object_id,
        points=points_arr,
        labels=labels,
    )

    if object_id in obj_ids:
        pos = obj_ids.index(object_id)
        mask_np = tensor_mask_to_numpy(masks[pos])
        mask_path = out_dir / frame_paths[annotation_idx].name
        if _save_mask(mask_np, mask_path):
            print(f"[sam2_prop_recur] Image saved for frame {annotation_idx}: {mask_path}")
        else:
            print(f"[sam2_prop_recur] Image NOT SAVED for frame {annotation_idx}: {mask_path}")
        if _show_overlay(anno_frame, mask_np, "SAM2 Recursive Masks", annotation_idx, len(frame_paths)):
            print("[sam2_prop_recur] User requested exit during initial overlay.")
            cv2.destroyAllWindows()
            return

    print("[sam2_prop_recur] Starting recursive propagation...")
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
            print(f"[sam2_prop_recur] Image saved for frame {out_idx}: {mask_path}")
        else:
            print(f"[sam2_prop_recur] Image NOT SAVED for frame {out_idx}: {mask_path}")

        frame = read_frame(frame_paths[out_idx])
        if _show_overlay(frame, mask_np, "SAM2 Recursive Masks", out_idx, len(frame_paths)):
            print(f"[sam2_prop_recur] User requested exit at frame {out_idx}.")
            break

        if samples_per_mask > 0:
            sampled_points = sample_points_recursive(
                mask_scores[pos],
                num_samples=samples_per_mask,
            )
            if sampled_points:
                current_points = sampled_points
                points_arr = np.asarray(sampled_points, dtype=np.float32)
                labels = np.ones(points_arr.shape[0], dtype=np.int32)
                predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=out_idx,
                    obj_id=object_id,
                    points=points_arr,
                    labels=labels,
                    clear_old_points=True,
                )
                predictor.propagate_in_video_preflight(inference_state)
                print(f"[sam2_prop_recur] Resampled {len(sampled_points)} points on frame {out_idx}.")

    cv2.destroyAllWindows()
    print("[sam2_prop_recur] Finished.")


def run_sam2_lkt(sequence_dir: Path, out_dir: Path) -> None:
    frame_paths = list_frame_paths(sequence_dir)
    if not frame_paths:
        print(f"[sam2_lkt] No frames found in {sequence_dir}")
        return

    annotation_idx_opt = _select_annotation_frame(frame_paths, "Select initial frame (sam2_lkt)")
    if annotation_idx_opt is None:
        print("[sam2_lkt] No frame selected; aborting.")
        return
    annotation_idx = int(annotation_idx_opt)
    print(f"[sam2_lkt] Selected frame index: {annotation_idx}")

    frames = load_image_sequence(sequence_dir)
    print(f"[sam2_lkt] Loaded {len(frames)} frames from {sequence_dir}")
    print(f"[sam2_lkt] Output directory: {out_dir}")

    sam_model = get_sam_model(
        checkpoint=LKT_DEFAULT_CKPT,
        model_config=LKT_DEFAULT_CFG,
    )

    points = get_points_lkt(frames[annotation_idx])
    if not points:
        print("[sam2_lkt] No points selected; aborting.")
        return

    mask, _ = compute_seg_mask(points, frames[annotation_idx], sam_model)
    mask_path = out_dir / frame_paths[annotation_idx].name
    if _save_mask(mask, mask_path):
        print(f"[sam2_lkt] Image saved for frame {annotation_idx}: {mask_path}")
    else:
        print(f"[sam2_lkt] Image NOT SAVED for frame {annotation_idx}: {mask_path}")
    if _show_overlay(frames[annotation_idx], mask, "SAM2 LKT Masks", annotation_idx, len(frames)):
        print("[sam2_lkt] User requested exit during initial overlay.")
        cv2.destroyAllWindows()
        return

    from utils.sampling_mask import jitter_mask_sampling as mask_sampling

    points = mask_sampling(mask, points, 10)
    prev_frame = frames[annotation_idx]

    for idx in range(annotation_idx + 1, len(frames)):
        frame = frames[idx]
        points = track_xy(points, frame, prev_frame)
        if not points:
            print(f"[sam2_lkt] Tracking lost at frame {idx}; stopping.")
            break
        prev_frame = frame

        mask, _ = compute_seg_mask(points, frame, sam_model)
        mask_path = out_dir / frame_paths[idx].name
        if _save_mask(mask, mask_path):
            print(f"[sam2_lkt] Image saved for frame {idx}: {mask_path}")
        else:
            print(f"[sam2_lkt] Image NOT SAVED for frame {idx}: {mask_path}")
        if _show_overlay(frame, mask, "SAM2 LKT Masks", idx, len(frames)):
            print(f"[sam2_lkt] User requested exit at frame {idx}.")
            break
        points = mask_sampling(mask, points, 10)

    cv2.destroyAllWindows()
    print("[sam2_lkt] Finished.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run SAM2 over an image sequence and save binary masks to "
            "/data/masks/{sequence_dir.name}/{timestamp}."
        )
    )
    parser.add_argument(
        "--sequence-dir",
        required=True,
        type=Path,
        help="Directory containing an ordered list of RGB frames.",
    )
    parser.add_argument(
        "--method",
        choices=("sam2_prop", "sam2_prop_recur", "sam2_lkt"),
        default="sam2_prop",
        help="Backend to use: plain SAM2 video, recursive SAM2, or LK+SAM2.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sequence_dir: Path = args.sequence_dir.expanduser().resolve()
    out_root = Path("/data/masks")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_root / sequence_dir.name / timestamp

    print(f"[main] Sequence dir: {sequence_dir}")
    print(f"[main] Method: {args.method}")
    print(f"[main] Masks will be saved under: {out_dir}")

    if args.method == "sam2_prop":
        run_sam2_prop(sequence_dir, out_dir)
    elif args.method == "sam2_prop_recur":
        run_sam2_prop_recursive(sequence_dir, out_dir)
    else:
        run_sam2_lkt(sequence_dir, out_dir)


if __name__ == "__main__":
    main()
