#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import tempfile
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
from utils.lk_tracker import load_image_sequence


def _save_mask(mask: np.ndarray, out_path: Path) -> bool:
    """
    Save a binary mask stack as .npy with shape [n, H, W].

    Returns:
        True if a file was written, False otherwise.
    """
    if mask is None:
        return False
    mask_arr = np.asarray(mask, dtype=bool)
    if mask_arr.ndim == 2:
        mask_arr = mask_arr[None, ...]
    if mask_arr.ndim != 3 or mask_arr.size == 0:
        return False

    # Remove empty slices to avoid writing placeholder objects.
    keep = mask_arr.reshape(mask_arr.shape[0], -1).any(axis=1)
    mask_arr = mask_arr[keep]
    if mask_arr.size == 0:
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), mask_arr)
    return True


def _select_annotation_frame(
    frame_paths: list[Path],
    window: str = "Select frame",
    init_idx: int = 0,
) -> int | None:
    """
    Let the user choose the initial frame.

    - SPACE: skip current frame and show the next one.
    - ENTER: select the current frame.
    - ESC/q: cancel selection and return None.
    """
    if not frame_paths:
        return None

    init_idx = max(0, min(int(init_idx), len(frame_paths) - 1))
    idx = init_idx
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
    frame_rgb = frame.copy()
    palette = np.array(
        [
            [0, 255, 0],
            [255, 0, 0],
            [0, 200, 255],
            [255, 165, 0],
            [255, 0, 255],
            [0, 255, 255],
        ],
        dtype=np.float32,
    )

    if mask.ndim == 3:
        color_mask = np.zeros((*mask.shape[1:], 3), dtype=np.float32)
        for i, obj_mask in enumerate(mask):
            if obj_mask.any():
                color_mask[obj_mask] = palette[i % len(palette)]
        mask_union = mask.any(axis=0)
    else:
        mask_union = mask.astype(bool)
        color_mask = np.zeros((*mask_union.shape, 3), dtype=np.float32)
        color_mask[mask_union] = palette[0]

    disp = frame_rgb.astype(np.float32)
    disp[mask_union] = (
        0.6 * color_mask[mask_union] + 0.4 * disp[mask_union]
    )
    disp = disp.astype(np.uint8)
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


def _prepare_sam2_video_dir(
    sequence_dir: Path,
    frame_paths: list[Path],
    out_dir: Path,
) -> tuple[Path, Path | None]:
    if not frame_paths:
        return sequence_dir, None
    if all(path.stem.isdigit() for path in frame_paths):
        return sequence_dir, None

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="sam2_numeric_frames_", dir=out_dir))
    width = max(4, len(str(len(frame_paths) - 1)))
    for idx, src in enumerate(frame_paths):
        dst = tmp_dir / f"{idx:0{width}d}{src.suffix.lower()}"
        try:
            dst.symlink_to(src.resolve())
        except OSError:
            shutil.copy2(src, dst)

    print(f"[sam2] Normalized frame names under {tmp_dir}")
    return tmp_dir, tmp_dir


def _collect_multi_object_points(
    anno_frame: np.ndarray,
) -> list[list[tuple[float, float]]]:
    """
    Let the user select click prompts for one or more objects on the same frame.

    Each round opens the usual point-selection UI; when finished with an object,
    the user presses ENTER/ESC/SPACE. After that, the user is asked in the
    terminal whether to add another object.
    """
    all_points: list[list[tuple[float, float]]] = []
    obj_idx = 1
    while True:
        print(f"[sam2_prop] Selecting points for object {obj_idx}.")
        print("[sam2_prop] Left-click to add points; press ENTER/ESC/SPACE when done.")
        points = get_points_video(anno_frame)
        if not points:
            if not all_points:
                print("[sam2_prop] No points selected.")
                try:
                    rsp = input("[sam2_prop] Retry point selection? [y/N]: ").strip().lower()
                except EOFError:
                    rsp = "n"
                if rsp in ("y", "yes"):
                    continue
                print("[sam2_prop] Aborting.")
            else:
                print(
                    "[sam2_prop] No points for this object; "
                    "finishing object selection."
                )
            break
        all_points.append(points)
        print(
            f"[sam2_prop] Collected {len(points)} points for object {obj_idx} "
            f"on annotation frame."
        )
        try:
            rsp = input(
                "[sam2_prop] Add another object? [y/N]: "
            ).strip().lower()
        except EOFError:
            rsp = "n"
        if rsp not in ("y", "yes"):
            break
        obj_idx += 1
    return all_points


def run_sam2_prop(sequence_dir: Path, out_dir: Path, init_idx: int = 0) -> None:
    frame_paths = list_frame_paths(sequence_dir)
    if not frame_paths:
        print(f"[sam2_prop] No frames found in {sequence_dir}")
        return

    print(f"[sam2_prop] Found {len(frame_paths)} frames in {sequence_dir}")
    print(f"[sam2_prop] Output directory: {out_dir}")

    annotation_idx_opt = _select_annotation_frame(
        frame_paths,
        "Select initial frame (sam2_prop)",
        init_idx=init_idx,
    )
    if annotation_idx_opt is None:
        print("[sam2_prop] No frame selected; aborting.")
        return
    annotation_idx = int(annotation_idx_opt)
    print(f"[sam2_prop] Selected frame index: {annotation_idx}")

    anno_frame = read_frame(frame_paths[annotation_idx])
    all_points = _collect_multi_object_points(anno_frame)
    if not all_points:
        print("[sam2_prop] No objects were defined; aborting.")
        return

    num_objects = len(all_points)
    print(
        f"[sam2_prop] Collected prompts for {num_objects} "
        f"object(s) on frame {annotation_idx}"
    )

    sam2_video_dir, cleanup_dir = _prepare_sam2_video_dir(
        sequence_dir,
        frame_paths,
        out_dir,
    )
    try:
        predictor = load_video_predictor(
            checkpoint=VID_DEFAULT_CKPT,
            model_config=str(VID_DEFAULT_CFG),
            device="cuda:0",
            vos_optimized=False,
        )
        inference_state = predictor.init_state(
            video_path=str(sam2_video_dir),
            offload_video_to_cpu=True,
            offload_state_to_cpu=False,
            async_loading_frames=True,
        )

        tracked_object_ids: list[int] = []
        initial_masks: dict[int, np.ndarray] = {}

        for idx, points in enumerate(all_points, start=1):
            obj_id = idx
            points_arr = np.asarray(points, dtype=np.float32)
            labels = np.ones(points_arr.shape[0], dtype=np.int32)

            frame_idx, obj_ids, masks = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=annotation_idx,
                obj_id=obj_id,
                points=points_arr,
                labels=labels,
            )

            if obj_id not in obj_ids:
                continue

            tracked_object_ids.append(obj_id)
            pos = obj_ids.index(obj_id)
            mask_np = tensor_mask_to_numpy(masks[pos])
            initial_masks[obj_id] = mask_np

        if not tracked_object_ids:
            print("[sam2_prop] No valid objects were initialized; aborting.")
            cv2.destroyAllWindows()
            return

        # Determine the common mask shape and build stacked masks [n, H, W]
        first_mask = next(iter(initial_masks.values()))
        mask_shape = first_mask.shape
        initial_stack: list[np.ndarray] = []
        combined_initial_mask = np.zeros(mask_shape, dtype=bool)
        for obj_id in tracked_object_ids:
            mask_np = initial_masks.get(obj_id)
            if mask_np is None:
                mask_np = np.zeros(mask_shape, dtype=bool)
            combined_initial_mask |= mask_np
            initial_stack.append(mask_np.astype(bool))

        stacked_initial = np.stack(initial_stack, axis=0)
        out_dir.mkdir(parents=True, exist_ok=True)
        init_out_path = out_dir / f"{frame_paths[annotation_idx].stem}.npy"
        np.save(str(init_out_path), stacked_initial)
        print(
            f"[sam2_prop] Saved stacked mask for frame {annotation_idx} "
            f"with shape {stacked_initial.shape} to {init_out_path}"
        )

        if combined_initial_mask is not None:
            if _show_overlay(
                anno_frame,
                stacked_initial,
                "SAM2 Masks",
                annotation_idx,
                len(frame_paths),
            ):
                print("[sam2_prop] User requested exit during initial overlay.")
                cv2.destroyAllWindows()
                return

        print("[sam2_prop] Starting propagation...")
        for out_idx, obj_ids, mask_scores in predictor.propagate_in_video(
            inference_state,
            start_frame_idx=annotation_idx,
            max_frame_num_to_track=None,
        ):
            combined_mask = np.zeros(mask_shape, dtype=bool)
            frame_masks: list[np.ndarray] = []
            for obj_id in tracked_object_ids:
                if obj_id in obj_ids:
                    pos = obj_ids.index(obj_id)
                    mask_np = tensor_mask_to_numpy(mask_scores[pos])
                else:
                    mask_np = np.zeros(mask_shape, dtype=bool)

                frame_masks.append(mask_np.astype(bool))
                combined_mask |= mask_np

            if frame_masks:
                stacked = np.stack(frame_masks, axis=0)
                out_path = out_dir / f"{frame_paths[out_idx].stem}.npy"
                np.save(str(out_path), stacked)
                print(
                    f"[sam2_prop] Saved stacked mask for frame {out_idx} "
                    f"with shape {stacked.shape} to {out_path}"
                )

            frame = read_frame(frame_paths[out_idx])
            if _show_overlay(
                frame,
                stacked if frame_masks else combined_mask,
                "SAM2 Masks",
                out_idx,
                len(frame_paths),
            ):
                print(f"[sam2_prop] User requested exit at frame {out_idx}.")
                break

        cv2.destroyAllWindows()
        print("[sam2_prop] Finished.")
    finally:
        if cleanup_dir is not None:
            shutil.rmtree(cleanup_dir, ignore_errors=True)


def run_sam2_prop_recursive(sequence_dir: Path, out_dir: Path, init_idx: int = 0) -> None:
    frame_paths = list_frame_paths(sequence_dir)
    if not frame_paths:
        print(f"[sam2_prop_recur] No frames found in {sequence_dir}")
        return

    print(f"[sam2_prop_recur] Found {len(frame_paths)} frames in {sequence_dir}")
    print(f"[sam2_prop_recur] Output directory: {out_dir}")

    annotation_idx_opt = _select_annotation_frame(
        frame_paths,
        "Select initial frame (sam2_prop_recur)",
        init_idx=init_idx,
    )
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

    sam2_video_dir, cleanup_dir = _prepare_sam2_video_dir(
        sequence_dir,
        frame_paths,
        out_dir,
    )
    try:
        predictor = load_video_predictor(
            checkpoint=VID_DEFAULT_CKPT,
            model_config=str(VID_DEFAULT_CFG),
            device="cuda:0",
            vos_optimized=False,
        )
        inference_state = predictor.init_state(
            video_path=str(sam2_video_dir),
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
            mask_path = out_dir / f"{frame_paths[annotation_idx].stem}.npy"
            if _save_mask(mask_np, mask_path):
                print(f"[sam2_prop_recur] Mask stack saved for frame {annotation_idx}: {mask_path}")
            else:
                print(f"[sam2_prop_recur] Mask stack NOT SAVED for frame {annotation_idx}: {mask_path}")
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
            mask_path = out_dir / f"{frame_paths[out_idx].stem}.npy"
            if _save_mask(mask_np, mask_path):
                print(f"[sam2_prop_recur] Mask stack saved for frame {out_idx}: {mask_path}")
            else:
                print(f"[sam2_prop_recur] Mask stack NOT SAVED for frame {out_idx}: {mask_path}")

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
    finally:
        if cleanup_dir is not None:
            shutil.rmtree(cleanup_dir, ignore_errors=True)


def run_sam2_lkt(sequence_dir: Path, out_dir: Path, init_idx: int = 0) -> None:
    frame_paths = list_frame_paths(sequence_dir)
    if not frame_paths:
        print(f"[sam2_lkt] No frames found in {sequence_dir}")
        return

    annotation_idx_opt = _select_annotation_frame(
        frame_paths,
        "Select initial frame (sam2_lkt)",
        init_idx=init_idx,
    )
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
    mask_path = out_dir / f"{frame_paths[annotation_idx].stem}.npy"
    if _save_mask(mask, mask_path):
        print(f"[sam2_lkt] Mask stack saved for frame {annotation_idx}: {mask_path}")
    else:
        print(f"[sam2_lkt] Mask stack NOT SAVED for frame {annotation_idx}: {mask_path}")
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
        mask_path = out_dir / f"{frame_paths[idx].stem}.npy"
        if _save_mask(mask, mask_path):
            print(f"[sam2_lkt] Mask stack saved for frame {idx}: {mask_path}")
        else:
            print(f"[sam2_lkt] Mask stack NOT SAVED for frame {idx}: {mask_path}")
        if _show_overlay(frame, mask, "SAM2 LKT Masks", idx, len(frames)):
            print(f"[sam2_lkt] User requested exit at frame {idx}.")
            break
        points = mask_sampling(mask, points, 10)

    cv2.destroyAllWindows()
    print("[sam2_lkt] Finished.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run SAM2 over an image sequence and save binary mask stacks (.npy, shape [n, H, W]) to "
            "<output-dir>/{sequence_dir.name}/{timestamp}.\n"
            "Example:\n"
            "  python create_masks_by_manual_prompts.py \\\n"
            "      --sequence-dir /workspace/assets/seq_images \\\n"
            "      --method sam2_prop \\\n"
            "      --output-dir /data/masks"
        )
    )
    parser.add_argument(
        "--sequence-dir",
        # required=True,
        default="/workspace/assets/seq_images",
        type=Path,
        help="Directory containing an ordered list of RGB frames.",
    )
    parser.add_argument(
        "--method",
        choices=("sam2_prop", "sam2_prop_recur", "sam2_lkt"),
        default="sam2_prop",
        help="Backend to use: plain SAM2 video, recursive SAM2, or LK+SAM2.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/masks"),
        help=(
            "Root directory where masks will be stored; "
            "masks go to <output-dir>/{sequence_dir.name}/{timestamp}."
        ),
    )
    parser.add_argument(
        "--init-idx",
        type=int,
        default=0,
        help=(
            "Initial frame index to start the frame selection UI from "
            "when choosing the first annotated frame."
        ),
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    sequence_dir: Path = args.sequence_dir.expanduser().resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = args.output_dir.expanduser().resolve()
    out_dir = out_root / sequence_dir.name / timestamp

    print(f"[main] Sequence dir: {sequence_dir}")
    print(f"[main] Method: {args.method}")
    print(f"[main] Masks will be saved under: {out_dir}")

    if args.method == "sam2_prop":
        run_sam2_prop(sequence_dir, out_dir, init_idx=args.init_idx)
    elif args.method == "sam2_prop_recur":
        run_sam2_prop_recursive(sequence_dir, out_dir, init_idx=args.init_idx)
    else:
        run_sam2_lkt(sequence_dir, out_dir, init_idx=args.init_idx)


if __name__ == "__main__":
    main()
