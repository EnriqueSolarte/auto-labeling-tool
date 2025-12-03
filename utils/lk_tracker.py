#!/usr/bin/env python3
"""
Lucas–Kanade (LK) keypoint tracking utilities.

The main workflow is:

1. Load a sequence of images from a directory.
2. Pick one frame (optionally at random) and let the user click UV points.
3. Track those UV points across the sequence using LK optical flow.

The interactive helpers rely on OpenCV's high-level GUI APIs, but the core
tracking logic is pure numpy plus cv2.calcOpticalFlowPyrLK so it can be
tested with simple stubs.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np

Point = Tuple[float, float]

DEFAULT_SEQ_DIR = Path(__file__).resolve().parent.parent / "assets" / "seq_images"


def _ensure_cv2():
    """Import OpenCV lazily to avoid a hard dependency at import time."""
    try:
        import cv2  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover - exercised only when cv2 is missing
        raise ImportError(
            "OpenCV (cv2) is required for LK tracking. "
            "Install it with `pip install opencv-python`."
        ) from exc
    return cv2


def _to_gray(frame: np.ndarray) -> np.ndarray:
    """Convert an RGB or grayscale frame to a single-channel uint8 array."""
    if frame.ndim == 2:
        gray = frame
    elif frame.ndim == 3 and frame.shape[2] >= 3:
        # Use a simple luminance transform; avoid depending on cv2 here.
        rgb = frame[..., :3].astype(np.float32)
        gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    else:
        raise ValueError("frame must be a 2D or 3D array representing an image")

    if np.issubdtype(gray.dtype, np.floating):
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    else:
        gray = gray.astype(np.uint8)
    return gray


def load_image_sequence(directory: Path | str = DEFAULT_SEQ_DIR) -> List[np.ndarray]:
    """
    Load an ordered sequence of RGB frames from a directory.

    Args:
        directory: Folder containing image files. Files are sorted lexicographically
            and filtered by common image extensions.

    Returns:
        List of images as numpy arrays in BGR (OpenCV) layout.
    """
    cv2 = _ensure_cv2()

    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Image sequence directory not found: {root}")

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    files = sorted(p for p in root.iterdir() if p.suffix.lower() in exts)
    if not files:
        raise ValueError(f"No image files found in directory: {root}")

    frames: List[np.ndarray] = []
    for path in files:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read image: {path}")
        frames.append(img)

    return frames

def loop_iter_image_sequence(directory: Path | str = DEFAULT_SEQ_DIR) -> List[np.ndarray]:
    """
    Lazily iterate over an ordered sequence of RGB frames from a directory.

    Args:
        directory: Folder containing image files. Files are sorted lexicographically
            and filtered by common image extensions.

    Returns:
        An iterator of images as numpy arrays in BGR (OpenCV) layout, one per file.
    """
    cv2 = _ensure_cv2()

    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Image sequence directory not found: {root}")

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    files = sorted(p for p in root.iterdir() if p.suffix.lower() in exts)
    if not files:
        raise ValueError(f"No image files found in directory: {root}")

    for path in files:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read image: {path}")
        yield img


def track_points_lk(
    frames: Sequence[np.ndarray],
    initial_points_uv: Sequence[Point],
    lk_params: dict | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Track 2D UV points across an image sequence using LK optical flow.

    Args:
        frames: Sequence of image frames (H, W) or (H, W, C) arrays.
        initial_points_uv: Iterable of (u, v) pixel coordinates on the first frame.
        lk_params: Optional dictionary forwarded to cv2.calcOpticalFlowPyrLK.

    Returns:
        trajectories: Array of shape (T, N, 2) with UV coordinates for each point
            across all T frames (including the first).
        statuses: Boolean array of shape (T, N) where True indicates a successfully
            tracked point for that frame.
    """
    if len(frames) < 2:
        raise ValueError("At least two frames are required for LK tracking.")
    if not initial_points_uv:
        raise ValueError("At least one UV point is required for LK tracking.")

    cv2 = _ensure_cv2()
    params = dict(lk_params or {})

    prev_gray = _to_gray(np.asarray(frames[0]))
    points = np.asarray(initial_points_uv, dtype=np.float32).reshape(-1, 1, 2)

    trajectories: List[np.ndarray] = [points.reshape(-1, 2)]
    statuses: List[np.ndarray] = [np.ones(points.shape[0], dtype=bool)]

    for frame in frames[1:]:
        next_gray = _to_gray(np.asarray(frame))
        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, points, None, **params)
        if p1 is None or st is None:
            raise RuntimeError("cv2.calcOpticalFlowPyrLK failed to return trajectories.")

        pts = p1.reshape(-1, 2)
        st_bool = st.reshape(-1).astype(bool)

        trajectories.append(pts)
        statuses.append(st_bool)

        prev_gray = next_gray
        points = p1

    traj_arr = np.stack(trajectories, axis=0)
    status_arr = np.stack(statuses, axis=0)
    return traj_arr, status_arr


def track_between_frames(
    frame_prev: np.ndarray,
    frame_next: np.ndarray,
    points_uv: Sequence[Point],
    lk_params: dict | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Track 2D UV points from one frame to the next using LK optical flow.

    Args:
        frame_prev: Previous image frame (H, W) or (H, W, C) array.
        frame_next: Next image frame (same spatial shape as frame_prev).
        points_uv: Iterable of (u, v) pixel coordinates defined on frame_prev.
        lk_params: Optional dictionary forwarded to cv2.calcOpticalFlowPyrLK.

    Returns:
        next_points: Array of shape (N, 2) with UV coordinates of the points on
            frame_next.
        statuses: Boolean array of shape (N,) indicating which points were
            successfully tracked.
    """
    points_arr = np.asarray(points_uv, dtype=np.float32).reshape(-1, 2)
    if points_arr.size == 0:
        raise ValueError("At least one UV point is required for LK tracking.")

    cv2 = _ensure_cv2()
    params = dict(lk_params or {})

    prev_gray = _to_gray(np.asarray(frame_prev))
    next_gray = _to_gray(np.asarray(frame_next))

    points = points_arr.reshape(-1, 1, 2)
    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, points, None, **params)
    # forward–backward consistency
    p2, _st, _err = cv2.calcOpticalFlowPyrLK(next_gray, prev_gray, p1, None, **params)
    d = abs(p1 - p2).reshape(-1, 2).max(-1)
    mask_in = d < 1
    if st is not None:
        st = st & mask_in.reshape(-1, 1)
    if p1 is None or st is None:
        raise RuntimeError("cv2.calcOpticalFlowPyrLK failed to return trajectories.")

    next_points = p1.reshape(-1, 2)
    statuses = st.reshape(-1).astype(bool)
    return next_points, statuses


def track_points_lk_pairwise(
    frames: Sequence[np.ndarray],
    initial_points_uv: Sequence[Point],
    lk_params: dict | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Track 2D UV points across an image sequence by iteratively calling
    track_between_frames on each consecutive pair of frames.

    The interface mirrors track_points_lk but exposes the pairwise stepping
    logic explicitly for easier testing and reuse.

    Args:
        frames: Sequence of image frames (H, W) or (H, W, C) arrays.
        initial_points_uv: Iterable of (u, v) pixel coordinates on the first frame.
        lk_params: Optional dictionary forwarded to cv2.calcOpticalFlowPyrLK.

    Returns:
        trajectories: Array of shape (T, N, 2) with UV coordinates for each point
            across all T frames (including the first).
        statuses: Boolean array of shape (T, N) where True indicates a successfully
            tracked point for that frame.
    """
    if len(frames) < 2:
        raise ValueError("At least two frames are required for LK tracking.")
    if not initial_points_uv:
        raise ValueError("At least one UV point is required for LK tracking.")

    points = np.asarray(initial_points_uv, dtype=np.float32).reshape(-1, 2)
    trajectories: List[np.ndarray] = [points.copy()]
    statuses: List[np.ndarray] = [np.ones(points.shape[0], dtype=bool)]

    for prev_frame, next_frame in zip(frames[:-1], frames[1:]):
        points, st = track_between_frames(prev_frame, next_frame, points, lk_params)
        trajectories.append(points)
        statuses.append(st)

    traj_arr = np.stack(trajectories, axis=0)
    status_arr = np.stack(statuses, axis=0)
    return traj_arr, status_arr


def _interactive_select_points(image: np.ndarray) -> List[Point]:
    """
    Display an image and let the user click UV points.

    Left-click to add points, press SPACE/ENTER/ESC to finish.
    """
    cv2 = _ensure_cv2()

    window_name = "Select UV points"
    img_display = image.copy()
    selected: List[Point] = []

    def on_mouse(event, x, y, flags, param):  # type: ignore[unused-argument]
        if event == cv2.EVENT_LBUTTONDOWN:
            selected.append((float(x), float(y)))
            cv2.circle(img_display, (x, y), 3, (0, 0, 255), -1)
            cv2.imshow(window_name, img_display)

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)
    cv2.imshow(window_name, img_display)

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 27, 32):  # ENTER, ESC, SPACE
            break

    cv2.destroyWindow(window_name)
    return selected


def _visualize_trajectories_loop(
    frames: Sequence[np.ndarray],
    trajectories: np.ndarray,
    statuses: np.ndarray | None = None,
    window_name: str = "LK Tracking",
    delay_ms: int = 50,
) -> None:
    """
    Visualize tracked points over all frames in a ping-pong loop.

    Press ESC or 'q' to exit the loop.
    """
    cv2 = _ensure_cv2()

    num_frames, num_points, _ = trajectories.shape
    if statuses is not None and statuses.shape != (num_frames, num_points):
        raise ValueError("statuses must have shape (T, N) matching trajectories.")

    idx = 0
    direction = 1  # 1 = forward, -1 = backward

    while True:
        frame = np.asarray(frames[idx])
        # Ensure 3-channel BGR for visualization.
        if frame.ndim == 2:
            frame_vis = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        elif frame.ndim == 3 and frame.shape[2] >= 3:
            frame_vis = frame[..., :3].copy()
        else:
            raise ValueError("Each frame must be 2D or 3D image array.")

        pts = trajectories[idx]
        if statuses is not None:
            st = statuses[idx]
        else:
            st = np.ones(pts.shape[0], dtype=bool)

        for (u, v), ok in zip(pts, st):
            if not ok:
                continue
            x = int(round(float(u)))
            y = int(round(float(v)))
            cv2.circle(frame_vis, (x, y), 3, (0, 0, 255), -1)

        cv2.imshow(window_name, frame_vis)
        key = cv2.waitKey(delay_ms) & 0xFF
        if key in (27, ord("q")):  # ESC or 'q'
            break

        idx += direction
        if idx >= num_frames:
            if num_frames == 1:
                idx = 0
            else:
                idx = num_frames - 2
                direction = -1
        elif idx < 0:
            if num_frames == 1:
                idx = 0
            else:
                idx = 1
                direction = 1

    cv2.destroyWindow(window_name)


def run_interactive_lk_tracker(sequence_dir: Path | str = DEFAULT_SEQ_DIR) -> Tuple[np.ndarray, np.ndarray] | Tuple[None, None]:
    """
    Full interactive demo: load a sequence, collect UV clicks on the first
    frame, then track them forward using LK across all frames.

    Returns:
        (trajectories, statuses) as from track_points_lk, or (None, None) if
        the user did not select any points.
    """
    frames = load_image_sequence(sequence_dir)
    if not frames:
        raise ValueError("No frames loaded for LK tracking.")

    first_frame = frames[0]

    points = _interactive_select_points(first_frame)
    if not points:
        return None, None

    sub_sequence = frames
    trajectories, statuses = track_points_lk(sub_sequence, points)

    # Show tracked points looping forward and backward through the sequence.
    _visualize_trajectories_loop(sub_sequence, trajectories, statuses)

    return trajectories, statuses


__all__ = [
    "DEFAULT_SEQ_DIR",
    "load_image_sequence",
    "track_points_lk",
    "track_between_frames",
    "track_points_lk_pairwise",
    "run_interactive_lk_tracker",
]
