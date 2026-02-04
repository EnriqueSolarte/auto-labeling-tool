#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, Sequence

import cv2


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def _list_image_files(root: Path) -> list[Path]:
    return [
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]


def _natural_key(text: str) -> list[object]:
    parts = re.split(r"(\d+)", text)
    key: list[object] = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())
    return key


def _sort_seq(paths: Iterable[Path], images_dir: Path) -> list[Path]:
    def sort_key(path: Path) -> tuple[object, ...]:
        rel = path.relative_to(images_dir)
        return (
            rel.parent.as_posix(),
            _natural_key(rel.stem),
            rel.suffix.lower(),
        )

    return sorted(paths, key=sort_key)


def _load_yolo_boxes(label_path: Path, img_shape: tuple[int, int]) -> list[tuple[int, int, int, int]]:
    h, w = img_shape
    if w <= 0 or h <= 0:
        return []

    boxes: list[tuple[int, int, int, int]] = []
    if not label_path.is_file():
        return boxes

    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                _, cx, cy, bw, bh = map(float, parts[:5])
            except ValueError:
                continue

            x_center = cx * w
            y_center = cy * h
            box_w = bw * w
            box_h = bh * h

            x1 = int(round(x_center - box_w / 2.0))
            y1 = int(round(y_center - box_h / 2.0))
            x2 = int(round(x_center + box_w / 2.0))
            y2 = int(round(y_center + box_h / 2.0))

            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))

            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))

    return boxes


def _draw_boxes(img, boxes: Sequence[tuple[int, int, int, int]], label: str) -> None:
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        img,
        label,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize a sequential YOLO dataset by sorting frames per sequence.\n"
            "Example:\n"
            "  python seq_dataset_visualization.py \\\n"
            "      --dataset-root /data/training_data/20251204_072806 \\\n"
            "      --split train \\\n"
            "      --output-video /tmp/train_vis.mp4"
        )
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        type=Path,
        help="Path to the dataset root (e.g. /data/training_data/20251204_072806).",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="train",
        help="Which split to visualize.",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction of frames to visualize from the sorted list (0 < fraction <= 1].",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional maximum number of frames to visualize (0 disables limit).",
    )
    parser.add_argument(
        "--output-video",
        type=Path,
        default=None,
        help="Optional output video path (e.g. /tmp/vis.mp4).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=20.0,
        help="FPS for output video.",
    )
    parser.add_argument(
        "--wait-ms",
        type=int,
        default=0,
        help="Delay per frame in ms for window playback (0 = wait for key).",
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Disable window visualization (useful for headless video export).",
    )
    return parser.parse_args()


def _open_writer(video_path: Path, fps: float, frame_shape: tuple[int, int]) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w = frame_shape
    return cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))


def main() -> None:
    args = parse_args()
    dataset_root: Path = args.dataset_root.expanduser().resolve()
    split: str = args.split
    fraction: float = float(args.fraction)

    if not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset_root is not a directory: {dataset_root}")
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")

    images_dir = dataset_root / "images" / split
    labels_dir = dataset_root / "labels" / split

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    image_paths = _list_image_files(images_dir)
    if not image_paths:
        print(f"No images found in {images_dir}")
        return

    sorted_paths = _sort_seq(image_paths, images_dir)
    if fraction < 1.0:
        num_to_show = max(1, int(len(sorted_paths) * fraction))
        sorted_paths = sorted_paths[:num_to_show]
    if args.max_frames and args.max_frames > 0:
        sorted_paths = sorted_paths[: args.max_frames]

    if not sorted_paths:
        print("No frames to visualize after filtering.")
        return

    if args.no_window and args.output_video is None:
        raise ValueError("Nothing to do: enable window visualization or provide --output-video.")

    window_name = f"{split} seq samples"
    writer = None
    first_frame_shape = None

    for idx, img_path in enumerate(sorted_paths, start=1):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        if first_frame_shape is None:
            first_frame_shape = img.shape[:2]
            if args.output_video is not None:
                args.output_video.parent.mkdir(parents=True, exist_ok=True)
                writer = _open_writer(args.output_video, float(args.fps), first_frame_shape)

        rel = img_path.relative_to(images_dir)
        label_path = labels_dir / rel.with_suffix(".txt")
        boxes = _load_yolo_boxes(label_path, img.shape[:2])

        label = f"{rel.as_posix()} ({idx}/{len(sorted_paths)})"
        _draw_boxes(img, boxes, label)

        if writer is not None:
            writer.write(img)

        if not args.no_window:
            cv2.imshow(window_name, img)
            key = cv2.waitKey(args.wait_ms) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break

    if writer is not None:
        writer.release()
    if not args.no_window:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
