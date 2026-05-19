#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, Sequence, cast

import cv2
import numpy as np

from auto_labeling_tool.app.create_training_dataset_from_masks import (
    LABEL_FORMAT_CHOICES,
    LabelFormat,
)


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
LABEL_FORMAT_ARG_CHOICES = ("auto", *LABEL_FORMAT_CHOICES)
MASK_OVERLAY_ALPHA = 0.85
MASK_CONTOUR_THICKNESS = 2


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


def _load_yolo_polygons(label_path: Path, img_shape: tuple[int, int]) -> list[list[tuple[int, int]]]:
    h, w = img_shape
    if w <= 0 or h <= 0:
        return []

    polygons: list[list[tuple[int, int]]] = []
    if not label_path.is_file():
        return polygons

    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7 or (len(parts) - 1) % 2 != 0:
                continue
            try:
                coords = [float(value) for value in parts[1:]]
            except ValueError:
                continue

            points: list[tuple[int, int]] = []
            for x_norm, y_norm in zip(coords[0::2], coords[1::2]):
                x = int(round(x_norm * w))
                y = int(round(y_norm * h))
                x = max(0, min(x, w - 1))
                y = max(0, min(y, h - 1))
                points.append((x, y))

            if len(points) >= 3:
                polygons.append(points)

    return polygons


def _label_row_format(parts: Sequence[str]) -> LabelFormat | None:
    try:
        _ = [float(value) for value in parts]
    except ValueError:
        return None
    if len(parts) == 5:
        return "bbox"
    if len(parts) >= 7 and (len(parts) - 1) % 2 == 0:
        return "mask"
    return None


def _infer_label_format(labels_dir: Path, requested: str = "auto") -> LabelFormat:
    if requested != "auto":
        if requested not in LABEL_FORMAT_CHOICES:
            raise ValueError(f"Unsupported label format: {requested}")
        return cast(LabelFormat, requested)

    detected: set[LabelFormat] = set()
    for label_path in sorted(labels_dir.rglob("*.txt")):
        with label_path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                label_format = _label_row_format(parts)
                if label_format is not None:
                    detected.add(label_format)
                if len(detected) > 1:
                    raise ValueError(
                        f"Mixed bbox and mask label rows found under {labels_dir}; "
                        "use a single label format per dataset."
                    )

    if detected:
        return next(iter(detected))

    raise ValueError(
        f"Could not infer label format from labels under {labels_dir}. "
        "Use --label-format bbox or --label-format mask."
    )


def _draw_boxes(
    img,
    boxes: Sequence[tuple[int, int, int, int]],
    label: str,
    color: tuple[int, int, int],
) -> None:
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        img,
        label,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2,
        cv2.LINE_AA,
    )


def _draw_polygons(
    img,
    polygons: Sequence[Sequence[tuple[int, int]]],
    label: str,
    color: tuple[int, int, int],
) -> None:
    overlay = img.copy()
    for polygon in polygons:
        points = np.array(polygon, dtype=np.int32)
        cv2.fillPoly(overlay, [points], color)
    if polygons:
        cv2.addWeighted(overlay, MASK_OVERLAY_ALPHA, img, 1.0 - MASK_OVERLAY_ALPHA, 0.0, img)
    for polygon in polygons:
        points = np.array(polygon, dtype=np.int32)
        cv2.polylines(
            img,
            [points],
            isClosed=True,
            color=color,
            thickness=MASK_CONTOUR_THICKNESS,
        )
    cv2.putText(
        img,
        label,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2,
        cv2.LINE_AA,
    )


def _draw_label(
    img,
    label_path: Path,
    label_format: LabelFormat,
    label: str,
    color: tuple[int, int, int],
) -> None:
    if label_format == "bbox":
        boxes = _load_yolo_boxes(label_path, img.shape[:2])
        _draw_boxes(img, boxes, label, color)
    elif label_format == "mask":
        polygons = _load_yolo_polygons(label_path, img.shape[:2])
        _draw_polygons(img, polygons, label, color)
    else:
        raise ValueError(f"Unsupported label format: {label_format}")


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
        "-d",
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
    parser.add_argument(
        "--label-format",
        choices=LABEL_FORMAT_ARG_CHOICES,
        default="auto",
        help="Label geometry to draw. Default auto infers from label rows.",
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
    label_format = _infer_label_format(labels_dir, str(args.label_format))

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

        label = f"{rel.as_posix()} ({idx}/{len(sorted_paths)})"
        _draw_label(img, label_path, label_format, label, (0, 200, 0))

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
