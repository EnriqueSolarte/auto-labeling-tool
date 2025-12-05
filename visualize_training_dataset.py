#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import cv2


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def _get_image_files(root: Path) -> list[Path]:
    return [
        p
        for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]


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


def _draw_boxes(img, boxes: Sequence[tuple[int, int, int, int]]):
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize YOLO training samples for a dataset created by create_training_dataset.py.\n"
            "Example: visualize_training_dataset.py --dataset-root /data/training_data/20251204_072806 --split train"
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
        choices=["train", "val"],
        default="train",
        help="Which split to visualize.",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.1,
        help="Fraction of images in the split to visualize (0 < fraction <= 1].",
    )
    return parser.parse_args()


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

    image_paths = _get_image_files(images_dir)
    if not image_paths:
        print(f"No images found in {images_dir}")
        return

    # Deterministic subset: take the first N images.
    num_to_show = max(1, int(len(image_paths) * fraction)) if fraction < 1.0 else len(image_paths)
    subset = image_paths[:num_to_show]

    print(f"Visualizing {len(subset)} of {len(image_paths)} images from '{split}' split...")

    window_name = f"{split} samples"
    for img_path in subset:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        label_path = labels_dir / (img_path.stem + ".txt")
        boxes = _load_yolo_boxes(label_path, img.shape[:2])
        _draw_boxes(img, boxes)

        cv2.imshow(window_name, img)
        key = cv2.waitKey(0) & 0xFF
        if key in (27, ord("q"), ord("Q")):  # ESC or 'q'/'Q' to quit
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
