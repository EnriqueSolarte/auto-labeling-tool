#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
MASK_EXTS = {".png", ".bmp", ".tiff", ".tif", ".jpg"}


@dataclass(frozen=True)
class Sample:
    image_path: Path
    mask_path: Path


def _get_files_with_ext(root: Path, exts: set[str]) -> list:
    exts_lower = {e.lower() for e in exts}
    list_path = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in exts_lower:
            list_path.append(path)
    return list_path

def _collect_samples(masks_dir: Path, images_dir: Path) -> list[Sample]:
    # Fast path: assume images mirror the masks directory structure.
    # This is O(#masks) and avoids scanning all images when images_dir is huge.
    samples_fast: list[Sample] = []
    mask_paths = _get_files_with_ext(masks_dir, MASK_EXTS)
    unmatched_masks: list[Path] = []

    for mask_path in mask_paths:
        rel = mask_path.relative_to(masks_dir)
        candidate_img = images_dir / rel
        if candidate_img.is_file() and candidate_img.suffix.lower() in IMAGE_EXTS:
            samples_fast.append(Sample(image_path=candidate_img, mask_path=mask_path))
        else:
            unmatched_masks.append(mask_path)

    # If all masks were matched using the mirrored-structure assumption, return early.
    if not unmatched_masks:
        return samples_fast

    # Fallback to the original behavior (match by filename only) when the directory
    # structures do not mirror each other. Complexity: O(#masks + #images).
    mask_index: dict[str, Path] = {}
    for mask_path in mask_paths:
        mask_index[mask_path.name] = mask_path

    samples: list[Sample] = []
    for img_path in _get_files_with_ext(images_dir, IMAGE_EXTS):
        mask_path = mask_index.get(img_path.name)
        if mask_path is not None:
            samples.append(Sample(image_path=img_path, mask_path=mask_path))
    return samples


def _compute_bbox_from_mask(mask_path: Path) -> tuple[float, float, float, float] | None:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    mask_bool = mask > 0
    ys, xs = np.where(mask_bool)
    if xs.size == 0:
        return None

    x_min = float(xs.min())
    x_max = float(xs.max())
    y_min = float(ys.min())
    y_max = float(ys.max())

    h, w = mask.shape
    box_w = x_max - x_min + 1.0
    box_h = y_max - y_min + 1.0
    cx = x_min + box_w / 2.0
    cy = y_min + box_h / 2.0

    if w <= 0 or h <= 0:
        return None

    return (
        cx / float(w),
        cy / float(h),
        box_w / float(w),
        box_h / float(h),
    )


def _split_train_val(samples: Sequence[Sample], val_ratio: float, seed: int) -> tuple[list[Sample], list[Sample]]:
    if not samples:
        return [], []

    rng = np.random.default_rng(seed)
    indices = np.arange(len(samples))
    rng.shuffle(indices)

    split_idx = int(len(samples) * (1.0 - float(val_ratio)))
    split_idx = min(max(split_idx, 0), len(samples))

    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]

    train = [samples[i] for i in train_idx]
    val = [samples[i] for i in val_idx]
    return train, val


def _symlink_safe(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(src)
    except OSError:
        # Fallback to copying if symlinks are not supported
        import shutil

        shutil.copy2(src, dst)


def _write_labels(samples: Sequence[Sample], labels_dir: Path, class_id: int) -> None:
    labels_dir.mkdir(parents=True, exist_ok=True)
    for s in samples:
        bbox = _compute_bbox_from_mask(s.mask_path)
        if bbox is None:
            continue
        cx, cy, bw, bh = bbox
        label_path = labels_dir / (s.image_path.stem + ".txt")
        with label_path.open("w", encoding="utf-8") as f:
            f.write(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


def _link_images(samples: Sequence[Sample], images_dir: Path) -> None:
    images_dir.mkdir(parents=True, exist_ok=True)
    for s in samples:
        dst = images_dir / s.image_path.name
        _symlink_safe(s.image_path, dst)


def _write_data_yaml(root: Path, class_name: str, nc: int = 1) -> None:
    # Use an absolute path so YOLO configs are location-independent of the
    # current working directory from which this script is invoked.
    root = root.expanduser().resolve()
    yaml_path = root / "data.yaml"
    content = [
        f"path: {root}",
        "train: images/train",
        "val: images/val",
        f"nc: {nc}",
        f"names: ['{class_name}']",
        "",
    ]
    yaml_path.write_text("\n".join(content), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a YOLO training dataset from mask annotations and RGB images.\n"
            "Masks may reside under a root like /data/masks or any nested subdirectory.\n"
            "Example:\n"
            "  python create_training_dataset_from_masks.py \\\n"
            "      --masks_dir /data/masks \\\n"
            "      --images_dir /data/with_bed \\\n"
            "      --output-root /data/training_data"
        )
    )
    parser.add_argument(
        "--masks_dir",
        default=Path("/data/masks"),
        type=Path,
        help="Root directory containing binary mask images (may include subdirectories).",
    )
    parser.add_argument(
        "--images_dir",
        default=Path("/data/with_bed"),
        type=Path,
        help="Root directory containing RGB images corresponding to the masks.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data/training_data"),
        help="Root directory where the training dataset will be created.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Fraction of samples reserved for validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for the train/val split.",
    )
    parser.add_argument(
        "--class-name",
        type=str,
        default="bed",
        help="Name of the single YOLO class.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    masks_dir: Path = args.masks_dir.expanduser().resolve()
    images_dir: Path = args.images_dir.expanduser().resolve()
    output_root: Path = args.output_root.expanduser().resolve()

    if not masks_dir.is_dir():
        raise FileNotFoundError(f"masks_dir is not a directory: {masks_dir}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"images_dir is not a directory: {images_dir}")

    samples = _collect_samples(masks_dir, images_dir)
    if not samples:
        raise RuntimeError("No samples found: ensure that mask files match image filenames.")

    train_samples, val_samples = _split_train_val(samples, val_ratio=float(args.val_ratio), seed=int(args.seed))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_root = output_root / timestamp

    images_train_dir = dataset_root / "images" / "train"
    images_val_dir = dataset_root / "images" / "val"
    labels_train_dir = dataset_root / "labels" / "train"
    labels_val_dir = dataset_root / "labels" / "val"

    _link_images(train_samples, images_train_dir)
    _link_images(val_samples, images_val_dir)
    _write_labels(train_samples, labels_train_dir, class_id=0)
    _write_labels(val_samples, labels_val_dir, class_id=0)
    _write_data_yaml(dataset_root, class_name=args.class_name, nc=1)

    print(f"Created dataset at: {dataset_root}")
    print(f"  Train samples: {len(train_samples)}")
    print(f"  Val samples:   {len(val_samples)}")


if __name__ == "__main__":
    main()
