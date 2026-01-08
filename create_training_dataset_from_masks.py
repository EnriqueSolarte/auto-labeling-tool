#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
from tqdm import tqdm


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
MASK_EXTS = {".png", ".bmp", ".tiff", ".tif", ".jpg", ".npy"}


@dataclass(frozen=True)
class Sample:
    image_path: Path
    mask_path: Path


def _log(msg: str) -> None:
    timestamp = datetime.now().isoformat(timespec="seconds")
    print(f"[{timestamp}] {msg}")


def _get_files_with_ext(root: Path, exts: set[str]) -> list:
    exts_lower = {e.lower() for e in exts}
    list_path = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in exts_lower:
            list_path.append(path)
    return list_path


def _filter_average_masks(mask_paths: list[Path], masks_dir: Path) -> list[Path]:
    """
    Keep only masks under 'average_mask' when such directories exist.

    If the user points directly to an average_mask directory, all masks pass
    through. If no average_mask paths are present, the original list is returned.
    """
    if masks_dir.name == "average_mask":
        return mask_paths

    has_average = any("average_mask" in p.parts for p in mask_paths)
    if not has_average:
        return mask_paths
    return [p for p in mask_paths if "average_mask" in p.parts]


def _load_mask_stack(mask_path: Path) -> np.ndarray | None:
    """
    Load a mask file as a boolean stack with shape [n, H, W].

    Supports both image masks and .npy stacks.
    """
    if mask_path.suffix.lower() == ".npy":
        try:
            arr = np.load(str(mask_path))
        except Exception:
            return None
        if arr.ndim == 2:
            arr = arr[None, ...]
        if arr.ndim != 3:
            return None
        mask_stack = arr.astype(bool)
    else:
        img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        mask_stack = (img > 0)[None, ...]

    # Drop empty slices.
    if mask_stack.size == 0:
        return None
    keep = mask_stack.reshape(mask_stack.shape[0], -1).any(axis=1)
    mask_stack = mask_stack[keep]
    return mask_stack if mask_stack.size > 0 else None


def _bboxes_from_mask_stack(mask_stack: np.ndarray) -> list[tuple[float, float, float, float]]:
    if mask_stack.ndim != 3:
        return []
    n, h, w = mask_stack.shape
    if h <= 0 or w <= 0 or n == 0:
        return []

    boxes: list[tuple[float, float, float, float]] = []
    for mask in mask_stack:
        ys, xs = np.where(mask)
        if xs.size == 0:
            continue

        x_min = float(xs.min())
        x_max = float(xs.max())
        y_min = float(ys.min())
        y_max = float(ys.max())

        box_w = x_max - x_min + 1.0
        box_h = y_max - y_min + 1.0
        cx = x_min + box_w / 2.0
        cy = y_min + box_h / 2.0

        boxes.append(
            (
                cx / float(w),
                cy / float(h),
                box_w / float(w),
                box_h / float(h),
            )
        )
    return boxes


def _collect_samples(masks_dir: Path, images_dir: Path) -> list[Sample]:
    raw_mask_paths = _get_files_with_ext(masks_dir, MASK_EXTS)
    _log(f"Discovered {len(raw_mask_paths)} mask files under {masks_dir}")
    mask_paths = _filter_average_masks(raw_mask_paths, masks_dir)
    if (
        masks_dir.name != "average_mask"
        and raw_mask_paths
        and not any("average_mask" in p.parts for p in raw_mask_paths)
    ):
        raise RuntimeError(
            f"No average_mask directory found under {masks_dir}; refusing to process other masks."
        )
    if len(mask_paths) != len(raw_mask_paths):
        _log(f"Filtered to {len(mask_paths)} masks within average_mask directories")
    samples_fast: list[Sample] = []
    unmatched_masks: list[Path] = []

    for mask_path in mask_paths:
        rel = mask_path.relative_to(masks_dir)
        rel_parts = list(rel.parts)
        if "average_mask" in rel_parts:
            idx = rel_parts.index("average_mask")
            rel = Path(*rel_parts[:idx] + rel_parts[idx + 1 :])
        # Prefer a mirrored directory with the same stem (extension may differ).
        candidate_dir = images_dir / rel.parent
        candidate_img = None
        for ext in IMAGE_EXTS:
            test_path = candidate_dir / f"{mask_path.stem}{ext}"
            if test_path.is_file():
                candidate_img = test_path
                break

        if candidate_img is not None:
            samples_fast.append(Sample(image_path=candidate_img, mask_path=mask_path))
        else:
            unmatched_masks.append(mask_path)

    samples: list[Sample] = list(samples_fast)
    if not unmatched_masks:
        return samples

    # Fallback: build an index of images by stem and match masks by stem.
    image_index: dict[str, Path] = {}
    for img_path in _get_files_with_ext(images_dir, IMAGE_EXTS):
        image_index.setdefault(img_path.stem, img_path)

    for mask_path in unmatched_masks:
        img_path = image_index.get(mask_path.stem)
        if img_path is not None:
            samples.append(Sample(image_path=img_path, mask_path=mask_path))

    _log(
        "Matched %d masks to images (%d fast matches, %d via stem lookup, %d unmatched)"
        % (
            len(samples),
            len(samples_fast),
            len(samples) - len(samples_fast),
            len(unmatched_masks) - (len(samples) - len(samples_fast)),
        )
    )
    return samples


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
    _log(f"Writing labels for {len(samples)} samples into {labels_dir}")
    for s in tqdm(samples, desc="Labels", unit="sample"):
        mask_stack = _load_mask_stack(s.mask_path)
        if mask_stack is None:
            continue

        boxes = _bboxes_from_mask_stack(mask_stack)
        if not boxes:
            continue
        label_path = labels_dir / (s.image_path.stem + ".txt")
        with label_path.open("w", encoding="utf-8") as f:
            for cx, cy, bw, bh in boxes:
                f.write(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


def _link_images(samples: Sequence[Sample], images_dir: Path) -> None:
    images_dir.mkdir(parents=True, exist_ok=True)
    _log(f"Linking {len(samples)} images into {images_dir}")
    for s in tqdm(samples, desc="Images", unit="image"):
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
    _log(f"Wrote dataset config to {yaml_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a YOLO training dataset from mask annotations and RGB images.\n"
            "Masks (binary images or .npy stacks shaped [n, H, W]) may reside under a root like /data/masks or any nested subdirectory.\n"
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
        help="Root directory containing mask files (.png or .npy stacks; may include subdirectories).",
    )
    parser.add_argument(
        "--images_dir",
        default=Path("/data/raw_data/with_bed"),
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
        default=0.3,
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

    _log("Starting dataset creation")
    _log(f"Masks dir:  {masks_dir}")
    _log(f"Images dir: {images_dir}")
    _log(f"Output:     {output_root}")

    if not masks_dir.is_dir():
        raise FileNotFoundError(f"masks_dir is not a directory: {masks_dir}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"images_dir is not a directory: {images_dir}")

    samples = _collect_samples(masks_dir, images_dir)
    if not samples:
        raise RuntimeError("No samples found: ensure that mask files match image filenames.")
    _log(f"Found {len(samples)} samples with matching masks and images")

    train_samples, val_samples = _split_train_val(samples, val_ratio=float(args.val_ratio), seed=int(args.seed))
    _log(
        "Performed train/val split: %d train, %d val (val_ratio=%.2f, seed=%d)"
        % (len(train_samples), len(val_samples), float(args.val_ratio), int(args.seed))
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_root = output_root / timestamp
    _log(f"Creating dataset folders under {dataset_root}")

    images_train_dir = dataset_root / "images" / "train"
    images_val_dir = dataset_root / "images" / "val"
    labels_train_dir = dataset_root / "labels" / "train"
    labels_val_dir = dataset_root / "labels" / "val"

    _link_images(train_samples, images_train_dir)
    _link_images(val_samples, images_val_dir)
    _write_labels(train_samples, labels_train_dir, class_id=0)
    _write_labels(val_samples, labels_val_dir, class_id=0)
    _write_data_yaml(dataset_root, class_name=args.class_name, nc=1)

    _log(f"Created dataset at: {dataset_root}")
    _log(f"Train samples: {len(train_samples)}")
    _log(f"Val samples:   {len(val_samples)}")


if __name__ == "__main__":
    main()
