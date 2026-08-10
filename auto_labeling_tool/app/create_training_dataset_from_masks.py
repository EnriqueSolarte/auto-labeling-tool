#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Literal, Sequence, get_args

import cv2
import numpy as np
from tqdm import tqdm


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
MASK_EXTS = {".npy"}
LabelFormat = Literal["bbox", "mask"]
LABEL_FORMAT_CHOICES: tuple[str, ...] = tuple(str(choice) for choice in get_args(LabelFormat))
DATASET_DIR_BY_LABEL_FORMAT: dict[LabelFormat, str] = {
    "bbox": "bbx-based",
    "mask": "mask-based",
}


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


def _load_mask_stack(mask_path: Path) -> tuple[np.ndarray | None, str | None]:
    """
    Load a mask file as a boolean stack with shape [n, H, W].

    Only .npy stacks are supported. A 2D array is treated as a single-object
    stack; a 3D array is treated as [n, H, W].
    """
    if mask_path.suffix.lower() != ".npy":
        return None, "unsupported mask extension"
    try:
        arr = np.load(str(mask_path))
    except Exception as exc:
        return None, f"could not load .npy mask: {exc}"
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3:
        return None, f"invalid mask shape {arr.shape}; expected [n, H, W]"
    mask_stack = arr.astype(bool)

    # Drop empty slices.
    if mask_stack.size == 0:
        return None, "empty mask stack"
    keep = mask_stack.reshape(mask_stack.shape[0], -1).any(axis=1)
    mask_stack = mask_stack[keep]
    if mask_stack.size == 0:
        return None, "all mask slices are empty"
    return mask_stack, None


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

        bbx = (
            cx / float(w),
            cy / float(h),
            box_w / float(w),
            box_h / float(h),
        )
        is_ok = compute_affordance_bbx(*bbx)
        if not is_ok:
            continue
        boxes.append(bbx)
    return boxes


def _normalize_point(x: float, y: float, width: int, height: int) -> tuple[float, float]:
    return (
        min(max(x / float(width), 0.0), 1.0),
        min(max(y / float(height), 0.0), 1.0),
    )


def _polygons_from_mask_stack(mask_stack: np.ndarray) -> list[list[tuple[float, float]]]:
    if mask_stack.ndim != 3:
        return []
    n, h, w = mask_stack.shape
    if h <= 0 or w <= 0 or n == 0:
        return []

    polygons: list[list[tuple[float, float]]] = []
    for mask in mask_stack:
        if not mask.any():
            continue

        mask_u8 = mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        def contour_key(contour: np.ndarray) -> tuple[float, int, int, int, int]:
            x, y, cw, ch = cv2.boundingRect(contour)
            return (float(cv2.contourArea(contour)), cw * ch, -y, -x)

        contour = max(contours, key=contour_key).reshape(-1, 2)
        if len(contour) < 3:
            continue

        polygon = [_normalize_point(float(x), float(y), w, h) for x, y in contour]
        if len(polygon) >= 3:
            polygons.append(polygon)
    return polygons


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
        _log(
            f"Filtered to {len(mask_paths)} masks within average_mask directories")
    samples_fast: list[Sample] = []
    unmatched_masks: list[Path] = []

    for mask_path in mask_paths:
        rel = mask_path.relative_to(masks_dir)
        rel_parts = list(rel.parts)
        if "average_mask" in rel_parts:
            idx = rel_parts.index("average_mask")
            rel = Path(*rel_parts[:idx] + rel_parts[idx + 1:])
        # Prefer a mirrored directory with the same stem (extension may differ).
        candidate_dir = images_dir / rel.parent
        candidate_img = None
        for ext in IMAGE_EXTS:
            test_path = candidate_dir / f"{mask_path.stem}{ext}"
            if test_path.is_file():
                candidate_img = test_path
                break

        if candidate_img is not None:
            samples_fast.append(
                Sample(image_path=candidate_img, mask_path=mask_path))
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


def _sample_key(sample: Sample) -> tuple[str, str]:
    return (sample.mask_path.as_posix(), sample.image_path.as_posix())


def _subsample_samples(samples: Sequence[Sample], sample_ratio: float | None, seed: int) -> list[Sample]:
    if sample_ratio is None:
        return list(samples)

    ratio = float(sample_ratio)
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"sample_ratio must be in (0, 1], got {sample_ratio}")
    if not samples:
        return []

    sorted_samples = sorted(samples, key=_sample_key)
    sample_count = math.ceil(len(sorted_samples) * ratio)
    sample_count = min(max(sample_count, 1), len(sorted_samples))

    rng = np.random.default_rng(seed)
    indices = np.arange(len(sorted_samples))
    rng.shuffle(indices)
    selected_indices = sorted(indices[:sample_count].tolist())
    return [sorted_samples[i] for i in selected_indices]


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
    from auto_labeling_tool.utils.file_links import hardlink_or_symlink_or_copy

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    hardlink_or_symlink_or_copy(src, dst)


def _write_labels(
    samples: Sequence[Sample],
    labels_dir: Path,
    class_id: int,
    label_format: LabelFormat = "bbox",
) -> None:
    labels_dir.mkdir(parents=True, exist_ok=True)
    _log(f"Writing {label_format} labels for {len(samples)} samples into {labels_dir}")
    for s in tqdm(samples, desc="Labels", unit="sample"):
        mask_stack, skip_reason = _load_mask_stack(s.mask_path)
        if mask_stack is None:
            _log(f"Skipping {s.mask_path}: {skip_reason}")
            continue

        if label_format == "bbox":
            rows = [
                f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
                for cx, cy, bw, bh in _bboxes_from_mask_stack(mask_stack)
            ]
        elif label_format == "mask":
            rows = []
            for polygon in _polygons_from_mask_stack(mask_stack):
                coords = " ".join(f"{coord:.6f}" for point in polygon for coord in point)
                rows.append(f"{class_id} {coords}")
        else:
            raise ValueError(f"Unsupported label_format: {label_format}")

        if not rows:
            _log(f"Skipping {s.mask_path}: no valid {label_format} labels")
            continue
        label_path = labels_dir / (s.image_path.stem + ".txt")
        with label_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(row + "\n")


def _dataset_dir_name(label_format: LabelFormat) -> str:
    try:
        return DATASET_DIR_BY_LABEL_FORMAT[label_format]
    except KeyError as exc:
        raise ValueError(f"Unsupported label_format: {label_format}") from exc


def _ensure_new_dataset_root(dataset_root: Path) -> None:
    if not dataset_root.exists():
        return
    if any(dataset_root.iterdir()):
        raise FileExistsError(
            f"Dataset directory already exists and is not empty: {dataset_root}. "
            "Move or remove it before regenerating this label type."
        )


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
            "Masks must be .npy stacks shaped [n, H, W] and may reside under a root like /data/masks or any nested subdirectory.\n"
            "Example:\n"
            "  python -m auto_labeling_tool.app.create_training_dataset_from_masks \\\n"
            "      --masks_dir /data/masks \\\n"
            "      --images_dir /data/with_bed \\\n"
            "      -o /data/training_data \\\n"
            "      --label-format bbox"
        )
    )
    parser.add_argument(
        "--masks_dir",
        required=True,
        type=Path,
        help="Root directory containing .npy mask stacks; may include subdirectories.",
    )
    parser.add_argument(
        "--images_dir",
        required=True,
        type=Path,
        help="Root directory containing RGB images corresponding to the masks.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        type=Path,
        default=Path("/data/training_data"),
        help="Root directory where the training dataset will be created.",
    )
    parser.add_argument(
        "--output-root",
        dest="output_dir",
        type=Path,
        help=argparse.SUPPRESS,
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
        "--sample-ratio",
        type=float,
        default=None,
        help="Optional deterministic raw sample fraction to use before train/val splitting.",
    )
    parser.add_argument(
        "--class-name",
        type=str,
        default="bed",
        help="Name of the single YOLO class.",
    )
    parser.add_argument(
        "--label-format",
        choices=LABEL_FORMAT_CHOICES,
        default="bbox",
        help=(
            "YOLO label format to write from .npy masks: bbox writes "
            "'class cx cy w h'; mask writes 'class x1 y1 x2 y2 ...'."
        ),
    )
    return parser.parse_args()


def compute_affordance_bbx(cx, cy, bbx_w, bbx_h, dist_thr=0.8, size_thr=0.05):
    return True


def main() -> None:
    args = parse_args()
    masks_dir: Path = args.masks_dir.expanduser().resolve()
    images_dir: Path = args.images_dir.expanduser().resolve()
    output_arg = getattr(args, "output_dir", None)
    if output_arg is None:
        output_arg = args.output_root
    output_root: Path = output_arg.expanduser().resolve()

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
        raise RuntimeError(
            "No samples found: ensure that mask files match image filenames.")
    _log(f"Found {len(samples)} samples with matching masks and images")

    samples = _subsample_samples(
        samples, sample_ratio=args.sample_ratio, seed=int(args.seed))
    _log(
        "Selected %d raw samples for splitting (sample_ratio=%s, seed=%d)"
        % (len(samples), args.sample_ratio, int(args.seed))
    )

    train_samples, val_samples = _split_train_val(
        samples, val_ratio=float(args.val_ratio), seed=int(args.seed))
    _log(
        "Performed train/val split: %d train, %d val (val_ratio=%.2f, seed=%d)"
        % (len(train_samples), len(val_samples), float(args.val_ratio), int(args.seed))
    )

    dataset_root = output_root / _dataset_dir_name(args.label_format)
    _ensure_new_dataset_root(dataset_root)
    _log(f"Creating dataset folders under {dataset_root}")

    images_train_dir = dataset_root / "images" / "train"
    images_val_dir = dataset_root / "images" / "val"
    labels_train_dir = dataset_root / "labels" / "train"
    labels_val_dir = dataset_root / "labels" / "val"

    _link_images(train_samples, images_train_dir)
    _link_images(val_samples, images_val_dir)
    _write_labels(train_samples, labels_train_dir, class_id=0, label_format=args.label_format)
    _write_labels(val_samples, labels_val_dir, class_id=0, label_format=args.label_format)
    _write_data_yaml(dataset_root, class_name=args.class_name, nc=1)

    _log(f"Created dataset at: {dataset_root}")
    _log(f"Train samples: {len(train_samples)}")
    _log(f"Val samples:   {len(val_samples)}")


if __name__ == "__main__":
    main()
