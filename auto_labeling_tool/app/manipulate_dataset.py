#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


@dataclass(frozen=True)
class Sample:
    image_path: Path
    label_path: Path | None
    original_split: str
    rel_image_path: Path


def _log(message: str) -> None:
    timestamp = datetime.now().isoformat(timespec="seconds")
    print(f"[{timestamp}] {message}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Redefine YOLO dataset splits by reassigning files under images/ and labels/.\n"
            "Examples:\n"
            "  python -m auto_labeling_tool.app.manipulate_dataset --dataset-dir /data/training_data/my_dataset --ratios 0.7 0.2 0.1\n"
            "  python -m auto_labeling_tool.app.manipulate_dataset --dataset-dir /data/training_data/my_dataset --ratios 0.7 0.2"
        )
    )
    parser.add_argument(
        "--dataset-dir",
        required=True,
        type=Path,
        help="Dataset root containing images/ and labels/ directories.",
    )
    parser.add_argument(
        "--ratios",
        required=True,
        nargs="+",
        type=float,
        help=(
            "Split ratios as train val [test]. If test is omitted, it is inferred as (1 - train - val)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for shuffling before split assignment.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned counts without changing files.",
    )
    return parser.parse_args()


def _resolve_ratios(raw: list[float]) -> tuple[list[str], list[float]]:
    if len(raw) not in {2, 3}:
        raise ValueError("--ratios expects 2 or 3 values: train val [test]")

    if any(r < 0.0 for r in raw):
        raise ValueError("Ratios must be non-negative")

    train, val = raw[0], raw[1]
    if len(raw) == 2:
        test = 1.0 - train - val
        if test < -1e-8:
            raise ValueError("train + val cannot exceed 1.0 when test ratio is omitted")
        test = max(0.0, test)
    else:
        test = raw[2]

    ratios = [train, val, test]
    total = sum(ratios)
    if total <= 0.0:
        raise ValueError("At least one ratio must be greater than 0")

    normalized = [r / total for r in ratios]
    split_names = ["train", "val", "test"]
    if normalized[2] <= 0.0:
        split_names = split_names[:2]
        normalized = normalized[:2]
        renorm_total = sum(normalized)
        normalized = [r / renorm_total for r in normalized]

    return split_names, normalized


def _list_images(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]


def _collect_samples(dataset_root: Path) -> tuple[list[Sample], int]:
    images_root = dataset_root / "images"
    labels_root = dataset_root / "labels"

    if not images_root.is_dir():
        raise FileNotFoundError(f"Missing images directory: {images_root}")
    if not labels_root.is_dir():
        raise FileNotFoundError(f"Missing labels directory: {labels_root}")

    image_split_dirs = sorted([p for p in images_root.iterdir() if p.is_dir()], key=lambda p: p.name)
    if not image_split_dirs:
        raise RuntimeError(f"No image split directories found under {images_root}")

    samples: list[Sample] = []
    seen_rel_paths: dict[str, Path] = {}
    for split_dir in image_split_dirs:
        split_name = split_dir.name
        for image_path in _list_images(split_dir):
            rel_image = image_path.relative_to(split_dir)
            rel_no_ext = rel_image.with_suffix("").as_posix()
            if rel_no_ext in seen_rel_paths:
                prev = seen_rel_paths[rel_no_ext]
                raise RuntimeError(
                    "Conflicting image relative paths across splits: "
                    f"{prev} and {image_path}. "
                    "Please ensure unique relative image paths before resplitting."
                )
            seen_rel_paths[rel_no_ext] = image_path

            candidate_label = labels_root / split_name / rel_image.with_suffix(".txt")
            samples.append(
                Sample(
                    image_path=image_path,
                    label_path=candidate_label if candidate_label.is_file() else None,
                    original_split=split_name,
                    rel_image_path=rel_image,
                )
            )

    return samples, len(image_split_dirs)


def _allocate_counts(total: int, ratios: list[float]) -> list[int]:
    raw = [total * r for r in ratios]
    counts = [int(v) for v in raw]
    remaining = total - sum(counts)
    order = sorted(range(len(ratios)), key=lambda idx: (raw[idx] - counts[idx]), reverse=True)
    for idx in order[:remaining]:
        counts[idx] += 1
    return counts


def _move_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


def main() -> None:
    args = _parse_args()
    dataset_root = args.dataset_dir.expanduser().resolve()
    split_names, ratios = _resolve_ratios(list(args.ratios))
    samples, discovered_splits = _collect_samples(dataset_root)

    if not samples:
        raise RuntimeError("No images found to split.")

    random.Random(args.seed).shuffle(samples)
    counts = _allocate_counts(len(samples), ratios)

    _log(f"Dataset: {dataset_root}")
    _log(f"Discovered {len(samples)} images across {discovered_splits} existing split directories")
    _log(
        "Target split ratios: "
        + ", ".join(f"{name}={ratio:.4f}" for name, ratio in zip(split_names, ratios))
    )
    _log(
        "Target image counts: "
        + ", ".join(f"{name}={count}" for name, count in zip(split_names, counts))
    )

    if args.dry_run:
        _log("Dry run requested. No file changes were made.")
        return

    images_root = dataset_root / "images"
    labels_root = dataset_root / "labels"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_images_root = dataset_root / f".images_backup_{timestamp}"
    backup_labels_root = dataset_root / f".labels_backup_{timestamp}"

    if backup_images_root.exists() or backup_labels_root.exists():
        raise RuntimeError("Backup directory already exists for the generated timestamp")

    _log(f"Creating backups: {backup_images_root.name}, {backup_labels_root.name}")
    images_root.rename(backup_images_root)
    labels_root.rename(backup_labels_root)

    images_root.mkdir(parents=True, exist_ok=False)
    labels_root.mkdir(parents=True, exist_ok=False)

    for split_name in split_names:
        (images_root / split_name).mkdir(parents=True, exist_ok=True)
        (labels_root / split_name).mkdir(parents=True, exist_ok=True)

    _log("Reassigning files into new split directories")
    cursor = 0
    for split_name, count in zip(split_names, counts):
        selected = samples[cursor : cursor + count]
        cursor += count
        for sample in selected:
            src_image = backup_images_root / sample.original_split / sample.rel_image_path
            new_image = images_root / split_name / sample.rel_image_path
            _move_file(src_image, new_image)

            src_label = backup_labels_root / sample.original_split / sample.rel_image_path.with_suffix(".txt")
            if src_label.is_file():
                new_label = labels_root / split_name / sample.rel_image_path.with_suffix(".txt")
                _move_file(src_label, new_label)

    _log("Preserving unmatched label files in labels/_orphaned/")
    for old_split_dir in sorted([p for p in backup_labels_root.iterdir() if p.is_dir()], key=lambda p: p.name):
        for label_path in old_split_dir.rglob("*.txt"):
            if not label_path.is_file():
                continue
            rel_label = label_path.relative_to(old_split_dir)
            orphan_dst = labels_root / "_orphaned" / old_split_dir.name / rel_label
            _move_file(label_path, orphan_dst)

    _log("Split manipulation complete")
    
    shutil.rmtree(backup_images_root)
    shutil.rmtree(backup_labels_root)
    

if __name__ == "__main__":
    main()
