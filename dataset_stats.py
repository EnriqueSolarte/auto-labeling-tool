#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Iterable


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def _list_files(root: Path, exts: set[str]) -> list[Path]:
    exts_lower = {ext.lower() for ext in exts}
    return [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in exts_lower
    ]


def _list_label_files(root: Path) -> list[Path]:
    return [path for path in root.rglob("*.txt") if path.is_file()]


def _rel_stems(paths: Iterable[Path], base: Path) -> set[str]:
    stems: set[str] = set()
    for path in paths:
        try:
            rel = path.relative_to(base)
        except ValueError:
            continue
        stems.add(rel.with_suffix("").as_posix())
    return stems


def _parse_label_file(path: Path) -> list[int]:
    class_ids: list[int] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                class_id = int(float(parts[0]))
                _ = float(parts[1])
                _ = float(parts[2])
                _ = float(parts[3])
                _ = float(parts[4])
            except ValueError:
                continue
            class_ids.append(class_id)
    return class_ids


def _try_load_names(data_yaml: Path) -> dict[int, str]:
    if not data_yaml.is_file():
        return {}

    text = data_yaml.read_text(encoding="utf-8", errors="ignore")
    names: dict[int, str] = {}

    try:
        import yaml  # type: ignore
    except Exception:
        yaml = None

    if yaml is not None:
        try:
            payload = yaml.safe_load(text)
        except Exception:
            payload = None
        if isinstance(payload, dict):
            raw = payload.get("names")
            if isinstance(raw, list):
                return {idx: str(name) for idx, name in enumerate(raw)}
            if isinstance(raw, dict):
                return {int(k): str(v) for k, v in raw.items() if str(k).isdigit()}

    lines = [line.split("#", 1)[0].strip() for line in text.splitlines()]
    joined = "\n".join(line for line in lines if line)

    match = re.search(r"names\s*:\s*\[(.*?)\]", joined)
    if match:
        items = [item.strip() for item in match.group(1).split(",") if item.strip()]
        for idx, item in enumerate(items):
            names[idx] = item.strip("'\"")
        return names

    in_names = False
    idx = 0
    for line in lines:
        if not line:
            continue
        if line.startswith("names:"):
            in_names = True
            line = line[len("names:") :].strip()
            if line:
                items = [item.strip() for item in line.split(",") if item.strip()]
                for item in items:
                    names[idx] = item.strip("'\"- ")
                    idx += 1
            continue
        if in_names:
            if line.startswith("-"):
                names[idx] = line.lstrip("-").strip().strip("'\"")
                idx += 1
            else:
                break
    return names


def _discover_splits(images_root: Path) -> list[str]:
    if not images_root.is_dir():
        return []
    return sorted([p.name for p in images_root.iterdir() if p.is_dir()])


def _format_class_name(class_id: int, names: dict[int, str]) -> str:
    if class_id in names:
        return f"{class_id} ({names[class_id]})"
    return str(class_id)


def _print_split_stats(split: str, images_dir: Path, labels_dir: Path, names: dict[int, str]) -> None:
    if not images_dir.is_dir():
        print(f"Split '{split}': images directory not found: {images_dir}")
        return

    image_paths = _list_files(images_dir, IMAGE_EXTS)
    label_paths = _list_label_files(labels_dir) if labels_dir.is_dir() else []

    image_stems = _rel_stems(image_paths, images_dir)
    label_stems = _rel_stems(label_paths, labels_dir) if labels_dir.is_dir() else set()

    missing_labels = image_stems - label_stems
    orphan_labels = label_stems - image_stems

    class_counts: Counter[int] = Counter()
    empty_label_files = 0
    images_with_boxes = set()

    for label_path in label_paths:
        class_ids = _parse_label_file(label_path)
        if not class_ids:
            empty_label_files += 1
            continue
        class_counts.update(class_ids)
        try:
            rel = label_path.relative_to(labels_dir)
        except ValueError:
            continue
        images_with_boxes.add(rel.with_suffix("").as_posix())

    total_boxes = sum(class_counts.values())
    labeled_images = len(image_stems & label_stems)
    images_with_boxes_count = len(images_with_boxes)

    print(f"Split '{split}':")
    print(f"  Images:              {len(image_paths)}")
    print(f"  Labels:              {len(label_paths)}")
    print(f"  Images w/ labels:    {labeled_images}")
    print(f"  Images missing lbls: {len(missing_labels)}")
    print(f"  Orphan labels:       {len(orphan_labels)}")
    print(f"  Empty label files:   {empty_label_files}")
    print(f"  Images w/ boxes:     {images_with_boxes_count}")
    print(f"  Total boxes:         {total_boxes}")

    if class_counts:
        print("  Boxes per class:")
        for class_id in sorted(class_counts):
            label = _format_class_name(class_id, names)
            print(f"    {label}: {class_counts[class_id]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Report dataset stats for a YOLO dataset generated by create_training_dataset_from_masks.py.\n"
            "Example:\n"
            "  python dataset_stats.py /data/training_data/20260113_065116"
        )
    )
    parser.add_argument(
        "--dataset-dir",
        required=True,
        type=Path,
        help="Path to the dataset root (e.g. /data/training_data/20260113_065116).",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="",
        help="Comma-separated split names (default: auto-detect under images/).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root: Path = args.dataset_dir.expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset_root is not a directory: {dataset_root}")

    data_yaml = dataset_root / "data.yaml"
    names = _try_load_names(data_yaml)

    images_root = dataset_root / "images"
    labels_root = dataset_root / "labels"

    if args.splits:
        splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    else:
        splits = _discover_splits(images_root)
        if not splits:
            splits = ["train", "val"]

    print(f"Dataset: {dataset_root}")
    if data_yaml.is_file():
        print(f"Config:  {data_yaml}")
    if names:
        formatted = ", ".join(f"{k}={v}" for k, v in sorted(names.items()))
        print(f"Classes: {formatted}")
    else:
        print("Classes: (not found; class IDs only)")
    print("")

    for split in splits:
        images_dir = images_root / split
        labels_dir = labels_root / split
        _print_split_stats(split, images_dir, labels_dir, names)
        print("")


if __name__ == "__main__":
    main()
