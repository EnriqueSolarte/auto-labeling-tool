#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
from datetime import datetime
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _log(message: str) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    print(f"[{now}] {message}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge multiple YOLO datasets into a new dataset directory.\n\n"
            "Example:\n"
            "  python -m auto_labeling_tool.app.merge_dataset \\\n"
            "    -d /data/training_data/dataset_a "
            "/data/training_data/dataset_b "
            "/data/training_data/dataset_c \\\n"
            "    -o /data/training_data"
        )
    )
    parser.add_argument(
        "-d",
        "--dataset",
        dest="dataset",
        nargs="+",
        type=Path,
        help="Dataset directories to merge.",
    )
    parser.add_argument(
        "--datasets",
        dest="dataset",
        nargs="+",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        type=Path,
        help="Parent directory where merge__{timestamp} will be created.",
    )
    parser.add_argument(
        "--output-root",
        dest="output_dir",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be merged without creating files.",
    )
    args = parser.parse_args()
    if args.dataset is None:
        parser.error("the following arguments are required: -d/--dataset")
    if args.output_dir is None:
        parser.error("the following arguments are required: -o/--output-dir")
    return args


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def _parse_data_yaml(dataset_dir: Path) -> tuple[int | None, list[str] | None]:
    yaml_path = dataset_dir / "data.yaml"
    if not yaml_path.is_file():
        return None, None

    nc: int | None = None
    names: list[str] | None = None

    for line in yaml_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("nc:"):
            try:
                nc = int(stripped.split(":", 1)[1].strip())
            except ValueError:
                nc = None
        if stripped.startswith("names:"):
            raw = stripped.split(":", 1)[1].strip()
            try:
                parsed = ast.literal_eval(raw)
                if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
                    names = parsed
            except (SyntaxError, ValueError):
                names = None

    return nc, names


def _validate_inputs(dataset_dirs: list[Path]) -> None:
    for dataset_dir in dataset_dirs:
        if not dataset_dir.is_dir():
            raise FileNotFoundError(f"Dataset not found: {dataset_dir}")
        if not (dataset_dir / "images").is_dir():
            raise FileNotFoundError(f"Missing images directory: {dataset_dir / 'images'}")
        if not (dataset_dir / "labels").is_dir():
            raise FileNotFoundError(f"Missing labels directory: {dataset_dir / 'labels'}")


def _collect_splits(dataset_dir: Path) -> list[str]:
    splits = sorted([p.name for p in (dataset_dir / "images").iterdir() if p.is_dir()])
    if not splits:
        raise RuntimeError(f"No image split directories found in {dataset_dir / 'images'}")
    return splits


def _write_data_yaml(output_dir: Path, splits: list[str], nc: int, names: list[str]) -> None:
    lines = [f"path: {output_dir}"]
    for split in ("train", "val", "test"):
        if split in splits:
            lines.append(f"{split}: images/{split}")
    lines.append(f"nc: {nc}")
    lines.append(f"names: {names}")
    (output_dir / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _symlink_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(src.resolve())


def _dedupe_rel_path(rel_path: Path, tag: str, index: int) -> Path:
    suffix = f"__{tag}" if index == 1 else f"__{tag}_{index}"
    return rel_path.with_name(f"{rel_path.stem}{suffix}{rel_path.suffix}")


def main() -> None:
    args = _parse_args()
    dataset_arg = getattr(args, "dataset", None)
    if dataset_arg is None:
        dataset_arg = args.datasets
    dataset_dirs = [d.expanduser().resolve() for d in dataset_arg]
    output_arg = getattr(args, "output_dir", None)
    if output_arg is None:
        output_arg = args.output_root
    output_root = output_arg.expanduser().resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / f"merge__{timestamp}"

    _validate_inputs(dataset_dirs)
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=False)

    discovered_splits: set[str] = set()
    sample_counts: dict[str, int] = {}
    total_images = 0
    total_labels = 0
    collision_count = 0
    nc_ref: int | None = None
    names_ref: list[str] | None = None
    used_rel_paths_per_split: dict[str, set[Path]] = {}

    for dataset_dir in dataset_dirs:
        split_names = _collect_splits(dataset_dir)
        discovered_splits.update(split_names)
        sample_counts[dataset_dir.name] = 0

        nc, names = _parse_data_yaml(dataset_dir)
        if nc is not None and names is not None:
            if nc_ref is None and names_ref is None:
                nc_ref, names_ref = nc, names
            elif nc_ref != nc or names_ref != names:
                raise RuntimeError(
                    f"Class metadata mismatch in {dataset_dir}. "
                    f"Expected nc={nc_ref}, names={names_ref}, got nc={nc}, names={names}."
                )

        for split in split_names:
            image_split_dir = dataset_dir / "images" / split
            label_split_dir = dataset_dir / "labels" / split

            for image_path in image_split_dir.rglob("*"):
                if not _is_image(image_path):
                    continue
                sample_counts[dataset_dir.name] += 1
                total_images += 1

                rel_image = image_path.relative_to(image_split_dir)
                used_rel_paths = used_rel_paths_per_split.setdefault(split, set())
                rel_target = rel_image
                dedupe_index = 1
                while rel_target in used_rel_paths:
                    rel_target = _dedupe_rel_path(rel_image, dataset_dir.name, dedupe_index)
                    dedupe_index += 1
                if rel_target != rel_image:
                    collision_count += 1
                used_rel_paths.add(rel_target)

                dst_image = output_dir / "images" / split / rel_target
                dst_label = output_dir / "labels" / split / rel_target.with_suffix(".txt")
                src_label = label_split_dir / rel_image.with_suffix(".txt")

                if args.dry_run:
                    if src_label.is_file():
                        total_labels += 1
                    continue

                _symlink_file(image_path, dst_image)

                if src_label.is_file():
                    _symlink_file(src_label, dst_label)
                    total_labels += 1

    sorted_splits = sorted(discovered_splits, key=lambda x: ("train", "val", "test").index(x) if x in {"train", "val", "test"} else 99)

    _log(f"Input datasets: {len(dataset_dirs)}")
    for name, count in sample_counts.items():
        _log(f"  {name}: {count} images")
    _log(f"Total images: {total_images}")
    _log(f"Total labels: {total_labels}")
    _log(f"Filename collisions resolved: {collision_count}")
    _log(f"Splits found: {', '.join(sorted_splits)}")
    _log(f"Target: {output_dir}")

    if args.dry_run:
        _log("Dry run requested. No files were created.")
        return

    if nc_ref is None or names_ref is None:
        nc_ref, names_ref = 1, ["class0"]
        _log("No valid data.yaml metadata found. Using fallback: nc=1, names=['class0']")

    _write_data_yaml(output_dir, sorted_splits, nc_ref, names_ref)

    _log("Merge complete.")


if __name__ == "__main__":
    main()
