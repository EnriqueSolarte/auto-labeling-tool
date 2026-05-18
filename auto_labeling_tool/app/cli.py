#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from auto_labeling_tool.config import (
    DEFAULT_CONFIG_NAME,
    default_config,
    load_config,
    load_raw_config,
    write_config,
)


COMPLETION_LINE = 'eval "$(register-python-argcomplete auto-label-tool)"'
COMPLETION_MARKER = "# auto-label-tool completion"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def _add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        type=Path,
        help="Path to the auto-label-tool YAML config.",
    )
    parser.add_argument(
        "--update-root",
        action="store_true",
        help=(
            "Update run.root to the config file parent directory if they differ. "
            "Without this flag, interactive shells ask; non-interactive runs fail."
        ),
    )


def _config_parent(path: Path) -> Path:
    return path.expanduser().resolve().parent


def _resolve_config_root(raw_root: object) -> Path | None:
    if raw_root is None:
        return None
    return Path(str(raw_root)).expanduser().resolve()


def _maybe_update_config_root(config_path: Path, force: bool = False) -> None:
    config_path = config_path.expanduser().resolve()
    raw = load_raw_config(config_path)
    current_root = _config_parent(config_path)
    run = raw.get("run", {})
    if not isinstance(run, dict):
        raise ValueError(f"Config `run` section must be a mapping: {config_path}")
    configured_root = _resolve_config_root(run.get("root"))
    if configured_root == current_root:
        return

    print(
        "[auto-label-tool] WARNING: config run.root does not match the config file location.",
        file=sys.stderr,
    )
    print(f"[auto-label-tool]   config:   {config_path}", file=sys.stderr)
    print(f"[auto-label-tool]   run.root: {configured_root}", file=sys.stderr)
    print(f"[auto-label-tool]   expected: {current_root}", file=sys.stderr)

    should_update = force
    if not should_update and sys.stdin.isatty():
        answer = input("Update run.root to the config file parent directory? [y/N = cancel]: ").strip().lower()
        should_update = answer in {"y", "yes"}
    elif not should_update:
        raise RuntimeError(
            "Config run.root mismatch in non-interactive mode. "
            "Run `auto-label-tool update-root -c "
            f"{config_path}` or pass `--update-root`."
        )

    if not should_update:
        raise RuntimeError(
            "Canceled because config run.root does not match the config file location. "
            "Update the config before running the pipeline."
        )
        return

    run = raw.setdefault("run", {})
    if not isinstance(run, dict):
        raise ValueError(f"Config `run` section must be a mapping: {config_path}")
    run["root"] = str(current_root)
    write_config(raw, config_path)
    print(f"[auto-label-tool] Updated run.root in {config_path}", file=sys.stderr)


def _load_cli_config(args: argparse.Namespace):
    if not getattr(args, "skip_root_check", False):
        _maybe_update_config_root(args.config, force=bool(getattr(args, "update_root", False)))
    return load_config(args.config, run_id=getattr(args, "run_id", None))


def _default_data_id(images_dir: Path) -> str:
    parts = images_dir.expanduser().resolve().parts
    selected = parts[-2:] if len(parts) >= 2 else parts
    data_id = "__".join(selected)
    if not data_id:
        raise ValueError(f"Cannot derive data id from images directory: {images_dir}")
    return data_id


def _assert_images_dir(images_dir: Path) -> None:
    if not images_dir.exists():
        raise FileNotFoundError(f"--images-dir does not exist: {images_dir}")
    if not images_dir.is_dir():
        raise NotADirectoryError(f"--images-dir is not a directory: {images_dir}")

    for path in images_dir.iterdir():
        # Path.is_file() follows symlinks, matching the downstream OpenCV read path.
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            return

    exts = ", ".join(sorted(IMAGE_EXTS))
    raise ValueError(
        f"--images-dir must contain at least one image file or image symlink "
        f"directly under it: {images_dir}\n"
        f"Supported extensions: {exts}"
    )


def _create_config(args: argparse.Namespace) -> None:
    images_dir = args.images_dir.expanduser().resolve()
    _assert_images_dir(images_dir)
    output_dir = args.output_dir.expanduser().resolve()
    data_id = args.data_id or _default_data_id(images_dir)
    run_root = output_dir / data_id
    config_path = args.config
    if config_path is None:
        config_path = run_root / DEFAULT_CONFIG_NAME
    else:
        config_path = config_path.expanduser().resolve()
        if config_path.parent != run_root:
            raise ValueError(
                "Explicit config path must be inside the derived run root so "
                "`run.root` matches the config file parent.\n"
                f"  expected parent: {run_root}\n"
                f"  config parent:   {config_path.parent}"
            )
    config = default_config(images_dir=images_dir, output_dir=run_root)
    config["run"]["data_id"] = data_id
    written = write_config(config, config_path)
    print(f"Data id: {data_id}")
    print(f"Run root: {run_root}")
    print(f"Wrote config: {written}")


def _create_masks(args: argparse.Namespace) -> None:
    from auto_labeling_tool.app.create_masks_by_manual_prompts import (
        run_sam2_lkt,
        run_sam2_prop,
        run_sam2_prop_recursive,
    )

    cfg = _load_cli_config(args)
    masking = cfg.data["masking"]
    method = str(masking["method"])
    out_dir = cfg.masks_root / cfg.run_id
    print(f"[auto-label-tool] images: {cfg.images_dir}")
    print(f"[auto-label-tool] masks:  {out_dir}")
    print(f"[auto-label-tool] method: {method}")

    if method == "sam2_prop":
        run_sam2_prop(
            cfg.images_dir,
            out_dir,
            init_idx=int(masking.get("init_idx", 0)),
            allow_multiple=bool(masking.get("multi_object", True)),
        )
    elif method == "sam2_prop_recur":
        run_sam2_prop_recursive(
            cfg.images_dir,
            out_dir,
            init_idx=int(masking.get("init_idx", 0)),
        )
    elif method == "sam2_lkt":
        run_sam2_lkt(
            cfg.images_dir,
            out_dir,
            init_idx=int(masking.get("init_idx", 0)),
        )
    else:
        raise ValueError(f"Unsupported masking.method: {method}")


def _aggregate(args: argparse.Namespace) -> None:
    from auto_labeling_tool.app.aggregate_masks import aggregate_all_episodes

    cfg = _load_cli_config(args)
    aggregation = cfg.data["aggregation"]
    debug = bool(aggregation.get("debug", False))
    debug_images_dir = aggregation.get("debug_images_dir")
    debug_rgb_root = Path(debug_images_dir).expanduser().resolve() if debug and debug_images_dir else None
    aggregate_all_episodes(
        cfg.masks_root,
        debug=debug,
        debug_rgb_root=debug_rgb_root,
        thr=float(aggregation.get("threshold", 0.5)),
        init_idx=aggregation.get("init_idx"),
    )


def _create_dataset(args: argparse.Namespace) -> None:
    from auto_labeling_tool.app.create_training_dataset_from_masks import (
        LABEL_FORMAT_CHOICES,
        _collect_samples,
        _dataset_dir_name,
        _ensure_new_dataset_root,
        _link_images,
        _log,
        _split_train_val,
        _subsample_samples,
        _write_data_yaml,
        _write_labels,
    )

    cfg = _load_cli_config(args)
    dataset = cfg.data["dataset"]
    masks_dir = cfg.masks_root
    images_dir = cfg.images_dir
    output_root = cfg.datasets_root

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

    val_ratio = float(dataset.get("val_ratio", 0.2))
    seed = int(dataset.get("seed", 42))
    sample_ratio = getattr(args, "sample_ratio", None)
    if sample_ratio is None:
        sample_ratio = dataset.get("sample_ratio")
    label_format = args.label_format or str(dataset.get("label_format", "bbox"))
    if label_format not in LABEL_FORMAT_CHOICES:
        raise ValueError(f"Unsupported dataset.label_format: {label_format}")
    samples = _subsample_samples(samples, sample_ratio=sample_ratio, seed=seed)
    _log(
        "Selected %d raw samples for splitting (sample_ratio=%s, seed=%d)"
        % (len(samples), sample_ratio, seed)
    )
    train_samples, val_samples = _split_train_val(samples, val_ratio=val_ratio, seed=seed)
    _log(
        "Performed train/val split: %d train, %d val (val_ratio=%.2f, seed=%d)"
        % (len(train_samples), len(val_samples), val_ratio, seed)
    )

    dataset_root = output_root / _dataset_dir_name(label_format)
    _ensure_new_dataset_root(dataset_root)
    _log(f"Creating dataset folders under {dataset_root}")
    _link_images(train_samples, dataset_root / "images" / "train")
    _link_images(val_samples, dataset_root / "images" / "val")
    _write_labels(train_samples, dataset_root / "labels" / "train", class_id=0, label_format=label_format)
    _write_labels(val_samples, dataset_root / "labels" / "val", class_id=0, label_format=label_format)
    _write_data_yaml(dataset_root, class_name=str(dataset.get("class_name", "bed")), nc=1)
    _log(f"Created dataset at: {dataset_root}")


def _validate(args: argparse.Namespace) -> None:
    from auto_labeling_tool.app.validate_dataset_annotations import (
        _draw_label,
        _draw_status,
        _ensure_label_file,
        _infer_label_format,
        _list_image_files,
        _prune_dataset,
        _read_blacklist,
        _sort_seq,
        _append_blacklist,
    )
    import cv2

    dataset_root: Path = args.dataset_dir.expanduser().resolve()
    split = str(args.split)
    fraction = float(args.fraction)
    max_frames = int(args.max_frames)
    wait_ms = int(args.wait_ms)
    idx = int(args.init_idx)

    if not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset_root is not a directory: {dataset_root}")
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")

    images_dir = dataset_root / "images" / split
    labels_dir = dataset_root / "labels" / split
    blacklist_path = dataset_root / "black_list.txt"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    image_paths = _list_image_files(images_dir)
    sorted_paths = _sort_seq(image_paths, images_dir)
    if fraction < 1.0:
        sorted_paths = sorted_paths[: max(1, int(len(sorted_paths) * fraction))]
    if max_frames > 0:
        sorted_paths = sorted_paths[:max_frames]
    if not sorted_paths:
        print("No frames to validate after filtering.")
        return

    blacklisted = _read_blacklist(blacklist_path)
    if getattr(args, "ensure_labels", False):
        for img_path in image_paths:
            rel = img_path.relative_to(images_dir)
            _ensure_label_file(labels_dir / rel.with_suffix(".txt"))
        print(f"Ensured labels for {len(image_paths)} images.")
        return
    if getattr(args, "prune", False):
        _prune_dataset(images_dir, labels_dir, blacklisted)
        return

    label_format = _infer_label_format(labels_dir, str(args.label_format))

    total = len(sorted_paths)
    if idx < 0 or idx >= total:
        raise ValueError(f"init-idx must be in [0, {total - 1}], got {idx}")

    window_name = f"{split} annotation review"
    while 0 <= idx < total:
        img_path = sorted_paths[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            idx += 1
            continue
        rel = img_path.relative_to(images_dir)
        rel_name = rel.name
        label_path = labels_dir / rel.with_suffix(".txt")
        _ensure_label_file(label_path)
        label = f"{rel_name} ({idx + 1}/{total})"
        if rel_name in blacklisted:
            box_color = (0, 0, 255)
            _draw_label(img, label_path, label_format, label, box_color)
            _draw_status(img, "REJECTED", box_color)
        else:
            box_color = (0, 200, 0)
            _draw_label(img, label_path, label_format, label, box_color)
            _draw_status(img, "ACCEPTED", box_color)

        cv2.imshow(window_name, img)
        key = cv2.waitKey(wait_ms) & 0xFF
        if key in (27, ord("q"), ord("Q")):
            break
        if key in (ord("x"), ord("X")):
            if rel_name not in blacklisted:
                _append_blacklist(blacklist_path, rel_name)
                blacklisted.add(rel_name)
                print(f"blacklisted: {rel_name}")
            idx += 1
            continue
        if key in (ord("a"), ord("A")):
            if rel_name in blacklisted:
                blacklisted.remove(rel_name)
                blacklist_path.write_text(
                    "\n".join(sorted(blacklisted)) + ("\n" if blacklisted else ""),
                    encoding="utf-8",
                )
                print(f"removed from blacklist: {rel_name}")
            idx += 1
            continue
        if key in (ord("j"), ord("J")):
            idx = max(0, idx - 1)
            continue
        if key in (ord("l"), ord("L")) or key == 255:
            idx += 1

    cv2.destroyAllWindows()


def _run(args: argparse.Namespace) -> None:
    steps = args.steps or ["create-masks", "aggregate-masks", "create-dataset"]
    cfg = _load_cli_config(args)
    handlers = {
        "create-masks": _create_masks,
        "aggregate-masks": _aggregate,
        "create-dataset": _create_dataset,
    }
    for step in steps:
        if step not in handlers:
            raise ValueError(f"Unsupported run step: {step}")
        print(f"[auto-label-tool] Running step: {step}")
        step_args = argparse.Namespace(**vars(args))
        step_args.skip_root_check = True
        step_args.run_id = cfg.run_id
        handlers[step](step_args)


def _completion(args: argparse.Namespace) -> None:
    print(COMPLETION_LINE)


def _install_completion(args: argparse.Namespace) -> None:
    bashrc = Path.home() / ".bashrc"
    block = f"\n{COMPLETION_MARKER}\n{COMPLETION_LINE}\n"
    if bashrc.is_file():
        text = bashrc.read_text(encoding="utf-8")
        if COMPLETION_LINE in text:
            print(f"Completion already installed in {bashrc}")
            return
    else:
        text = ""

    bashrc.write_text(text.rstrip() + block, encoding="utf-8")
    print(f"Installed completion in {bashrc}")
    print("Open a new shell or run:")
    print(f"  {COMPLETION_LINE}")


def _update_root(args: argparse.Namespace) -> None:
    _maybe_update_config_root(args.config, force=True)


def build_parser() -> argparse.ArgumentParser:
    from auto_labeling_tool.app.create_training_dataset_from_masks import LABEL_FORMAT_CHOICES

    parser = argparse.ArgumentParser(prog="auto-label-tool")
    sub = parser.add_subparsers(dest="command", required=True)

    create_config = sub.add_parser("create-config", help="Create a YAML config file.")
    create_config.add_argument(
        "-d",
        "--images-dir",
        required=True,
        type=Path,
        metavar="DIR",
        help="Required. Directory containing RGB input frames.",
    )
    create_config.add_argument(
        "-o",
        "--output-dir",
        required=True,
        type=Path,
        metavar="DIR",
        help=(
            "Required. Parent output directory. The run root becomes "
            "OUTPUT_DIR/DATA_ID."
        ),
    )
    create_config.add_argument(
        "--id",
        dest="data_id",
        default=None,
        metavar="DATA_ID",
        help=(
            "Optional data id. Default: last two --images-dir path parts joined "
            "with '__', e.g. _0/cam_0 -> _0__cam_0."
        ),
    )
    create_config.add_argument(
        "-c",
        "--config",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Optional. Config output path. Default: "
            f"OUTPUT_DIR/DATA_ID/{DEFAULT_CONFIG_NAME}."
        ),
    )
    create_config.set_defaults(func=_create_config)

    create_masks = sub.add_parser("create-masks", help="Create mask versions from RGB frames.")
    _add_config_arg(create_masks)
    create_masks.set_defaults(func=_create_masks)

    aggregate = sub.add_parser("aggregate-masks", help="Aggregate mask versions.")
    _add_config_arg(aggregate)
    aggregate.set_defaults(func=_aggregate)

    create_dataset = sub.add_parser("create-dataset", help="Create a YOLO dataset from masks.")
    _add_config_arg(create_dataset)
    create_dataset.add_argument(
        "--label-format",
        choices=LABEL_FORMAT_CHOICES,
        default=None,
        help="Override dataset.label_format for this run. Default comes from config.",
    )
    create_dataset.add_argument(
        "--sample-ratio",
        type=float,
        default=None,
        help="Override dataset.sample_ratio for this run. Default comes from config.",
    )
    create_dataset.set_defaults(func=_create_dataset)

    validate = sub.add_parser("validate", help="Review generated YOLO labels.")
    validate.add_argument(
        "-d",
        "--dataset-dir",
        required=True,
        type=Path,
        help="Dataset root containing images/ and labels/ directories.",
    )
    validate.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="train",
        help="Which split to validate.",
    )
    validate.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction of frames to validate from the sorted list (0 < fraction <= 1].",
    )
    validate.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional maximum number of frames to validate (0 disables limit).",
    )
    validate.add_argument(
        "--wait-ms",
        type=int,
        default=0,
        help="Delay per frame in ms for window playback (0 = wait for key).",
    )
    validate.add_argument(
        "--init-idx",
        type=int,
        default=0,
        help="Start index in the sorted frame list (0-based).",
    )
    validate.add_argument("--ensure-labels", action="store_true")
    validate.add_argument("--prune", action="store_true")
    validate.add_argument(
        "--label-format",
        choices=("auto", *LABEL_FORMAT_CHOICES),
        default="auto",
        help="Label geometry to draw. Default auto infers from label rows.",
    )
    validate.set_defaults(func=_validate)

    run = sub.add_parser("run", help="Run create-masks, aggregate-masks, and create-dataset.")
    _add_config_arg(run)
    run.add_argument(
        "--label-format",
        choices=LABEL_FORMAT_CHOICES,
        default=None,
        help="Override dataset.label_format when the create-dataset step runs.",
    )
    run.add_argument(
        "--sample-ratio",
        type=float,
        default=None,
        help="Override dataset.sample_ratio when the create-dataset step runs.",
    )
    run.add_argument(
        "--steps",
        nargs="+",
        choices=["create-masks", "aggregate-masks", "create-dataset"],
        help="Optional subset/order of steps to run.",
    )
    run.set_defaults(func=_run)

    completion = sub.add_parser(
        "completion",
        help="Print the Bash completion hook.",
    )
    completion.set_defaults(func=_completion)

    install_completion = sub.add_parser(
        "install-completion",
        help="Install Bash completion into ~/.bashrc.",
    )
    install_completion.set_defaults(func=_install_completion)

    update_root = sub.add_parser(
        "update-root",
        help="Update config run.root to the config file parent directory.",
    )
    update_root.add_argument(
        "-c",
        "--config",
        required=True,
        type=Path,
        help="Path to the auto-label-tool YAML config.",
    )
    update_root.set_defaults(func=_update_root)
    return parser


def main() -> None:
    parser = build_parser()
    try:
        import argcomplete
    except ImportError:
        argcomplete = None
    if argcomplete is not None:
        argcomplete.autocomplete(parser)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
