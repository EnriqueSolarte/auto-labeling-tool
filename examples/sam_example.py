#!/usr/bin/env python3
"""
Example script that runs a Segment Anything Model 2 (SAM 2) prediction using
utils.* helpers.

This module is intended as a runnable example, separate from the pytest
suite. The tests import helpers from here where needed.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from utils import get_segmentation, load_sam_model, overlay_mask

DEFAULT_IMAGE_PATH = Path(__file__).resolve().parent.parent / "assets" / "image.jpg"


def _find_sam_checkpoint() -> Path | None:
    """
    Try to locate a SAM 2 checkpoint for integration-style runs.

    Priority:
      1. SAM_CHECKPOINT environment variable, if it points to an existing file.
      2. The default path used by the CLI (/models/sam2/sam2.1_hiera_tiny.pt), if it exists.
    """
    env_path = os.getenv("SAM_CHECKPOINT")
    if env_path:
        candidate = Path(env_path).expanduser()
        if candidate.exists():
            return candidate

    default = Path("/models/sam2/sam2.1_hiera_tiny.pt")
    if default.exists():
        return default

    return None


def _resolve_model_config(argument: str | None) -> str:
    """
    Resolve the requested SAM 2 Hydra config, defaulting to the tiny model.
    """
    if argument:
        return argument
    env_config = os.getenv("SAM_CONFIG")
    if env_config:
        return env_config
    default = Path("/models/sam2/sam2.1_hiera_t.yaml")
    if default.exists():
        return str(default)
    return "configs/sam2.1/sam2.1_hiera_t.yaml"


def _parse_points(points_str: str):
    """
    Parse a string formatted as "x1,y1 x2,y2 ..." into a list of tuples.
    """
    if not points_str:
        raise ValueError("You must provide at least one keypoint via --points.")
    coords = []
    for token in points_str.split():
        try:
            x_str, y_str = token.split(",")
        except ValueError as exc:
            raise ValueError(
                f"Failed to parse keypoint '{token}'. Expected format 'x,y'."
            ) from exc
        coords.append((float(x_str), float(y_str)))
    return coords


def _save_mask(mask: np.ndarray, destination: Path) -> None:
    """
    Persist the boolean mask as a grayscale 0/255 PNG for quick visualization.
    """
    img = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    img.save(destination)


def _resolve_output_path(path_argument: str | None, image_path: Path, filename: str) -> Path:
    """Return the requested path or default to the image directory."""
    if path_argument:
        return Path(path_argument).expanduser().resolve()
    return image_path.with_name(filename).resolve()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quick SAM 2 segmentation sanity check.")
    parser.add_argument(
        "--image",
        default=str(DEFAULT_IMAGE_PATH),
        help=f"Path to the RGB image to segment (default: {DEFAULT_IMAGE_PATH}).",
    )
    parser.add_argument(
        "--checkpoint",
        default="/models/sam2/sam2.1_hiera_tiny.pt",
        help="Path to the SAM 2 .pt checkpoint.",
    )
    parser.add_argument(
        "--model-config",
        default="configs/sam2.1/sam2.1_hiera_t.yaml",
        help=(
            "Hydra config name or YAML path for the SAM 2 checkpoint "
            "(e.g., 'configs/sam2.1/sam2.1_hiera_t.yaml' or /models/sam2/*.yaml). "
            "Defaults to $SAM_CONFIG, otherwise /models/sam2/sam2.1_hiera_t.yaml when present."
        ),
    )
    parser.add_argument(
        "--points",
        default="400,200 400,230 400,300",
        help="Space-delimited UV points 'x,y'. Example: \"320,200 360,230\"",
    )
    parser.add_argument(
        "--output-mask",
        default=None,
        help="Destination PNG file for the predicted mask (defaults to <image-dir>/mask.png).",
    )
    parser.add_argument(
        "--output-overlay",
        default=None,
        help="Destination image for the RGB visualization (defaults to overlay_<image-name>).",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device to run on (e.g., 'cuda', 'cuda:0', 'cpu'). Defaults to cuda:0.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    image_path = Path(args.image).expanduser().resolve()

    points = _parse_points(args.points)
    predictor = load_sam_model(
        checkpoint_path=args.checkpoint,
        model_config=_resolve_model_config(args.model_config),
        device=args.device,
    )
    mask, score = get_segmentation(
        image=str(image_path),
        keypoints_uv=points,
        predictor=predictor,
        point_labels=None,
        multimask_output=False,
    )

    destination = _resolve_output_path(args.output_mask, image_path, "mask.png")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _save_mask(mask, destination)

    overlay_path = _resolve_output_path(
        args.output_overlay,
        image_path,
        f"overlay_{image_path.name}",
    )
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    composite = overlay_mask(str(image_path), mask)
    Image.fromarray(composite).save(overlay_path)
    print(f"Saved overlay to {overlay_path}")

    print(f"Saved mask to {destination} (score={score:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
