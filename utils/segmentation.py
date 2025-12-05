#!/usr/bin/env python3
"""
Helpers to generate segmentation masks from the Segment Anything Model 2 (SAM 2).

Use load_sam_model() once to create a predictor, then reuse get_segmentation()
for multiple images by passing that predictor alongside UV keypoints.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

ImageSource = Union[str, Path, np.ndarray, Image.Image]
Point = Union[Tuple[float, float], Tuple[int, int], np.ndarray]

SAM2_DEFAULT_DIR = Path("/models/sam2")
DEFAULT_SAM2_CONFIG = SAM2_DEFAULT_DIR / "configs/sam2.1/sam2.1_hiera_l.yaml"
DEFAULT_SAM2_CHECKPOINT = SAM2_DEFAULT_DIR / "sam2.1_hiera_large.pt"


def _ensure_sam2_predictor():
    """Import the SAM 2 modules lazily to avoid a hard dependency at import time."""
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError as exc:
        raise ImportError(
            "SAM-2 is not installed. Install it with "
            "`pip install git+https://github.com/facebookresearch/sam2.git`."
        ) from exc
    return SAM2ImagePredictor, build_sam2


def _default_model_config() -> str:
    """Return the default SAM 2 config path."""
    return str(DEFAULT_SAM2_CONFIG)


def _apply_postprocessing_overrides(cfg) -> None:
    """Mimic the postprocessing tweaks from sam2.build_sam.build_sam2()."""
    from omegaconf import OmegaConf, open_dict

    OmegaConf.set_struct(cfg, False)
    with open_dict(cfg):
        if "model" not in cfg or cfg.model is None:
            raise ValueError("Invalid SAM 2 config: missing 'model' section.")
        model_cfg = cfg.model
        with open_dict(model_cfg):
            if (
                "sam_mask_decoder_extra_args" not in model_cfg
                or model_cfg.sam_mask_decoder_extra_args is None
            ):
                model_cfg.sam_mask_decoder_extra_args = {}
            decoder_cfg = model_cfg.sam_mask_decoder_extra_args
            with open_dict(decoder_cfg):
                decoder_cfg.dynamic_multimask_via_stability = True
                decoder_cfg.dynamic_multimask_stability_delta = 0.05
                decoder_cfg.dynamic_multimask_stability_thresh = 0.98


def _build_sam2_from_local_config(
    config_path: Path,
    checkpoint: Path,
    device: Optional[str],
    **build_kwargs: Any,
):
    """Instantiate SAM 2 using a YAML file stored outside the package configs."""
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    import torch

    cfg = OmegaConf.load(str(config_path))

    apply_postprocessing = bool(build_kwargs.pop("apply_postprocessing", True))
    mode = build_kwargs.pop("mode", "eval")
    hydra_overrides_extra = build_kwargs.pop("hydra_overrides_extra", None)
    if hydra_overrides_extra:
        raise ValueError(
            "hydra_overrides_extra is not supported when using an explicit config path."
        )
    if build_kwargs:
        unsupported = ", ".join(build_kwargs.keys())
        raise ValueError(f"Unsupported build kwargs for local config: {unsupported}")

    if apply_postprocessing:
        _apply_postprocessing_overrides(cfg)

    model = instantiate(cfg.model, _recursive_=True)

    state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)["model"]
    missing_keys, unexpected_keys = model.load_state_dict(state)
    if missing_keys:
        raise RuntimeError(f"Missing checkpoint keys: {missing_keys}")
    if unexpected_keys:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected_keys}")

    if device is not None:
        model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def _coerce_image(image: ImageSource) -> np.ndarray:
    """Convert different image sources to an RGB uint8 numpy array."""
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")
    if isinstance(image, Image.Image):
        image = np.asarray(image)
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy array, PIL image, or path-like object")
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(
            "image array must have shape (H, W, 3) or (H, W, 4) with channels-last layout"
        )
    if image.shape[2] == 4:
        image = image[..., :3]
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating):
            image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    return image


def _prepare_points(points_uv: Sequence[Point], image_shape: Tuple[int, int]) -> np.ndarray:
    """Validate and convert UV keypoints to the format SAM 2 expects."""
    if not points_uv:
        raise ValueError("At least one keypoint is required to seed SAM.")

    coords = []
    width, height = image_shape[1], image_shape[0]
    for u, v in points_uv:
        coords.append(
            (
                float(np.clip(u, 0, width - 1)),
                float(np.clip(v, 0, height - 1)),
            )
        )
    return np.asarray(coords, dtype=np.float32)


def load_sam_model(
    checkpoint_path: Union[str, Path],
    model_config: Optional[str] = None,
    device: Optional[str] = None,
    **build_kwargs: Any,
):
    """
    Load a SAM 2 checkpoint and return a configured SAM2ImagePredictor instance.

    Args:
        checkpoint_path: Filesystem path to the SAM 2 .pt checkpoint.
        model_config: Either a Hydra config name (e.g., "configs/sam2.1/sam2.1_hiera_l.yaml")
            or an absolute path to a YAML config file. Defaults to the large SAM 2.1 config
            under `/models/sam2/configs/sam2.1/sam2.1_hiera_l.yaml`.
        device: Optional torch device string (e.g., "cuda", "cuda:0", "cpu").
        **build_kwargs: Extra keyword arguments forwarded to sam2.build_sam.build_sam2.
    """

    SAM2ImagePredictor, build_sam2 = _ensure_sam2_predictor()

    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"SAM 2 checkpoint not found at {checkpoint}")

    if model_config is None:
        model_config = _default_model_config()

    config_path = Path(model_config)
    if config_path.exists():
        sam_model = _build_sam2_from_local_config(
            config_path=config_path,
            checkpoint=checkpoint,
            device=device,
            **build_kwargs,
        )
    else:
        build_params = dict(config_file=model_config, ckpt_path=str(checkpoint))
        if device is not None:
            build_params["device"] = device
        build_params.update(build_kwargs)
        sam_model = build_sam2(**build_params)
    return SAM2ImagePredictor(sam_model)


def get_segmentation(
    image: ImageSource,
    keypoints_uv: Sequence[Point],
    predictor: Any,
    point_labels: Optional[Sequence[int]] = None,
    multimask_output: bool = False,
) -> Tuple[np.ndarray, float]:
    """
    Segment an object in an image using SAM and UV keypoints as positive prompts.

    Args:
        image: The source image. Accepts numpy arrays, PIL images, or filesystem paths.
        keypoints_uv: Iterable of UV (x, y) pixel coordinates that should lie on the target.
        predictor: A SAM2ImagePredictor instance (see load_sam_model()) already loaded on the desired device.
        point_labels: Optional labels (1=positive, 0=negative) per keypoint. Defaults to all ones.
        multimask_output: Whether to return multiple mask proposals from SAM. When False, only the
            highest-scoring mask is returned.

    Returns:
        (mask, score) tuple where mask is a boolean array with shape (H, W) and score is the
        SAM confidence score for the returned mask.
    """

    image_np = _coerce_image(image)

    point_coords = _prepare_points(keypoints_uv, image_np.shape[:2])
    if point_labels is None:
        point_labels = np.ones(len(point_coords), dtype=np.int32)
    else:
        if len(point_labels) != len(point_coords):
            raise ValueError("point_labels must match the length of keypoints_uv.")
        point_labels = np.asarray(point_labels, dtype=np.int32)

    predictor.set_image(image_np)

    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=multimask_output,
    )

    best_idx = int(np.argmax(scores))
    best_mask = masks[best_idx] > 0.0
    best_score = float(scores[best_idx])
    return best_mask, best_score


def overlay_mask(
    image: ImageSource,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.5,
) -> np.ndarray:
    """Overlay a segmentation mask on an image for quick visualization.

    Args:
        image: Source image (path, PIL image, or numpy array) to overlay onto.
        mask: Boolean or 0/1 array with shape (H, W) defining the region of interest.
        color: RGB tuple for the mask tint. Defaults to red.
        alpha: Blend factor in [0, 1]; higher values emphasize the mask.

    Returns:
        Numpy array with the same shape as the input image showing the overlay.
    """

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")

    base = _coerce_image(image)
    if mask.ndim != 2:
        raise ValueError("mask must be a 2D array")
    if mask.shape != base.shape[:2]:
        raise ValueError("mask shape must match the image height/width")

    overlay_color = np.asarray(color, dtype=np.float32)
    if overlay_color.shape != (3,):
        raise ValueError("color must be a 3-element RGB tuple")

    mask_bool = mask.astype(bool)
    if not mask_bool.any():
        return base.copy()

    result = base.astype(np.float32)
    overlay = result.copy()
    overlay[mask_bool] = (
        (1.0 - alpha) * result[mask_bool] + alpha * overlay_color
    )

    return overlay.astype(np.uint8)


__all__ = ["load_sam_model", "get_segmentation", "overlay_mask"]
