from __future__ import annotations

from pathlib import Path

import numpy as np

from auto_labeling_tool.app.create_training_dataset_from_masks import (
    Sample,
    _dataset_dir_name,
    _ensure_new_dataset_root,
    _load_mask_stack,
    _write_labels,
)


def _write_mask(path: Path, mask: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, mask)
    return path


def _write_image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake image")
    return path


def test_dataset_dir_name_comes_from_label_format() -> None:
    assert _dataset_dir_name("bbox") == "bbx-based"
    assert _dataset_dir_name("mask") == "mask-based"


def test_existing_non_empty_dataset_root_is_rejected(tmp_path: Path) -> None:
    dataset_root = tmp_path / "bbx-based"
    dataset_root.mkdir()
    (dataset_root / "stale.txt").write_text("old run", encoding="utf-8")

    try:
        _ensure_new_dataset_root(dataset_root)
    except FileExistsError as exc:
        assert "not empty" in str(exc)
    else:
        raise AssertionError("expected non-empty dataset root to be rejected")


def test_write_labels_defaults_to_bbox_format(tmp_path: Path) -> None:
    mask = np.zeros((1, 6, 8), dtype=bool)
    mask[0, 1:4, 2:5] = True
    mask_path = _write_mask(tmp_path / "masks" / "frame.npy", mask)
    image_path = _write_image(tmp_path / "images" / "frame.jpg")

    _write_labels([Sample(image_path=image_path, mask_path=mask_path)], tmp_path / "labels", class_id=0)

    assert (tmp_path / "labels" / "frame.txt").read_text(encoding="utf-8") == (
        "0 0.437500 0.416667 0.375000 0.500000\n"
    )


def test_write_labels_can_explicitly_write_bbox_format(tmp_path: Path) -> None:
    mask = np.zeros((1, 6, 8), dtype=bool)
    mask[0, 1:4, 2:5] = True
    mask_path = _write_mask(tmp_path / "masks" / "frame.npy", mask)
    image_path = _write_image(tmp_path / "images" / "frame.jpg")

    _write_labels(
        [Sample(image_path=image_path, mask_path=mask_path)],
        tmp_path / "labels",
        class_id=0,
        label_format="bbox",
    )

    assert (tmp_path / "labels" / "frame.txt").read_text(encoding="utf-8") == (
        "0 0.437500 0.416667 0.375000 0.500000\n"
    )


def test_write_labels_writes_rectangular_mask_polygon(tmp_path: Path) -> None:
    mask = np.zeros((1, 6, 8), dtype=bool)
    mask[0, 1:4, 2:5] = True
    mask_path = _write_mask(tmp_path / "masks" / "frame.npy", mask)
    image_path = _write_image(tmp_path / "images" / "frame.jpg")

    _write_labels(
        [Sample(image_path=image_path, mask_path=mask_path)],
        tmp_path / "labels",
        class_id=0,
        label_format="mask",
    )

    row = (tmp_path / "labels" / "frame.txt").read_text(encoding="utf-8").strip()
    parts = row.split()
    assert parts[0] == "0"
    assert len(parts) == 9
    points = list(zip([float(value) for value in parts[1::2]], [float(value) for value in parts[2::2]]))
    assert set(points) == {
        (0.250000, 0.166667),
        (0.250000, 0.500000),
        (0.500000, 0.500000),
        (0.500000, 0.166667),
    }


def test_stacked_mask_writes_one_segmentation_row_per_slice(tmp_path: Path) -> None:
    mask = np.zeros((2, 8, 8), dtype=bool)
    mask[0, 1:4, 1:4] = True
    mask[1, 4:7, 4:7] = True
    mask_path = _write_mask(tmp_path / "masks" / "frame.npy", mask)
    image_path = _write_image(tmp_path / "images" / "frame.jpg")

    _write_labels(
        [Sample(image_path=image_path, mask_path=mask_path)],
        tmp_path / "labels",
        class_id=0,
        label_format="mask",
    )

    rows = (tmp_path / "labels" / "frame.txt").read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 2


def test_empty_mask_stack_is_skipped(tmp_path: Path) -> None:
    mask = np.zeros((2, 6, 8), dtype=bool)
    mask_path = _write_mask(tmp_path / "masks" / "frame.npy", mask)
    image_path = _write_image(tmp_path / "images" / "frame.jpg")

    _write_labels(
        [Sample(image_path=image_path, mask_path=mask_path)],
        tmp_path / "labels",
        class_id=0,
        label_format="mask",
    )

    assert not (tmp_path / "labels" / "frame.txt").exists()


def test_invalid_mask_shape_is_skipped(tmp_path: Path) -> None:
    mask_path = _write_mask(tmp_path / "masks" / "frame.npy", np.zeros((2, 3, 4, 5), dtype=bool))
    image_path = _write_image(tmp_path / "images" / "frame.jpg")

    mask_stack, skip_reason = _load_mask_stack(mask_path)
    assert mask_stack is None
    assert "invalid mask shape" in str(skip_reason)

    _write_labels(
        [Sample(image_path=image_path, mask_path=mask_path)],
        tmp_path / "labels",
        class_id=0,
        label_format="mask",
    )

    assert not (tmp_path / "labels" / "frame.txt").exists()


def test_mask_label_output_is_deterministic(tmp_path: Path) -> None:
    mask = np.zeros((1, 6, 8), dtype=bool)
    mask[0, 1:4, 2:5] = True
    mask_path = _write_mask(tmp_path / "masks" / "frame.npy", mask)
    image_path = _write_image(tmp_path / "images" / "frame.jpg")
    sample = Sample(image_path=image_path, mask_path=mask_path)

    _write_labels([sample], tmp_path / "labels_a", class_id=0, label_format="mask")
    _write_labels([sample], tmp_path / "labels_b", class_id=0, label_format="mask")

    assert (tmp_path / "labels_a" / "frame.txt").read_text(encoding="utf-8") == (
        tmp_path / "labels_b" / "frame.txt"
    ).read_text(encoding="utf-8")
