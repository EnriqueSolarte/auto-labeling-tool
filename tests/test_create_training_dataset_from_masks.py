from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import auto_labeling_tool.app.create_training_dataset_from_masks as dataset_module
from auto_labeling_tool.app.create_training_dataset_from_masks import (
    Sample,
    _dataset_dir_name,
    _ensure_new_dataset_root,
    _load_mask_stack,
    _subsample_samples,
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


def _sample_ids(samples: list[Sample]) -> list[str]:
    return [sample.image_path.stem for sample in samples]


def _make_samples(root: Path, count: int) -> list[Sample]:
    return [
        Sample(
            image_path=root / "images" / f"frame_{idx:03d}.jpg",
            mask_path=root / "masks" / f"frame_{idx:03d}.npy",
        )
        for idx in range(count)
    ]


def test_dataset_dir_name_comes_from_label_format() -> None:
    assert _dataset_dir_name("bbox") == "bbx-based"
    assert _dataset_dir_name("mask") == "mask-based"


def test_subsample_none_keeps_all_samples_in_input_order(tmp_path: Path) -> None:
    samples = _make_samples(tmp_path, 5)

    selected = _subsample_samples(samples, sample_ratio=None, seed=42)

    assert selected == samples


def test_subsample_ratio_one_keeps_all_samples(tmp_path: Path) -> None:
    samples = list(reversed(_make_samples(tmp_path, 5)))

    selected = _subsample_samples(samples, sample_ratio=1.0, seed=42)

    assert set(_sample_ids(selected)) == set(_sample_ids(samples))
    assert len(selected) == len(samples)


def test_subsample_ratio_uses_ceil_for_nonzero_selection(tmp_path: Path) -> None:
    samples = _make_samples(tmp_path, 5)

    selected = _subsample_samples(samples, sample_ratio=0.01, seed=42)

    assert len(selected) == 1


def test_subsample_is_deterministic_for_same_seed(tmp_path: Path) -> None:
    samples = _make_samples(tmp_path, 8)

    first = _subsample_samples(samples, sample_ratio=0.5, seed=123)
    second = _subsample_samples(samples, sample_ratio=0.5, seed=123)

    assert _sample_ids(first) == _sample_ids(second)


def test_subsample_can_select_different_samples_for_different_seeds(tmp_path: Path) -> None:
    samples = _make_samples(tmp_path, 8)

    first = _subsample_samples(samples, sample_ratio=0.5, seed=1)
    second = _subsample_samples(samples, sample_ratio=0.5, seed=2)

    assert set(_sample_ids(first)) != set(_sample_ids(second))


def test_subsample_does_not_depend_on_input_order(tmp_path: Path) -> None:
    samples = _make_samples(tmp_path, 8)

    selected = _subsample_samples(samples, sample_ratio=0.5, seed=123)
    selected_from_reversed = _subsample_samples(list(reversed(samples)), sample_ratio=0.5, seed=123)

    assert _sample_ids(selected) == _sample_ids(selected_from_reversed)


@pytest.mark.parametrize("sample_ratio", [0.0, -0.1, 1.1])
def test_subsample_rejects_invalid_ratios(tmp_path: Path, sample_ratio: float) -> None:
    samples = _make_samples(tmp_path, 3)

    with pytest.raises(ValueError, match="sample_ratio must be in"):
        _subsample_samples(samples, sample_ratio=sample_ratio, seed=42)


def test_main_subsamples_before_train_val_split(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    samples = _make_samples(tmp_path, 5)
    split_input_lengths = []

    monkeypatch.setattr(
        dataset_module,
        "parse_args",
        lambda: dataset_module.argparse.Namespace(
            masks_dir=tmp_path / "masks",
            images_dir=tmp_path / "images",
            output_root=tmp_path / "output",
            val_ratio=0.5,
            seed=42,
            sample_ratio=0.4,
            class_name="bed",
            label_format="bbox",
        ),
    )
    monkeypatch.setattr(dataset_module, "_collect_samples", lambda _masks_dir, _images_dir: samples)
    monkeypatch.setattr(dataset_module, "_ensure_new_dataset_root", lambda _dataset_root: None)
    monkeypatch.setattr(dataset_module, "_link_images", lambda _samples, _images_dir: None)
    monkeypatch.setattr(dataset_module, "_write_labels", lambda _samples, _labels_dir, class_id, label_format: None)
    monkeypatch.setattr(dataset_module, "_write_data_yaml", lambda _root, class_name, nc=1: None)

    def split_spy(split_samples: list[Sample], val_ratio: float, seed: int) -> tuple[list[Sample], list[Sample]]:
        split_input_lengths.append(len(split_samples))
        return [], split_samples

    monkeypatch.setattr(dataset_module, "_split_train_val", split_spy)
    (tmp_path / "masks").mkdir()
    (tmp_path / "images").mkdir()

    dataset_module.main()

    assert split_input_lengths == [2]


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
