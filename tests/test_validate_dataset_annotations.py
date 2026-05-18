from __future__ import annotations

from pathlib import Path

from auto_labeling_tool.app.validate_dataset_annotations import (
    _infer_label_format,
    _load_yolo_boxes,
    _load_yolo_polygons,
)


def test_load_yolo_boxes_parses_bbox_rows(tmp_path: Path) -> None:
    label_path = tmp_path / "frame.txt"
    label_path.write_text("0 0.500000 0.500000 0.250000 0.500000\n", encoding="utf-8")

    assert _load_yolo_boxes(label_path, (100, 200)) == [(75, 25, 125, 75)]


def test_load_yolo_polygons_parses_mask_rows(tmp_path: Path) -> None:
    label_path = tmp_path / "frame.txt"
    label_path.write_text(
        "0 0.250000 0.100000 0.250000 0.500000 0.750000 0.500000 0.750000 0.100000\n",
        encoding="utf-8",
    )

    assert _load_yolo_polygons(label_path, (100, 200)) == [
        [(50, 10), (50, 50), (150, 50), (150, 10)]
    ]


def test_infer_label_format_from_bbox_label_rows(tmp_path: Path) -> None:
    labels_dir = tmp_path / "labels" / "train"
    labels_dir.mkdir(parents=True)
    (labels_dir / "frame.txt").write_text(
        "0 0.500000 0.500000 0.250000 0.500000\n",
        encoding="utf-8",
    )

    assert _infer_label_format(labels_dir) == "bbox"


def test_infer_label_format_from_mask_label_rows(tmp_path: Path) -> None:
    labels_dir = tmp_path / "labels" / "train"
    labels_dir.mkdir(parents=True)
    (labels_dir / "frame.txt").write_text(
        "0 0.250000 0.100000 0.250000 0.500000 0.750000 0.500000 0.750000 0.100000\n",
        encoding="utf-8",
    )

    assert _infer_label_format(labels_dir) == "mask"


def test_explicit_label_format_overrides_inference(tmp_path: Path) -> None:
    assert _infer_label_format(tmp_path / "missing-labels", "bbox") == "bbox"


def test_infer_label_format_rejects_mixed_label_rows(tmp_path: Path) -> None:
    labels_dir = tmp_path / "labels" / "train"
    labels_dir.mkdir(parents=True)
    (labels_dir / "bbox.txt").write_text(
        "0 0.500000 0.500000 0.250000 0.500000\n",
        encoding="utf-8",
    )
    (labels_dir / "mask.txt").write_text(
        "0 0.250000 0.100000 0.250000 0.500000 0.750000 0.500000 0.750000 0.100000\n",
        encoding="utf-8",
    )

    try:
        _infer_label_format(labels_dir)
    except ValueError as exc:
        assert "Mixed bbox and mask" in str(exc)
    else:
        raise AssertionError("expected mixed label rows to be rejected")
