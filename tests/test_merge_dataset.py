from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from auto_labeling_tool.app import merge_dataset


class FrozenDatetime(datetime):
    @classmethod
    def now(cls) -> "FrozenDatetime":
        return cls(2026, 5, 19, 4, 0, 0)


def _write_dataset(root: Path, name: str, image_name: str = "frame.jpg") -> Path:
    dataset_dir = root / name
    image_dir = dataset_dir / "images" / "train"
    label_dir = dataset_dir / "labels" / "train"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    (image_dir / image_name).write_bytes(b"fake image")
    (label_dir / Path(image_name).with_suffix(".txt")).write_text(
        "0 0.500000 0.500000 0.250000 0.250000\n",
        encoding="utf-8",
    )
    (dataset_dir / "data.yaml").write_text(
        f"path: {dataset_dir}\ntrain: images/train\nnc: 1\nnames: ['bed']\n",
        encoding="utf-8",
    )
    return dataset_dir


def test_merge_dataset_creates_hardlinked_dataset_and_dedupes_names(
    monkeypatch, tmp_path: Path
) -> None:
    dataset_a = _write_dataset(tmp_path / "inputs", "dataset_a")
    dataset_b = _write_dataset(tmp_path / "inputs", "dataset_b")
    output_root = tmp_path / "merged"
    monkeypatch.setattr(
        merge_dataset,
        "_parse_args",
        lambda: argparse.Namespace(
            datasets=[dataset_a, dataset_b],
            output_root=output_root,
            dry_run=False,
        ),
    )
    monkeypatch.setattr(merge_dataset, "datetime", FrozenDatetime)

    merge_dataset.main()

    output_dir = output_root / "merge__20260519_040000"
    output_image = output_dir / "images" / "train" / "frame.jpg"
    output_label = output_dir / "labels" / "train" / "frame.txt"
    duplicate_image = output_dir / "images" / "train" / "frame__dataset_b.jpg"
    duplicate_label = output_dir / "labels" / "train" / "frame__dataset_b.txt"
    assert not output_image.is_symlink()
    assert not output_label.is_symlink()
    assert output_image.samefile(dataset_a / "images" / "train" / "frame.jpg")
    assert output_label.samefile(dataset_a / "labels" / "train" / "frame.txt")
    assert duplicate_image.samefile(dataset_b / "images" / "train" / "frame.jpg")
    assert duplicate_label.samefile(dataset_b / "labels" / "train" / "frame.txt")
    assert (output_dir / "data.yaml").read_text(encoding="utf-8") == (
        f"path: {output_dir}\n"
        "train: images/train\n"
        "nc: 1\n"
        "names: ['bed']\n"
    )


def test_merge_dataset_hardlinks_resolved_sources(monkeypatch, tmp_path: Path) -> None:
    source_image = tmp_path / "source" / "frame.jpg"
    source_label = tmp_path / "source" / "frame.txt"
    source_image.parent.mkdir()
    source_image.write_bytes(b"fake image")
    source_label.write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")

    intermediate_image = tmp_path / "intermediate" / "frame.jpg"
    intermediate_label = tmp_path / "intermediate" / "frame.txt"
    intermediate_image.parent.mkdir()
    intermediate_image.symlink_to(source_image)
    intermediate_label.symlink_to(source_label)

    dataset = tmp_path / "inputs" / "dataset_a"
    (dataset / "images" / "train").mkdir(parents=True)
    (dataset / "labels" / "train").mkdir(parents=True)
    (dataset / "images" / "train" / "frame.jpg").symlink_to(intermediate_image)
    (dataset / "labels" / "train" / "frame.txt").symlink_to(intermediate_label)
    (dataset / "data.yaml").write_text(
        f"path: {dataset}\ntrain: images/train\nnc: 1\nnames: ['bed']\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "merged"
    monkeypatch.setattr(
        merge_dataset,
        "_parse_args",
        lambda: argparse.Namespace(
            datasets=[dataset],
            output_root=output_root,
            dry_run=False,
        ),
    )
    monkeypatch.setattr(merge_dataset, "datetime", FrozenDatetime)

    merge_dataset.main()

    output_dir = output_root / "merge__20260519_040000"
    assert (output_dir / "images" / "train" / "frame.jpg").samefile(source_image)
    assert (output_dir / "labels" / "train" / "frame.txt").samefile(source_label)


def test_merge_dataset_dry_run_creates_no_output(monkeypatch, tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path / "inputs", "dataset_a")
    output_root = tmp_path / "merged"
    monkeypatch.setattr(
        merge_dataset,
        "_parse_args",
        lambda: argparse.Namespace(
            datasets=[dataset],
            output_root=output_root,
            dry_run=True,
        ),
    )

    merge_dataset.main()

    assert not output_root.exists()
