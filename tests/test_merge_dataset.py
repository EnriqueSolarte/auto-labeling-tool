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


def test_merge_dataset_creates_symlinked_dataset_and_dedupes_names(
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
    assert (output_dir / "images" / "train" / "frame.jpg").is_symlink()
    assert (output_dir / "labels" / "train" / "frame.txt").is_symlink()
    assert (output_dir / "images" / "train" / "frame__dataset_b.jpg").is_symlink()
    assert (output_dir / "labels" / "train" / "frame__dataset_b.txt").is_symlink()
    assert (output_dir / "data.yaml").read_text(encoding="utf-8") == (
        f"path: {output_dir}\n"
        "train: images/train\n"
        "nc: 1\n"
        "names: ['bed']\n"
    )


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
