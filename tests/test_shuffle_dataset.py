from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from auto_labeling_tool.app import shuffle_dataset


class FrozenDatetime(datetime):
    @classmethod
    def now(cls) -> "FrozenDatetime":
        return cls(2026, 5, 19, 4, 0, 0)


def test_shuffle_dataset_hardlinks_nested_symlinks_to_resolved_sources(
    monkeypatch, tmp_path: Path
) -> None:
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

    dataset = tmp_path / "dataset"
    (dataset / "images" / "train").mkdir(parents=True)
    (dataset / "labels" / "train").mkdir(parents=True)
    (dataset / "images" / "train" / "frame.jpg").symlink_to(intermediate_image)
    (dataset / "labels" / "train" / "frame.txt").symlink_to(intermediate_label)

    monkeypatch.setattr(
        shuffle_dataset,
        "_parse_args",
        lambda: argparse.Namespace(
            dataset=dataset,
            ratios=[1.0, 0.0],
            seed=42,
            dry_run=False,
        ),
    )
    monkeypatch.setattr(shuffle_dataset, "datetime", FrozenDatetime)

    shuffle_dataset.main()

    assert (dataset / "images" / "train" / "frame.jpg").samefile(source_image)
    assert (dataset / "labels" / "train" / "frame.txt").samefile(source_label)
