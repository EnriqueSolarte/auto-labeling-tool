from __future__ import annotations

import argparse
from pathlib import Path

from auto_labeling_tool import config as config_module
from auto_labeling_tool.app import cli


def test_create_config_run_invokes_pipeline_with_created_config(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    monkeypatch.setattr(config_module, "get_config_version", lambda: "vtest")
    captured_run_args: list[argparse.Namespace] = []
    monkeypatch.setattr(cli, "_run", captured_run_args.append)

    images_dir = tmp_path / "raw" / "images"
    images_dir.mkdir(parents=True)
    (images_dir / "frame_000.jpg").write_bytes(b"fake image")
    output_dir = tmp_path / "runs"

    cli._create_config(
        argparse.Namespace(
            images_dir=images_dir,
            output_dir=output_dir,
            data_id=None,
            config=None,
            run=True,
        )
    )

    written = output_dir / "raw__images" / "cfg.yaml"
    assert written.is_file()
    assert len(captured_run_args) == 1
    assert captured_run_args[0].config == written.resolve()
    assert captured_run_args[0].update_root is False
    assert captured_run_args[0].label_format is None
    assert captured_run_args[0].sample_ratio is None
    assert captured_run_args[0].steps is None

    stdout = capsys.readouterr().out
    assert "Running pipeline with created config" in stdout
    assert "Namespace(" not in stdout
