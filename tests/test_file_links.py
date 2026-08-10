import os

from auto_labeling_tool.utils.file_links import (
    LinkFallbackAuthorizer,
    hardlink_or_symlink_or_copy,
)


def test_prefers_hard_links(tmp_path):
    source = tmp_path / "source.txt"
    destination = tmp_path / "destination.txt"
    source.write_text("mask data", encoding="utf-8")

    result = hardlink_or_symlink_or_copy(source, destination)

    assert result == "hardlink"
    assert os.stat(source).st_ino == os.stat(destination).st_ino
    assert destination.read_text(encoding="utf-8") == "mask data"


def test_requests_confirmation_before_symlink_fallback(tmp_path, monkeypatch):
    source = tmp_path / "source.txt"
    destination = tmp_path / "destination.txt"
    source.write_text("mask data", encoding="utf-8")
    monkeypatch.setattr(
        "auto_labeling_tool.utils.file_links.os.link",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("cross-device")),
    )
    monkeypatch.setattr(
        "auto_labeling_tool.utils.file_links.sys.stdin.isatty", lambda: True
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: "yes")

    result = hardlink_or_symlink_or_copy(
        source, destination, authorizer=LinkFallbackAuthorizer()
    )

    assert result == "symlink"
    assert destination.is_symlink()
