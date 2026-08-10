"""Efficient, portable file linking helpers."""
from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LinkFallbackAuthorizer:
    """Ask once per process before allowing each less-efficient fallback."""

    symlink_approved: bool | None = None
    copy_approved: bool | None = None

    @staticmethod
    def _ask(prompt: str) -> bool:
        if not sys.stdin.isatty():
            raise RuntimeError(
                "A file-link fallback requires confirmation, but standard input is not interactive."
            )
        return input(f"{prompt} [y/N]: ").strip().lower() in {"y", "yes"}

    def allow_symlink(self, src: Path, dst: Path, error: OSError) -> bool:
        if self.symlink_approved is None:
            self.symlink_approved = self._ask(
                "WARNING: hard link is unavailable for "
                f"{src} -> {dst} ({error}). Fall back to a symlink?"
            )
        return self.symlink_approved

    def allow_copy(self, src: Path, dst: Path, error: OSError) -> bool:
        if self.copy_approved is None:
            self.copy_approved = self._ask(
                "WARNING: symlink is unavailable for "
                f"{src} -> {dst} ({error}). Fall back to copying the file?"
            )
        return self.copy_approved


_DEFAULT_AUTHORIZER = LinkFallbackAuthorizer()


def hardlink_or_symlink_or_copy(
    src: Path,
    dst: Path,
    *,
    authorizer: LinkFallbackAuthorizer | None = None,
) -> str:
    """Create ``dst`` from ``src``, preferring a hard link.

    Hard links avoid duplicating data. Less-efficient fallbacks require an
    interactive confirmation once per process.
    """
    authorizer = authorizer or _DEFAULT_AUTHORIZER
    source = src.resolve()
    try:
        os.link(source, dst)
        return "hardlink"
    except OSError as hardlink_error:
        if not authorizer.allow_symlink(source, dst, hardlink_error):
            raise RuntimeError(
                f"Hard link unavailable for {source}; symlink fallback was declined."
            ) from hardlink_error
    try:
        dst.symlink_to(source)
        return "symlink"
    except OSError as symlink_error:
        if not authorizer.allow_copy(source, dst, symlink_error):
            raise RuntimeError(
                f"Symlink unavailable for {source}; copy fallback was declined."
            ) from symlink_error
        shutil.copy2(source, dst)
        return "copy"
