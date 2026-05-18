from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - exercised only in missing envs.
    raise RuntimeError(
        "PyYAML is required for config files. Install with `pip install -e .` "
        "or add `PyYAML>=6.0` to your environment."
    ) from exc


DEFAULT_CONFIG_NAME = "cfg.yaml"


def get_config_version() -> str:
    try:
        return f"v{version('auto-labeling-tool')}"
    except PackageNotFoundError:
        raise RuntimeError(
            "Cannot determine config version because package metadata for "
            "`auto-labeling-tool` is not installed. Run `pip install -e .` "
            "from the repository root."
        ) from None


def _assert_config_version(data: dict[str, Any], path: Path) -> None:
    expected = get_config_version()
    actual = data.get("version")
    if actual != expected:
        raise ValueError(
            f"Config version mismatch in {path}: expected {expected}, got {actual!r}. "
            "Regenerate the config with `auto-label-tool create-config` or update it "
            "to match the installed package version."
        )


def default_config(images_dir: Path, output_dir: Path) -> dict[str, Any]:
    return {
        "version": get_config_version(),
        "run": {
            "root": str(output_dir),
        },
        "paths": {
            "images_dir": str(images_dir),
            "masks_root": "{run.root}/masks",
            "datasets_root": "{run.root}/datasets",
            "dataset_root": "latest",
        },
        "masking": {
            "method": "sam2_prop",
            "init_idx": 0,
            "multi_object": True,
        },
        "aggregation": {
            "threshold": 0.5,
            "init_idx": None,
            "debug": False,
            "debug_images_dir": "{paths.images_dir}",
        },
        "dataset": {
            "class_name": "bed",
            "label_format": "bbox",
            "val_ratio": 0.2,
            "seed": 42,
        },
    }


def write_config(config: dict[str, Any], path: Path) -> Path:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return path


def load_raw_config(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return data


def _merge_defaults(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_defaults(merged[key], value)
        else:
            merged[key] = value
    return merged


def _flatten(data: dict[str, Any], prefix: str = "") -> dict[str, str]:
    values: dict[str, str] = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            values.update(_flatten(value, name))
        elif value is not None:
            values[name] = str(value)
    return values


_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z0-9_.-]+)\}")


def _interpolate_value(value: Any, values: dict[str, str]) -> Any:
    if isinstance(value, str):
        return _PLACEHOLDER_RE.sub(lambda match: values.get(match.group(1), match.group(0)), value)
    if isinstance(value, dict):
        return {key: _interpolate_value(item, values) for key, item in value.items()}
    if isinstance(value, list):
        return [_interpolate_value(item, values) for item in value]
    return value


def _interpolate(data: dict[str, Any]) -> dict[str, Any]:
    resolved = deepcopy(data)
    for _ in range(5):
        values = _flatten(resolved)
        next_resolved = _interpolate_value(resolved, values)
        if next_resolved == resolved:
            return resolved
        resolved = next_resolved
    return resolved


def _path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _latest_child_dir(root: Path) -> Path:
    if not root.is_dir():
        raise FileNotFoundError(f"Directory not found while resolving latest dataset: {root}")
    children = [p for p in root.iterdir() if p.is_dir()]
    if not children:
        raise FileNotFoundError(f"No dataset directories found under: {root}")
    return max(children, key=lambda p: p.stat().st_mtime)


@dataclass(frozen=True)
class ToolConfig:
    source_path: Path
    data: dict[str, Any]
    _run_id: str

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def images_dir(self) -> Path:
        return _path(self.data["paths"]["images_dir"])

    @property
    def masks_root(self) -> Path:
        return _path(self.data["paths"]["masks_root"])

    @property
    def datasets_root(self) -> Path:
        return _path(self.data["paths"]["datasets_root"])

    @property
    def dataset_root(self) -> Path:
        raw = str(self.data["paths"].get("dataset_root", "latest"))
        if raw == "latest":
            return _latest_child_dir(self.datasets_root)
        return _path(raw)


def load_config(path: Path, run_id: str | None = None) -> ToolConfig:
    raw = load_raw_config(path)
    _assert_config_version(raw, path)
    defaults = default_config(
        Path(raw.get("paths", {}).get("images_dir", ".")),
        Path(raw.get("run", {}).get("root", raw.get("paths", {}).get("output_dir", "."))),
    )
    merged = _merge_defaults(defaults, raw)
    resolved = _interpolate(merged)
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ToolConfig(source_path=path.expanduser().resolve(), data=resolved, _run_id=run_id)
