"""YAML-based configuration loader utilities for the ACP runtime."""

from pathlib import Path
from typing import Any, Dict

import yaml


class ConfigError(RuntimeError):
    """Raised when configuration documents are missing or malformed."""


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML configuration file and return its mapping representation."""

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        try:
            data = yaml.safe_load(handle) or {}
        except yaml.YAMLError as exc:  # pragma: no cover - rewrap for clarity
            raise ConfigError(f"Failed to parse YAML config: {config_path}") from exc

    if not isinstance(data, dict):
        raise ConfigError("Configuration document must be a mapping at the top level.")

    return data
