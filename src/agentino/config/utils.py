"""Config utilities — YAML loading, env var resolution, markdown parsing."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any


def resolve_env_vars(text: str) -> str:
    """Replace ${VAR_NAME} with environment variable values."""

    def _replace(match: re.Match) -> str:
        var = match.group(1)
        return os.environ.get(var, match.group(0))

    return re.sub(r"\$\{(\w+)\}", _replace, text)


def resolve_config_values(obj: Any) -> Any:
    """Recursively resolve environment variables in config values."""
    if isinstance(obj, str):
        return resolve_env_vars(obj)
    if isinstance(obj, dict):
        return {k: resolve_config_values(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [resolve_config_values(item) for item in obj]
    return obj


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file with env var resolution."""
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML required: pip install pyyaml")
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return resolve_config_values(data)


def parse_instructions_md(text: str) -> str:
    """Parse SOUL.md / instructions file — strip YAML frontmatter, return content."""
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            return parts[2].strip()
    return text.strip()
