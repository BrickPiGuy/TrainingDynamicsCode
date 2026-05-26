from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["_config_path"] = str(config_path)
    config["_repo_root"] = str(_repo_root_from_config(config_path))
    return config


def repo_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return Path(config["_repo_root"]) / path


def _repo_root_from_config(config_path: Path) -> Path:
    for parent in [config_path.parent, *config_path.parents]:
        if (parent / "experiment").exists() and (parent / "scripts").exists():
            return parent
    return Path.cwd().resolve()
