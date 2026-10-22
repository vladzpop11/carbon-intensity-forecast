"""The project's settings (configs/config.yaml, with the weather settings it points to) and
paths relative to the project root."""

import json
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def project_path(relative: str | Path) -> Path:
    return ROOT / relative


def load_config(path: str | Path = "configs/config.yaml") -> dict:
    cfg = yaml.safe_load(project_path(path).read_text())
    cfg["weather"] = json.loads(project_path(cfg["data"]["weather"]).read_text())
    return cfg


def env_value(name: str) -> str:
    """A secret from the environment, else from .env as `make` reads it (KEY=value lines,
    optional quotes or `export`). Never printed."""
    if os.environ.get(name):
        return os.environ[name]
    env_file = project_path(".env")
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            key, sep, value = line.removeprefix("export ").partition("=")
            if sep and key.strip() == name:
                return value.strip().strip("'\"")
    raise KeyError(f"{name} is not set (environment or .env)")
