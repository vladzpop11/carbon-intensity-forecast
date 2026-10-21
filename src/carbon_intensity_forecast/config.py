"""The project's settings (configs/config.yaml, with the weather settings it points to) and
paths relative to the project root."""

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def project_path(relative: str | Path) -> Path:
    return ROOT / relative


def load_config(path: str | Path = "configs/config.yaml") -> dict:
    cfg = yaml.safe_load(project_path(path).read_text())
    cfg["weather"] = json.loads(project_path(cfg["data"]["weather"]).read_text())
    return cfg
