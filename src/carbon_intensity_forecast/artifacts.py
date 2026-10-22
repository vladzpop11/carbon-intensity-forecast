"""Small helpers for the files the pipeline writes."""

import json
import subprocess
from pathlib import Path

import yaml


def write_json(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n")


def provenance() -> dict:
    """The git commit, whether the code that matters differs from it, and the data version."""

    def git(*args: str) -> str:
        done = subprocess.run(["git", *args], capture_output=True, text=True,
                              cwd=_project_path("."), check=True)  # fmt: skip
        return done.stdout.strip()

    commit = git("rev-parse", "HEAD")
    # What reproduces a model: the code, the configuration and the locked dependencies (the
    # data version is data_md5; DVC pointers change with every pipeline run).
    dirty = git("status", "--porcelain", "--untracked-files=no", "--",
                "src", "configs", "pyproject.toml", "uv.lock")  # fmt: skip
    data = yaml.safe_load(_project_path("data/raw.dvc").read_text())["outs"][0]["md5"]
    return {"git_commit": commit, "git_dirty": str(bool(dirty)), "data_md5": data}


def _project_path(relative: str) -> Path:
    from carbon_intensity_forecast.config import project_path

    return project_path(relative)
