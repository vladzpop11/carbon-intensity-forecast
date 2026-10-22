"""Export the champion from the MLflow registry to a folder, for serving where MLflow is
not reachable (the image, Kubernetes, CI): MODEL_URI=<folder>.

The folder also records the registry version and the last day the model was trained
through (the API refuses issue days up to it). It is replaced atomically: the new export is
downloaded next to it and renamed into place, so a failed download leaves the old one.

python -m carbon_intensity_forecast.export_model [OUT_DIR]   (default models/champion)
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

import mlflow

from carbon_intensity_forecast.config import load_config, project_path


def exported_version(folder: Path) -> str | None:
    meta = folder / "registry_version.txt"
    return meta.read_text().strip() if meta.exists() else None


def export(out: Path, cfg: dict) -> str:
    name = cfg["registry"]["model_name"]
    v = mlflow.MlflowClient().get_model_version_by_alias(name, "champion")
    staging = Path(tempfile.mkdtemp(prefix=".champion-", dir=out.parent))
    try:
        mlflow.artifacts.download_artifacts(f"models:/{name}/{v.version}", dst_path=str(staging))
        (staging / "registry_version.txt").write_text(f"{v.version}\n")
        (staging / "trained_through.txt").write_text(f"{v.tags['trained_through']}\n")
        old = out.with_name(out.name + ".old")
        if out.exists():
            out.rename(old)
        staging.rename(out)
        shutil.rmtree(old, ignore_errors=True)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return v.version


def main() -> None:
    cfg = load_config()
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"]))
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else project_path("models/champion")
    print(f"champion version {export(out, cfg)} -> {out}")


if __name__ == "__main__":
    main()
