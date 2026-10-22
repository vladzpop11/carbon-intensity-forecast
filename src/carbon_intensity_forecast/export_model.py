"""Export the champion from the MLflow registry to a folder, for serving where MLflow is
not reachable (the Kubernetes deployment, CI): MODEL_URI=<folder>.

python -m carbon_intensity_forecast.export_model [OUT_DIR]   (default models/champion)
"""

import os
import shutil
import sys
from pathlib import Path

import mlflow

from carbon_intensity_forecast.config import load_config, project_path


def main() -> None:
    cfg = load_config()
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"]))
    name = cfg["registry"]["model_name"]
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else project_path("models/champion")
    version = mlflow.MlflowClient().get_model_version_by_alias(name, "champion").version
    if out.exists():
        shutil.rmtree(out)
    mlflow.artifacts.download_artifacts(f"models:/{name}/{version}", dst_path=str(out))
    (out / "registry_version.txt").write_text(f"{version}\n")
    print(f"champion version {version} -> {out}")


if __name__ == "__main__":
    main()
