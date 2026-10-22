"""Model registry operations beyond training: show the aliases, and roll the champion back
to the version it replaced.

python -m carbon_intensity_forecast.registry status
python -m carbon_intensity_forecast.registry rollback
python -m carbon_intensity_forecast.registry promote VERSION   (a manual decision, recorded)
"""

import os
import sys
from datetime import UTC, datetime

import mlflow
from mlflow import MlflowClient

from carbon_intensity_forecast.config import load_config


def client_for(cfg: dict) -> MlflowClient:
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"]))
    return MlflowClient()


def status(client: MlflowClient, name: str) -> None:
    for alias in ["champion", "challenger"]:
        try:
            v = client.get_model_version_by_alias(name, alias)
            print(f"{alias}: version {v.version}, trained through {v.tags.get('trained_through')}, "
                  f"holdout MAE {v.tags.get('holdout_mae_challenger')}")  # fmt: skip
        except mlflow.exceptions.MlflowException:
            print(f"{alias}: none")


def rollback(client: MlflowClient, name: str) -> str:
    """Move `champion` back to the version the current champion replaced."""
    current = client.get_model_version_by_alias(name, "champion")
    previous = current.tags.get("previous_champion")
    if not previous:
        raise SystemExit(
            f"version {current.version} replaced no earlier champion: nothing to roll back to"
        )
    client.set_registered_model_alias(name, "champion", previous)
    stamp = f"{datetime.now(UTC):%Y-%m-%dT%H:%M:%SZ}"
    client.set_model_version_tag(name, current.version, "rolled_back_at", stamp)
    client.set_model_version_tag(name, previous, "restored_at", stamp)
    print(f"champion: version {current.version} -> version {previous}")
    return previous


def promote(client: MlflowClient, name: str, version: str) -> None:
    """Make `version` the champion by hand, bypassing the gates; recorded on the version."""
    try:
        current = client.get_model_version_by_alias(name, "champion").version
    except mlflow.exceptions.MlflowException:
        current = None
    client.set_registered_model_alias(name, "champion", version)
    stamp = f"{datetime.now(UTC):%Y-%m-%dT%H:%M:%SZ}"
    client.set_model_version_tag(name, version, "promoted_manually_at", stamp)
    if current and current != version:
        client.set_model_version_tag(name, version, "previous_champion", current)
    print(f"champion: version {current} -> version {version} (manual)")


def main() -> None:
    cfg = load_config()
    client = client_for(cfg)
    name = cfg["registry"]["model_name"]
    command, *args = sys.argv[1:]
    {"status": status, "rollback": rollback, "promote": promote}[command](client, name, *args)


if __name__ == "__main__":
    main()
