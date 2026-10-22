"""The pipeline as Prefect flows.

- training: ingest -> validate -> version the data -> train -> evaluate -> register. Validation
  is part of ingestion (the data contracts): a batch that breaks them stops the flow before
  anything is trained on it.
- daily: ingest and version, then monitoring (issue, benchmark and score live forecasts,
  drift report, retraining decision); if a trigger fires, train; if a new champion is
  promoted, export it for the image and push it to DVC. The API reloads the champion itself.

The data version is DVC's checksum of data/raw, pushed to the DagsHub remote and recorded on
every MLflow run and model version (data_md5), so a model's exact data can be pulled again
by writing that checksum into data/raw.dvc. The flows do not commit to git.

Needs MLflow and Postgres (make up) and DagsHub credentials in .env.

python -m carbon_intensity_forecast.flows [training|daily|serve]
  serve: run `daily` now (catching up on days missed while the machine was off), then every
  day at 09:30 UTC, after NESO's forecast is archived (needs a Prefect server).
"""

import json
import os
import subprocess
import sys

import yaml
from prefect import flow, get_run_logger, task

from carbon_intensity_forecast import export_model, ingest, monitor, train
from carbon_intensity_forecast.config import project_path

DAILY_CRON = "30 9 * * *"


def dagshub_env() -> dict:
    """Environment for DVC with the DagsHub token from .env (read, never printed)."""
    values = {}
    for line in project_path(".env").read_text().splitlines():
        key, sep, value = line.partition("=")
        if sep and key.strip():
            values[key.strip()] = value.strip()
    token = values["DAGSHUB_TOKEN"]
    return {**os.environ, "AWS_ACCESS_KEY_ID": token, "AWS_SECRET_ACCESS_KEY": token}


def dvc(*args: str) -> None:
    subprocess.run(["dvc", *args], cwd=project_path("."), check=True, capture_output=True,
                   env=dagshub_env())  # fmt: skip


@task(retries=2, retry_delay_seconds=300)
def ingest_and_validate() -> None:
    """Download, check the contracts, save data/raw (raises if a contract fails)."""
    ingest.main()


@task(retries=2, retry_delay_seconds=60)
def version_data() -> str:
    """Track the new data/raw with DVC and push it to the remote; -> its checksum."""
    dvc("add", "data/raw")
    dvc("push", "data/raw.dvc")
    return yaml.safe_load(project_path("data/raw.dvc").read_text())["outs"][0]["md5"]


@task
def train_and_register() -> bool:
    """-> whether a new champion was promoted."""
    train.main()
    return json.loads(project_path("reports/training.json").read_text())["promoted"]


@task(retries=1, retry_delay_seconds=60)
def monitor_live() -> str | None:
    return monitor.main()


@task(retries=2, retry_delay_seconds=60)
def publish_champion() -> None:
    """Export the champion for the image and push it with DVC."""
    export_model.main()
    dvc("add", "models/champion")
    dvc("push", "models/champion.dvc")


@flow(name="training", log_prints=True)
def training() -> None:
    ingest_and_validate()
    get_run_logger().info(f"data version {version_data()}")
    if train_and_register():
        publish_champion()


@flow(name="daily", log_prints=True)
def daily() -> None:
    logger = get_run_logger()
    ingest_and_validate()
    logger.info(f"data version {version_data()}")
    trigger = monitor_live()
    if trigger is None:
        logger.info("no retraining")
        return
    logger.info(f"retraining: {trigger}")
    if train_and_register():
        publish_champion()
        logger.info("new champion promoted and published")
    else:
        logger.info("the challenger did not pass the gates; the champion stays")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "training"
    if command == "serve":
        daily()  # catch up first: monitoring fills in every missed issue day
        daily.serve(name="daily", cron=DAILY_CRON)
    else:
        {"training": training, "daily": daily}[command]()
