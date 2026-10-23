"""The pipeline as Prefect flows.

- training: ingest -> validate -> version the data -> train -> evaluate -> register. Validation
  is part of ingestion (the data contracts): a batch that breaks them stops the flow before
  anything is trained on it.
- daily: ingest and version, then monitoring (issue, benchmark and score live forecasts,
  drift report, retraining decision); if a trigger fires, train. Every run then makes sure
  the exported champion (for the image) is the registry's. The API reloads the champion itself.

The data version is DVC's checksum of data/raw, pushed to the DagsHub remote and recorded on
every MLflow run and model version (data_md5), so a model's exact data can be pulled again
by writing that checksum into data/raw.dvc. The flows do not commit to git.

Needs MLflow and Postgres (make up) and DagsHub credentials in .env.

python -m carbon_intensity_forecast.flows [training|daily|serve]
  serve: catch up (a snapshot if it is near 09:00 UTC; `daily`, which fills in days missed
  while the machine was off; failures are logged and do not stop the schedule), then every
  day: `snapshot` at 09:00 UTC and `daily` at 09:30 UTC, one run of each at a time (needs a
  Prefect server).
"""

import json
import logging
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta

import mlflow
import yaml
from prefect import flow, get_run_logger, serve, task

from carbon_intensity_forecast import archive, export_model, ingest, monitor, train
from carbon_intensity_forecast.config import env_value, load_config, project_path

DAILY_CRON = "30 9 * * *"
SNAPSHOT_CRON = "0 9 * * *"  # NESO's forecast and the inputs, as published at the issue time


def dvc(*args: str) -> None:
    token = env_value("DAGSHUB_TOKEN")
    env = {**os.environ, "AWS_ACCESS_KEY_ID": token, "AWS_SECRET_ACCESS_KEY": token}
    done = subprocess.run(["dvc", *args], cwd=project_path("."), capture_output=True, text=True,
                          env=env)  # fmt: skip
    if done.returncode != 0:  # DVC's own message, so a failure can be diagnosed from the logs
        raise RuntimeError(f"dvc {' '.join(args)} failed: {done.stderr.strip()[-2000:]}")


@task(retries=2, retry_delay_seconds=300)
def ingest_and_validate() -> None:
    """Download, check the contracts, save data/raw (raises if a contract fails)."""
    ingest.main()


@task(retries=2, retry_delay_seconds=60)
def version_data() -> str:
    """Track the new data/raw with DVC and push it to the remote; -> its checksum."""
    dvc("add", "data/raw")
    dvc("push", "data/raw.dvc")
    if project_path("data/archive").exists():
        dvc("add", "data/archive")
        dvc("push", "data/archive.dvc")
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
def publish_champion() -> str:
    """Export the champion for the image and push it with DVC, unless the export is already
    the registry's champion (so a failed export, a rollback or a manual promotion is caught
    up by the next run)."""
    cfg = load_config()
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"]))
    name = cfg["registry"]["model_name"]
    champion = mlflow.MlflowClient().get_model_version_by_alias(name, "champion").version
    folder = project_path("models/champion")
    if export_model.exported_version(folder) == champion:
        return f"the export is already version {champion}"
    export_model.export(folder, cfg)
    dvc("add", "models/champion")
    dvc("push", "models/champion.dvc")
    return f"exported and pushed version {champion}"


@flow(name="snapshot", log_prints=True)
def snapshot() -> None:
    """Save what is published at the issue time (archive.py) to the local archive, as the
    GitHub Action does; GitHub's scheduler can start hours late, this one runs on time
    whenever the machine is on. The folder is versioned with DVC by the daily flow."""
    out = project_path(load_config()["monitoring"]["local_archive_dir"])
    try:
        archive.main(str(out))
    except SystemExit as exc:
        if exc.code:
            raise RuntimeError("a snapshot failed (the others were saved)") from exc


@flow(name="training", log_prints=True)
def training() -> None:
    ingest_and_validate()
    get_run_logger().info(f"data version {version_data()}")
    train_and_register()
    get_run_logger().info(publish_champion())


@flow(name="daily", log_prints=True)
def daily() -> None:
    logger = get_run_logger()
    ingest_and_validate()
    logger.info(f"data version {version_data()}")
    trigger = monitor_live()
    if trigger is None:
        logger.info("no retraining")
    else:
        logger.info(f"retraining: {trigger}")
        promoted = train_and_register()
        logger.info("new champion promoted" if promoted
                    else "the challenger did not pass the gates; the champion stays")  # fmt: skip
    logger.info(publish_champion())


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "training"
    if command == "serve":
        log = logging.getLogger(__name__)
        now = datetime.now(UTC)
        issue_time = now.replace(hour=load_config()["task"]["issue_hour_utc"], minute=0,
                                 second=0, microsecond=0)  # fmt: skip
        # Catch up first. A snapshot is only worth taking near the issue time; monitoring
        # fills in every missed issue day. Failures are logged; the schedule starts anyway.
        near_issue_time = abs(now - issue_time) <= timedelta(hours=1)
        for name, run, wanted in [("snapshot", snapshot, near_issue_time), ("daily", daily, True)]:
            if wanted:
                try:
                    run()
                except Exception:  # noqa: BLE001
                    log.exception("the catch-up %s run failed; serving anyway", name)
        serve(
            snapshot.to_deployment(name="snapshot", cron=SNAPSHOT_CRON, concurrency_limit=1),
            daily.to_deployment(name="daily", cron=DAILY_CRON, concurrency_limit=1),
        )
    else:
        {"training": training, "daily": daily}[command]()
